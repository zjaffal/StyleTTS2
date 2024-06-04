import wandb
import torch
import pandas as pd

SAMPLE_RATE=24000

def get_speaker_mapping(train_path, test_path=None):
    """Generate an identity mapping if speaker names aren't provided

    Args:
        train_path: list of training data
        test_path: list of test data

    Returns:
        Identity mapping of speakers
    """
    speaker_mapping = {}
    train_df = pd.read_csv(train_path, sep="|", names=["audio", "text", "id"])

    unique_ids = train_df["id"].astype(str).unique().tolist()
    for id in unique_ids:
        speaker_mapping[id] = id

    if test_path:
        test_df = pd.read_csv(test_path, sep="|", names=["audio", "text", "id"])

        unique_ids = test_df["id"].astype(str).unique().tolist()
        for id in unique_ids:
            speaker_mapping[id] = id

    return speaker_mapping

def generate_samples(
    model,
    sampler,
    bib,
    ref_s,
    t_en,
    d_en,
    texts,
    bert_dur,
    input_lengths,
    text_mask,
    multispeaker,
):
    """
    Generate samples following the same logic in inference. This code is largely attributed from `train_second.py` file
    """
    if multispeaker:
        s_pred = sampler(
            noise=torch.randn((1, 256)).unsqueeze(1).to(texts.device),
            embedding=bert_dur[bib].unsqueeze(0),
            embedding_scale=1,
            features=ref_s[bib].unsqueeze(
                0
            ),  # reference from the same speaker as the embedding
            num_steps=5,
        ).squeeze(1)
    else:
        s_pred = sampler(
            noise=torch.randn((1, 256)).unsqueeze(1).to(texts.device),
            embedding=bert_dur[bib].unsqueeze(0),
            embedding_scale=1,
            num_steps=5,
        ).squeeze(1)

    s = s_pred[:, 128:]
    ref = s_pred[:, :128]

    d = model.predictor.text_encoder(
        d_en[bib, :, : input_lengths[bib]].unsqueeze(0),
        s,
        input_lengths[bib, ...].unsqueeze(0),
        text_mask[bib, : input_lengths[bib]].unsqueeze(0),
    )

    x, _ = model.predictor.lstm(d)
    duration = model.predictor.duration_proj(x)

    duration = torch.sigmoid(duration).sum(axis=-1)
    pred_dur = torch.round(duration.squeeze()).clamp(min=1)

    pred_dur[-1] += 5

    pred_aln_trg = torch.zeros(input_lengths[bib], int(pred_dur.sum().data))
    c_frame = 0
    for i in range(pred_aln_trg.size(0)):
        pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].data)] = 1
        c_frame += int(pred_dur[i].data)

    # encode prosody
    en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(texts.device)
    F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
    return model.decoder(
        (
            t_en[bib, :, : input_lengths[bib]].unsqueeze(0)
            @ pred_aln_trg.unsqueeze(0).to(texts.device)
        ),
        F0_pred,
        N_pred,
        ref.squeeze().unsqueeze(0),
    )

class SpeakerRecord:
    """Record of each speaker and their audios

    This is used when displaying items in wandb
    """

    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.true_recordings = []
        self.predicted_recordings = []

    def __len__(self):
        """length of recordings in speaker

        Returns:
            the length of the recordings, true and predicted must match
        """
        assert len(self.true_recordings) == len(self.predicted_recordings)
        return len(self.predicted_recordings)

    def add(self, true_recording, predicted_recording):
        """Add data to speaker

        Args:
            true_recording: true value
            predicted_recording: generated value
        """
        self.true_recordings.append(
            wandb.Audio(true_recording, sample_rate=SAMPLE_RATE)
        )
        self.predicted_recordings.append(
            wandb.Audio(predicted_recording, sample_rate=SAMPLE_RATE)
        )

    def get_data(self):
        """get data from object

        Returns:
            list of tuples of true and predicted recordings
        """
        return list(zip(self.true_recordings, self.predicted_recordings))

    def clear(self):
        """clear the lists"""
        self.predicted_recordings.clear()
        self.true_recordings.clear()


class AudioSampleLogger:
    """
    General class for logging audio information to wandb.
    Each table in wandb consists of columns and rows of information.
    This class is a wrapper around that functionality.

    The tables generated in wandb should have a number of recordings for each speaker
    """

    def __init__(self, table_name, speaker_mappings: dict, num_per_speaker=5):
        self.speaker_mappings = speaker_mappings
        self.speaker_data = {}
        for speaker_id, speaker_name in speaker_mappings.items():
            self.speaker_data[speaker_id] = SpeakerRecord(speaker_id, speaker_name)
        self.table_name = table_name
        self.table = wandb.Table(
            columns=["id", "Speaker", "Reference audio", "Predicted Audio"]
        )
        self.num_per_speaker = num_per_speaker

    def add_data(self, reference_audio, predicted_audio, speaker_id):
        """Add data to wandb table given for a specific speaker

        log relevant recordings to speaker

        Args:
            reference_audio: numpy array of reference audio
            predicted_audio: numpy array of generated audio
            speaker_id: speaker id for the relevant speaker
        """
        speaker_record = self.speaker_data[speaker_id]
        if len(speaker_record) < self.num_per_speaker:
            speaker_record.add(reference_audio, predicted_audio)

    def log_info(self):
        """Upload table to wandb

        Called at the end of a batch or epoch to upload the data to wandb
        and prepare the table for the next run.
        """

        # Generate the table from the speaker data logged through training/validation.
        for speaker_id, speaker in self.speaker_data.items():
            if len(speaker) == 0:
                continue
            for idx, (ref_audio, pred_audio) in enumerate(speaker.get_data()):
                # wandb specific ID.
                tb_id = f"{speaker_id}_{idx}"
                self.table.add_data(tb_id, speaker.name, ref_audio, pred_audio)
        # upload the data.
        wandb.log({self.table_name: self.table})

        # empty speaker data necessary when logging the following run.
        for _, speaker in self.speaker_data.items():
            speaker.clear()
