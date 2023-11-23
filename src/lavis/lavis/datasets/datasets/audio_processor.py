import os
import math
import torch
import torchaudio
from torch import Tensor
from random import randint, uniform, betavariate
from omegaconf import OmegaConf
from torchaudio.transforms import TimeMasking, FrequencyMasking

# from utilities import set_logger

# log = set_logger(__name__, level="debug")


class fbankProcessor:
    r"""Processor for fbank extraction."""
    _OUTPUT_FORMAT_ = ["waveform", "fbank"]

    def __init__(
        self,
        audio_config: OmegaConf,
        summary_processor: bool = True,
    ) -> None:
        self.sampling_rate = audio_config.get("sampling_rate")
        self.n_mels = audio_config.get("n_mels")
        self.target_length = audio_config.get("target_length")
        self.num_mels = audio_config.get("num_mels")

        self.resample = audio_config.get("resample")
        self.normalize = audio_config.get("normalize")

        self.timem = audio_config.get("timem")
        self.freqm = audio_config.get("freqm")
        self.mixup = audio_config.get("mixup")
        self.add_noise = audio_config.get("add_noise")

        if summary_processor:
            self.summary_processor()

    def __call__(self, filename, filename2=None):
        return self.extract_features(filename, filename2)

    def extract_features(self, filename, filename2=None):
        r"""Return (`waveform`, `fbank`, `mix_lamda`)."""
        # FIXME: enable resample in fbank extraction
        wav, _ = self.load_wav(filename)
        wav2 = self.load_wav(filename2)[0] if filename2 != None else None

        fbank, mix_lamda = self.wav2fbank(wav, wav2)

        # Cut and pad by `target_length - n_frames`
        fbank = self.pad_or_clip_fbank(fbank, self.target_length)
        r"""Data augmentation should be turned off in the evaluation."""
        # SpacAug
        if self.timem != 0 or self.freqm != 0:
            fbank = torch.transpose(fbank, 0, 1)
            fbank = fbank.unsqueeze(0)

            try:
                fbank = self._time_mask_fbank(fbank, self.timem)
            except TypeError:
                pass

            try:
                fbank = self._frequency_mask_fbank(fbank, self.freqm)
            except TypeError:
                pass

            fbank = fbank.squeeze(0)
            fbank = torch.transpose(fbank, 0, 1)

        # Adding noise
        if self.add_noise:
            fbank = self._add_noise(fbank)
            fbank = self._time_roll(fbank, self.target_length)

        return wav, fbank.unsqueeze(dim=0), mix_lamda

    def load_wav(
        self,
        wav_file: str,
        resample: bool = True,
        normalize: bool = True,
    ) -> list:
        r"""Return (torch.Tensor, float), Tensor shape = (c, n_temporal_step)."""
        audio, sr = torchaudio.load(wav_file)
        # Resample the audio if `resample` = True
        if resample and (sr != self.sampling_rate):
            audio = torchaudio.functional.resample(
                audio,
                orig_freq=sr,
                new_freq=self.sampling_rate,
            )
            sr = self.sampling_rate  # point to the current sampling rate.

        if normalize:
            try:
                audio = self._normalize_wav(audio)
            except RuntimeError as e:
                # log.warning(f"{e}: {wav_file} is empty.")
                print(f"{e}: {wav_file} is empty.")

        return audio, sr

    @staticmethod
    def _normalize_wav(waveform: Tensor, eps=torch.tensor(1e-8)):
        r"""Return wavform with mean=0, std=0.5."""
        waveform = waveform - waveform.mean()
        waveform = waveform / torch.max(waveform.abs() + eps)

        return waveform * 0.5  # manually limit the maximum amplitude into 0.5

    @staticmethod
    def mix_wavs(waveform1, waveform2, alpha=10, beta=10):
        mix_lambda = betavariate(alpha, beta)
        mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2

        return __class__._normalize_wav(mix_waveform), mix_lambda

    @staticmethod
    def torchaudio_to_byte(
        audio: torch.Tensor,
        sampling_rate: int,
        cache_path="./.tmp.flac",
    ):
        torchaudio.save(
            filepath=cache_path,
            src=audio,
            sample_rate=sampling_rate,
        )

        with open(cache_path, "rb") as f:
            audio_stream = f.read()

        os.remove(cache_path)

        return audio_stream

    @staticmethod
    def segment_wavs(waveform, target_length, padding_mode="zeros"):
        r"""Args: `waveform` is a 2d channel-first tensor."""
        segmented_wavs = []
        n_channels, wav_length = waveform.size()
        for stt_idx in range(0, wav_length, target_length):
            end_idx = stt_idx + target_length
            if end_idx > wav_length:
                # NOTE: Drop the last seg if it is too short
                if (wav_length - stt_idx) < 0.1 * target_length:
                    break
                # Pad the last seg with the content in the previous one
                if padding_mode == "replicate":
                    segmented_wavs.append(waveform[:, -target_length:])
                else:
                    assert padding_mode == "zeros"
                    _tmp_wav = waveform[:, stt_idx:]
                    _padded_wav = torch.zeros(n_channels, wav_length)
                    _padded_wav[:, : _tmp_wav.size(dim=-1)] += _tmp_wav
                    segmented_wavs.append(_padded_wav)
            else:
                segmented_wavs.append(waveform[:, stt_idx:end_idx])

        return segmented_wavs

    @staticmethod
    def trim_wav(
        waveform,
        target_length,
        truncation="right",
    ):
        r"""Return trimed wav and the start time of the segmentation."""
        assert truncation in ["left", "right", "random"]

        waveform_length = waveform.shape[-1]
        assert waveform_length > 100, "Waveform is too short, %s" % waveform_length

        # Too short
        if (waveform_length - target_length) <= 0:
            return waveform, 0

        if truncation == "left":
            start_index = waveform_length - target_length
        elif truncation == "right":
            start_index = 0
        else:
            start_index = randint(0, waveform_length - target_length)
        return waveform[:, start_index : start_index + target_length], start_index

    @staticmethod
    def pad_wav(waveform, target_length, pad_last=True):
        waveform_length = waveform.shape[-1]
        assert waveform_length > 100, "Waveform is too short, {waveform_length}"

        if waveform_length == target_length:
            return waveform

        # Pad
        output_wav = torch.zeros((1, target_length), dtype=torch.float32)

        if not pad_last:
            rand_start = randint(0, target_length - waveform_length)
        else:
            rand_start = 0

        output_wav[:, rand_start : rand_start + waveform_length] = waveform
        return output_wav

    @staticmethod
    def detect_active_wav(waveform):
        if waveform.abs().max() < 0.0001:
            return waveform

        def detect_leading_silence(waveform, threshold=0.0001):
            chunk_size = 1000
            waveform_length = waveform.shape[0]

            start = 0
            while start + chunk_size < waveform_length:
                if waveform[start : start + chunk_size].abs().max() < threshold:
                    start += chunk_size
                else:
                    break

            return start

        def detect_ending_silence(waveform, threshold=0.0001):
            chunk_size = 1000
            waveform_length = waveform.shape[0]
            start = waveform_length

            while start - chunk_size > 0:
                if waveform[start - chunk_size : start].abs().max() < threshold:
                    start -= chunk_size
                else:
                    break

            if start == waveform_length:
                return start
            else:
                return start + chunk_size

        start = detect_leading_silence(waveform)
        end = detect_ending_silence(waveform)

        return waveform[start:end]

    def wav2fbank(self, wav, wav2=None):
        if wav2 != None:  # w/ mixup
            if wav.shape[1] != wav2.shape[1]:
                wav_length = wav.shape[1]
                if wav2.shape[1] < wav_length:
                    wav2 = self.pad_wav(wav2, wav_length, pad_last=True)
                else:
                    # Cutting
                    wav2, _ = self.random_trim_wav(wav2, wav_length)

            wav, mix_lamda = self.mix_wavs(wav, wav2)

        try:
            fbank = torchaudio.compliance.kaldi.fbank(
                wav,
                htk_compat=True,
                sample_frequency=self.sampling_rate,
                use_energy=False,
                window_type="hanning",
                num_mel_bins=self.n_mels,
                dither=0.0,
                frame_shift=10,
            )
        except AssertionError as e:
            fbank = torch.zeros([self.target_length, self.n_mels]) + 0.01
            # log.warning(f"A empty fbank loaded as {e}.")

        return fbank, (None if wav2 == None else mix_lamda)

    @staticmethod
    def _pad_fbank(fbank, padding_length):
        m = torch.nn.ZeroPad2d((0, 0, 0, padding_length))
        return m(fbank)

    @staticmethod
    def _clip_fbank(fbank, target_length):
        return fbank[0:target_length, :]

    @staticmethod
    def pad_or_clip_fbank(fbank, target_length):
        p = target_length - fbank.shape[0]  # target_length - n_frames
        if p > 0:
            return __class__._pad_fbank(fbank, p)
        else:
            return __class__._clip_fbank(fbank, target_length)

    @staticmethod
    def fbank_splits(fbank, target_length):
        n_frames = fbank.size(dim=1)

        splits = torch.tensor_split(
            fbank,
            math.ceil(n_frames / target_length),
            dim=1,
        )

        segments = []
        for s in splits:
            p = target_length - s.size(dim=1)
            if p > 0:
                segments.append(__class__._pad_fbank(fbank, p))
            else:
                segments.append(s)

        return segments

    @staticmethod
    def _add_noise(fbank, noise_magnitude=uniform(0, 1) * 0.1):
        d0, d1 = fbank.size()
        return fbank + torch.rand(d0, d1) * noise_magnitude

    @staticmethod
    def _time_roll(fbank, rolling_step=None):
        return torch.roll(fbank, randint(-rolling_step, rolling_step - 1), 0)

    @staticmethod
    def _time_mask_fbank(fbank, timem=0):
        m = TimeMasking(timem)
        return m(fbank)

    @staticmethod
    def _frequency_mask_fbank(fbank, freqm=0):
        m = FrequencyMasking(freqm)
        return m(fbank)

    @classmethod
    def build_processor(cls, audio_configs: OmegaConf = None):
        if audio_configs is None:
            audio_configs = OmegaConf.create()

        audio_configs["sampling_rate"] = audio_configs.get("sampling_rate", 32000)
        audio_configs["n_mels"] = audio_configs.get("n_mels", 128)
        audio_configs["target_length"] = audio_configs.get("target_length", 8192)
        audio_configs["resample"] = audio_configs.get("resample", True)
        audio_configs["normalize"] = audio_configs.get("normalize", True)
        audio_configs["timem"] = audio_configs.get("timem", 0)
        audio_configs["freqm"] = audio_configs.get("freqm", 0)
        audio_configs["mixup"] = audio_configs.get("mixup", 0)
        audio_configs["add_noise"] = audio_configs.get("add_noise", False)

        return cls(audio_configs)

    def summary_processor(self):
        # log.info(
        print(
            f"""
                Extract fbank from audio:
                    ---Processor information---
                    sampling rate: {self.sampling_rate}.
                    resample waveform after loading: {self.resample}.
                    normalize after loading: {self.normalize}.
                    fbank temporal length: {self.target_length}. 
                    number of mel bins: {self.n_mels}.
                    ---Audio augmentation---
                    spectrum masking: {self.timem:d} time, {self.freqm:d} freq.
                    mix-up with rate: {self.mixup:f}'.
                    adding noise: {self.add_noise}.
                    """
                        )


if __name__ == "__main__":
    audio_processor = fbankProcessor.build_processor({})
    print(audio_processor)
