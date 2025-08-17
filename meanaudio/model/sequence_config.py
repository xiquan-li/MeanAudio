import dataclasses
import math


@dataclasses.dataclass
class SequenceConfig:
    # general
    duration: float

    # audio
    sampling_rate: int
    spectrogram_frame_rate: int
    latent_downsample_rate: int = 2

    @property
    def num_audio_frames(self) -> int:
        # we need an integer number of latents
        return self.latent_seq_len * self.spectrogram_frame_rate * self.latent_downsample_rate

    @property
    def latent_seq_len(self) -> int:
        return int(
            math.ceil(self.duration * self.sampling_rate / self.spectrogram_frame_rate /
                      self.latent_downsample_rate))

CONFIG_16K = SequenceConfig(duration=9.975, sampling_rate=16000, spectrogram_frame_rate=256)  # !TODO fix sequnce config here -> Latent length = 312
CONFIG_44K = SequenceConfig(duration=9.975, sampling_rate=44100, spectrogram_frame_rate=512)
CONFIG_44K_30 = SequenceConfig(duration=29.975, sampling_rate=44100, spectrogram_frame_rate=512)

if __name__ == '__main__':
    # assert CONFIG_16K.latent_seq_len == 312
    # assert CONFIG_16K.clip_seq_len == 64
    # assert CONFIG_16K.sync_seq_len == 192
    # assert CONFIG_16K.num_audio_frames == 128000  # 312 * 256 * 2

    print(CONFIG_44K_30.latent_seq_len)
