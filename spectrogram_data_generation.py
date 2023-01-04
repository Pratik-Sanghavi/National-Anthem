import torchaudio
import torchaudio.transforms as T
import librosa
import matplotlib
import matplotlib.pyplot as plt
import os

def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Spectrogram (db)')
  axs.set_ylabel(ylabel)
  axs.set_xlabel('frame')
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)
  plt.show()

DATA_PATH = 'national_anthem_scrape/national_anthem_dataset/audio_files'
n_fft = 1024
win_length = None
hop_length = 512

files = os.listdir(DATA_PATH)
for file in files[1:]:
    waveform, sample_rate = torchaudio.load(os.path.join(DATA_PATH, file))
    spectrogram = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode = 'reflect',
        power = 2.0
    )
    spec = spectrogram(waveform)
    # plot_spectrogram(spec=spec[0], title = file.split(".mp3")[0])
    break
