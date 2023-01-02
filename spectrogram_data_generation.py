import torchaudio
import os

DATA_PATH = 'national_anthem_scrape/national_anthem_dataset/audio_files'
files = os.listdir(DATA_PATH)
for file in files:
    metadata = torchaudio.info(os.path.join(DATA_PATH, file))
    print(metadata.sample_rate)
    break
