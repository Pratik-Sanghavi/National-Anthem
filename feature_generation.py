from national_anthem_feature_engineering.getAudioFeatures import AudioFeatureGeneration

DATA_PATH = 'national_anthem_scrape/national_anthem_dataset/audio_files'
OUTFILE = 'national_anthem_scrape/national_anthem_dataset/features.csv'

aud_features = AudioFeatureGeneration(data_path=DATA_PATH, out_file=OUTFILE)
print(aud_features.getDataset())