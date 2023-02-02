from national_anthem_feature_engineering.getTextFeatures import TextFeatureGeneration

DATA_PATH = 'national_anthem_scrape/national_anthem_dataset/nationalanthems_lyrics_and_flag.csv'
COUNTRY_COLUMN = 'nation'
ANTHEM_COLUMN = 'lyrics_english'
OUTFILE = 'national_anthem_scrape/national_anthem_dataset/text_features.csv'

text_features = TextFeatureGeneration(data_path=DATA_PATH, country_column=COUNTRY_COLUMN, anthem_column=ANTHEM_COLUMN)
text_features.textFeaturization()
text_features.writeFile(out_file_path = OUTFILE)