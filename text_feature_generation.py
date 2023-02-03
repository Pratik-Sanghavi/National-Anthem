from national_anthem_feature_engineering.getTextFeatures import TextFeatureGeneration
from settings import text_data_path, text_feat_write_path
COUNTRY_COLUMN = 'Country'
ANTHEM_COLUMN = 'English_Translation'

text_features = TextFeatureGeneration(data_path=text_data_path, country_column=COUNTRY_COLUMN, anthem_column=ANTHEM_COLUMN)
text_features.textFeaturization(max_features = 20)
text_features.writeFile(out_file_path = text_feat_write_path)