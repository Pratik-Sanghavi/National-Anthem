from national_anthem_feature_engineering.getAudioFeatures import AudioFeatureGeneration
from settings import audio_data_path, audio_feat_write_path

aud_features = AudioFeatureGeneration(data_path=audio_data_path)
aud_features.writeFile(out_file=audio_feat_write_path)