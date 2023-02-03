import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

BASE_PATH = os.environ.get("BASE_PATH")

# AUDIO FEATURES VARS
audio_data_path = join(BASE_PATH, 'audio_files')
audio_feat_write_path = join(BASE_PATH, 'audio_features.csv')

# TEXT FEATURES VARS
text_data_path = join(BASE_PATH, 'key.csv')
text_feat_write_path = join(BASE_PATH, 'text_features.csv')