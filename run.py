import pandas as pd
from tqdm import tqdm
import os
from national_anthem_scrape.national_anthem import NationalAnthem

try:
    with NationalAnthem() as bot:
        bot.land_first_page()
        bot.get_menu_link()
except Exception as e:
    print(f"Exception: \n{e}\nencountered! Please fix before rerunning.")