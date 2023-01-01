from selenium import webdriver
from selenium.webdriver.common.by import By
from tqdm import tqdm
import os
import national_anthem_scrape.constants as constants
import time
import pandas as pd

class NationalAnthem(webdriver.Chrome):
    def __init__(self,
                 driver_path=r'C:/Users/Pratik Sanghavi/Desktop/Drivers/ChromeDriver',
                 teardown=False,
                 download_directory = "C:\\Users\\Pratik Sanghavi\\Desktop\\Projects\\National_Anthem\\national_anthem_scrape\\national_anthem_dataset\\audio_files"
                 ):
        self.driver_path = driver_path
        self.teardown = teardown
        self.download_directory = download_directory
        os.environ["PATH"] = driver_path
        options = webdriver.ChromeOptions()
        prefs = {"download.default_directory": download_directory, "download.directory_upgrade": True, "download.prompt_for_download": False, "safebrowsing.enabled":True}
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        options.add_experimental_option("prefs", prefs)
        options.add_argument("--headless")
        # options.add_experimental_option("detach", True)
        super(NationalAnthem, self).__init__(options=options)
        self.implicitly_wait(10)
        self.maximize_window()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.teardown:
            self.quit()
    
    def land_first_page(self):
        self.get(constants.BASE_URL)
    
    def get_menu_link(self):
        sub_menus = self.find_elements(
            By.CSS_SELECTOR,
            'ul[class="sub-menu"]'
        )
        links = []
        for sub_menu in tqdm(sub_menus):
            for link_list in sub_menu.find_elements(By.TAG_NAME,'li'):
                for link in link_list.find_elements(By.TAG_NAME, 'a'):
                    links.append(link.get_attribute('href'))
        return list(set(links))

    def download_music(self, links):
        names = []
        mp3_files = []
        for link in tqdm(links):
            self.get(link)
            try:
                music_link = self.find_element(By.CSS_SELECTOR, 'a[id^="mp3"]')
                title = self.find_element(By.TAG_NAME, 'h1')
                music_link.click()
                names.append(title.get_attribute('innerHTML'))
                mp3_files.append(music_link.get_attribute('href').split('/')[-1])
                time.sleep(5)
            except:
                print(f"Not present for: {link}")
        key_df = pd.DataFrame({
            'Country': names,
            'Audio_File': mp3_files
        })
        key_df['File_Location'] = self.download_directory.replace("\\", "/") + "/" + key_df["Audio_File"]
        key_location = self.download_directory.replace("\\", "/").replace("audio_files", "")
        key_df.to_csv(f'{key_location}/key.csv', index = False)