from selenium import webdriver
from selenium.webdriver.common.by import By
from tqdm import tqdm
import os
import national_anthem_scrape.constants as constants
from datetime import datetime

class NationalAnthem(webdriver.Chrome):
    def __init__(self, driver_path=r'C:/Users/Pratik Sanghavi/Desktop/Drivers/ChromeDriver', teardown=False):
        self.driver_path = driver_path
        self.teardown = teardown
        self.links = None
        os.environ["PATH"] = driver_path
        options = webdriver.ChromeOptions()
        prefs = {"download.default_directory": "C:/Users/Pratik Sanghavi/Desktop/Projects/National_Anthem_Dataset/national_anthem_scrape/national_anthem_dataset"}
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        options.add_experimental_option("prefs", prefs)
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
        
