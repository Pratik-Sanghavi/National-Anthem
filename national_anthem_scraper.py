from national_anthem_scrape.national_anthem import NationalAnthem

try:
    with NationalAnthem() as bot:
        bot.land_first_page()
        links = bot.get_menu_link()
        bot.download_music(links)
except Exception as e:
    print(f"Exception: \n{e}\nencountered! Please fix before rerunning.")