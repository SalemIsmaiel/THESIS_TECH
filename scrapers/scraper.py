import requests
from bs4 import BeautifulSoup
import urllib3


class Scraper:
    pool: urllib3.PoolManager = urllib3.PoolManager()

    def __init__(self):
        pass


# https://requests.readthedocs.io/en/latest/user/quickstart/#custom-headers
headers = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36"
}

base_url = "https://play.google.com/store/apps/details?id="


def fetch_description(package):
    params = {
        "id": package,  # app name
        "gl": "US",  # country of the search
        "hl": "en_US"  # language of the search
    }

    html = requests.get("https://play.google.com/store/apps/details", params=params, headers=headers, timeout=30)

    if html.status_code >= 400:
        print(str(html.status_code) + ": https://play.google.com/store/apps/details?id=" + package)
        return None

    soup = BeautifulSoup(html.text, "lxml")

    description = soup.find_all("div", {"data-g-id": "description"})[0].text

    return description
