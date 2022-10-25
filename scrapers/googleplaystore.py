from urllib.parse import urlencode

import urllib3
from bs4 import BeautifulSoup
from urllib3 import PoolManager

from scrapers.scraper import Scraper


class GooglePlayStoreScraper(Scraper):
    def __init__(self):
        super().__init__()

    def fetch_descriptions(self, packages):
        retries = urllib3.Retry(status=10, backoff_factor=0.5, status_forcelist=[429])

        self.pool = urllib3.PoolManager(retries=retries)

        results = list()

        for _, package, genre in packages.itertuples():
            p, d = self.fetch_description_pool(self.pool, package)

            if d is not None:
                results.append((p, genre, d))

        return results

    def fetch_description_pool(self, http: PoolManager, package):
        base_url = "https://play.google.com/store/apps/details?id="
        headers = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/105.0.0.0 Safari/537.36 "
        }
        params = {
            "id": package,  # app name
            "gl": "US",  # country of the search
            "hl": "en_US"  # language of the search
        }

        response = http.request("GET", "https://play.google.com/store/apps/details", fields=params, headers=headers)

        if response.status == 429:
            print("HTTP: 429")
            return package, None
        elif response.status >= 400:
            return package, None

        soup = BeautifulSoup(response.data.decode("utf-8"), "lxml")

        description = soup.find_all("div", {"data-g-id": "description"})[0].text

        return package, description

    async def fetch_description(self, session, package):
        base_url = "https://play.google.com/store/apps/details?id="
        headers = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/105.0.0.0 Safari/537.36 "
        }
        params = {
            "id": package,  # app name
            "gl": "US",  # country of the search
            "hl": "en_US"  # language of the search
        }

        async with session.get("https://play.google.com/store/apps/details", params=params,
                               headers=headers) as response:
            if response.status == 429:
                return package, None
            elif response.status >= 400:
                return package, None

            soup = BeautifulSoup(await response.text(), "lxml")

            description = soup.find_all("div", {"data-g-id": "description"})[0].text

            return package, description
