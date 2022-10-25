import asyncio

import aiohttp
from bs4 import BeautifulSoup

from scrapers.scraper import Scraper


class MicrosoftStoreScraper(Scraper):
    def __init__(self):
        super().__init__()

    async def fetch_description(self, package: str):
        async with aiohttp.ClientSession() as session:
            _, description = await asyncio.ensure_future(self.fetch_description_async(session, package))

            if description is None:
                return None

            return description

    async def fetch_description_async(self, session, package):
        base_url = "https://apps.microsoft.com/store/detail/{}"
        headers = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/105.0.0.0 Safari/537.36 "
        }
        params = {
            "gl": "us",  # country of the search
            "hl": "en-us"  # language of the search
        }

        async with session.get(base_url.format(package), params=params, headers=headers) as response:
            if response.status == 429:
                return package, None
            elif response.status >= 400:
                return package, None

            soup = BeautifulSoup(await response.text(), "lxml")

            description = soup.find("meta", {"name": "description"})['content']

            return package, description
