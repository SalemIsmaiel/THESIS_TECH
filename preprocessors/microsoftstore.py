import asyncio
import os
from typing import Optional

import aiohttp


import pandas as pd

from preprocessors.preprocesser import Preprocessor
from scrapers.microsoftstore import MicrosoftStoreScraper


class MicrosoftStorePreprocessor(Preprocessor):
    def __init__(self):
        super().__init__()

    store_scraper = MicrosoftStoreScraper()

    def read_dataset(self, file_path: str):
        if os.path.exists("preprocessed/preprocessed_ms.pkl"):
            print("Loading file... : preprocessed_ms.pkl")

            return pd.read_pickle("preprocessed/preprocessed_ms.pkl")

        print("Reading dataset... : {}".format(file_path))
        dataset = pd.read_csv(file_path)

        print("Processing results...")

        output = list()
        for description, category in zip(dataset["Description"], dataset["Category"]):
            words = self.preprocess(description)

            if "nokia" in words or "lumia" in words:
                continue

            output.append({"package": None,
                           "description": words,
                           "category": category
                           })

        print("Writing DataFrame...")
        df = pd.DataFrame(output, columns=["package", "description", "category"])

        print("Saving file... : preprocessed_ms.csv")
        df.to_csv("preprocessed/preprocessed_ms.csv")

        print("Saving file... : preprocessed_ms.pkl")
        df.to_pickle("preprocessed/preprocessed_ms.pkl")

        return df

    def split_by_genre(self, genre: str, sub_genres: [str], df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if os.path.exists("preprocessed/preprocessed_ms_{}.pkl".format(genre)):
            print("Loading file... : preprocessed_ms_{}.pkl".format(genre))
            return pd.read_pickle("preprocessed/preprocessed_ms_{}.pkl".format(genre))

        if df is None:
            print("Loading file... : preprocessed_ms.pkl")
            df = self.read_dataset("preprocessed/preprocessed_ms.pkl")

        print("Splitting genre from DataFrame...")
        df_category = df[df.category.isin(sub_genres)]

        print("Saving file... : preprocessed_ms_{}.csv".format(genre))

        print(genre)
        df_category.to_csv("preprocessed/preprocessed_ms_{}.csv".format(genre))

        print("Saving file... : preprocessed_ms_{}.pkl".format(genre))
        df_category.to_pickle("preprocessed/preprocessed_ms_{}.pkl".format(genre))

        return df_category
