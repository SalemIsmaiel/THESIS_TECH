import os

import pandas as pd
import numpy as np

from preprocessors.preprocesser import Preprocessor
from scrapers.googleplaystore import GooglePlayStoreScraper


class GooglePlayStorePreprocessor(Preprocessor):
    def __init__(self):
        super().__init__()

    store_scraper = GooglePlayStoreScraper()

    def read_dataset(self, file_path: str, parts: int = 1, part: int = 0):
        if os.path.exists("preprocessed/preprocessed.pkl"):
            print("Loading preprocessed dataset...")
            return pd.read_pickle("preprocessed/preprocessed.pkl")

        print("Reading dataset...")

        dataset = pd.read_csv(file_path)

        print("Fetching description...")

        packages = dataset[["pkgname", "Genre"]]

        if parts > 1:
            packages = np.array_split(packages, parts)[part]

        print("Processing results...")

        output = list()

        results = self.store_scraper.fetch_descriptions(packages)

        for package, genre, description in results:
            output.append({"package": package,
                           "genre": genre,
                           "description": self.preprocess(description)
                           })

        print("Writing DataFrame...")

        df = pd.DataFrame(output, columns=["package", "genre", "description"])

        df.to_csv("preprocessed/preprocessed_{}.csv".format(part))
        df.to_pickle("preprocessed/preprocessed_{}.pkl".format(part))

    @staticmethod
    def merge_preprocessed(parts: int):
        dfs = list()

        for part in range(parts):
            dfs.append(pd.read_pickle("preprocessed/preprocessed_{}.pkl".format(part)))

        df = pd.concat(dfs)

        df.to_csv("preprocessed/preprocessed.csv")
        df.to_pickle("preprocessed/preprocessed.pkl")
