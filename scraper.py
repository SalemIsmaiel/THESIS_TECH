import sys

from preprocessors.googleplaystore import GooglePlayStorePreprocessor


def scrape_playstore():
    preprocessor = GooglePlayStorePreprocessor()

    preprocessor.read_dataset("dataset/Clean-ContextualData22Values.csv", parts=10, part=int(sys.argv[1]))


if __name__ == '__main__':
    scrape_playstore()
