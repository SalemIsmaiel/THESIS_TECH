import asyncio
import sys

from predictor import Predictor
from preprocessors.googleplaystore import GooglePlayStorePreprocessor
from preprocessors.microsoftstore import MicrosoftStorePreprocessor


async def train_playstore():
    preprocessor = GooglePlayStorePreprocessor()

    dataset = await preprocessor.read_dataset("dataset/Clean-ContextualData22Values.csv", limit=1000)
    print(dataset)

if __name__ == '__main__':
    asyncio.run(train_playstore())


