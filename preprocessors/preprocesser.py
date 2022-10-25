import asyncio
import csv
import os

import nltk
from bs4 import BeautifulSoup
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re


class Preprocessor:
    def __init__(self):
        nltk.download('wordnet')
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('omw-1.4')

    common_words = ["get", "use", "app", "apps", "game", "free", "games", "play", "new", "features", "www", "com",
                    "like", "do", "n't", "one", "want", "need", "help", "support", "contact", "email", "website", "web",
                    "site", "page", "pages", "'s", "http", "https"]

    url_pattern = r'http\S+'
    email_pattern = r'\S*@\S*\s?'

    lemmatizer = WordNetLemmatizer()

    def is_email_or_url(self, word: str):
        return re.search(self.url_pattern, word) or re.search(self.email_pattern, word)

    def preprocess(self, text: str) -> list[str]:
        word_tokens = self.tokinize(text)
        filtered = self.remove_stopwords(word_tokens)
        lemmatized = self.lemmatize(filtered)

        return lemmatized

    def lemmatize(self, word_tokens: list[str]) -> list[str]:
        return [self.lemmatizer.lemmatize(w) for w in word_tokens]

    def tokinize(self, text: str) -> list[str]:
        word_tokens = word_tokenize(text.lower())

        return word_tokens

    def remove_stopwords(self, word_tokens: list[str]) -> list[str]:
        stop_words = set(stopwords.words('english')).union(self.common_words)

        filtered = [w for w in word_tokens if not w and any(c.isalpha() for c in w) in stop_words]

        for w in word_tokens:
            if not self.is_email_or_url(w) and w not in stop_words and any(c.isalpha() for c in w):
                filtered.append(w)

        return filtered
