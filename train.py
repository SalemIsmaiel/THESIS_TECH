import asyncio
import os.path
import sys
from typing import List, Dict

from predictor import Predictor
from preprocessors.googleplaystore import GooglePlayStorePreprocessor
from preprocessors.microsoftstore import MicrosoftStorePreprocessor


def print_model_info(predictor: Predictor):
    model = predictor.get_lda_model()

    print(model)
    print(model.print_topics())
    print(model.get_topics())
    print(model.num_topics)

    coherence = predictor.get_coherence_model()

    print(coherence)
    print(coherence.get_coherence())
    print(coherence.get_coherence_per_topic())


async def train_playstore(is_test: bool):
    preprocessor = GooglePlayStorePreprocessor()

    predictor = Predictor("models/playstore")

    if predictor.get_lda_model() is None:
        print("Preprocessing data set...")
        dataset = await preprocessor.read_dataset("dataset/Clean-ContextualData22Values.csv", limit=1000)

        print(dataset)

        print("Training model")

        predictor.train_lda_model(dataset["description"], passes=1)
        predictor.train_coherence_model(dataset["description"])

    print_model_info(predictor)

    predictor.save_lda_topics("topics_ps.csv")


def train_microsoft_store(is_test: bool = False):
    preprocessor = MicrosoftStorePreprocessor()

    predictor = Predictor("models/microsoftstore")

    if predictor.get_lda_model() is None:
        print("Preprocessing data set...")

        dataset = preprocessor.read_dataset("dataset/bjarkarim.csv")

        print(dataset)

        print("Training model")

        if is_test:
            subset = dataset["description"].head(1000)
            predictor.train_lda_model(subset, passes=1)
            predictor.train_coherence_model(subset)
        else:
            training_set = dataset["description"]

            predictor.train_lda_model(training_set)
            predictor.train_coherence_model(training_set)

    print_model_info(predictor)

    predictor.save_lda_topics("topics_ms.csv")


MICROSOFT_STORE_CATEGORIES: List[Dict[str, List[str]]] = [
    {"games": ["Action & adventure", "Puzzle & trivia", "Card & board", "Family & kids", "Racing & flying", "Shooter",
               "Simulation", "Multi-Player Online Battle Arena", "Role playing", "Classics", "Fighting", "Strategy",
               "Platformer", "Tools", "Other"]},
    {"entertainment": ["Entertainment"]},
    {"music": ["Music"]},
    {"productivity": ["Productivity"]},
    {"photo": ["Photo & video"]},
    {"books": ["Books & reference", "Books & reference  >  Reference", "Books & reference  >  Fiction",
               "Books & reference  >  Non-fiction", "Books & reference  >  E-reader"]},
    {"sports": ["Sports"]},
    {"health": ["Health & fitness"]},
    {"social": ["Social"]},
    {"kids": ["Kids & family"]},
    {"travel": ["Travel", "Travel  >  Hotels", "Travel  >  City guides"]},
    {"lifestyle": ["Lifestyle", "Lifestyle  >  Special interest", "Lifestyle  >  Style & fashion",
                   "Lifestyle  >  Home & garden", "Lifestyle  >  Automotive", "Lifestyle  >  Relationships",
                   "Lifestyle  >  DIY", "LifeStyle"]},
    {"education": ["Education", "Educational", "Education  >  Instructional tools", "Education  >  Early learning",
                   "Education  >  Books & reference", "Education  >  Study guides", "Education  >  Language"]},
    {"maps": ["Navigation & maps"]},
    {"food": ["Food & dining"]},
    {"news": ["News & weather", "News & weather  >  News", "News & weather  >  Weather"]},
    {"medical": ["Medical"]},
    {"shopping": ["Shopping"]},
    {"casino": ["Casino"]},
    {"finance": ["Personal finance", "Personal finance  >  Banking & investments",
                 "Personal finance  >  Budgeting & taxes"]},
    {"business": ["Business", "Business  >  Sales & marketing", "Business  >  Accounting & finance",
                  "Business  >  Collaboration", "Business  >  Data & analytics", "Business  >  Project management",
                  "Business  >  CRM", "Business  >  Time & expenses", "Business  >  Remote desktop",
                  "Business  >  Inventory & logistics", "Business  >  File management", "Business  >  Legal & HR"]},
    {"design": ["Multimedia design", "Multimedia design  >  Illustration & graphic design",
                "Multimedia design  >  Photo & video production", "Multimedia design  >  Music production"]},
    {"developer": ["Developer tools", "Developer tools  >  Utilities", "Developer tools  >  Design tools",
                   "Developer tools  >  Development kits", "Developer tools  >  Networking",
                   "Developer tools  >  Servers", "Developer tools  >  Web hosting",
                   "Developer tools  >  Device Portal Providers", "Developer tools  >  Database",
                   "Developer tools  >  Reference & training"]},
    {"security": ["Security", "Security  >  PC protection", "Security  >  Personal security"]},
    {"personalization": ["Personalization", "Personalization  >  Ringtones & sounds", "Personalization  >  Fonts",
                         "Personalization  >  Local Experience Packs", "Personalization  >  Stickers",
                         "Personalization  >  Wallpaper & lock screens", "Personalization  >  Themes"]}
]


def train_microsoft_store_genre(category=None, is_test: bool = False):
    preprocessor = MicrosoftStorePreprocessor()

    print("Preprocessing data set...")
    dataset = preprocessor.read_dataset("dataset/bjarkarim.csv")

    print("Processing per genre...")
    categories = list()

    if category is None:
        categories = MICROSOFT_STORE_CATEGORIES
    else:
        categories.append(MICROSOFT_STORE_CATEGORIES[category])

    for category in categories:
        category_name = list(category.keys())[0]
        sub_categories = category[category_name]

        print("Genre: {}, sub-genres {}".format(category_name, str(sub_categories)))
        dataset_for_category = preprocessor.split_by_genre(genre=category_name, sub_genres=sub_categories, df=dataset)

        print("Training model")
        predictor = Predictor("models/microsoftstore_{}".format(category_name))

        if is_test:
            subset = dataset_for_category["description"].head(100)

            predictor.train_lda_model(subset, num_topics=20, passes=1)
            predictor.train_coherence_model(subset)
        else:
            training_set = dataset_for_category["description"]

            predictor.train_lda_model(training_set, num_topics=20)
            predictor.train_coherence_model(training_set)

        predictor.save_lda_topics("results/topics_ms_{}.csv".format(category_name))


def parse_args():
    if not os.path.exists("models"):
        os.mkdir("models")

    if not os.path.exists("preprocessed"):
        os.mkdir("preprocessed")

    if not os.path.exists("results"):
        os.mkdir("results")

    playstore = False
    microsoftStore = False
    test = False

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == "playstore":
                playstore = True
            elif arg == "microsoft":
                microsoftStore = True
            elif arg == "test":
                test = True

    return playstore, microsoftStore, test


if __name__ == '__main__':
    # playstore, microsoftStore, test = parse_args()

    cat = None

    if len(sys.argv) > 1:
        cat = int(sys.argv[1]) - 1

    train_microsoft_store_genre(category=cat)

    # if test:
    #     asyncio.run(train_microsoft_store(is_test=True))
    # else:
    #     asyncio.run(train_microsoft_store())
    # asyncio.run(train_playstore())
