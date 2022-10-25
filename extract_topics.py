from pprint import pprint

import gensim
from gensim import corpora
import csv
from gensim.models import CoherenceModel

import warnings

warnings.simplefilter("ignore")


class Extract_Topic():
    def __init__(self, dataset, passes, num_topics):
        self.dataset = dataset
        self.num_topics = num_topics
        self.passes = passes

        # First we create a dictionary that labels every word with unique id
        self.dictionary = corpora.Dictionary(self.dataset)

        # Then create a corpus which list the frequency of each word
        self.corpus = [self.dictionary.doc2bow(row) for row in self.dataset]

    def run_lda_model(self):
        lda_model = gensim.models.LdaMulticore(corpus=self.corpus,
                                               id2word=self.dictionary,
                                               num_topics=self.num_topics,
                                               passes=self.passes,
                                               workers=5)
        return lda_model

    def calculate_coherence_score(self, lda_model):
        coherence_model = CoherenceModel(model=lda_model, texts=self.dataset
                                         , dictionary=self.dictionary, coherence='c_v')
        print("Coherence score with " + str(self.passes) + " passes and " + str(self.num_topics) + " topics: " + str(
            coherence_model.get_coherence()))

    # Write the extracted topics to a csv file
    def save_lda_topics(self, csv_path, lda_model):
        pprint(lda_model.print_topics())
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Topic Number", "Words"])

        with open(csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            for idx in range(0, self.num_topics):
                joint_words = ", ".join([str(word) for word in lda_model.show_topic(idx)])
                writer.writerow([str(idx), joint_words])

    # Topic distribution for each document
    def topic_distribution_each_document(self, csv_path, apps_df, dataset_number, lda_model):
        # Write the header first
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["App Pkgname", "Topic Percentage", "Genre"])

        if dataset_number == 1:
            for i in range(0, len(self.corpus)):
                with open(csv_path, 'a', newline='') as file:
                    topics = sorted(lda_model[self.corpus[i]], key=lambda x: x[1], reverse=True)
                    writer = csv.writer(file)
                    if str(apps_df.at[apps_df.index[i], 'Type']) == 'GAME':
                        writer.writerow([str(apps_df.at[apps_df.index[i], 'Pkgname']), topics, 'Game'])
                    else:
                        genre = str(apps_df.at[apps_df.index[i], 'Genre']).split(',')[0]
                        writer.writerow([str(apps_df.at[apps_df.index[i], 'Pkgname']), topics, genre])
        elif dataset_number == 2:
            games = ['Puzzle', 'Action &amp; Adventure', 'Casual', 'Brain Games', 'Action', 'Simulation',
                     'Creativity', 'Puzzle', 'Role Playing', 'Adventure', 'Strategy', 'Arcade', 'Trivia',
                     'Card', 'Racing', 'Casino', 'Pretend Play', 'Educational', 'Sports', 'Word', 'Board']
            for i in range(0, len(self.corpus)):
                with open(csv_path, 'a', newline='') as file:
                    topics = sorted(lda_model[self.corpus[i]], key=lambda x: x[1], reverse=True)
                    writer = csv.writer(file)
                    pkgname = str(apps_df.at[apps_df.index[i], 'app_id'])
                    if any(item in str(apps_df.at[apps_df.index[i], 'genre']) for item in games):
                        writer.writerow([pkgname, topics, 'Game'])
                    elif str(apps_df.at[apps_df.index[i], 'genre']) is not None:
                        genre = str(apps_df.at[apps_df.index[i], 'genre'])
                        writer.writerow([pkgname, topics, genre])
        else:
            games = ['Puzzle', 'Action & Adventure', 'Casual', 'Brain Games', 'Action', 'Simulation',
                     'Creativity', 'Puzzle', 'Role Playing', 'Adventure', 'Strategy', 'Arcade', 'Trivia',
                     'Card', 'Racing', 'Casino', 'Pretend Play', 'Educational', 'Sports', 'Word', 'Board']
            for i in range(0, len(self.corpus)):
                with open(csv_path, 'a', newline='') as file:
                    topics = sorted(lda_model[self.corpus[i]], key=lambda x: x[1], reverse=True)
                    writer = csv.writer(file)
                    pkgname = str(apps_df.at[apps_df.index[i], 'pkgname'])
                    if any(item in str(apps_df.at[apps_df.index[i], 'Genre']) for item in games):
                        writer.writerow([pkgname, topics, 'Game'])
                    elif str(apps_df.at[apps_df.index[i], 'Genre']) is not None:
                        genre = str(apps_df.at[apps_df.index[i], 'Genre'])
                        writer.writerow([pkgname, topics, genre])
