import csv
import os

import gensim
from gensim import corpora
from gensim.models import CoherenceModel, LdaMulticore

from preprocessors.preprocesser import Preprocessor


class Predictor:
    def __init__(self, model_name):
        self.model_name = model_name
        self.dictionary = None
        self.corpus = None
        self.lda_model = None
        self.coherence_model = None
        self.preprocessor = Preprocessor()

        if os.path.exists(model_name):
            self.lda_model = LdaMulticore.load(model_name)
            self.dictionary = corpora.Dictionary.load(model_name + ".dict")

        if os.path.exists(model_name + ".coherence"):
            self.coherence_model = CoherenceModel.load(model_name + ".coherence")

    def create_dictionary(self, dataset):
        self.dictionary = corpora.Dictionary(dataset)
        self.dictionary.save(self.model_name + ".dict")

    def create_corpus(self, dataset):
        self.corpus = [self.dictionary.doc2bow(row) for row in dataset]

    def train_lda_model(self, dataset, num_topics=40, passes=40, chunksize=100, alpha=0.1, eta=0.001, workers=8):
        self.create_dictionary(dataset)
        self.create_corpus(dataset)

        if self.lda_model is None:
            lda_model = LdaMulticore(corpus=self.corpus,
                                     id2word=self.dictionary,
                                     num_topics=num_topics,
                                     passes=passes,
                                     chunksize=chunksize,
                                     alpha=alpha,
                                     eta=eta,
                                     workers=workers)

            self.lda_model = lda_model

            self.lda_model.save(self.model_name)

        return self.lda_model

    def get_lda_model(self):
        return self.lda_model

    def train_coherence_model(self, dataset):
        if self.coherence_model is None:
            self.dictionary = corpora.Dictionary(dataset)
            self.coherence_model = CoherenceModel(model=self.lda_model, texts=dataset, dictionary=self.dictionary, coherence='c_v')

            self.coherence_model.save(self.model_name + ".coherence")

        return self.coherence_model

    def get_coherence_model(self):
        return self.coherence_model

    def infer(self, description: str):
        doc = self.preprocessor.preprocess(description)
        doc_bow = self.dictionary.doc2bow(doc)

        # y = self.lda_model.get_document_topics(doc_bow, minimum_probability=0)

        # y = [tup[1] for tup in y]
        return self.lda_model.get_document_topics(doc_bow, minimum_probability=0)

    def save_lda_topics(self, csv_path):
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Topic Number", "Words"])

        with open(csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            for idx in range(0, self.lda_model.num_topics):
                joint_words = ", ".join([str(word) for word in self.lda_model.show_topic(idx)])
                writer.writerow([str(idx), joint_words])
