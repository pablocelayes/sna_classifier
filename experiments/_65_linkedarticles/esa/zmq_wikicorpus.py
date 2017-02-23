#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copyright (C) Cogfor <info@cogfor.com>
Copyright (C) 2014 Alejandro Naser Pastoriza <alejandro@cogfor.com>

Some implementations are taken from wikicorpus.py by
Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
Copyright (C) 2012 Lars Buitinck <larsmans@gmail.com>
Copyright (C) 2012 Karsten Jeschkies <jeskar@web.de>

This is an implementation of a gensim.corpus to work on
wikipedia articles stored in a Redis database where
each article is stored as a list with the article name as key.
"""

import os
import sys
import time
import redis
import logging
import multiprocessing

from esamodel import EsaModel
from ngram_tokenizer import Tokenizer
from document_titles import DocumentTitles

from datetime import datetime, timedelta
from optparse import OptionParser
from gensim import utils, corpora, models

LOGGER = logging.getLogger("feature_extractor")
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
LOGGER.info("Running %s", ' '.join(sys.argv))

NUM_TOPICS = 1000

def tokenize_text(article_name):
    """
    Tokenizes a wikipedia document stored in a Redis database
    """
    doc = REDIS.lindex(article_name, 0)
    tokens = Tokenizer().tokenize(doc)
    return article_name, tokens


class CleanCorpus(corpora.TextCorpus):
    """
    Tokenizes each document in a Redis database.
    Each value in Redis is regarded as a document, the value is the
    corresponding text from Wikipedia Documents should be text files.
    Stems all words and removes stop words.
    """
    def __init__(self, documents, dictionary=None, no_below=20, keep_n=50000):
        """
        Args:
            documents: Documents keys on the Redis database.
            no_below:  Words that appear less than this are neglected.
            keep_n: Maximum dictionary size (default: 50000)
        """
        super(CleanCorpus, self).__init__()
        self.documents = documents
        self.start_time = time.time()
        self.article_names = []
        self.total_articles = len(documents)
        self.dictionary = dictionary or \
            self.corpus_dictionary(no_below, keep_n)

    def report_progress(self, position, total):
        """
        Calculates overall progress and estimated end time
        Args:
            position:   number of records processed
            total:      total number of records
        Returns:
            Human readable description of the elapsed time and
            estimated remaining time as a string
            (days, hours and minutes); estimated time of completion
            as a string containing a full date.
        """
        time_now = time.time()
        elapsed_time = time_now - self.start_time
        estimated_remaining = elapsed_time * total / position
        estimated_end_time = time_now + estimated_remaining
        time_delta = timedelta(seconds=elapsed_time)

        elapsed_time = ':'.join(str(time_delta).split(':')[:2])
        time_delta = timedelta(seconds=estimated_remaining)
        estimated_remaining = ':'.join(str(time_delta).split(':')[:2])
        estimated_end_time = datetime.fromtimestamp(estimated_end_time).\
            strftime("%d %B %Y - %H:%M")

        return elapsed_time, estimated_remaining, estimated_end_time

    def get_texts(self):
        """
        Texts from Redis send to the tokenizer in parallel batches.
        """
        LOGGER.info("Scanning %d files.", self.total_articles)
        articles_processed = 0
        processes = multiprocessing.cpu_count()
        chunksize = 10 * processes
        pool = multiprocessing.Pool(processes)
        groups = utils.chunkize_serial(self.documents, chunksize=chunksize)
        for group in groups:
            for article_name, tokens in pool.imap(tokenize_text, group):
                articles_processed += 1
                name = None
                try:
                    if articles_processed % 5000 == 0:
                        progress = self.report_progress(
                            articles_processed, self.total_articles)
                        LOGGER.info(
                            "#articles: %s, time elapsed: %s, "
                            "remaining time: %s, ETA: %s",
                            articles_processed, progress[0],
                            progress[1], progress[2])
                    name = article_name.strip("\n").decode("UTF-8")
                except UnicodeDecodeError as error:
                    LOGGER.error("Could not decode %s: %s",
                                 article_name, error)
                    exit(1)
                except Exception as error:
                    LOGGER.error("Unknown error processing article: '%s' (%s)",
                                 article_name, error)
                    exit(1)
                self.article_names.append(name)
                yield tokens
        pool.terminate()
        LOGGER.info("Processed %d articles", articles_processed)

    def corpus_dictionary(self, no_below, keep_n):
        """
        Calculate the Dictionary.
        This is a mapping between words and their integer ids.
        """
        texts = self.get_texts()
        dictionary = corpora.Dictionary(texts)
        dictionary.filter_extremes(
            no_below=no_below, keep_n=keep_n, no_above=0.1)
        return dictionary

    def save_article_names(self, file_path):
        """
        Save titles of Wikipedia articles (keys) to
        file system for later use
        """
        LOGGER.info("Saving article names to %s", file_path)
        with open(file_path, "wb") as output_file:
            for name in self.article_names:
                output_file.write("%s\n" % name.encode("UTF-8"))

    def load_article_names(self, file_path):
        """
        Load titles of Wikipedia articles from file system
        """
        LOGGER.info("Loading article names from %s", file_path)
        self.article_names = []
        with open(file_path, "r") as input_file:
            for line in input_file:
                article_name = line.strip("\n").decode("UTF-8")
                self.article_names.append(article_name)


if __name__ == '__main__':
    LANGUAGES = ['nl' 'en', 'es', 'de']
    REDIS_INSTANCES = {'en': 0, 'es': 1, 'de': 2, 'nl': 3}

    PARSER = OptionParser()
    PARSER.add_option('-o', '--output-prefix', action='store', dest='prefix',
                      help="specify path prefix where everything "
                      "should be saved")
    (OPTIONS, _) = PARSER.parse_args()
    if not OPTIONS.prefix:
        PARSER.error('Prefix not given')

    for language in LANGUAGES:
        REDIS = redis.StrictRedis(
            host='redis', port=6379, db=REDIS_INSTANCES[language])
        DOCS = REDIS.keys("*")

        if len(DOCS) < 10000:
            LOGGER.info("Insufficient data to generate %s-models", language)
            continue

        DICTIONARY_PATH = os.path.join(
            OPTIONS.prefix, language + "_wordids.dict")
        DICTIONARY_AS_TEXT_PATH = os.path.join(
            OPTIONS.prefix, language + "_wordids.dict.txt")
        if not os.path.exists(DICTIONARY_PATH):
            CORPUS = CleanCorpus(DOCS)
            CORPUS.dictionary.save(DICTIONARY_PATH)
        else:
            DICTIONARY = corpora.Dictionary.load(DICTIONARY_PATH)
            CORPUS = CleanCorpus(DOCS, dictionary=DICTIONARY)
        if not os.path.exists(DICTIONARY_AS_TEXT_PATH):
            CORPUS.dictionary.save_as_text(DICTIONARY_AS_TEXT_PATH)
        LOGGER.info("Finished %s-Dictionary creation", language)

        ARTICLES_PATH = os.path.join(
            OPTIONS.prefix, language + "_articles.txt")
        if not os.path.exists(ARTICLES_PATH):
            CORPUS.save_article_names(ARTICLES_PATH)
        LOGGER.info("Finished Saving %s Articles", language)

        BAG_OF_WORDS_PATH = os.path.join(
            OPTIONS.prefix, language + "_bow_corpus.mm")
        if not os.path.exists(BAG_OF_WORDS_PATH):
            corpora.MmCorpus.serialize(
                BAG_OF_WORDS_PATH, CORPUS, progress_cnt=10000)
        MM_BOW = corpora.MmCorpus(BAG_OF_WORDS_PATH)
        LOGGER.info("Finished %s-Bag-Of-Words creation", language)

        TF_IDF_PATH = os.path.join(
            OPTIONS.prefix, language + "_tfidf.model")
        if not os.path.exists(TF_IDF_PATH):
            TF_IDF = models.TfidfModel(
                MM_BOW, id2word=CORPUS.dictionary, normalize=True)
            TF_IDF.save(TF_IDF_PATH)
        else:
            TF_IDF = models.TfidfModel.load(TF_IDF_PATH)
        TF_IDF_CORPUS_PATH = os.path.join(
            OPTIONS.prefix, language + "_tfidf_corpus.mm")
        if not os.path.exists(TF_IDF_CORPUS_PATH):
            corpora.MmCorpus.serialize(
                TF_IDF_CORPUS_PATH, TF_IDF[MM_BOW], progress_cnt=10000)
        MM_TF_IDF = corpora.MmCorpus(TF_IDF_CORPUS_PATH)
        LOGGER.info("Finished %s-TF-IDF Model Generation", language)

        ESA_PATH = os.path.join(
            OPTIONS.prefix, language + "_esa_on_tfidf.model")
        if not os.path.exists(ESA_PATH):
            ARTICLE_TITLES = DocumentTitles.load(ARTICLES_PATH)
            ESA = EsaModel(MM_TF_IDF, document_titles=ARTICLE_TITLES)
            ESA.save(ESA_PATH)
        LOGGER.info("Finished %s-ESA Model Generation", language)

        if language == 'en':
            SMALL_EN_ESA_PATH = os.path.join(
                OPTIONS.prefix, "small_en_esa_on_tfidf.model")
            if not os.path.exists(SMALL_EN_ESA_PATH):
                ESA = EsaModel(MM_TF_IDF, document_titles=ARTICLE_TITLES, num_concepts=NUM_TOPICS)
                ESA.save(SMALL_EN_ESA_PATH)
            LOGGER.info("Finished small en-ESA Model Generation")


        LOGGER.info("Finished ALL Transforming Activity")
