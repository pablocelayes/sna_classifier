#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copyright (C) Cogfor <info@cogfor.com>
Copyright (C) Graham Ashton <graham@effectif.com>


"""

from nltk.corpus import stopwords


class Counter(dict):
    def add(self, other):
        for ngram in other.iterkeys():
            self[ngram] = self.get(ngram, 0) + other[ngram]

    def remove_subphrases(self):
        """Remove subphrases from n-gram

        Removes n-grams that are part of a longer n-gram if the shorter n-gram appears just as frequently as the longer
        n-gram (which means that it can only be present within the longer n-gram)
        """
        builder = NgramBuilder()
        to_remove = []
        for phrase in self.keys():
            for length in range(1, len(phrase.split(' '))):
                for subphrase in builder.find_ngrams(phrase, length):
                    if self.get(subphrase, '') == self[phrase]:
                        to_remove.append(subphrase)
        for subphrase in set(to_remove):
            del self[subphrase]


class NgramBuilder(object):
    def __init__(self, stopwords=None):
        self.stopwords = stopwords
        self.ngrams = {}
        self.unigram_cache = {}

    def find_ngrams(self, text, length):
        counter = Counter()
        num_unigrams, unigrams = self.split_into_unigrams(text.lower())

        for i in xrange(num_unigrams - length + 1):
            if self.stopwords and (
                            unigrams[i] in self.stopwords or \
                                unigrams[i + length - 1] in self.stopwords):
                continue
            ngram = ' '.join(unigrams[i:i + length])
            counter[ngram] = counter.get(ngram, 0) + 1

        return counter

    def split_into_unigrams(self, text):
        unigrams = []

        for token in text.split():
            if token in self.unigram_cache:
                unigram = self.unigram_cache[token]
            else:
                unigram = self.token_to_unigram(token)
                self.unigram_cache[token] = unigram

            if unigram:
                unigrams.append(unigram)

        return len(unigrams), unigrams

    def token_to_unigram(self, token):
        token = token.strip(" ,.:;!|&-_()[]<>{}/\"'")

        def has_no_char(token):
            for c in token:
                if c.isalpha():
                    return False
            return True

        if len(token) == 1 or token.isdigit() or has_no_char(token):
            return None

        return token


class Tokenizer(object):
    """
    Tokenizes in n-grams, removes subphrases
    """

    def __init__(self):
        self.counter = Counter()
        self.stopwords = stopwords.words('dutch')
        self.builder = NgramBuilder(self.stopwords)

    def tokenize(self, document):
        if not type(document) is unicode:
            document = document.decode("UTF-8")
        # list of English stopwords
        # create dictionary containing n-grams of one, two or three words in length
        # mapped to the number of times that each n-gram occurred.
        for i in range(1, 3):
            self.counter.add(
                self.builder.find_ngrams(document, i))
        # remove n-grams that are part of a longer n-gram if the shorter n-gram
        # appears just as frequently as the longer n-gram
        self.counter.remove_subphrases()
        return self.counter.keys()  # list

# IMPORTANT: The tokenizer defined here must be used in the model training phase
# as well as in the feature extractors
TOKENIZER = Tokenizer()
