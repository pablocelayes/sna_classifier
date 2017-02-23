#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pandas import DataFrame
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.data import load
from nltk.stem import SnowballStemmer
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer

spanish_tokenizer = load('tokenizers/punkt/spanish.pickle')

# stopword list to use
spanish_stopwords = stopwords.words('spanish')

# spanish stemmer
stemmer = SnowballStemmer('spanish')

# punctuation to remove
non_words = list(punctuation)
# we add spanish punctuation
non_words.extend(['¿', '¡'])
non_words.extend(map(str, range(10)))

stemmer = SnowballStemmer('spanish')


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text, stem=True):
    text = text.lower()

    result = []
    for sentence in spanish_tokenizer.tokenize(text):
        # remove punctuation
        text = ''.join([c for c in sentence if c not in non_words])
        # tokenize
        tokens = word_tokenize(text)
        # stem
        if stem:
            try:
                stems = stem_tokens(tokens, stemmer)
            except Exception as e:
                print(e)
                print(text)
                stems = ['']
            result += stems
        else:
            result += tokens
    
    return result

vectorizer = TfidfVectorizer(
    analyzer='word',
    tokenizer=tokenize,
    lowercase=True,
    stop_words=spanish_stopwords)


if __name__ == "__main__":
    text = """Definitivamente, había que estar en Caibarién. Sin temor al sol ni a la carretera, había que llegarse al norteño municipio de Villa Clara, a respirar la brisa marina y vivir el adiós a los diamantes de Ariel Pestano Valdés, el más grande receptor que ha visto la generación de quienes nacimos después de 1989."""
    vectorized_text = vectorizer.fit_transform([text])
    print(DataFrame(vectorized_text.A,
                    columns=vectorizer.get_feature_names()).to_string())
