#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copyright (C) Cogfor <info@cogfor.com>
Copyright (C) karsten jeschkies <jeskar@web.de>
"""

import numpy as np
import logging
import os

from scipy import sparse
from gensim import interfaces, utils, matutils, corpora, models
import pickle

from nyan.shared_modules.kmedoids_clustering import KMedoidsClustering as KMedoids
from nyan.shared_modules.feature_extractor.esa.document_titles import DocumentTitles


from config import CONFIG

LOGGER = logging.getLogger('gensim.models.esamodel')

# TODO: change to parameterized prefix
PREFIX = os.path.join(CONFIG['prefix'], 'reduce_2000_kmed_lda')

class EsaModel(interfaces.TransformationABC):
    """Transform a TF-IDF model to Explicit Semantic Analysis (ESA)

    Objects of this class realize the transformation between concepts (docs)
    represented in the TF-IDF model and the ESA model.
    The transformation is done by multiplying the the doc in TF-IDF model by
    the ESA interpreter matrix. Each column of the ESA matrix is
    one concept, each row is
    one token An entry is the TF-IDF value of a token and the concept.
    The concepts are usually Wikipedia articles.

    The main methods are:

    1. constructor, which loads an interpreter
        matrix with the TF-IDF value for each token in each concept/doc.

    2. the [] method, which transforms a simple TF-IDF
        representation into the ESA space.

    >>> esa = EsaModel(tfidf_corpus)
    >>> print = esa[some_doc]
    >>> esa.save('/tmp/foo.esa_model', '/tmp/foo.esa_concept_dict')

    Model persistency is achieved via its load/save methods.

    Args:
        corpus:             The document corpus as TF-IDF
        document_titles:    The names of each concept (doc) in corpus.
        num_features:       The number of features of corpus
    """

    def __init__(self, corpus, document_titles, num_features=None, 
                num_concepts=None, lang=None):
        """    
        Args:
            corpus: is our interpreter matrix. It is sparse and each row represents
                    a doc that is seen as a concept
            document_titles: gives the names of each concept (doc) in corpus.
            num_features: gives the number of features of corpus
            
            num_concepts: if present indicates how many concepts to use as reduced
                concept corpus. If None all documents are used.
            lang: required language parameter used for concept reduction. Must be present
                if num_concepts is not None
        """
        self.corpus = corpus
        self.document_titles = document_titles
        
        if num_features is None:
            LOGGER.info("scanning corpus to determine the number of features")
            # num_features = 1 + utils.get_max_id(corpus)
            num_features = corpus.num_terms
        self.num_features = num_features
        LOGGER.info("%d features found" % num_features)

        if num_concepts:
            self.reduce_concept_space(num_concepts, lang, num_lda_features=1000)
            self.reduced = True

        else:
            num_concepts = corpus.num_docs
            self.reduced = False
        self.num_concepts = num_concepts
        self.lang = lang

        self.compute_sparse_corpus()

    def reduce_concept_space(self, num_concepts, lang, num_lda_features=None):
        """
            This reduces the concept space (corpus) by
            applying KMedoids clustering to the set of 
            concept vectors and keeping only the clusters medoids as
            reduced set of concepts

            if a num_features parameter is given, a previous LDA
            dimensionality reduction step is applied before performing
            the clustering, to improve efficiency

            In any case, the returned reduced corpus retains its
            original feature_number, reducing only the number of concepts
        """
        LOGGER.info("Applying KMedoids clustering to reduce number of concepts from %d to %d" %
            (self.corpus.num_docs, num_concepts))

        if num_lda_features:
            LOGGER.info("applying LDA to reduce dimensionality")            
            
            # create LDA-reduced corpus only if it doesn't exist yet
            PASSES = 1
            lda_corpus_path = os.path.join(PREFIX, '%s_lda%d_corpus_p%d.mm' % (lang, num_lda_features, PASSES))
            if not os.path.exists(lda_corpus_path):
                LOGGER.info("Generating LDA-reduced corpus")
                # init corpus reader and word -> id map
                id2token = corpora.Dictionary.load(CONFIG['prefix'] + "%s_wordids.dict" % lang)

                # build lda model
                lda = models.LdaMulticore(corpus=self.corpus, id2word=id2token,
                                            num_topics=num_lda_features,
                                            chunksize=10000,
                                            passes=PASSES,
                                            workers=3)

                # save trained model
                lda.save(PREFIX + '%s_lda%d.model' % (lang, num_lda_features))
                LOGGER.info("finished LDA model generation")
                # save corpus as lda vectors in matrix market format
                corpora.MmCorpus.serialize(lda_corpus_path,
                                lda[self.corpus], progress_cnt=10000)

            # init lda-corpus reader
            clustering_corpus = corpora.MmCorpus(lda_corpus_path)
            clustering_num_features = num_lda_features           
        else:
            clustering_corpus = self.corpus
            clustering_num_features = self.num_features
        
        # Create medoids or load previous ones
        clusters_path = os.path.join(PREFIX, '%s_esa_kmedoids.pickle' % lang)
        if not os.path.exists(clusters_path):
            clusterer = KMedoids(corpus=clustering_corpus,
                                 num_features=clustering_num_features,
                                 num_clusters=num_concepts,
                                 max_iterations=10)
            clusters = clusterer.cluster()
            with open(clusters_path, 'w') as f:
                pickle.dump(clusters, f)

        with open(clusters_path) as f:
            clusters = pickle.load(f)

        # set the corpus to the original representations of
        # of the cluster medoids

        #reduce document titles
        reduced_document_titles = DocumentTitles()
        reduced_document_titles.file_path = os.path.join(PREFIX, "%s_%d_articles.txt" % (lang, num_concepts))
        medoid_ids = sorted(clusters.keys())
        for medoid_id in medoid_ids:
            reduced_document_titles.append(self.document_titles[medoid_id] or "no title")

        self.document_titles = reduced_document_titles

        # Reduce corpus using medoid ids
        reduced_corpus_path = os.path.join(PREFIX, '%s_%d_corpus.pickle' % (lang, num_concepts))
        if not os.path.exists(reduced_corpus_path):
            LOGGER.info("Picking vectors for reduced corpus from original corpus")
            reduced_corpus = []
            for doc_id, doc in enumerate(self.corpus):
                if doc_id in medoid_ids:
                    reduced_corpus.append(doc)
                if doc_id % 1000 == 0:
                    LOGGER.info("%.2f %%" % ((doc_id * 100.0) / len(self.corpus)))
            with open(reduced_corpus_path, 'w') as f:
                pickle.dump(reduced_corpus, f)
        with open(reduced_corpus_path) as f:
            reduced_corpus = pickle.load(f) 

        self.corpus = reduced_corpus

    def __str__(self):
        """
        Return document titles as string
        """
        return "EsaModel(numConcepts=%s, numFeatures=%s)" % \
            (len(self.document_titles), self.num_features)

    def get_concept_titles(self, doc_vec):
        """
        Converts numerical ids from document vector to concept titles (topics).
        """
        concept_titles = [(self.document_titles[concept_id], weight)
                          for concept_id, weight in doc_vec]
        return concept_titles

    def compute_sparse_corpus(self):
        LOGGER.info("Started computing sparse corpus")
        i = 0
        I = []
        J = []
        data = []
        for concept_tfidf in self.corpus:
            for token_id, weight in concept_tfidf:
                I.append(i)
                J.append(token_id)
                data.append(weight)
            i += 1
        nrows = len(self.document_titles)
        ncols = self.num_features
        self.sparse_corpus = sparse.coo_matrix(
            (data, (I, J)), shape=(nrows, ncols))
        LOGGER.info("Finished computing sparse corpus")

    def __getitem__(self, tf_idf, eps=1e-12):
        """
        Return ESA representation of the input vector and/or corpus.
        """
        # If the input vector is in fact a corpus,
        # return a transformed corpus as a result
        is_corpus, _ = utils.is_corpus(tf_idf)
        if is_corpus:
            return self._apply(tf_idf)
        
        I = []
        data = []
        ncols = self.num_features
        for (pos, weight) in tf_idf:
            if weight > eps:
                data.append(weight)
                I.append(pos)
        J = [0] * len(I)
        tf_idf = sparse.coo_matrix((data, (I, J)), shape=(ncols, 1))
        
        vector = self.sparse_corpus * tf_idf
        vector = matutils.unitvec(vector).transpose()
        
        return vector

    def save(self, fname,
             separately=None, sep_limit=10485760, ignore=frozenset([])):
        """
        Save corpus index in for persistency
        """
        self.document_titles.save()
        path = fname + '.npz'
        LOGGER.info("Storing %s object to %s and %s",
                    self.__class__.__name__, fname, path)
        # Remove the index from self.__dict__,
        # so it doesn't get pickled
        index = self.sparse_corpus
        del self.corpus
        del self.sparse_corpus
        try:
            utils.pickle(self, fname)
            LOGGER.info("Finished pickling EsaModel")
            np.savez(path, row=index.row, col=index.col,
                     data=index.data, shape=index.shape)
            LOGGER.info("Finished saving sparse corpus")
            # Not needed? Check if they are saved properly with the rest of the model
            # pickle.dump(self.document_titles, open(fname + '_doc_titles.pickle', 'w'))
            # LOGGER.info("Finished saving (reduced) doc titles")
        finally:
            self.corpus = index

    @classmethod
    def load(cls, fname, mmap=None):
        """
        Load a previously corpus index from file.
        """
        LOGGER.info("Loading %s object from %s", cls.__name__, fname)
        result = utils.unpickle(fname)
        LOGGER.info("Finished unpickling EsaModel")
        path = fname + '.npz'
        sc = np.load(path)
        result.sparse_corpus = sparse.coo_matrix(
            (sc['data'], (sc['row'], sc['col'])),
            shape=sc['shape'])
        LOGGER.info("Finished loading sparse corpus")

        return result
