#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copyright (C) Cogfor <info@cogfor.com>
Copyright (C) karsten jeschkies <jeskar@web.de>
"""

# import langid
import logging
import pickle
import sys
import os
from scipy import sparse

from gensim import corpora, models, matutils

from tw_dataset.settings import COGFOR_REPO_PATH, PREFIX
sys.path.append(os.path.join(COGFOR_REPO_PATH, 'nyan/shared_modules/feature_extractor/esa'))
# Required for gensim unpickling of ESA model to work

from nyan.shared_modules.feature_extractor.esa.esamodel import EsaModel
from nyan.shared_modules.feature_extractor.esa.cosine_esamodel import \
    CosineEsaModel

from nyan.shared_modules.feature_extractor.ngram_tokenizer import TOKENIZER

import numpy as np
from nyan.shared_modules.utils.helper import coo_vector_to_tuples

logger = logging.getLogger("extractor")
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.ERROR)


def get_feature_extractor():
    """
        This is used to create the common feature extractor
        being used throughout the whole project
    """
    vendor_extractor = VendorFeatureExtractor(prefix=PREFIX)
    
    nlp_extractor_class = {"esa": EsaFeatureExtractor, 
                "lda": LdaFeatureExtractor}[NLP_TOPIC_MODEL]
    
    feature_extractor = nlp_extractor_class(prefix=PREFIX) # prefix=target_prefix)

    return feature_extractor

"""
An extractor creates a feature vector from a text. This implementation relies
heavily on gensim. Som additions were made to gensim. See esamodel.

All extractors have to load at least one feature model.
"""

class Extractor:
    """
    Extractor interface
    """

    def __init__(self):
        pass

    def get_features(self, article):
        logger.debug("get_features not implemented!")
        raise NotImplementedError()

    def get_feature_number(self):
        logger.debug("get_feature_number not implemented!")
        raise NotImplementedError()

    @classmethod
    def get_version(cls, article):
        logger.debug("get_version not implemented!")
        raise NotImplementedError()


#####
# NLP Extractors
#####
class TfidfFeatureExtractor(Extractor):
    """The Tf-Idf feature extractor creates a feature vector from a text using tf-idf as a weighting factor

    The tf-idf value increases proportionally to the number of times a word appears in the document, but is offset by
    the frequency of the word in the corpus. This implementation relies heavily on gensim and is adapted to support
    n-grams via a ngram tokenizer. Set TOKENIZER to the ngrams needed (default: trigrams)
    """

    def __init__(self, prefix):
        # Extractor.__init__(self)
        logger.info("Load dictionary and tfidf model with prefix %s"
                    % prefix)
        self.dictionary = corpora.Dictionary.load(prefix + "_wordids.dict")
        self.tfidf_model = models.TfidfModel.load(prefix + "_tfidf.model")

    def get_features(self, article=None, text=None):
        """Extract tf-idf features from a document, based on a bag-of-ngrams

        A ngram tokenizer is used to extract n-grams which are converted to a tf-idf model using Gensim. A dictionary
        of n-grams and the resulting tf-idf space are stored for later use.

        Args:
            document: a text document (utf-8) with no formatting

        Returns:
            doc_tfidf: Tf-idf representation of the document as Gensim model object
        """
        # create list of tokens from doc
        logger.debug("Tokenize document.")
        if text is None:
            text = article.text.encode("utf8")
        tokens = TOKENIZER.tokenize(text)

        # create bow of doc from token list
        logger.debug("Create bag-of-words representation from article.")
        doc_bow = self.dictionary.doc2bow(tokens)

        # create tfidf representation from bag-of-words
        logger.debug("Transform to tfidf.")
        doc_tfidf = self.tfidf_model[doc_bow]

        return doc_tfidf

    def get_feature_number(self):
        """Get length of the tfidf dictionary"""
        return len(self.dictionary)

    @classmethod
    def get_version(self):
        """Get version of tfidf feature extractor"""
        return u"TF-IDF-1.2"


class LdaFeatureExtractor(Extractor):
    """LDA transformation on a document, based on an existing tf-idf space and dictionary"""

    def __init__(self, prefix, n_topics):
        Extractor.__init__(self)
        logger.info("Load dictionary and tfidf and lda model with prefix %s"
                    % prefix)
        self.dictionary = corpora.Dictionary.load(prefix + "_wordids.dict")
        self.tfidf_model = models.TfidfModel.load(prefix + "_tfidf.model")
        self.lda_model = models.LdaModel.load(prefix + "_lda%d.model" % n_topics)
        self.lda_model.iterations = 50
        self.lda_model.gamma_threshold = 0.001

    def get_features(self, article=None, text=None):
        """Extract LDA features from a document, transforming existing tf-idf models

        a tf-idf space is transformed to a LDA model using Gensim. 
        The implementation uses the previously, in TfidfFeatureExtractor,
        generated dictionary (bag-of-ngrams) and tf-idf model space to apply LDA transformation.

        Args:
            article: an article object, 
            text: a text document (utf-8) with no formatting

        Returns:
            doc_tfidf: LDA representation of the document as Gensim model object
        """
        # create list of tokens from doc
        logger.debug("Tokenize document.")
        if text is None:
            text = article.clean_content

        tokens = TOKENIZER.tokenize(text)

        # create bow of doc from token list
        logger.debug("Create bag-of-words representation from article.")
        doc_bow = self.dictionary.doc2bow(tokens)

        # create tfidf representation from bag-of-words
        logger.debug("Transform to tfidf.")
        doc_tfidf = self.tfidf_model[doc_bow]

        #create lda representation from tfidf
        logger.debug("Transform to lda")
        doc_lda = self.lda_model[doc_tfidf]

        return doc_lda

    def get_feature_number(self):
        """Get the number of features in the LDA dictionary"""
        return self.lda_model.num_topics

    @classmethod
    def get_version(self):
        """Get version of LDA feature extractor"""
        return u"LDA-0.4"


class LdaBowFeatureExtractor(Extractor):
    """LDA transformation on a document, based on an existing bag-of-words/ngrams dictionary"""

    def __init__(self, prefix):
        Extractor.__init__(self)
        logger.info("Load dictionary and lda model with prefix %s"
                    % prefix)
        self.dictionary = corpora.Dictionary.load(prefix + "_wordids.dict")
        self.lda_model = models.LdaModel.load(prefix + "_lda_on_bow.model")

    def get_features(self, article):
        """Extract LDA features from a document, transforming an existing bag-of-ngrams as dictionary

        A bag-of-words/ngrams space is transformed to a LDA model using Gensim. The implementation uses the previously,
        in TfidfFeatureExtractor, generated dictionary (bag-of-ngrams) to apply LDA transformation. This method differs
        from LdaFeatureExtractor in that it uses a bag-of-ngrams instead of tf-idf as model space to apply
        transformation

        Args:
            document: a text document (utf-8) with no formatting

        Returns:
            doc_tfidf: LDA representation of the document as Gensim model object
        """
        # create list of tokens from doc
        logger.debug("Tokenize document.")
        document = article.clean_content.encode("utf8")
        tokens = TOKENIZER.tokenize(document)

        # create bow of doc from token list
        logger.debug("Create bag-of-words representation from article.")
        doc_bow = self.dictionary.doc2bow(tokens)

        # create lda representation from tfidf
        logger.debug("Transform to lda")
        doc_lda = self.lda_model[doc_bow]

        return doc_lda

    def get_feature_number(self):
        """Get number of topics in the LDA-model"""
        return self.lda_model.num_topics

    @classmethod
    def get_version(self):
        """Get version of LDA feature extractor"""
        return u"LDA-on-BOW-50"


class EsaFeatureExtractor(Extractor):
    """Explicit Semantic Analysis transformation on a document, based on an existing tf-idf space"""

    def __init__(self, prefix):
        Extractor.__init__(self)
        logger.info(
            "Load dictionary, tfidf model, lda model and esa model with prefix %s" % prefix)
        self.dictionary = corpora.Dictionary.load(prefix + "_wordids.dict")
        self.tfidf_model = models.TfidfModel.load(prefix + "_tfidf.model")
        # self.lda_model = models.LdaModel.load(prefix + "_lda.model")
        self.esa_model = EsaModel.load(prefix + "_esa1000_on_tfidf.model")

    def get_features(self, article=None, text=None):
        """Extract ESA features from a document, transforming an existing tf-idf model space

        A tf-idf model space is transformed to an ESA model using Gensim. The implementation uses the previously,
        in TfidfFeatureExtractor, generated tf-idf model space.

        Args:
            document: a text document (utf-8) with no formatting

        Returns:
            doc_tfidf: ESA representation of the document as Gensim model object
        """
        # create list of tokens from doc
        logger.debug("Tokenize document.")
        if text is None:
            text = article.text
        document = text.encode("utf8")
        tokens = TOKENIZER.tokenize(document)

        # create bow of doc from token list
        logger.debug("Create bag-of-words representation from article.")
        doc_bow = self.dictionary.doc2bow(tokens)

        # create tfidf representation from bag-of-words
        logger.debug("Transform to tfidf.")
        doc_tfidf = self.tfidf_model[doc_bow]

        #create lda representation from tfidf
        # logger.debug("Transform to lda")
        # doc_lda = self.lda_model[doc_tfidf]

        #create esa representation from lda
        logger.debug("Transform to esa")
        doc_esa = self.esa_model[doc_tfidf]

        return doc_esa

    def get_feature_number(self):
        """Get number of topics in the ESA-model"""
        return len(self.esa_model.document_titles)

    @classmethod
    def get_version(self):
        """Get version of ESA feature extractor"""
        return u"ESA-1.1"


class cEsaFeatureExtractor(Extractor):
    """Cosine Explicit Semantic Analysis transformation on a document, based on an existing tf-idf space

    Cosine Explicit Semantic Analysis reduces the dimension of the tf-idf matrix and applies ESA transformation on
    top of this. THe reduction is obtained by taking the cosine similarity between two features vectors and use this
    as a weight for a new feacture vector.
    """

    def __init__(self, prefix):
        Extractor.__init__(self)
        logger.info(
            "Load dictionary, tfidf model, lda model and cosine esa model with prefix %s"
            % prefix)
        self.dictionary = corpora.Dictionary.load(prefix + "_wordids.dict")
        self.tfidf_model = models.TfidfModel.load(prefix + "_tfidf.model")
        self.cesa_model = CosineEsaModel.load(prefix + "_cesa.model")

    def get_features(self, article):
        """Extract cosine ESA features from a document, transforming an existing tf-idf model space

        A tf-idf model space is transformed to an ESA model using Gensim. The implementation uses the previously,
        in TfidfFeatureExtractor, generated tf-idf model space, but reduces the dimensionality of the vector spaces by
        taking the cosine similarity of nearby veactors.

        Args:
            document: a text document (utf-8) with no formatting

        Returns:
            doc_tfidf: cosine ESA representation of the document as Gensim model object
        """
        # create list of tokens from doc
        logger.debug("Tokenize document.")
        document = article.clean_content.encode("utf8")
        tokens = TOKENIZER.tokenize(document)

        # create bow of doc from token list
        logger.debug("Create bag-of-words representation from article.")
        doc_bow = self.dictionary.doc2bow(tokens)

        # create tfidf representation from bag-of-words
        logger.debug("Transform to tfidf.")
        doc_tfidf = self.tfidf_model[doc_bow]

        #create cosine esa representation from lda
        logger.debug("Transform to cesa")
        doc_cesa = self.cesa_model[doc_tfidf]

        return doc_cesa

    def get_feature_number(self):
        """Get number of topics in the cosine ESA-model"""
        return len(self.cesa_model.document_titles)

    @classmethod
    def get_version(self):
        """Get version of the cosine ESA feature extractor"""
        return u"cESA-1.2"


class CLEsaFeatureExtractor(Extractor):
    def __init__(self, prefix, target_prefix=""):
        """
            prefix: path where models are saved
            target_prefix: additional prefix to select different
                variations of the target EN-ESA model
                ["lda_", "kmedoidsN_", "kbestN_", "small_"]

                (full EN-ESA is used if no prefix is given)
        """
        Extractor.__init__(self)
        self.prefix = prefix
        self.target_prefix = target_prefix
        logger.info("Started loading required data")
        self.load_dictionaries()
        self.load_tf_idf_models()
        self.load_esa_models()
        self.load_mappings()
        logger.info("Finished loading required data")

    def load_dictionaries(self):
        logger.info("Loading dictionaries")
        self.dictionaries = {
            # 'en': corpora.Dictionary.load(self.prefix + 'en_wordids.dict'),
            'es': corpora.Dictionary.load(self.prefix + 'es_wordids.dict'),
            # 'de': corpora.Dictionary.load(self.prefix + 'de_wordids.dict'),
            # 'nl': corpora.Dictionary.load(self.prefix + 'nl_wordids.dict')
        }

    def load_tf_idf_models(self):
        logger.info("Loading TF-IDF models")
        self.tf_idfs = {
            # 'en': models.TfidfModel.load(self.prefix + "en_tfidf.model"),
            'es': models.TfidfModel.load(self.prefix + "es_tfidf.model"),
            # 'de': models.TfidfModel.load(self.prefix + "de_tfidf.model"),
            # 'nl': models.TfidfModel.load(self.prefix + "nl_tfidf.model")
        }

    def load_esa_models(self):
        logger.info("Loading ESA models")

        self.esas = {
            # 'en': EsaModel.load(self.prefix + self.target_prefix + "en_esa_on_tfidf.model"),
            'es': EsaModel.load(self.prefix + "es_esa_on_tfidf.model"),
            # 'de': EsaModel.load(self.prefix + "de_esa_on_tfidf.model"),
            # 'nl': EsaModel.load(self.prefix + "nl_esa_on_tfidf.model")
        }

    def load_mappings(self):
        def to_coo(mapping):
            mapping = sparse.coo_matrix((mapping['data'], 
                (mapping['row'], mapping['col'])), shape=mapping['shape'])
            return mapping

        logger.info("Loading mapping functions")
        
        mapping_paths = {lang: self.prefix + '%s_to_%sen.npz' % (lang, self.target_prefix)\
                        for lang in ["es", "de", "nl"]}
        
        logger.info("...ES to (small)EN")
        es_mapping = to_coo(np.load(mapping_paths["es"]))
        
        # logger.info("...DE to (small)EN")
        # de_mapping = to_coo(np.load(mapping_paths["de"]))
        
        # logger.info("...NL to (small)EN")
        # nl_mapping = to_coo(np.load(mapping_paths["nl"]))
        
        self.transformations = {
            'es': es_mapping,
            # 'de': de_mapping,
            # 'nl': nl_mapping,
        }

    def get_features(self, article=None, text=None, lang=None):
        """Extract CL-ESA features for an article object or plain text
        Args:
            article: an article object, 
            text: a text document (utf-8) with no formatting

        Returns:
            vec: CL-ESA representation of the document as Gensim model object
        """
        if text is None:
            text = article.clean_content

        if lang is None:
            lang = langid.classify(text)[0]
        
        # if lang in CONFIG["supported_languages"]:
        logger.debug("Tokenizing document")
        tokenized_text = TOKENIZER.tokenize(text)
        
        logger.debug("Creating bag-of-words representation for article")
        bow = self.dictionaries[lang].doc2bow(tokenized_text)
        
        logger.debug("Transforming to TF-IDF model")
        tf_idf = self.tf_idfs[lang][bow]
        
        logger.debug("Transforming to ESA model")
        esa = self.esas[lang][tf_idf]
        if lang != 'en':
            esa = matutils.unitvec(esa * self.transformations[lang])
        # esa = coo_vector_to_tuples(sparse.coo_matrix(esa))

        return esa
        # else:
        #     logger.debug("Unknown language (%s) detected, articled ignored", lang)
        #     return None

    def get_feature_number(self):
        """Get the number of features"""
        return len(self.esas['en'].document_titles)

    @classmethod
    def get_version(self):
        """Get version of CL-ESA feature extractor"""
        return u"CL-ESA-0.1"


#####
# Extra Feature Extractors
#####
class VendorFeatureExtractor(Extractor):
    """
        Implements extraction of vendor feature
    """
    def __init__(self, prefix):
        Extractor.__init__(self)
        self.vendor_list_filename = prefix + "_vendor.list"
        self.load_or_create_vendor_list()
        new_vendor_list = [v.name for v in Vendor.objects(name__not__in=self.vendor_list)]
        self.vendor_list += new_vendor_list
        self.save_vendor_list()

    def get_features(self, article):
        vendor = article.vendor
        try:
            feature = self.vendor_list.index(article.vendor)
        except ValueError:
            self.vendor_list.append(article.vendor)
            self.save_vendor_list()
            feature = self.vendor_list.index(article.vendor)

        return [(0, feature)]

    def load_or_create_vendor_list(self):
        try:
            vendor_list_file = open(self.vendor_list_filename, 'r')
            self.vendor_list = pickle.load(vendor_list_file)
            vendor_list_file.close()
        except Exception:
            # Since we use a sparse representation for the features
            # index 0 must mean absence of this feature
            self.vendor_list = ["NO-VENDOR"]

    def save_vendor_list(self):
        vendor_list_file = open(self.vendor_list_filename, 'w')
        pickle.dump(self.vendor_list, vendor_list_file)
        vendor_list_file.close()

    def get_feature_number(self):
        return 1

    @classmethod
    def get_version(self):
        return u'VendorFeatures 0.9'
