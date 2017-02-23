from config import CONFIG
import os
from gensim import models
from gensim import corpora
from nyan.shared_modules.feature_extractor.esa.esamodel import EsaModel
from nyan.shared_modules.feature_extractor.esa.document_titles import DocumentTitles

from wikiextract.mappings import generate_mappings

language = "en"
NUM_TOPICS = 2000

TF_IDF_PATH = os.path.join(CONFIG['prefix'], language + "_tfidf.model")
TF_IDF = models.TfidfModel.load(TF_IDF_PATH)

TF_IDF_CORPUS_PATH = os.path.join(CONFIG['prefix'], language + "_tfidf_corpus.mm")
MM_TF_IDF = corpora.MmCorpus(TF_IDF_CORPUS_PATH)

ARTICLES_PATH = os.path.join(CONFIG['prefix'], language + "_articles.txt")
ARTICLE_TITLES = DocumentTitles.load(ARTICLES_PATH)

SMALL_EN_ESA_PATH = os.path.join(CONFIG['prefix'], "en_esa%d_on_tfidf.model" % NUM_TOPICS)
SMALL_EN_ESA = EsaModel(MM_TF_IDF, document_titles=ARTICLE_TITLES, num_features=50000,
                num_concepts=NUM_TOPICS, lang='en')
SMALL_EN_ESA.save(SMALL_EN_ESA_PATH)

SMALL_EN_ESA = EsaModel.load(SMALL_EN_ESA_PATH)

# Regenerate mappings
generate_mappings()
