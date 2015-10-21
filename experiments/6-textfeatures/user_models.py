#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copyright (C) Cogfor <info@cogfor.com>
Copyright (C) karsten jeschkies <jeskar@web.de>
"""

"""
Several different user models for capturing a user's interests.

"""
import cPickle
from itertools import chain, izip
import logging

from random import sample

import numpy as np
from gensim import matutils, similarities

# from naive_bayes import GaussianNB #iterative GaussianNB
from sklearn.naive_bayes import GaussianNB
from sklearn import svm, tree
from smote import borderlineSMOTE

logger = logging.getLogger("user_models")

import os
from tw_dataset.settings import USER_MODELS_FOLDER
from tw_dataset.dbmodels import *

class UserModelBase(object):
    def __init__(self, user_id, extractor):
        self.user = User.objects(id=user_id).first()
        self.extractor = extractor
        self.num_features_ = self.extractor.get_feature_number()

    def train(self):
        logger.debug("train() not implemented!")
        raise NotImplementedError()

    def save(self):
        logger.debug("save() not implemented!")
        raise NotImplementedError()

    def load(self):
        logger.debug("load() not implemented!")
        raise NotImplementedError()

    def rank(self, doc):
        """
        Ranks a document with learnt model
        """
        logger.debug("rank() not implemented!")
        raise NotImplementedError()

    @classmethod
    def get_version(cls):
        logger.debug("get_version not implemented!")
        raise NotImplementedError()

    def get_features(self, tweet):
        """
        Returns full features vector from tweet.
        tweet should be an Tweet.
        """
        if feature_version != self.extractor.get_version():
            tweet.id = tweet.url
            tweet = Tweet.get(id=tweet.id)
            clean_content = tweet.clean_content

            clean_content.encode("utf8")

            # get new features
            new_features = self.extractor.get_features(tweet)

            #save new features
            features = Features(version=self.extractor.get_version(),
                                data=new_features)
            tweet.features = features
            try:
                tweet.save()
            except queryset.OperationError as e:
                logger.error(
                    "Could not save tweet with id %s: %s" % (tweet.id, e))

        # sparse2full converts list of 2-tuples to numpy array
        tweet_features_as_full_vec = matutils.sparse2full(
            tweet.features.data, self.num_features_)

        return tweet_features_as_full_vec


class UserModelCentroid(UserModelBase):
    """
    Trains a user model base on user feedback.

    It uses the centroid based user model described in:
    Han und Karypis "Centroid-Based Document Classification:
    Analysis & Experimental Results" ,2000
    """

    # Class vars
    READ = 2
    UNREAD = 1

    @classmethod
    def get_version(cls):
        return "UserModelCentroid-1.0"

    def __init__(self, user_id, extractor):
        super(UserModelCentroid, self).__init__(user_id, extractor)

        # instance vars
        self.user_model_features = []
        self.learned_user_model = UserModel.objects(
            user_id=self.user.id).first()

    def train(self, rt_tweet_ids=None, nort_tweet_ids=None):
        # Load user feedback if needed
        if rt_tweet_ids is None:
            rt_tweet_ids = (r.tweet_id for r in
                                ReadTweetFeedback.objects(
                                    user_id=self.user.id).only("tweet"))

        user_feedback = Tweet.objects(id__in=rt_tweet_ids)

        #TODO: cluster feedback tweets and save more than one profile

        num_loaded_tweets = 0
        centroid = np.zeros(self.num_features_, dtype=np.float32)

        for tweet in user_feedback:
            try:
                tweet_features_as_full_vec = self.get_features(tweet)
            except Exception as inst:
                logger.error("Could not get features for tweet %s: %s" % (
                    tweet.id, inst))
                continue

            #do we need this?
            tmp_doc = matutils.unitvec(tweet_features_as_full_vec)

            #add up tmp_doc
            centroid = np.add(centroid, tmp_doc)
            num_loaded_tweets += 1

        #average each element
        if num_loaded_tweets != 0:
            centroid = centroid / num_loaded_tweets

        centroid = matutils.full2sparse(centroid)

        #set user model data
        self.user_model_features = [centroid]

    def save(self):
        # replace old user model with new
        try:
            #replace profile
            #pickle classifier and decode it to utf-8
            #pickled_classifier = cPickle.dumps(self.user_model_features).decode('utf-8')
            UserModel.objects(user_id=self.user.id).update(upsert=True,
                                                           set__user_id=self.user.id,
                                                           set__data=self.user_model_features,
                                                           set__version=self.get_version())
        except Exception as inst:
            logger.error(
                "Could not save learned user model due to unknown error %s: %s" % (
                    type(inst), inst))

    def load(self):
        """
        Loads user model from self.user. Not yet from UserModel

        NOTE: No feature conversion is done!
        """
        user_model = UserModel.objects(user_id=self.user.id).first()

        if user_model is None:
            logger.debug("UserModel for user %s is empty." % self.user.id)
            self.clf = None
            return

        # learned_user_model = UserModel.objects(user_id=self.user.id).first()
        #
        #if learned_user_model is None:
        #    self.user_model_features = []
        #    return
        #
        #get learned profile/model
        #convert features to list of tuples.
        #we make a double list because we will have more than one model soon.
        #self.user_model_features = [[tuple(a) for a in learned_user_model.data] for profile in self.user.learned_profile]

        #unpickle classifier. it was saved as a utf-8 string.
        #get the str object by encoding it.
        #pickled_classifier = user_model.data.encode('utf-8')
        #self.clf = cPickle.loads(pickled_classifier)

        self.user_model_features = [tuple(a) for a in
                                    self.learned_user_model.data]

    def rank(self, doc):
        """
        Returns a ranking of the document to learnt model.

        doc should be instance of mongodb_models.Tweet
        """

        self.load()

        if len(self.user_model_features) == 0:
            logger.error("Learned user model seems to be empty.")
            return None

        index = similarities.SparseMatrixSimilarity(self.user_model_features,
                                                    num_terms=self.num_features_,
                                                    num_best=1,
                                                    num_features=self.num_features_,
                                                    num_docs=len(
                                                        self.user_model_features))

        # convert features to list of tuples
        news_tweet_features = list(tuple(a) for a in doc.features.data)

        logger.debug("converted news tweet features")

        #calculate similarities of tweet to each user model
        #will return best fit
        sim = index[news_tweet_features]
        logger.debug("Sim: %s" % list(sim))
        logger.debug("created similarities")

        # TODO: add check on sim if it is empty.
        #convert sim from numpy.float32 to native float
        try:
            native_sim = np.asscalar(sim[0][1])
        except IndexError as e:
            logger.error(
                "Could not access similarity: %s. Similarity: %s. User model: %s"
                % (e, sim, self.user_model_features))
            return self.UNREAD

        if native_sim > 0.3:
            return self.READ
        else:
            return self.UNREAD


class NoClassifier(Exception):
    pass


class UserModelBayes(UserModelBase):
    """
    Trains a user model base on user feedback.

    A Naive Bayes classifier is trained to decide if an tweet belongs to the
    "read" tweets or not.

    Does not use SMOTE
    """

    READ = 2
    UNREAD = 1

    def __init__(self, user_id, extractor):
        super(UserModelBayes, self).__init__(user_id, extractor)

    @classmethod
    def get_version(cls):
        return "UserModelBayes-1.0"

    class AllTweets(object):

        def __init__(self,
                     rt_tweets,
                     nort_tweets,
                     get_features_function):
            """
            Parameters:
            rt_tweets : Tweet Queryset
            nort_tweet : Tweet Queryset
            get_features_function : should be a function that takes an tweet
                                    as Tweet instance and returns the full
                                    features vector
            """
            self.rt_tweets = rt_tweets
            self.nort_tweets = nort_tweets
            self.get_features = get_features_function

        def _iter_features_and_marks(self):
            marked_rt_tweets = ((tweet, UserModelBayes.READ) for tweet
                                    in self.rt_tweets)
            marked_nort_tweets = ((tweet, UserModelBayes.UNREAD) for
                                      tweet in self.nort_tweets)

            all_tweets = chain(marked_rt_tweets, marked_nort_tweets)

            for tweet, mark in all_tweets:
                try:
                    tweet_features_as_full_vec = self.get_features(tweet)
                    yield tweet_features_as_full_vec, mark
                except AttributeError as e:
                    logger.error("Tweet %s does not have attribute: %s." % (
                        tweet.id, e))

        def __iter__(self):
            for a, _ in self._iter_features_and_marks():
                yield a

        def get_marks(self):
            for _, mark in self._iter_features_and_marks():
                yield mark

    def train(self, rt_tweet_ids=None, nort_tweet_ids=None):
        """
        Trains the Bayes Classifier.
        rt_tweet_ids should be an iterable over read tweet ids
        nort_tweet_ids should be an iterable over unread tweet ids

        If one is None it will be loaded from database.
        """

        # Load user feedback if needed
        if rt_tweet_ids is None:
            rt_tweet_ids = set(r.tweet_id for r
                                   in ReadTweetFeedback.objects(
                user_id=self.user.id).only("tweet_id"))
        else:
            rt_tweet_ids = set(rt_tweet_ids)

        logger.info(
            "Use %d read tweets for learning." % len(rt_tweet_ids))
        rt_tweets = Tweet.objects(id__in=rt_tweet_ids)

        #Get all tweets the user did not read.
        if nort_tweet_ids is None:
            all_ranked_tweet_ids = (r.tweet_id for r in RankedTweet.objects(
                user_id=self.user.id).only("tweet_id"))
            nort_tweet_ids = all_ranked_tweet_ids - rt_tweet_ids
            
        #undersample unreads
        logger.info(
            "Use %d unread tweets for learning." % (len(nort_tweet_ids)))

        nort_tweets = Tweet.objects(id__in=nort_tweet_ids)

        #convert all tweet features
        all_tweets = UserModelBayes.AllTweets(rt_tweets,
                                                  nort_tweets,
                                                  self.get_features)

        self.clf = GaussianNB()
        self.clf.fit(np.array(list(all_tweets)),
                     np.array(list(all_tweets.get_marks())))

    def save(self):
        # replace old user model with new
        try:
            #pickle classifier and decode it to utf-8
            pickled_classifier = cPickle.dumps(self.clf).decode('utf-8')

            #replace profile
            UserModel.objects(user_id=self.user.id).update(upsert=True,
                                                           set__user_id=self.user.id,
                                                           set__data=pickled_classifier,
                                                           set__version=self.get_version())

        except Exception as inst:
            logger.error(
                "Could not save learned user model due to unknown error %s: %s" % (
                    type(inst), inst))

    def load(self):
        try:
            if self.clf is not None:
                return

            user_model = UserModel.objects(user_id=self.user.id).first()

            if user_model is None:
                logger.debug("UserModel for user %s is empty." % self.user.id)
                self.clf = None
                return

            # ensure right version
            if user_model.version != self.get_version():
                logger.debug(
                    "UserModel for user %s has wrong version." % self.user.id)
                self.clf = None
                return

            #unpickle classifier. it was saved as a utf-8 string.
            #get the str object by encoding it.
            pickled_classifier = user_model.data.clf.encode('utf-8')

            pickled_classifier = user_model.data.encode('utf-8')
            self.clf = cPickle.loads(pickled_classifier)

        except Exception as inst:
            logger.error(
                "Could not load learned user model due to unknown error %s: %s" % (
                    type(inst), inst))

    def rank(self, doc):
        """
        doc should be instance of mongodb_models.Tweet
        """

        self.load()

        if self.clf is None:
            logger.error("No classifier for user %s." % self.user.id)
            raise NoClassifier(
                "Bayes Classifier for user %s seems to be None." % self.user.id)

        data = np.empty(shape=(1, self.num_features_), dtype=np.float32)

        data[0] = self.get_features(doc)
        prediction = self.clf.predict(data)

        return prediction[0]


class UserModelSVM(UserModelBayes):
    READ = 1
    UNREAD = 0

    @classmethod
    def get_version(cls):
        return "UserModelSVM-1.0"

    def __init__(self, user_id, extractor):
        self.set_samples_sizes()
        self.clf = None
        self.theta_ = None
        self.sigma_ = None
        super(UserModelSVM, self).__init__(user_id, extractor)

    def _calculate_mean_and_std_deviation(self, X):
        """
        Calculates mean and standard deviation of sample features.

        Parameters
        ----------
        X : array-like, samples, shape = (n_samples, n_features)
        """

        _, n_features = X.shape

        self.theta_ = np.zeros((n_features))
        self.sigma_ = np.zeros((n_features))
        epsilon = 1e-9

        self.theta_[:] = np.mean(X[:, :], axis=0)
        self.sigma_[:] = np.std(X[:, :], axis=0) + epsilon

    def _normalize(self, X):
        """
        Normalizes sample features.

        self.theta_ and self.sigma_ have to be set.

        Parameters
        ----------
        X : array-like, samples, shape = (n_samples, n_features)
        """
        n_samples, n_features = X.shape

        new_X = np.zeros(shape=(n_samples, n_features), dtype=np.float32)

        try:
            new_X[:, :] = (X[:, :] - self.theta_[:]) / self.sigma_[:]
        except AttributeError as e:
            logger.error(
                "theta_ or sigma_ are not set. Call _calculate_mean_and_std_deviation. Error: %s" % e)
            raise AttributeError()

        return new_X

    def _get_samples(self, rt_tweet_ids, nort_tweet_ids,
        p_synthetic_samples=300, p_majority_samples=500, k=5):
        """
        rt_tweet_ids : Set
        nort_tweet_ids : Set
        p_synthetic_samples : Percentage of synthetic samples, 300 for 300%
                              If None no are created
        p_majority_samples : Size of majority sample = p_majority_samples/n_minority_sample,
                             500 for 500%
                             If None under sampling ist not done
        k : neighbourhood for k nearest neighbour, standard 5

        Returns
        -------
        array-like full vector samples, shape = [n_features, n_samples]
        array-like marks, shape = [n_samples]
        """

        # Under-sample unread ids
        if p_majority_samples is not None:
            nort_tweet_ids = set(sample(nort_tweet_ids,
                                            min(p_majority_samples / 100 * len(
                                                rt_tweet_ids),
                                                len(nort_tweet_ids))
            )
            )

        #Create unread tweet vectors
        nort_marks = np.empty(len(nort_tweet_ids))
        nort_marks.fill(UserModelSVM.UNREAD)
        nort_tweets = np.empty(
            shape=(len(nort_tweet_ids), self.num_features_))

        s = open_session()
        nort_tweets = s.query(Tweet).filter(Tweet.id.in_(nort_tweet_ids)).all()
        for i, tweet in enumerate(nort_tweets):
            try:
                tweet_features_as_full_vec = self.get_features(tweet)
                nort_tweets[i, :] = tweet_features_as_full_vec[:]
            except AttributeError as e:
                logger.error("Tweet %s does not have attribute: %s. Index %d"
                             % (nort_tweet_ids[i], e, i))

        #Create read tweet vectors
        rt_marks = np.empty(len(rt_tweet_ids))
        rt_marks.fill(UserModelSVM.READ)
        rt_tweets = np.empty(
            shape=(len(rt_tweet_ids), self.num_features_))

        rt_tweets = s.query(Tweet).filter(Tweet.id.in_(rt_tweet_ids)).all()
        for i, tweet in enumerate(rt_tweets):
            try:
                tweet_features_as_full_vec = self.get_features(tweet)
                rt_tweets[i, :] = tweet_features_as_full_vec[:]
            except AttributeError as e:
                logger.error(
                    "Tweet %s does not have attribute: %s." % (tweet.id, e))
        s.close()

        #SMOTE sample minorities
        #synthetic_rt_tweets = SMOTE(rt_tweets, p_synthetic_samples, k)

        #borderlineSMOTE sample minorities if p_synthetic_samples not None
        X = np.concatenate((rt_tweets, nort_tweets))

        self._calculate_mean_and_std_deviation(X)
        X = self._normalize(X)

        y = np.concatenate((rt_marks, nort_marks))
        if p_synthetic_samples is None:
            return X, y
        else:
            new_rt_tweets, synthetic_rt_tweets, danger_rt_tweets = \
                borderlineSMOTE(X=X, y=y, minority_target=UserModelSVM.READ,
                                N=p_synthetic_samples, k=k)

            #Create synthetic read samples
            synthetic_marks = np.zeros(len(synthetic_rt_tweets))
            synthetic_marks.fill(UserModelSVM.READ)

            rt_marks = np.empty(len(new_rt_tweets))
            rt_marks.fill(UserModelSVM.READ)

            danger_rt_marks = np.empty(len(danger_rt_tweets))
            danger_rt_marks.fill(UserModelSVM.READ)

            logger.info(
                "Use %d read, %d unread, %d danger reads and %d synthetic samples." %
                (len(rt_marks), len(nort_marks),
                 len(danger_rt_marks), len(synthetic_marks)))

            return (np.concatenate((new_rt_tweets,
                                    synthetic_rt_tweets,
                                    danger_rt_tweets,
                                    nort_tweets)),
                    np.concatenate((rt_marks,
                                    synthetic_marks,
                                    danger_rt_marks,
                                    nort_marks))
            )

    def set_samples_sizes(self, p_synthetic_samples=None,
                          p_majority_samples=None):
        self.p_synthetic_samples = p_synthetic_samples
        self.p_majority_samples = p_majority_samples

    def train(self, rt_tweet_ids=None, nort_tweet_ids=None):
        """
        Trains the SVM Classifier.
        rt_tweet_ids should be an iterable over read tweet ids
        nort_tweet_ids should be an iterable over unread tweet ids

        If one is None it will be loaded from database.
        """
        # Load user feedback if needed
        if rt_tweet_ids is None:
            feedback = ReadTweetFeedback.objects(user_id=self.user.id, score=1.0).only("tweet_id")
            rt_tweet_ids = set(r.tweet_id for r in feedback if r.tweet_id)
        else:
            rt_tweet_ids = set(rt_tweet_ids)

        #Get tweets that were shown to the user but she did not read.
        if nort_tweet_ids is None:
            shown_tweets = ReadTweetFeedback.objects(user_id=self.user.id).only("tweet_id")
            # TODO: figure out why some tweet_id's can be None
            shown_tweet_ids = list(a.tweet_id for a in shown_tweets if a.tweet_id)
            shown_tweet_ids = set(a["id"] for a in Tweet.objects(id__in=shown_tweet_ids).only("id"))
            nort_tweet_ids = shown_tweet_ids - rt_tweet_ids

        # TODO: remove this patch
        # happens only for old users that haven't used frontend recently
        # (old frontend didn't record feedback for unread tweets)
        if not nort_tweet_ids:
            nread = len(rt_tweet_ids)
            enough_tweet_ids = Tweet.get_ids(nresults=2 * nread)
            nort_tweet_ids = [id for id in enough_tweet_ids if not id in rt_tweet_ids]

            # FIX: some tweets with no content are throwing errors
            # nort_tweet_ids = [id for id in enough_tweet_ids if Tweet.get(id=id).clean_content]            
            nort_tweet_ids = nort_tweet_ids[:nread]
      
        #convert all tweet features
        if not rt_tweet_ids or not nort_tweet_ids:
            logger.info("Not enough read/unread information to learn")
        else:
            all_tweets, marks = self._get_samples(rt_tweet_ids,
                                                    nort_tweet_ids,
                                                    p_synthetic_samples=self.p_synthetic_samples,
                                                    p_majority_samples=self.p_majority_samples)

            logger.info("Learn on %d samples." % len(marks))

            self.clf = svm.SVC(kernel='linear')
            logger.info("Tweets found: %s" % all_tweets)
            logger.info("Classes found: %s" % marks)

            self.clf.fit(all_tweets, marks)

    def save(self):
        # replace old user model with new
        try:
            #pickle classifier and decode it to utf-8
            pickled_classifier = cPickle.dumps(self.clf).decode('utf-8')

            pickled_theta = cPickle.dumps(self.theta_).decode('utf-8')
            pickled_sigma = cPickle.dumps(self.sigma_).decode('utf-8')

            data = {'clf': pickled_classifier,
                    'theta': pickled_theta,
                    'sigma': pickled_sigma,
                    'version': self.get_version()}

            user_model_path = os.path.join(USER_MODELS_FOLDER, "%s.pickle" % self.user.id)
            cPickle.dump(data, open(user_model_path, 'wb'))

            #replace profile
            # UserModel.objects(user_id=self.user.id).update(upsert=True,
            #                                                set__user_id=self.user.id,
            #                                                set__data=data,
            #                                                set__version=self.get_version())

        except Exception as inst:
            logger.error(
                "Could not save learned user model due to unknown error %s: %s" % (
                    type(inst), inst))

    def load(self):
        logger.debug("Load user model data from MongoDB")
        logger.debug("CLF: %s" % self.clf)
        # try:
        if self.clf is not None:
            logger.debug("CLF is not None")
            return

        logger.debug("Get User_Model")
        try:
            user_model_path = os.path.join(USER_MODELS_FOLDER, "%s.pickle" % self.user.id)
            user_model_data = cPickle.load(open(user_model_path, 'rb'))
        except IOError:
            user_model_data = None
        
        if user_model_data is None:
            logger.debug("UserModel for user %s is empty." % self.user.id)
            self.clf = None
            return

        logger.debug("Check user_model version")
        #ensure right version
        if user_model_data["version"] != self.get_version():
            logger.debug(
                "UserModel for user %s has wrong version." % self.user.id)
            self.clf = None
            return

        logger.debug("Load Pickled CLF, Theta and Sigma")
        #unpickle classifier. it was saved as a utf-8 string.
        #get the str object by encoding it.
        pickled_classifier = user_model_data['clf'].encode('utf-8')
        pickled_theta = user_model_data['theta'].encode('utf-8')
        pickled_sigma = user_model_data['sigma'].encode('utf-8')

        logger.debug("unpickle CLF, Theta and Sigma")
        self.clf = cPickle.loads(pickled_classifier)
        self.theta_ = cPickle.loads(pickled_theta)
        self.sigma_ = cPickle.loads(pickled_sigma)

        logger.debug("Theta: %s" % self.theta_)
        logger.debug("Sigma: %s" % self.sigma_)

        # except Exception as inst:
        #     logger.error("Could not load learned user model due to unknown error %s: %s" % (type(inst), inst))

    def rank(self, doc):
        """
        doc should be instance of mongodb_models.Tweet
        """
        logger.debug("Loading User Model")

        self.load()
        # try:
        # self.load()
        # except Exception:
        #     logger.error("Failed to load User Model")

        logger.debug("Start Ranking")
        logger.debug(self.theta_)
        logger.debug(self.sigma_)

        if self.clf is None:
            logger.error("No classifier for user %s." % self.user.id)
            raise NoClassifier(
                "SVM Classifier for user %s seems to be None." % self.user.id)

        data = np.empty(shape=(1, self.num_features_), dtype=np.float32)

        data[0] = self.get_features(doc)
        data = self._normalize(data)
        prediction = self.clf.predict(data)

        return prediction[0]


class UserModelTree(UserModelSVM):
    READ = 2
    UNREAD = 1

    @classmethod
    def get_version(cls):
        return "UserModelMeta-1.0"

    def train(self, rt_tweet_ids=None, nort_tweet_ids=None):
        """
        Trains the DecisionTree Classifier.
        rt_tweet_ids should be an iterable over read tweet ids
        nort_tweet_ids should be an iterable over unread tweet ids

        If one is None it will be loaded from database.
        """

        # Load user feedback if needed
        if rt_tweet_ids is None:
            rt_tweet_ids = set(r.tweet_id
                                   for r in ReadTweetFeedback.objects(
                user_id=self.user.id).only("tweet_id"))
        else:
            rt_tweet_ids = set(rt_tweet_ids)

        # Get all tweets the user did not read.
        if nort_tweet_ids is None:
            ranked_tweet_ids = (a.tweet_id
                                  for a in RankedTweet.objects(
                user_id=self.user.id).only("tweet_id"))
            all_tweet_ids = set(a["id"] for a in Tweet.objects(
                id__in=ranked_tweet_ids).only("id"))
            nort_tweet_ids = all_tweet_ids - rt_tweet_ids

        # convert all tweet features
        all_tweets, marks = self._get_samples(rt_tweet_ids,
                                                nort_tweet_ids,
                                                p_synthetic_samples=self.p_synthetic_samples,
                                                p_majority_samples=self.p_majority_samples)

        logger.debug("Learn on %d samples." % len(marks))

        self.clf = tree.DecisionTreeClassifier()
        self.clf.fit(all_tweets, marks)


class UserModelMeta(UserModelSVM):
    READ = 2
    UNREAD = 1

    @classmethod
    def get_version(cls):
        return "UserModelMeta-1.0"

    def _call_classifiers(self, classifiers, parameters):
        """
        Calls

        tweets, marks = self._get_samples(rt_tweet_ids,
                                            nort_tweet_ids,
                                            p_synthetic_samples = 300,
                                            p_majority_samples = 500,
                                            k = 10)
        clf = classifier(kernel='rbf')
        clf.fit(tweets, marks)

        for each classifier and parameter set

        Parameters
        ----------
        classifiers : iterable of classifier classes (not instances)
        parameters : an iterable of dictionaries of
                     form {rt_tweet_ids,
                           nort_tweet_ids,
                           p_synthetic_samples = 300,
                           p_majority_samples = 500,
                           k = 10}
        """
        self.classifiers_ = list()
        for classifier, param_dict in izip(classifiers, parameters):
            tweets, marks = self._get_samples(**param_dict)
            clf = classifier()
            clf.fit(tweets, marks)
            self.classifiers_.append(clf)

    def train(self, rt_tweet_ids=None, nort_tweet_ids=None):
        """
        Trains the several SVM and Naive Bayes Classifiers.
        rt_tweet_ids should be an iterable over read tweet ids
        nort_tweet_ids should be an iterable over unread tweet ids

        If one is None it will be loaded from database.
        """

        # Load user feedback if needed
        if rt_tweet_ids is None:
            rt_tweet_ids = set(r.tweet_id
                                   for r in ReadTweetFeedback.objects(
                user_id=self.user.id).only("tweet_id"))
        else:
            rt_tweet_ids = set(rt_tweet_ids)

        #Get all tweets the user did not read.
        if nort_tweet_ids is None:
            ranked_tweet_ids = (a.tweet_id
                                  for a in RankedTweet.objects(
                user_id=self.user.id).only("tweet_id"))
            all_tweet_ids = set(a["id"] for a in Tweet.objects(
                id__in=ranked_tweet_ids).only("id"))
            nort_tweet_ids = all_tweet_ids - rt_tweet_ids

        classifiers = [lambda: svm.SVC(kernel='rbf'),
                       lambda: svm.SVC(kernel='rbf'),
                       lambda: svm.SVC(kernel='rbf'),
                       lambda: svm.SVC(kernel='rbf'),
                       lambda: svm.SVC(kernel='rbf'),
                       GaussianNB,
                       GaussianNB,
                       GaussianNB,
                       GaussianNB]

        parameters = [  # SVM
                        {'rt_tweet_ids': rt_tweet_ids,
                         'nort_tweet_ids': nort_tweet_ids,
                         'p_synthetic_samples': 100,
                         'p_majority_samples': 200,
                         'k': 10},
                        {'rt_tweet_ids': rt_tweet_ids,
                         'nort_tweet_ids': nort_tweet_ids,
                         'p_synthetic_samples': 200,
                         'p_majority_samples': 300,
                         'k': 10},
                        {'rt_tweet_ids': rt_tweet_ids,
                         'nort_tweet_ids': nort_tweet_ids,
                         'p_synthetic_samples': 300,
                         'p_majority_samples': 400,
                         'k': 10},
                        {'rt_tweet_ids': rt_tweet_ids,
                         'nort_tweet_ids': nort_tweet_ids,
                         'p_synthetic_samples': 400,
                         'p_majority_samples': 500,
                         'k': 10},
                        {'rt_tweet_ids': rt_tweet_ids,
                         'nort_tweet_ids': nort_tweet_ids,
                         'p_synthetic_samples': 500,
                         'p_majority_samples': 600,
                         'k': 10},
                        # Naive Bayes
                        {'rt_tweet_ids': rt_tweet_ids,
                         'nort_tweet_ids': nort_tweet_ids,
                         'p_synthetic_samples': 100,
                         'p_majority_samples': 100,
                         'k': 10},
                        {'rt_tweet_ids': rt_tweet_ids,
                         'nort_tweet_ids': nort_tweet_ids,
                         'p_synthetic_samples': 100,
                         'p_majority_samples': 200,
                         'k': 10},
                        {'rt_tweet_ids': rt_tweet_ids,
                         'nort_tweet_ids': nort_tweet_ids,
                         'p_synthetic_samples': 300,
                         'p_majority_samples': 500,
                         'k': 10},
                        {'rt_tweet_ids': rt_tweet_ids,
                         'nort_tweet_ids': nort_tweet_ids,
                         'p_synthetic_samples': 600,
                         'p_majority_samples': 600,
                         'k': 10}]

        self._call_classifiers(classifiers, parameters)

    def rank(self, doc):
        """
        doc should be instance of mongodb_models.Tweet
        """

        # self.load()

        #check if classifiers were loaded

        data = np.empty(shape=(1, self.num_features_), dtype=np.float32)

        data[0] = self.get_features(doc)
        predictions = np.empty(shape=(len(self.classifiers_)))
        for i, clf in enumerate(self.classifiers_):
            predictions[i] = clf.predict(data)

        #Evaluate votes
        uniques = np.unique(predictions)

        if len(uniques) == 1:
            return uniques[0]
        else:  # So far all classifiers have to vote for READ to have it READ
            return self.UNREAD
