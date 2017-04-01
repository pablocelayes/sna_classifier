
# coding: utf-8

# In[3]:

from experiments._1_one_user_learn_neighbours.try_some_users import *
from experiments.utils import *
from experiments.datasets import *
import json


# In[4]:

with open('nlp_users.json') as f:
    users = json.load(f)


# In[6]:

uid, f = users[0]; uid = int(uid)


# In[7]:

X_train, X_valid, X_test, y_train, y_valid, y_test = load_small_validation_dataframe(uid)


# In[8]:

s = open_session()

train_tweets = [s.query(Tweet).get(twid) for twid in X_train.index]

valid_tweets = [s.query(Tweet).get(twid) for twid in X_valid.index]


# # Load TFIDF model

# In[9]:

import pickle
with open('tfidf.pickle', 'rb') as f:
    tfidf = pickle.load(f)

with open('vec.pickle', 'rb') as f:
    vec = pickle.load(f)


# ## explore tfidf features

# In[10]:

text = train_tweets[0].text


# In[11]:

tvec = tfidf.transform(vec.transform([text]))


# In[12]:

def tfidf_feats(text):
    tvec = tfidf.transform(vec.transform([text]))

    ivocab = {i: w for (w,i) in vec.vocabulary_.items()}

    return dict(zip([ivocab[i] for i in tvec.indices], tfidf.idf_[tvec.indices]))

tfidf_feats(train_tweets[0].text)


# # Generate TFIDF features

# In[13]:

X_train_f = tfidf.transform(vec.transform([t.text for t in train_tweets]))


# In[14]:

X_valid_f = tfidf.transform(vec.transform([t.text for t in valid_tweets]))


# In[15]:

X_train_f.shape


# In[16]:

X_valid_f.shape


# In[10]:

# fname = join(DATASETS_FOLDER, "tfidf_%d.pickle" % uid)
# dataset = (X_train, X_test, y_train, y_test)
# pickle.dump(dataset, open(fname, 'wb'))


# In[ ]:

# dataset = pickle.load(open(fname, 'rb'))
# X_train, X_test, y_train, y_test = dataset


# ## normalize features

# In[17]:

from sklearn.preprocessing import StandardScaler


# In[20]:

from scipy.sparse import vstack


def scale(X_train, X_test):
    train_size = X_train.shape[0]
    X = np.concatenate((X_train.todense(), X_test.todense()))
    X = StandardScaler().fit_transform(X)
    X_train = X[:train_size,:]
    X_test = X[train_size:,:]

    return X_train, X_test


# ## class balancing sample weights

# In[21]:

# weights for class balancing
w1 = sum(y_train)/len(y_train)
w0 = 1 - w1
sample_weights = np.array([w0 if x==0 else w1 for x in y_train])


# In[22]:

import scipy.sparse as sp

X_train_combined = sp.hstack((X_train,X_train_f), format='csr')


# In[23]:

X_valid_combined = sp.hstack((X_valid,X_valid_f), format='csr')


# In[24]:

X_train_combined, X_valid_combined = scale(X_train_combined, X_valid_combined)


# In[25]:

clf = RandomForestClassifier(n_jobs=-1, n_estimators=50)     
clf.fit(X_train_combined, y_train, sample_weight=sample_weights)
evaluate_model(clf, X_train_combined, X_valid_combined, y_train, y_valid)


# In[ ]:

ds_comb = (X_train_combined, X_valid_combined, y_train, y_valid)
# comb_clf = model_select_svc(ds_comb, n_jobs=4)


# In[28]:

from sklearn.svm import SVC

clf = SVC(**{'kernel': 'rbf', 'C': 1, 'gamma': 0.1, 'class_weight': 'balanced'})
clf.fit(X_train_combined, y_train, sample_weight=sample_weights)
evaluate_model(clf, X_train_combined, X_valid_combined, y_train, y_valid)


# In[26]:

uid


# In[ ]:



