from experiments._1_one_user_learn_neighbours.try_some_users import *
from experiments.utils import *
from collections import defaultdict

uid = 37226353

X_train, X_test, y_train, y_test = load_or_create_dataset(uid)

X = X_train
y = y_train
Xy = merge_Xy(X, y)

pos_samples = [r for r in Xy_u if r[-1] == 1]
neg_samples = [r for r in Xy_u if r[-1] == 0]

np.mean([sum(x[:-1]) for x in pos_samples])
np.mean([sum(x[:-1]) for x in neg_samples])

np.max([sum(x[:-1]) for x in neg_samples])
np.max([sum(x[:-1]) for x in pos_samples])


neighbours = get_neighbourhood(uid)

miss_clf_counts, details = count_doomed_samples(X_train, y_train, neighbours)

# Esto es lo que perdemos de recall en la clase 1
print miss_clf_counts[1] / sum(y_train)
