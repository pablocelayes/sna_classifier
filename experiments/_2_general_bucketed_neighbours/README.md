More general approach:

For each central user, we take all other users (or a set of relevant neighbors)
and we group them in buckets according to a certain affinity metric relative
to the chosen central user.

Now for each tweet, features are the numbers of users that posted the tweet within each bucket. This feature model is not based on a fixed set of users
and can be applied to different central users.

We will use to try to train a classifier that performs well in the general
case. Even though the model is more general, it is worth noting that a tweet
will be assigned different feature vectors for each central user.

============
First try: we reuse the datasets from the previous experiment (one user learn neighbours)
and see how the classifier performs for those users.




============
Next step: Build a more representative dataset based on more users.

Train on one user test on others