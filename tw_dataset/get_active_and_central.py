from experiments._1_one_user_learn_neighbours.try_some_users import *
from experiments.utils import get_most_central_twids

len(TEST_USERS_ALL)
s = open_session()
most_central = s.query(User).filter(User.id.in_(most_central_twids)).all()
most_central_twids = get_most_central_twids(1000)
most_central_twids
sess = open_session()
rtcounts = {u: len(u.retweets) for u in sess.query(User).all()}
most_active = sorted(rtcounts.items(), key=lambda x:-x[1])
most_active
most_active = [u.id for u in most_active]
most_active_twids = [u[0].id for u in most_active]
most_active_twids
len(most_active_twids )
most_active = most_active[:1000]
most_active_twids = [u[0].id for u in most_active]
len(most_active_twids)
len(most_central_twids)
most_active_twids
most_central_twids
most_central_twids = [int(x) for x in most_central_twids]
active_and_central_ids = list(set(most_active_ids).intersection(set(most_central_ids)))
active_and_central_ids = list(set(most_active_twids).intersection(set(most_central_twids)))
len(active_and_central_ids)
len(most_active_twids)
len(most_central_twids)
