import json
from copy import deepcopy

prec_social = json.load(open('../../_1_one_user_learn_neighbours/scores/precisions_testv_svc.json'))
recall_social = json.load(open('../../_1_one_user_learn_neighbours/scores/recalls_testv_svc.json'))

# prec_lda10 = json.load(open('tw_t10_precisions_test_svc.json'))
# recall_lda10 = json.load(open('tw_t10_recalls_test_svc.json'))

prec_lda15 = json.load(open('tw_t15_precisions_test_svc.json'))
recall_lda15 = json.load(open('tw_t15_recalls_test_svc.json'))

# prec_lda20 = json.load(open('tw_t20_precisions_test_svc.json'))
# recall_lda20 = json.load(open('tw_t20_recalls_test_svc.json'))

# prec_social_twlda10 = deepcopy(prec_social)
# prec_social_twlda10.update(prec_lda10)

# recall_social_twlda10 = deepcopy(recall_social)
# recall_social_twlda10.update(recall_lda10)

prec_social_twlda15 = deepcopy(prec_social)
prec_social_twlda15.update(prec_lda15)

recall_social_twlda15 = deepcopy(recall_social)
recall_social_twlda15.update(recall_lda15)

# prec_social_twlda20 = deepcopy(prec_social)
# prec_social_twlda20.update(prec_lda20)

# recall_social_twlda20 = deepcopy(recall_social)
# recall_social_twlda20.update(recall_lda20)

# with open('social_lda/recall_social_twlda10_test.json', 'w') as f:
#     json.dump(recall_social_twlda10, f)

# with open('social_lda/prec_social_twlda10_test.json', 'w') as f:
#     json.dump(prec_social_twlda10, f)

with open('social_lda/recall_social_twlda15_test.json', 'w') as f:
    json.dump(recall_social_twlda15, f)

with open('social_lda/prec_social_twlda15_test.json', 'w') as f:
    json.dump(prec_social_twlda15, f)

# with open('social_lda/recall_social_twlda20_test.json', 'w') as f:
#     json.dump(recall_social_twlda20, f)

# with open('social_lda/prec_social_twlda20_test.json', 'w') as f:
#     json.dump(prec_social_twlda20, f)
