import json
prec_social = json.load(open('scores/precisions_testv_svc.json'))
recall_social = json.load(open('scores/recalls_testv_svc.json'))
cd ..
cd _60_nlpmodels/
ls
cd scores/
ls
prec_lda10 = json.load(open('t10_precisions_test_svc.json'))
recall_lda10 = json.load(open('t10_recalls_test_svc.json'))
prec_lda20 = json.load(open('t20_precisions_test_svc.json'))
recall_lda20 = json.load(open('t20_recalls_test_svc.json'))
from copy import deepcopy
prec_social_lda10 = deepcopy(prec_social)
prec_social_lda10.update(prec_lda10)
prec_social_lda20 = deepcopy(prec_social)
prec_social_lda20.update(prec_lda20)
%hist
recall_social_lda10 = deepcopy(recall_social)
recall_social_lda10.update(recall_lda10)
recall_social_lda20 = deepcopy(recall_social)
recall_social_lda20.update(recall_lda20)
ls
pwd
ls
mkdir social_lda
cd social_lda/
with open('recall_social_lda10_test.json', 'w') as f:json.dump(recall_social_lda10, f)
ls
with open('recall_social_lda10_test.json', 'w') as f:
    json.dump(recall_social_lda10, f)

with open('prec_social_lda10_test.json', 'w') as f:
    json.dump(prec_social_lda10, f)

with open('recall_social_lda20_test.json', 'w') as f:
    json.dump(recall_social_lda20, f)

with open('prec_social_lda20_test.json', 'w') as f:
    json.dump(prec_social_lda20, f)
