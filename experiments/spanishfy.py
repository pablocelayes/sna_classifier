#!/usr/bin/env python
# -*- coding: utf-8 -*-
from datasets import *

tu_path = "/home/pablo/Proyectos/tesiscomp/experiments/_1_one_user_learn_neighbours/active_and_central.json"
TEST_USERS_ALL = json.load(open(tu_path))

def spanishfy_dataframe(uid):

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    s = open_session()
    for uid, _, _ in TEST_USERS_ALL:
        print uid        
        ds = load_dataframe(uid)
        if not ds:
            continue

        X_train, X_test, y_train, y_test = ds
        n_train = len(X_train)
        n_test = len(X_test)

        es_train_twids = [t.id for t in s.query(Tweet.id).\
                                        filter(Tweet.id.in_(X_train.index)).\
                                        filter(Tweet.lang=='es')]
        X_train['y'] = y_train
        X_train = X_train.loc[es_train_twids]

        es_test_twids = [t.id for t in s.query(Tweet.id).\
                                        filter(Tweet.id.in_(X_test.index)).\
                                        filter(Tweet.lang=='es')]
        X_test['y'] = y_test
        X_test = X_test.loc[es_test_twids]

        n_nones = (n_train + n_test) - (len(X_train) + len(X_test))
        if n_nones:
            Xall = pd.concat((X_train,X_test))
            y = Xall['y']; del Xall['y']

            X_train, X_test, y_train, y_test = train_test_split(Xall, y,
                                            test_size=0.3, random_state=42)

            print "%d Non ES tweets found for user %d" % (n_nones, uid)
            Xtrain_fname = join(DATAFRAMES_FOLDER, "dfXtrain_%d.pickle" % uid)
            Xtest_fname = join(DATAFRAMES_FOLDER, "dfXtest_%d.pickle" % uid)
            ys_fname = join(DATAFRAMES_FOLDER, "ys_%d.pickle" % uid)

            X_train.to_pickle(Xtrain_fname)
            X_test.to_pickle(Xtest_fname)
            pickle.dump((y_train, y_test), open(ys_fname, 'wb'))

# 228252737
# 142800528
# 18623370
# 37226353
# 195412602
# 54943340
# 1311735576
# 80462161
# 110325813
# 254316467
# 263780425
# 778653451
# 146431317
# 90453671
# 60239907
# 226011872
# 214777467
# 147078123
# 171990897
# 178731620
# 188525384
# 219788847
# 562601076
# 171110904
# 203030351
# 214986628
# 263872477
# 36997365
# 321389148
# 283762641
# 155304314
# 201413739
# 334364707
# 117183228
# 312686640
# 186584578
# 208575740
# 158116807
# 152613501
# 101047126
# 126527644
# 114582574
# 144310601
# 20611326
# 213047203
# 173222615
# 237994358
# 147800890
# 150783792
# 241991145
# 935088068
# 188895783
# 623334075
# 5033321
# 138835591
# 20292293
# 187453746
# 188617212
# 160768274
# 118500492
# 143685841
# 199982337
# 157523545
# 156351223
# 176669543
# 146209361
# 133945128
# 132704571
# 119748967
# 118547936
# 130959700
# 79063686
# 178432723
# 146715656
# 74660411
# 166576483
# 163481473
# 50361675
# 173550390
# 45220870
# 173348160
# 136487930
# 186595337
# 601015661
# 20489477
# 35710956
# 53738889
# 147249049
# 62255061
# 23486396
# 200302248
# 146655020
# 139914117
# 9527062
# 107176039
# 28784256
# 148193563
# 152821325
# 138098855
# 270144571
# 57686497
# 155919261
# 7888482
# 66145322
# 124222509
# 258968955
# 152769475
# 121039771
# 716152460
# 16469876
# 323417560
# 186693730
# 212650503
# 8366982
# 43564780
# 628409060
# 17620434
# 28668781
# 154738110
# 186998381
# 83119297
# 229480408
# 130234556
# 166670881
# 111866614
# 818229818
# 145709292
# 18145859
# 96022814
# 49651047
# 151540361
# 188376648
# 2722181
# 47781723
# 25066392
# 164824570
# 356464236
# 329410435
# 58209067
# 153127426
# 179673885
# 125363239
# 367853828
# 309812848
# 91919237
# 49041045
# 42976687
# 54313 Non ES tweets found for user 42976687
# 1625949817
# 146453553
# 127163121
# 67699758
# 183463551
# 263741712
# 75397992
# 118085767
# 169806496
# 171971212
# 87599044
# 130941579
# 74153376
# 7261072
# 164236576
# 154225689
# 87606356
# 221934374
# 256674213
# 223811224
# 157589057
# 1622441
# 213462875
# 151317587
# 166394396
# 295873172
# 186570446
# 143524017
# 148125585
# 42097354
# 210630945
# 50661819
# 822112
# 151058509
# 125734567
# 1383498264
# 13438282
# 197176019
# 37857874
# 128386349
# 71303860
# 196478764
# 1566310694
# 59857143
# 76684633
# 85123028
# 54987976