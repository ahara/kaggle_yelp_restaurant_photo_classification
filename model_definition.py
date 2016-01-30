from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier


model_definition = dict()

# Supported data sets
model_definition['top10'] = {}
model_definition['top20'] = {}
model_definition['top10_max'] = {}
model_definition['top10__iv3'] = {}

# Top 10 InceptionBN
model_definition['top10']['knn33'] = KNeighborsClassifier(n_neighbors=33, weights='distance',
                                                          random_state=0)  # 0.7562
model_definition['top10']['knn21'] = KNeighborsClassifier(n_neighbors=21, weights='distance',
                                                          random_state=0)  # 0.7547
model_definition['top10']['mnb'] = OneVsRestClassifier(MultinomialNB())  # 0.7548
model_definition['top10']['ada'] = OneVsRestClassifier(AdaBoostClassifier(n_estimators=180,
                                                                          random_state=0))  # 0.7617
model_definition['top10']['rf'] = RandomForestClassifier(n_estimators=500, bootstrap=False,
                                                         random_state=0, n_jobs=4)  # 7729
model_definition['top10']['svc'] = OneVsRestClassifier(SVC(C=.05, kernel='linear', random_state=0,
                                                           probability=True))  # 0.7482
model_definition['top10']['baglr'] = OneVsRestClassifier(BaggingClassifier(
    base_estimator=LogisticRegression(C=0.5, penalty='l1', fit_intercept=False, random_state=23),
    n_estimators=50, max_samples=1., max_features=0.6, bootstrap=True, random_state=23))  # 0.7877
model_definition['top10']['gbm'] = OneVsRestClassifier(GradientBoostingClassifier(
    n_estimators=145, learning_rate=0.1, random_state=1))  # 0.7918

# Top 20 InceptionBN
model_definition['top20']['knn34'] = KNeighborsClassifier(n_neighbors=34, weights='distance',
                                                          random_state=0)  # 0.7594
model_definition['top20']['knn20'] = KNeighborsClassifier(n_neighbors=20, weights='distance',
                                                          random_state=0)  # 0.7566
model_definition['top20']['mnb'] = OneVsRestClassifier(MultinomialNB())  # 0.7546
model_definition['top20']['ada'] = OneVsRestClassifier(AdaBoostClassifier(n_estimators=180,
                                                                          random_state=0))  # 0.7639
model_definition['top20']['rf'] = RandomForestClassifier(n_estimators=490, bootstrap=False,
                                                         random_state=6, n_jobs=4)  # 0.7737
model_definition['top20']['svc'] = OneVsRestClassifier(
    SVC(C=.05, kernel='linear', random_state=0, probability=True))  # 0.7481
model_definition['top20']['baglr'] = OneVsRestClassifier(BaggingClassifier(
    base_estimator=LogisticRegression(C=0.7, penalty='l1', fit_intercept=False, random_state=21),
    n_estimators=50, max_samples=1., max_features=0.6, bootstrap=True, random_state=21))  # 0.7901
model_definition['top20']['gbm'] = OneVsRestClassifier(GradientBoostingClassifier(
    n_estimators=215, learning_rate=0.1, random_state=1))  # 0.7888

# Top 10 max InceptionBN
model_definition['top10_max']['knn30'] = KNeighborsClassifier(n_neighbors=30, weights='distance',
                                                              random_state=0)  # 0.7519
model_definition['top10_max']['knn22'] = KNeighborsClassifier(n_neighbors=22, weights='distance',
                                                              random_state=0)  # 0.7528
model_definition['top10_max']['mnb'] = OneVsRestClassifier(MultinomialNB())  # 7588
model_definition['top10_max']['ada'] = OneVsRestClassifier(AdaBoostClassifier(n_estimators=160,
                                                                              random_state=0))  # 0.7555
model_definition['top10_max']['rf'] = RandomForestClassifier(n_estimators=420, bootstrap=False,
                                                             random_state=3, n_jobs=4)  # 7680
model_definition['top10_max']['svc'] = OneVsRestClassifier(
    SVC(C=.05, kernel='linear', random_state=0, probability=True))  # 0.7862
model_definition['top10_max']['baglr'] = OneVsRestClassifier(BaggingClassifier(
    base_estimator=LogisticRegression(C=.8, penalty='l1', fit_intercept=False, random_state=21),
    n_estimators=50, max_samples=1., max_features=0.6, bootstrap=True, random_state=21))  # 0.7853
model_definition['top10_max']['gbm'] = OneVsRestClassifier(GradientBoostingClassifier(
    n_estimators=185, learning_rate=0.1, random_state=1))  # 0.7886

# Top 10 Inception v3
model_definition['top10__iv3']['knn32'] = KNeighborsClassifier(n_neighbors=32, weights='distance',
                                                               random_state=0)  # 0.7275
model_definition['top10__iv3']['knn14'] = KNeighborsClassifier(n_neighbors=14, weights='distance',
                                                               random_state=0)  # 0.7290
model_definition['top10__iv3']['mnb'] = OneVsRestClassifier(MultinomialNB())  # 0.7475
model_definition['top10__iv3']['ada'] = OneVsRestClassifier(AdaBoostClassifier(n_estimators=220,
                                                                               random_state=0))  # 0.7667
model_definition['top10__iv3']['rf'] = RandomForestClassifier(n_estimators=500, bootstrap=False,
                                                              random_state=0, n_jobs=4)  # 7730
model_definition['top10__iv3']['svc'] = OneVsRestClassifier(SVC(C=.03, kernel='linear',
                                                                random_state=0, probability=True))  # 0.7493
model_definition['top10__iv3']['baglr'] = OneVsRestClassifier(BaggingClassifier(
    base_estimator=LogisticRegression(C=.4, penalty='l1', fit_intercept=False, random_state=20),
    n_estimators=50, max_samples=1., max_features=0.6, bootstrap=True, random_state=20))  # 0.7872
model_definition['top10__iv3']['gbm'] = OneVsRestClassifier(GradientBoostingClassifier(
    n_estimators=205, learning_rate=0.1, random_state=1))  # 0.7877