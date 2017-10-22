import matplotlib.pyplot as plt
from textblob import TextBlob
import pandas
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold


def split_into_tokens(message):
    if type(message) is float:
        message = str(message)
    return TextBlob(message).words


def traintest(messages):
    msg_train, msg_test, label_train, label_test = \
        train_test_split(messages['CONTENT'] + messages['AUTHOR'] + messages['DATE'], messages['CLASS'], test_size=0.3)
    return [msg_train, msg_test, label_train, label_test]



eminem = pandas.read_csv('Youtube04-Eminem.csv',encoding='utf-8')
katy = pandas.read_csv('Youtube02-KatyPerry.csv',encoding='utf-8')
lmfao = pandas.read_csv('Youtube03-LMFAO.csv',encoding='utf-8')
psy = pandas.read_csv('Youtube01-Psy.csv',encoding='utf-8')
shakira = pandas.read_csv('Youtube05-Shakira.csv',encoding='utf-8')

frames = [eminem,katy,lmfao,psy,shakira]
combined = pandas.concat(frames)

frames4 = [eminem,katy,lmfao,shakira]
combined4 = pandas.concat(frames)

def traintest2(messages):
    msg_train, msg_test, label_train, label_test = \
        train_test_split(messages['CONTENT'] + messages['AUTHOR'] + messages['DATE'], messages['CLASS'], test_size=0)
    return [msg_train, label_train]

def prepro(input):
    bow = CountVectorizer(max_features =750)
    bow2 = bow.fit_transform(input)
    return bow2

def pltcon(classifier,title,label_test,msg_test):
    plt.matshow(confusion_matrix(label_test, classifier.predict(msg_test)), cmap=plt.cm.binary, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    plt.ylabel('expected label')
    plt.xlabel('predicted label')
    name = title+'.png'
    plt.savefig(name)

def tonyFunction(msg_train, msg_test, label_train, label_test, info):
    # msg_train = prepro(msg_train)
    # msg_test = prepro(msg_test)

    pipeline = Pipeline([
        ('bow', CountVectorizer(analyzer=split_into_tokens)),  # strings to token integer counts
        ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
        ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
    ])

    params = {
        'tfidf__use_idf': (True, False),
        # 'bow__analyzer': (split_into_tokens, split_into_tokens),
    }

    grid = GridSearchCV(
        pipeline,  # pipeline from above
        params,  # parameters to tune via cross validation
        refit=True,  # fit using all available data at the end, on the best found param combination
        n_jobs=1,  # number of cores to use for parallelization; -1 for "all cores"
        scoring='accuracy',  # what score are we optimizing?
        cv=StratifiedKFold(n_splits=5),  # what type of cross validation to use
    )

    nb_detector = grid.fit(msg_train, label_train)
    print('\nTest Results')
    print('\nMultinomial NaiveBayes '+info)

    predictions = nb_detector.predict(msg_test)
    print ('\nConfusion Matrix')
    print (confusion_matrix(label_test, predictions))
    print (classification_report(label_test, predictions))

    #KNN Implementation
    knn_pipeline = Pipeline([
        ('bow', CountVectorizer(analyzer=split_into_tokens)),
        ('tfidf', TfidfTransformer()),
        ('classifier', KNeighborsClassifier(n_neighbors=1)),  # <== change here
    ])

    knn_grid = GridSearchCV(
        knn_pipeline,  # pipeline from above
        params,  # parameters to tune via cross validation
        refit=True,  # fit using all available data at the end, on the best found param combination
        n_jobs=1,  # number of cores to use for parallelization; -1 for "all cores"
        scoring='accuracy',  # what score are we optimizing?
        cv=StratifiedKFold(n_splits=5),  # what type of cross validation to use
    )

    knn_detector = knn_grid.fit(msg_train, label_train) # find the best combination from param_svm
    print('K-NearestNeighbour '+info)
    print ('\nConfusion Matrix')
    print (confusion_matrix(label_test, knn_detector.predict(msg_test)))
    print (classification_report(label_test, knn_detector.predict(msg_test)))


    #SVM Implementation
    pipeline_svm = Pipeline([
        ('bow', CountVectorizer(analyzer=split_into_tokens)),
        ('tfidf', TfidfTransformer()),
        ('classifier', SVC()),  # <== change here
    ])

    # pipeline parameters to automatically explore and tune / limited for speed now
    param_svm = [
      {'classifier__C': [1], 'classifier__kernel': ['linear']},
     ]

    grid_svm = GridSearchCV(
        pipeline_svm,  # pipeline from above
        param_grid=param_svm,  # parameters to tune via cross validation
        refit=True,  # fit using all data, on the best detected classifier
        n_jobs=1,  # number of cores to use for parallelization; -1 for "all cores"
        scoring='accuracy',  # what score are we optimizing?
        cv=StratifiedKFold(n_splits=5),  # what type of cross validation to use
    )

    svm_detector = grid_svm.fit(msg_train, label_train) # find the best combination from param_svm
    print('SVM '+info)


    print ('\nConfusion Matrix')
    print (confusion_matrix(label_test, svm_detector.predict(msg_test)))
    print (classification_report(label_test, svm_detector.predict(msg_test)))

    nb = 'NaiveBayes '+info
    pltcon(nb_detector,nb,label_test,msg_test)
    knn = 'KNN ' + info
    pltcon(knn_detector,knn,label_test,msg_test)
    svm = 'SVM '+info
    pltcon(svm_detector,svm,label_test,msg_test)


msg_trainp, msg_testp, label_trainp, label_testp = traintest(psy)
msg_traink, msg_testk, label_traink, label_testk = traintest(katy)
msg_trainl, msg_testl, label_trainl, label_testl = traintest(lmfao)
msg_traine, msg_teste, label_traine, label_teste = traintest(eminem)
msg_trains, msg_tests, label_trains, label_tests = traintest(shakira)
msg_trainc, msg_testc, label_trainc, label_testc = traintest(combined)
msg_train4, msg_test4, label_train4, label_test4 = traintest(combined4)

msg_trains1, label_trains1 = traintest2(shakira)
msg_testp1, label_testp1 = traintest2(psy)
msg_test4, label_test4 = traintest2(combined4)
print('\nPsy')


def grace(pro):
    tonyFunction(msg_trainp,msg_testp, label_trainp, label_testp,pro+' Psy')

    print('\nKaty Perry')
    tonyFunction(msg_traink, msg_testk, label_traink, label_testk,pro+' Katy Perry')

    print('\nLMFAO')
    tonyFunction(msg_trainl, msg_testl, label_trainl, label_testl,pro+' LMFAO')

    print('\nEminem')
    tonyFunction(msg_traine, msg_teste, label_traine, label_teste,pro+' Eminem')

    print('\nShakira')
    tonyFunction(msg_trains, msg_tests, label_trains, label_tests,pro+' Shakira')

    print('\nTrained on Shakira Tested on Psy')
    tonyFunction(msg_trains1, msg_testp1, label_trains1, label_testp1,pro+' Trained on Shakira Tested on Psy')

    print('\nCombined')
    tonyFunction(msg_trainc, msg_testc, label_trainc, label_testc,pro+' Combined')

    print('\nCombined Minus Psy')
    tonyFunction(msg_train4, msg_testp1, label_train4, label_testp1,pro+' Combined Tested on Psy')


grace('No Preprocessing')
