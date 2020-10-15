## parse pan data.

# AUGMENTETD
# 49-
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from feature_construction import *
import json
import pandas as pd
from functools import reduce
import random
import numpy as np
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import SGDRegressor
from random import randint

def augment_years(year):
    if  year <= 1958:
        return 0
    if year >= 1959 and year <= 1966:
        return 1
    if year >= 1967 and year <= 1973:
        return 3
    if year >= 1974 and year <= 1980:
        return 4
    if year >= 1981 and year <= 1986:
        return 5
    if year >= 1987 and year <= 1991:
        return 6
    if year >= 1992 and year <= 1995:
        return 7
    if year >= 1996:
        return 8

def reverse_augment(in_): 
    ins = [ int(1958), 
            randint(1959,1966), 
            randint(1967,1973), 
            randint(1974,1980),
            randint(1981,1986),
            randint(1987,1991),
            randint(1987,1991),
            randint(1992,1995),
            randint(1996,1999)            
            ]
    return ins[in_]        
def export_labels(labels_file):
    labels = {}
    labels['occupations']=[]
    labels['gender']=[]
    labels['birthyear']=[]
    labels['tens'] = []
    labels['hundrets'] = []
    labels['years_clf'] = []
    ins = set()
    genders= {'male':0,'female':1}
    occupations = {'sports' : 0, 'performer' : 1 , 'creator' : 2, 'politics':3}
    with open(labels_file) as lf:
        for line in lf:
            lab_di = json.loads(line)
            curr = lab_di['birthyear']
            labels['occupations'].append(occupations[lab_di['occupation']])
            labels['gender'].append(genders[lab_di['gender']])
            labels['birthyear'].append(int(lab_di['birthyear']))
            labels['hundrets'].append((int(lab_di['birthyear'])/100)/10)
            labels['years_clf'].append(augment_years(int(curr)))
            labels['tens'].append(int(lab_di['birthyear'])%10)
            print("YEAR: "+curr+" CLASS: "+str(augment_years(int(curr))))
            ins.add(lab_di['birthyear'])
    print("Parsed labels..")
    print(sorted(ins))
    print(len(ins))
    with open(os.path.join("../train_data/","labels.pkl"),mode='wb') as f:
        pickle.dump(labels,f)
    
def parse_feeds(fname, labels_file, all=False):
    documents = {}
    taken = 0
    with open(os.path.join(fname,"follower-feeds.ndjson")) as fnx:
        for line in tqdm(fnx,total=args.tweets):
            #if random.random() > 0.3:
            lx = json.loads(line)
            lx['text'] = np.array((lx['text']))            
            persons = random.sample(range(len(lx['text'])),min(len(lx['text']),args.persons))
            out = []
            for c in persons:
                idxs = random.sample(range(len(lx['text'][c])), min(args.idxs,len(lx['text'][c])))     
                for i in idxs:
                    out.append(lx['text'][c][i])
                tokens_word = " ".join(out) #lx['text'][0:30])#lx['text'][0:args.num_samples][0:args.num_samples])
                documents[lx['id']] = tokens_word
            taken = taken + 1
            if taken > args.tweets:
                break       
    ## pan features
    print("Building Dataframe")
    dataframe = build_dataframe(documents)
    print('Dataframe built')
    with open(os.path.join("../train_data/","dataframe_f.pkl"),mode='wb') as f:
        pickle.dump(dataframe,f)
    nfeat = 10000
    dim = 512
    tokenizer, feature_names, data_matrix = get_features(dataframe, max_num_feat = nfeat, labels = None)
    reducer = TruncatedSVD(n_components = min(dim, nfeat * len(feature_names)-1))
    data_matrix = reducer.fit_transform(data_matrix)

    #print("{} Performed with {}".format(target,f1))  
    with open(os.path.join("../train_data/","tokenizer_f.pkl"),mode='wb') as f:
        pickle.dump(tokenizer,f)
    with open(os.path.join("../train_data/","reducer_f.pkl"),mode='wb') as f:
        pickle.dump(reducer,f)
    with open(os.path.join("../train_data/","data_matrix_f.pkl"),mode='wb') as f:
        pickle.dump(data_matrix,f)

        
def _import(path_in="../train_data/"):
    """Imports tokenizer,clf,reducer from param(path_in, default is ../models)"""
    tokenizer = pickle.load(open(os.path.join(path_in,"tokenizer_f.pkl"),'rb'))
    reducer = pickle.load(open(os.path.join(path_in,"reducer_f.pkl"),'rb'))
    data_matrix = pickle.load(open(os.path.join(path_in,"data_matrix_f.pkl"),'rb'))
    labels = pickle.load(open(os.path.join(path_in,"labels.pkl"),'rb'))
    return tokenizer,reducer,data_matrix,labels

def fit_import(path_in="../train_data/"):
    """Imports tokenizer,clf,reducer from param(path_in, default is ../models)"""
    tokenizer = pickle.load(open(os.path.join(path_in,"tokenizer_f.pkl"),'rb'))
    clf = pickle.load(open(os.path.join(path_in,"clf-f.pkl"),'rb'))
    reducer = pickle.load(open(os.path.join(path_in,"reducer_f.pkl"),'rb'))
    return tokenizer,clf,reducer

def clf_find():
    tokenizer,reducer,data_matrix,labels = _import()
    clfs = {}
    for label in tqdm(['years_clf','occupations','gender'],total=3):
        print(label)
        if not label == 'birthyear' :
            X_train, X_test, y_train, y_test = train_test_split(data_matrix, labels[label][0:args.tweets+1], train_size=0.9, test_size=0.1)    
            parameters = {'kernel':["linear","poly"], 'C':[0.1, 1, 10, 100, 500],"gamma":["scale","auto"],"class_weight":["balanced",None]}
            svc = svm.SVC(verbose=0)
            clf1 = GridSearchCV(svc, parameters, verbose = 0, n_jobs = -1)
            clf1.fit(X_train, y_train)
            logging.info(str(max(clf1.cv_results_['mean_test_score'])) +" training configuration with best score (SVM)")
            predictions = clf1.predict(X_test)
            acc_svm = accuracy_score(predictions,y_test)
            logging.info("Test accuracy score SVM {}".format(acc_svm))
            parameters = {"C":[0.1,1,10,25,50,100,500],"penalty":["l2"]}
            svc = LogisticRegression(max_iter = 100000)
            clf2 = GridSearchCV(svc, parameters, verbose = 0, n_jobs = -1)
            clf2.fit(X_train, y_train)
            logging.info(str(max(clf2.cv_results_['mean_test_score'])) + " training configuration with best score (LR)")
            predictions = clf2.predict(X_test)
            acc_lr = accuracy_score(predictions,y_test)
            logging.info("Test accuracy score LR {}".format(acc_lr))
            if acc_lr > acc_svm:
                clfs[label] = clf2
            else:
                clfs[label] = clf1            
        else:
            X_train, X_test, y_train, y_test = train_test_split(data_matrix,  labels[label][0:args.tweets+1], train_size=0.8, test_size=0.2)
            parameters = {"loss":["squared_loss"],"penalty":["elasticnet"],"alpha":[0.01,0.001,0.0001,0.0005],"l1_ratio":[0,0.05,0.25,0.3,0.6,0.8,0.95,1],"power_t":[0.5,0.1,0.9]}
            sgd = SGDRegressor(max_iter=10000)         
            clf_grid = GridSearchCV(estimator=sgd , param_grid=parameters,n_jobs=-1, scoring='neg_mean_absolute_error',cv = 3, verbose=2)
            clf_grid.fit(X_train,y_train)
            predictions = clf_grid.predict(X_test)
            acc0 =  mean_absolute_error(predictions,y_test)
            logging.info("Test accuracy MAE score SGD regression {}".format(acc0))
            parameters = {'kernel': ('linear', 'rbf','poly'), 
                     'C':[1.5,10],
                     'gamma': [1e-7, 1e-4],
                     'epsilon':[0.1,0.2,0.5,0.3],
                     }
            svr = svm.SVR()
            clf2 = GridSearchCV(svr, parameters, cv = 3,
                        n_jobs = -1,
                        verbose=True, scoring='neg_mean_absolute_error')
            clf2.fit(X_train, y_train)
            logging.info(str(sum(clf2.cv_results_['mean_test_score'])/len(clf2.cv_results_['mean_test_score'])) +" training configuration with best score (SVM)")
            predictions = clf2.predict(X_test)
            acc2 =  mean_absolute_error(predictions,y_test)
            logging.info("Test accuracy MAE score SVM regression {}".format(acc2))
            if acc0 < acc2:
                clf2 = clf_grid
                acc2 = acc0
            n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
            max_features = ['auto', 'sqrt']
            max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
            max_depth.append(None)
            min_samples_split = [2, 5, 10]
            min_samples_leaf = [1, 2, 4]
            bootstrap = [True, False]
            random_grid = {'n_estimators': n_estimators,'max_features': max_features,'max_depth': max_depth,
                           'min_samples_split': min_samples_split,'min_samples_leaf': min_samples_leaf,'bootstrap': bootstrap}
            rf = RandomForestRegressor()
            clf1 = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 3, cv = 3, verbose=2, random_state=42, n_jobs = -1, scoring='neg_mean_absolute_error')
            clf1.fit(X_train, y_train)
            predictions = clf1.predict(X_test)
            acc = mean_absolute_error(y_test, predictions)
            logging.info("Test accuracy MAE score SVM regression {}".format(acc))
            print(acc)
            #predictions = clf1.predict()
            #acc_svm = accuracy_score(predictions,y_test)
            #logging.info("Test accuracy score RF {}".format(acc_svm))
            if acc2 > acc:
                clfs[label] = clf2
            else: 
                clfs[label] = clf1
                
    with open(os.path.join("../train_data/","clf-f.pkl"),mode='wb') as f:
        pickle.dump(clfs,f) 
        
def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")

def fit(path,out_path="../out"):
    tic()
    test_texts = {}
    inv_g = {0:'male',1:'female'}
    inv_o = {0:'sports',1:'performer',2:'creator',3:'politics'}
    tokenizer,clfs,reducer = fit_import("/home/koloski20/ff_clf_all/train_data")
    """
    with open(path) as fnx:
        for line in fnx: 
            lx = json.loads(line)
            tokens_word = " ".join(lx['text'][0:30])
            test_texts[lx['id']] = tokens_word    
    """
    with open(os.path.join(path,"follower-feeds.ndjson")) as fnx:
        for line in fnx:
                lx = json.loads(line)
                lx['text'] = np.array((lx['text']))            
                persons = random.sample(range(len(lx['text'])),min(len(lx['text']),args.persons))
                out = []
                for c in persons:
                    idxs = random.sample(range(len(lx['text'][c])), min(args.idxs,len(lx['text'][c])))     
                    for i in idxs:
                        out.append(lx['text'][c][i])
                tokens_word = " ".join(out) #lx['text'][0:30])#lx['text'][0:args.num_samples][0:args.num_samples])
                test_texts[lx['id']] = tokens_word
    f = open(os.path.join(out_path,"labels.ndjson"),mode='w')    
    df_text = build_dataframe(test_texts)
    matrix_form = tokenizer.transform(df_text)
    reduced_matrix_form = reducer.transform(matrix_form)
    gender = clfs['gender'].predict(reduced_matrix_form)
    occupation = clfs['occupations'].predict(reduced_matrix_form)
    birthyear = clfs['years_clf'].predict(reduced_matrix_form)
    cnt = 0
    for x in test_texts:                 
        o = inv_o[int(occupation[cnt])]
        g = inv_g[int(gender[cnt])]
        y = reverse_augment(birthyear[cnt])
        item = {"id": x, "occupation": o, "gender":g, "birthyear": max(1949,min(1999,int(y)))}
        print(item)
        v = json.dumps(item)
        cnt = cnt + 1
        f.write(v + '\n')  
    toc()
        
if __name__ == "__main__":
    from scipy import io
    import argparse
    data_inpt = "../../data/pan20-celebrity-profiling-training-dataset-2020-02-28"
    labels_inpt = "../../data/pan20-celebrity-profiling-training-dataset-2020-02-28/labels.ndjson"
    datafolder = "../train_data"    
    
    argparser = argparse.ArgumentParser(description='Author Profiling Evaluation')
    argparser.add_argument('-o', '--output', dest='output', type=str, default='../out',
                           help='Choose output directory')
    argparser.add_argument('-i', '--input', dest='input', type=str,
                           default=data_inpt,
                           help='Choose input dataset')
    argparser.add_argument('-p', '--persons', dest='persons', type=int,
                           default=20,
                           help='Choose dataset size dataset')
    argparser.add_argument('-idx', '--idxs', dest='idxs', type=int,
                           default=2,
                           help='Choose dataset size dataset')
    argparser.add_argument('-t', '--tweets', dest='tweets', type=int,
                           default=1920,
                           help='Choose dataset size dataset')
    argparser.add_argument('-s', '--s', dest='seed', type=int,
                           default=42,
                           help='Choose dataset size dataset')
    args = argparser.parse_args()
        
    path = args.input
    path_out = args.output
    seed_ = args.seed
    random.seed(seed_)
    #print(args)
    #a = parse_feeds(data_inpt, labels_inpt, all=True)
    #export_labels(labels_inpt)
    #clf_find()
    fit(path,path_out)
