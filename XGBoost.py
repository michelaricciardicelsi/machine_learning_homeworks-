#15S:


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import itertools
import operator
import csv
import numpy as np
import pandas as pd
import xgboost as xgb

from matplotlib import pylab as plt
from sklearn import cross_validation
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.cross_validation import cross_val_score 
from sklearn.model_selection import StratifiedKFold

def preprocess(X, y, threshold):
    for name in set(y):
        count = y.count(name)
        if y.count(name) < threshold:
            for i in range(0, count):
                idx = y.index(name)
                y.pop(idx)
                X.pop(idx)
                
    toshuffle = np.column_stack((X, y))
    np.random.shuffle(toshuffle)
    result = np.hsplit(toshuffle, np.array([1]))
    
    X, y = result[0].transpose()[0], result[1].transpose()[0]
    vectorizer = DictVectorizer()
    vecX = vectorizer.fit_transform(X)
    nfeatures = vectorizer.get_feature_names()
    return vecX, y, nfeatures


def load_dataset(dataset_path, malware_file_path, max_samples):
    X = []
    y = []
    malware_hash = load_malware(malware_file_path)
    for file in os.listdir(dataset_path):
        if max_samples > 0:
            if file in malware_hash:
                    X.append(parse_file(dataset_path, file))
                    y.append(malware_hash.get(file,malware_hash[file]))
                    max_samples-= 1
        else:
            break
    return X, y


def parse_file(dataset_path, file_name):
    file_dict = {}
    with open(dataset_path + "/" + file_name, "r") as f:
        for line in f:
            file_dict[line.strip()] = True
    return file_dict

def load_malware(malware_file_path):
    malware_hash = {}
    with open(malware_file_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            malware_hash[row[0]] = row[1]

    return malware_hash

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure(figsize=(18, 15))
    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def plot_progress(progress):
    epochs = len(progress['test']['merror'])
    x_axis = range(0, epochs)
    # plot log loss
    fig, ax = plt.subplots()
    ax.plot(x_axis, progress['train']['mlogloss'], label='Train')
    ax.plot(x_axis, progress['test']['mlogloss'], label='Test')
    ax.legend()
    plt.ylabel('Multiclass Log Loss')
    plt.title('XGBoost Multiclass Log Loss')
    plt.show()
    # plot classification error
    fig, ax = plt.subplots()
    ax.plot(x_axis, progress['train']['merror'], label='Train')
    ax.plot(x_axis, progress['test']['merror'], label='Test')
    ax.legend()
    plt.ylabel('Multiclass Classification Error')
    plt.title('XGBoost Multiclass Classification Error')
    plt.show()
    

def plot_importance(bst):
    
    importance = bst.get_fscore()
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    df['feature'].replace('*', '[',inplace=True)
    df['feature'].replace('^',']',inplace=True)
    df['feature'].replace('#','<',inplace=True)
    plt.figure()
    df.plot()
    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 200))
    plt.title('XGBoost Feature Importance')

X, y = load_dataset("drebin/feature_vectors",
                        "drebin/sha256_family.csv", 5561)
                        
malware_hash = load_malware("drebin/sha256_family.csv")

vecX, y, nfeatures = preprocess(X,y,16)

labels = set(y)
cat = pd.Categorical(y).codes

CROSS_VALIDATION = True
if CROSS_VALIDATION:
              
    model = xgb.XGBClassifier(
                  objective='multi:softmax',
                  eval_metric='mlogloss',
                  num_class=len(labels),
                  eta=0.01,
                  max_depth=8,
                  subsample=0.7,
                  colsample_bytree=0.7,
                  gamma=0.02,
                  min_child_weight=1,
                  num_round=1000,
                  nthread=8,
                  silent=1,
                  early_stopping_rounds=50,
                  random_state=1518)
    
    
    kfold = StratifiedKFold(n_splits = 10, random_state=1518)
    splits = kfold.get_n_splits(vecX, y)

    
    results = cross_val_score(model, vecX, cat, cv=splits)
    print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

else:
    nf_1 = [s.replace('[', '*') for s in nfeatures] # remove all the 8s 
    nf_2 = [s.replace(']', '^') for s in nf_1]
    nf_3 = [s.replace('<', '#') for s in nf_2]
    
    train_X, test_X, train_Y, test_Y = cross_validation.train_test_split(vecX, cat, test_size=0.3, random_state=1518)
    
    xg_train = xgb.DMatrix(train_X, label=train_Y, feature_names = nf_3)
    xg_test = xgb.DMatrix(test_X, label=test_Y,feature_names = nf_3)
                   
    
    param = {}
    param['objective'] = 'multi:softmax'
    param['eval_metric'] = ['merror','mlogloss']
    param['num_class'] = len(labels)
    param['eta'] = 0.01
    param['max_depth'] = 8
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['gamma'] = 0.02
    param['min_child_weight'] = 1
    param['num_round'] = 10
    param['nthread'] = 8
    param['silent'] = 1
    param['early_stopping_rounds'] = 50
    param['random_state'] = 1518
    progress = dict()
    
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    bst = xgb.train(param, xg_train, 10, evals = watchlist, evals_result = progress)
    
    pred = bst.predict(xg_test)
    error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
    print('Test error using softmax = {}'.format(error_rate))
    
    # evaluate predictions
    accuracy = accuracy_score(test_Y, pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    
    # retrieve performance metrics
    plot_progress(progress)
    plot_importance(bst)
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(test_Y, pred)
    np.set_printoptions(precision=2)
    
    # Plot normalized confusion matrix
    plot_confusion_matrix(cnf_matrix, classes=labels, normalize=True,
                          title='Normalized confusion matrix')





#30S:
    
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import itertools
import operator
import csv
import numpy as np
import pandas as pd
import xgboost as xgb

from matplotlib import pylab as plt
from sklearn import cross_validation
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.cross_validation import cross_val_score 
from sklearn.model_selection import StratifiedKFold

def preprocess(X, y, treshold):
    for name in set(y):
        count = y.count(name)
        if y.count(name) < treshold:
            for i in range(0, count):
                idx = y.index(name)
                y.pop(idx)
                X.pop(idx)
                
    toshuffle = np.column_stack((X, y))
    np.random.shuffle(toshuffle)
    result = np.hsplit(toshuffle, np.array([1]))
    
    X, y = result[0].transpose()[0], result[1].transpose()[0]
    vectorizer = DictVectorizer()
    vecX = vectorizer.fit_transform(X)
    nfeatures = vectorizer.get_feature_names()
    return vecX, y, nfeatures


def load_dataset(dataset_path, malware_file_path, max_samples):
    X = []
    y = []
    malware_hash = load_malware(malware_file_path)
    for file in os.listdir(dataset_path):
        if max_samples > 0:
            if file in malware_hash:
                    X.append(parse_file(dataset_path, file))
                    y.append(malware_hash.get(file,malware_hash[file]))
                    max_samples-= 1
        else:
            break
    return X, y


def parse_file(dataset_path, file_name):
    file_dict = {}
    with open(dataset_path + "/" + file_name, "r") as f:
        for line in f:
            file_dict[line.strip()] = True
    return file_dict

def load_malware(malware_file_path):
    malware_hash = {}
    with open(malware_file_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            malware_hash[row[0]] = row[1]

    return malware_hash

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure(figsize=(18, 15))
    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def plot_progress(progress):
    epochs = len(progress['test']['merror'])
    x_axis = range(0, epochs)
    # plot log loss
    fig, ax = plt.subplots()
    ax.plot(x_axis, progress['train']['mlogloss'], label='Train')
    ax.plot(x_axis, progress['test']['mlogloss'], label='Test')
    ax.legend()
    plt.ylabel('Multiclass Log Loss')
    plt.title('XGBoost Multiclass Log Loss')
    plt.show()
    # plot classification error
    fig, ax = plt.subplots()
    ax.plot(x_axis, progress['train']['merror'], label='Train')
    ax.plot(x_axis, progress['test']['merror'], label='Test')
    ax.legend()
    plt.ylabel('Multiclass Classification Error')
    plt.title('XGBoost Multiclass Classification Error')
    plt.show()
    

def plot_importance(bst):
    
    importance = bst.get_fscore()
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    df['feature'].replace('*', '[',inplace=True)
    df['feature'].replace('^',']',inplace=True)
    df['feature'].replace('#','<',inplace=True)
    plt.figure()
    df.plot()
    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 200))
    plt.title('XGBoost Feature Importance')

X, y = load_dataset("drebin/feature_vectors",
                        "drebin/sha256_family.csv", 5561)
                        
malware_hash = load_malware("drebin/sha256_family.csv")

vecX, y, nfeatures = preprocess(X,y,31)

labels = set(y)
cat = pd.Categorical(y).codes

CROSS_VALIDATION = True
if CROSS_VALIDATION:
              
    model = xgb.XGBClassifier(
                  objective='multi:softmax',
                  eval_metric='mlogloss',
                  num_class=len(labels),
                  eta=0.01,
                  max_depth=8,
                  subsample=0.7,
                  colsample_bytree=0.7,
                  gamma=0.02,
                  min_child_weight=1,
                  num_round=1000,
                  nthread=8,
                  silent=1,
                  early_stopping_rounds=50,
                  random_state=1518)
    
    
    kfold = StratifiedKFold(n_splits = 10, random_state=1518)
    splits = kfold.get_n_splits(vecX, y)

    
    results = cross_val_score(model, vecX, cat, cv=splits)
    print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

else:
    nf_1 = [s.replace('[', '*') for s in nfeatures] # remove all the 8s 
    nf_2 = [s.replace(']', '^') for s in nf_1]
    nf_3 = [s.replace('<', '#') for s in nf_2]
    
    train_X, test_X, train_Y, test_Y = cross_validation.train_test_split(vecX, cat, test_size=0.3, random_state=1518)
    
    xg_train = xgb.DMatrix(train_X, label=train_Y, feature_names = nf_3)
    xg_test = xgb.DMatrix(test_X, label=test_Y,feature_names = nf_3)
                   
    
    param = {}
    param['objective'] = 'multi:softmax'
    param['eval_metric'] = ['merror','mlogloss']
    param['num_class'] = len(labels)
    param['eta'] = 0.01
    param['max_depth'] = 8
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['gamma'] = 0.02
    param['min_child_weight'] = 1
    param['num_round'] = 10
    param['nthread'] = 8
    param['silent'] = 1
    param['early_stopping_rounds'] = 50
    param['random_state'] = 1518
    progress = dict()
    
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    bst = xgb.train(param, xg_train, 10, evals = watchlist, evals_result = progress)
    
    pred = bst.predict(xg_test)
    error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
    print('Test error using softmax = {}'.format(error_rate))
    
    # evaluate predictions
    accuracy = accuracy_score(test_Y, pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    
    # retrieve performance metrics
    plot_progress(progress)
    plot_importance(bst)
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(test_Y, pred)
    np.set_printoptions(precision=2)
    
    # Plot normalized confusion matrix
    plot_confusion_matrix(cnf_matrix, classes=labels, normalize=True,
                          title='Normalized confusion matrix')


#40S:

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import itertools
import operator
import csv
import numpy as np
import pandas as pd
import xgboost as xgb

from matplotlib import pylab as plt
from sklearn import cross_validation
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.cross_validation import cross_val_score 
from sklearn.model_selection import StratifiedKFold

def preprocess(X, y, treshold):
    for name in set(y):
        count = y.count(name)
        if y.count(name) < treshold:
            for i in range(0, count):
                idx = y.index(name)
                y.pop(idx)
                X.pop(idx)
                
    toshuffle = np.column_stack((X, y))
    np.random.shuffle(toshuffle)
    result = np.hsplit(toshuffle, np.array([1]))
    
    X, y = result[0].transpose()[0], result[1].transpose()[0]
    vectorizer = DictVectorizer()
    vecX = vectorizer.fit_transform(X)
    nfeatures = vectorizer.get_feature_names()
    return vecX, y, nfeatures


def load_dataset(dataset_path, malware_file_path, max_samples):
    X = []
    y = []
    malware_hash = load_malware(malware_file_path)
    for file in os.listdir(dataset_path):
        if max_samples > 0:
            if file in malware_hash:
                    X.append(parse_file(dataset_path, file))
                    y.append(malware_hash.get(file,malware_hash[file]))
                    max_samples-= 1
        else:
            break
    return X, y


def parse_file(dataset_path, file_name):
    file_dict = {}
    with open(dataset_path + "/" + file_name, "r") as f:
        for line in f:
            file_dict[line.strip()] = True
    return file_dict

def load_malware(malware_file_path):
    malware_hash = {}
    with open(malware_file_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            malware_hash[row[0]] = row[1]

    return malware_hash

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure(figsize=(18, 15))
    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def plot_progress(progress):
    epochs = len(progress['test']['merror'])
    x_axis = range(0, epochs)
    # plot log loss
    fig, ax = plt.subplots()
    ax.plot(x_axis, progress['train']['mlogloss'], label='Train')
    ax.plot(x_axis, progress['test']['mlogloss'], label='Test')
    ax.legend()
    plt.ylabel('Multiclass Log Loss')
    plt.title('XGBoost Multiclass Log Loss')
    plt.show()
    # plot classification error
    fig, ax = plt.subplots()
    ax.plot(x_axis, progress['train']['merror'], label='Train')
    ax.plot(x_axis, progress['test']['merror'], label='Test')
    ax.legend()
    plt.ylabel('Multiclass Classification Error')
    plt.title('XGBoost Multiclass Classification Error')
    plt.show()
    

def plot_importance(bst):
    
    importance = bst.get_fscore()
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    df['feature'].replace('*', '[',inplace=True)
    df['feature'].replace('^',']',inplace=True)
    df['feature'].replace('#','<',inplace=True)
    plt.figure()
    df.plot()
    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 200))
    plt.title('XGBoost Feature Importance')

X, y = load_dataset("drebin/feature_vectors",
                        "drebin/sha256_family.csv", 5561)
                        
malware_hash = load_malware("drebin/sha256_family.csv")

vecX, y, nfeatures = preprocess(X,y,41)

labels = set(y)
cat = pd.Categorical(y).codes

CROSS_VALIDATION = True
if CROSS_VALIDATION:
              
    model = xgb.XGBClassifier(
                  objective='multi:softmax',
                  eval_metric='mlogloss',
                  num_class=len(labels),
                  eta=0.01,
                  max_depth=8,
                  subsample=0.7,
                  colsample_bytree=0.7,
                  gamma=0.02,
                  min_child_weight=1,
                  num_round=1000,
                  nthread=8,
                  silent=1,
                  early_stopping_rounds=50,
                  random_state=1518)
    
    
    kfold = StratifiedKFold(n_splits = 10, random_state=1518)
    splits = kfold.get_n_splits(vecX, y)

    
    results = cross_val_score(model, vecX, cat, cv=splits)
    print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

else:
    nf_1 = [s.replace('[', '*') for s in nfeatures] # remove all the 8s 
    nf_2 = [s.replace(']', '^') for s in nf_1]
    nf_3 = [s.replace('<', '#') for s in nf_2]
    
    train_X, test_X, train_Y, test_Y = cross_validation.train_test_split(vecX, cat, test_size=0.3, random_state=1518)
    
    xg_train = xgb.DMatrix(train_X, label=train_Y, feature_names = nf_3)
    xg_test = xgb.DMatrix(test_X, label=test_Y,feature_names = nf_3)
                   
    
    param = {}
    param['objective'] = 'multi:softmax'
    param['eval_metric'] = ['merror','mlogloss']
    param['num_class'] = len(labels)
    param['eta'] = 0.01
    param['max_depth'] = 8
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['gamma'] = 0.02
    param['min_child_weight'] = 1
    param['num_round'] = 10
    param['nthread'] = 8
    param['silent'] = 1
    param['early_stopping_rounds'] = 50
    param['random_state'] = 1518
    progress = dict()
    
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    bst = xgb.train(param, xg_train, 10, evals = watchlist, evals_result = progress)
    
    pred = bst.predict(xg_test)
    error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
    print('Test error using softmax = {}'.format(error_rate))
    
    # evaluate predictions
    accuracy = accuracy_score(test_Y, pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    
    # retrieve performance metrics
    plot_progress(progress)
    plot_importance(bst)
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(test_Y, pred)
    np.set_printoptions(precision=2)
    
    # Plot normalized confusion matrix
    plot_confusion_matrix(cnf_matrix, classes=labels, normalize=True,
                          title='Normalized confusion matrix')





#15S:

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import itertools
import operator
import csv
import numpy as np
import pandas as pd
import xgboost as xgb

from matplotlib import pylab as plt
from sklearn import cross_validation
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.cross_validation import cross_val_score 
from sklearn.model_selection import StratifiedKFold

def preprocess(X, y, treshold):
    for name in set(y):
        count = y.count(name)
        if y.count(name) < treshold:
            for i in range(0, count):
                idx = y.index(name)
                y.pop(idx)
                X.pop(idx)
                
    toshuffle = np.column_stack((X, y))
    np.random.shuffle(toshuffle)
    result = np.hsplit(toshuffle, np.array([1]))
    
    X, y = result[0].transpose()[0], result[1].transpose()[0]
    vectorizer = DictVectorizer()
    vecX = vectorizer.fit_transform(X)
    nfeatures = vectorizer.get_feature_names()
    return vecX, y, nfeatures


def load_dataset(dataset_path, malware_file_path, max_samples):
    X = []
    y = []
    malware_hash = load_malware(malware_file_path)
    for file in os.listdir(dataset_path):
        if max_samples > 0:
            if file in malware_hash:
                    X.append(parse_file(dataset_path, file))
                    y.append(malware_hash.get(file,malware_hash[file]))
                    max_samples-= 1
        else:
            break
    return X, y


def parse_file(dataset_path, file_name):
    file_dict = {}
    with open(dataset_path + "/" + file_name, "r") as f:
        for line in f:
            file_dict[line.strip()] = True
    return file_dict

def load_malware(malware_file_path):
    malware_hash = {}
    with open(malware_file_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            malware_hash[row[0]] = row[1]

    return malware_hash

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure(figsize=(18, 15))
    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def plot_progress(progress):
    epochs = len(progress['test']['merror'])
    x_axis = range(0, epochs)
    # plot log loss
    fig, ax = plt.subplots()
    ax.plot(x_axis, progress['train']['mlogloss'], label='Train')
    ax.plot(x_axis, progress['test']['mlogloss'], label='Test')
    ax.legend()
    plt.ylabel('Multiclass Log Loss')
    plt.title('XGBoost Multiclass Log Loss')
    plt.show()
    # plot classification error
    fig, ax = plt.subplots()
    ax.plot(x_axis, progress['train']['merror'], label='Train')
    ax.plot(x_axis, progress['test']['merror'], label='Test')
    ax.legend()
    plt.ylabel('Multiclass Classification Error')
    plt.title('XGBoost Multiclass Classification Error')
    plt.show()
    

def plot_importance(bst):
    
    importance = bst.get_fscore()
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    df['feature'].replace('*', '[',inplace=True)
    df['feature'].replace('^',']',inplace=True)
    df['feature'].replace('#','<',inplace=True)
    plt.figure()
    df.plot()
    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 200))
    plt.title('XGBoost Feature Importance')

X, y = load_dataset("drebin/feature_vectors",
                        "drebin/sha256_family.csv", 5561)
                        
malware_hash = load_malware("drebin/sha256_family.csv")

vecX, y, nfeatures = preprocess(X,y,16)

labels = set(y)
cat = pd.Categorical(y).codes

CROSS_VALIDATION = False
if CROSS_VALIDATION:
              
    model = xgb.XGBClassifier(
                  objective='multi:softmax',
                  eval_metric='mlogloss',
                  num_class=len(labels),
                  eta=0.01,
                  max_depth=8,
                  subsample=0.7,
                  colsample_bytree=0.7,
                  gamma=0.02,
                  min_child_weight=1,
                  num_round=1000,
                  nthread=8,
                  silent=1,
                  early_stopping_rounds=50,
                  random_state=1518)
    
    
    kfold = StratifiedKFold(n_splits = 10, random_state=1518)
    splits = kfold.get_n_splits(vecX, y)

    
    results = cross_val_score(model, vecX, cat, cv=splits)
    print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

else:
    nf_1 = [s.replace('[', '*') for s in nfeatures] # remove all the 8s 
    nf_2 = [s.replace(']', '^') for s in nf_1]
    nf_3 = [s.replace('<', '#') for s in nf_2]
    
    train_X, test_X, train_Y, test_Y = cross_validation.train_test_split(vecX, cat, test_size=0.3, random_state=1518)
    
    xg_train = xgb.DMatrix(train_X, label=train_Y, feature_names = nf_3)
    xg_test = xgb.DMatrix(test_X, label=test_Y,feature_names = nf_3)
                   
    
    param = {}
    param['objective'] = 'multi:softmax'
    param['eval_metric'] = ['merror','mlogloss']
    param['num_class'] = len(labels)
    param['eta'] = 0.01
    param['max_depth'] = 8
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['gamma'] = 0.02
    param['min_child_weight'] = 1
    param['num_round'] = 10
    param['nthread'] = 8
    param['silent'] = 1
    param['early_stopping_rounds'] = 50
    param['random_state'] = 1518
    progress = dict()
    
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    bst = xgb.train(param, xg_train, 10, evals = watchlist, evals_result = progress)
    
    pred = bst.predict(xg_test)
    error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
    print('Test error using softmax = {}'.format(error_rate))
    
    # evaluate predictions
    accuracy = accuracy_score(test_Y, pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    
    # retrieve performance metrics
    plot_progress(progress)
    plot_importance(bst)
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(test_Y, pred)
    np.set_printoptions(precision=2)
    
    # Plot normalized confusion matrix
    plot_confusion_matrix(cnf_matrix, classes=labels, normalize=True,
                          title='Normalized confusion matrix')


#30S:

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import itertools
import operator
import csv
import numpy as np
import pandas as pd
import xgboost as xgb

from matplotlib import pylab as plt
from sklearn import cross_validation
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.cross_validation import cross_val_score 
from sklearn.model_selection import StratifiedKFold

def preprocess(X, y, treshold):
    for name in set(y):
        count = y.count(name)
        if y.count(name) < treshold:
            for i in range(0, count):
                idx = y.index(name)
                y.pop(idx)
                X.pop(idx)
                
    toshuffle = np.column_stack((X, y))
    np.random.shuffle(toshuffle)
    result = np.hsplit(toshuffle, np.array([1]))
    
    X, y = result[0].transpose()[0], result[1].transpose()[0]
    vectorizer = DictVectorizer()
    vecX = vectorizer.fit_transform(X)
    nfeatures = vectorizer.get_feature_names()
    return vecX, y, nfeatures


def load_dataset(dataset_path, malware_file_path, max_samples):
    X = []
    y = []
    malware_hash = load_malware(malware_file_path)
    for file in os.listdir(dataset_path):
        if max_samples > 0:
            if file in malware_hash:
                    X.append(parse_file(dataset_path, file))
                    y.append(malware_hash.get(file,malware_hash[file]))
                    max_samples-= 1
        else:
            break
    return X, y


def parse_file(dataset_path, file_name):
    file_dict = {}
    with open(dataset_path + "/" + file_name, "r") as f:
        for line in f:
            file_dict[line.strip()] = True
    return file_dict

def load_malware(malware_file_path):
    malware_hash = {}
    with open(malware_file_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            malware_hash[row[0]] = row[1]

    return malware_hash

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure(figsize=(18, 15))
    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def plot_progress(progress):
    epochs = len(progress['test']['merror'])
    x_axis = range(0, epochs)
    # plot log loss
    fig, ax = plt.subplots()
    ax.plot(x_axis, progress['train']['mlogloss'], label='Train')
    ax.plot(x_axis, progress['test']['mlogloss'], label='Test')
    ax.legend()
    plt.ylabel('Multiclass Log Loss')
    plt.title('XGBoost Multiclass Log Loss')
    plt.show()
    # plot classification error
    fig, ax = plt.subplots()
    ax.plot(x_axis, progress['train']['merror'], label='Train')
    ax.plot(x_axis, progress['test']['merror'], label='Test')
    ax.legend()
    plt.ylabel('Multiclass Classification Error')
    plt.title('XGBoost Multiclass Classification Error')
    plt.show()
    

def plot_importance(bst):
    
    importance = bst.get_fscore()
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    df['feature'].replace('*', '[',inplace=True)
    df['feature'].replace('^',']',inplace=True)
    df['feature'].replace('#','<',inplace=True)
    plt.figure()
    df.plot()
    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 200))
    plt.title('XGBoost Feature Importance')

X, y = load_dataset("drebin/feature_vectors",
                        "drebin/sha256_family.csv", 5561)
                        
malware_hash = load_malware("drebin/sha256_family.csv")

vecX, y, nfeatures = preprocess(X,y,31)

labels = set(y)
cat = pd.Categorical(y).codes

CROSS_VALIDATION = False
if CROSS_VALIDATION:
              
    model = xgb.XGBClassifier(
                  objective='multi:softmax',
                  eval_metric='mlogloss',
                  num_class=len(labels),
                  eta=0.01,
                  max_depth=8,
                  subsample=0.7,
                  colsample_bytree=0.7,
                  gamma=0.02,
                  min_child_weight=1,
                  num_round=1000,
                  nthread=8,
                  silent=1,
                  early_stopping_rounds=50,
                  random_state=1518)
    
    
    kfold = StratifiedKFold(n_splits = 10, random_state=1518)
    splits = kfold.get_n_splits(vecX, y)

    
    results = cross_val_score(model, vecX, cat, cv=splits)
    print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

else:
    nf_1 = [s.replace('[', '*') for s in nfeatures] # remove all the 8s 
    nf_2 = [s.replace(']', '^') for s in nf_1]
    nf_3 = [s.replace('<', '#') for s in nf_2]
    
    train_X, test_X, train_Y, test_Y = cross_validation.train_test_split(vecX, cat, test_size=0.3, random_state=1518)
    
    xg_train = xgb.DMatrix(train_X, label=train_Y, feature_names = nf_3)
    xg_test = xgb.DMatrix(test_X, label=test_Y,feature_names = nf_3)
                   
    
    param = {}
    param['objective'] = 'multi:softmax'
    param['eval_metric'] = ['merror','mlogloss']
    param['num_class'] = len(labels)
    param['eta'] = 0.01
    param['max_depth'] = 8
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['gamma'] = 0.02
    param['min_child_weight'] = 1
    param['num_round'] = 10
    param['nthread'] = 8
    param['silent'] = 1
    param['early_stopping_rounds'] = 50
    param['random_state'] = 1518
    progress = dict()
    
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    bst = xgb.train(param, xg_train, 10, evals = watchlist, evals_result = progress)
    
    pred = bst.predict(xg_test)
    error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
    print('Test error using softmax = {}'.format(error_rate))
    
    # evaluate predictions
    accuracy = accuracy_score(test_Y, pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    
    # retrieve performance metrics
    plot_progress(progress)
    plot_importance(bst)
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(test_Y, pred)
    np.set_printoptions(precision=2)
    
    # Plot normalized confusion matrix
    plot_confusion_matrix(cnf_matrix, classes=labels, normalize=True,
                          title='Normalized confusion matrix')


#40S:

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import itertools
import operator
import csv
import numpy as np
import pandas as pd
import xgboost as xgb

from matplotlib import pylab as plt
from sklearn import cross_validation
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.cross_validation import cross_val_score 
from sklearn.model_selection import StratifiedKFold

def preprocess(X, y, treshold):
    for name in set(y):
        count = y.count(name)
        if y.count(name) < treshold:
            for i in range(0, count):
                idx = y.index(name)
                y.pop(idx)
                X.pop(idx)
                
    toshuffle = np.column_stack((X, y))
    np.random.shuffle(toshuffle)
    result = np.hsplit(toshuffle, np.array([1]))
    
    X, y = result[0].transpose()[0], result[1].transpose()[0]
    vectorizer = DictVectorizer()
    vecX = vectorizer.fit_transform(X)
    nfeatures = vectorizer.get_feature_names()
    return vecX, y, nfeatures


def load_dataset(dataset_path, malware_file_path, max_samples):
    X = []
    y = []
    malware_hash = load_malware(malware_file_path)
    for file in os.listdir(dataset_path):
        if max_samples > 0:
            if file in malware_hash:
                    X.append(parse_file(dataset_path, file))
                    y.append(malware_hash.get(file,malware_hash[file]))
                    max_samples-= 1
        else:
            break
    return X, y


def parse_file(dataset_path, file_name):
    file_dict = {}
    with open(dataset_path + "/" + file_name, "r") as f:
        for line in f:
            file_dict[line.strip()] = True
    return file_dict

def load_malware(malware_file_path):
    malware_hash = {}
    with open(malware_file_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            malware_hash[row[0]] = row[1]

    return malware_hash

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure(figsize=(18, 15))
    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def plot_progress(progress):
    epochs = len(progress['test']['merror'])
    x_axis = range(0, epochs)
    # plot log loss
    fig, ax = plt.subplots()
    ax.plot(x_axis, progress['train']['mlogloss'], label='Train')
    ax.plot(x_axis, progress['test']['mlogloss'], label='Test')
    ax.legend()
    plt.ylabel('Multiclass Log Loss')
    plt.title('XGBoost Multiclass Log Loss')
    plt.show()
    # plot classification error
    fig, ax = plt.subplots()
    ax.plot(x_axis, progress['train']['merror'], label='Train')
    ax.plot(x_axis, progress['test']['merror'], label='Test')
    ax.legend()
    plt.ylabel('Multiclass Classification Error')
    plt.title('XGBoost Multiclass Classification Error')
    plt.show()
    

def plot_importance(bst):
    
    importance = bst.get_fscore()
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    df['feature'].replace('*', '[',inplace=True)
    df['feature'].replace('^',']',inplace=True)
    df['feature'].replace('#','<',inplace=True)
    plt.figure()
    df.plot()
    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 200))
    plt.title('XGBoost Feature Importance')

X, y = load_dataset("drebin/feature_vectors",
                        "drebin/sha256_family.csv", 5561)
                        
malware_hash = load_malware("drebin/sha256_family.csv")

vecX, y, nfeatures = preprocess(X,y,41)

labels = set(y)
cat = pd.Categorical(y).codes

CROSS_VALIDATION = False
if CROSS_VALIDATION:
              
    model = xgb.XGBClassifier(
                  objective='multi:softmax',
                  eval_metric='mlogloss',
                  num_class=len(labels),
                  eta=0.01,
                  max_depth=8,
                  subsample=0.7,
                  colsample_bytree=0.7,
                  gamma=0.02,
                  min_child_weight=1,
                  num_round=1000,
                  nthread=8,
                  silent=1,
                  early_stopping_rounds=50,
                  random_state=1518)
    
    
    kfold = StratifiedKFold(n_splits = 10, random_state=1518)
    splits = kfold.get_n_splits(vecX, y)

    
    results = cross_val_score(model, vecX, cat, cv=splits)
    print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

else:
    nf_1 = [s.replace('[', '*') for s in nfeatures] # remove all the 8s 
    nf_2 = [s.replace(']', '^') for s in nf_1]
    nf_3 = [s.replace('<', '#') for s in nf_2]
    
    train_X, test_X, train_Y, test_Y = cross_validation.train_test_split(vecX, cat, test_size=0.3, random_state=1518)
    
    xg_train = xgb.DMatrix(train_X, label=train_Y, feature_names = nf_3)
    xg_test = xgb.DMatrix(test_X, label=test_Y,feature_names = nf_3)
                   
    
    param = {}
    param['objective'] = 'multi:softmax'
    param['eval_metric'] = ['merror','mlogloss']
    param['num_class'] = len(labels)
    param['eta'] = 0.01
    param['max_depth'] = 8
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['gamma'] = 0.02
    param['min_child_weight'] = 1
    param['num_round'] = 10
    param['nthread'] = 8
    param['silent'] = 1
    param['early_stopping_rounds'] = 50
    param['random_state'] = 1518
    progress = dict()
    
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    bst = xgb.train(param, xg_train, 10, evals = watchlist, evals_result = progress)
    
    pred = bst.predict(xg_test)
    error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
    print('Test error using softmax = {}'.format(error_rate))
    
    # evaluate predictions
    accuracy = accuracy_score(test_Y, pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    
    # retrieve performance metrics
    plot_progress(progress)
    plot_importance(bst)
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(test_Y, pred)
    np.set_printoptions(precision=2)
    
    # Plot normalized confusion matrix
    plot_confusion_matrix(cnf_matrix, classes=labels, normalize=True,
                          title='Normalized confusion matrix')

