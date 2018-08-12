#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import matplotlib.pyplot as plt
from matplotlib  import cm

import numpy as np

### Section 1: Feature selection

list1=['poi','salary', 'bonus','from_poi_to_this_person', 'from_this_person_to_poi','shared_receipt_with_poi', 'exercised_stock_options','restricted_stock_deferred']
list2=['poi','salary', 'bonus','from_poi_to_this_person', 'from_this_person_to_poi','shared_receipt_with_poi', 'ratio']
list3=['poi','salary', 'bonus','ratio','from_poi_to_this_person', 'from_this_person_to_poi','shared_receipt_with_poi', 'exercised_stock_options','restricted_stock_deferred']
list4=['poi','salary', 'bonus','from_poi_to_this_person', 'from_this_person_to_poi','shared_receipt_with_poi']

features_list = list1
my_feature_list = []
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
### Section 2: Remove outliers
data_dict.pop('TOTAL', 0 )
### Section 3: Create new features
### Store to my_dataset for easy export below.
my_dataset = data_dict
rat = []
#count = 0
for k in my_dataset:
    stock = my_dataset[k]["total_stock_value"]
    pay = my_dataset[k]["total_payments"]
    if pay == 'NaN' or stock == 'NaN':
        ratio = 0
    elif my_dataset[k]["salary"] == 'NaN' and my_dataset[k]["bonus"] == 'NaN':
        ratio = 0
    else:
        ratio = float(stock) / float(pay)
    my_dataset[k]["ratio"] = ratio
    rat.append(ratio)
#    if ratio > 0:
#        count += 1
my_feature_list.append(rat)
#print count

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Section 4: Classifier algorithms
from sklearn.pipeline import Pipeline
from sklearn import decomposition
from sklearn import preprocessing

from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
gauss = GaussianNB()
sv = svm.SVC(kernel='rbf')
pca = decomposition.PCA()
min_max_scaler = preprocessing.MinMaxScaler()
clf = Pipeline(steps=[('scaler',min_max_scaler),('pca', pca), ('svm', sv)])
#clf = Pipeline(steps=[('scaler',min_max_scaler),('pca', pca), ('tree', tree)])

### Section 5: Tuning
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn import metrics

kf = KFold(len(labels),3,random_state=5)

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    
criterion = ["gini", "entropy"]
min_samples_split = [2, 10, 20]
max_depth = [None, 2, 5, 10]
min_samples_leaf = [1, 5, 10]
max_leaf_nodes = [None, 5, 10, 20]
    
Cs = [5e2,1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6]
gammas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, .5, 1, 5, 7.5]

n_components = np.arange(1,len(features_list))


grid = GridSearchCV(clf, dict(pca__n_components=n_components, svm__C=Cs,svm__gamma=gammas), scoring='recall')
#grid = GridSearchCV(clf, dict(pca__n_components=n_components, tree__criterion = criterion, tree__min_samples_split=min_samples_split, tree__max_depth=max_depth, tree__min_samples_leaf = min_samples_leaf,
#                              tree__max_leaf_nodes = max_leaf_nodes),scoring='f1')
grid.fit(features_train,labels_train)
prediction = grid.predict(features_test)

print metrics.f1_score(labels_test, prediction)
print metrics.recall_score(labels_test, prediction)
print grid.best_params_
    
f1_all = []
precision_all = []
recall_all = []    
for train_index, test_index in kf:
    features_train_kf = [features[ii] for ii in train_index]
    features_test_kf = [features[ii] for ii in test_index]
    labels_train_kf = [labels[ii] for ii in train_index]
    labels_test_kf = [labels[ii] for ii in test_index]
    
    clf.set_params(pca__n_components=5, svm__C=500000,svm__gamma=1).fit(features_train_kf,labels_train_kf)
    #clf.set_params(tree__criterion = 'entropy', tree__min_samples_split = 2, pca__n_components = 1, tree__min_samples_leaf = 1, tree__max_leaf_nodes = 10,
    #               tree__max_depth = 5).fit(features_train_kf,labels_train_kf)
    pred = clf.predict(features_test_kf)
    f1_all.append(metrics.f1_score(labels_test_kf, pred))
    recall_all.append(metrics.recall_score(labels_test_kf, pred))
    precision_all.append(metrics.precision_score(labels_test_kf, pred))
prec = np.average(precision_all)
recall = np.average(recall_all)
f1 = np.average(f1_all)

'''print prec, recall, f1
print pca.explained_variance_ratio_
print np.sum(pca.explained_variance_ratio_)'''

### Section 6: Dump classifier, dataset, and features_list

dump_classifier_and_data(clf, my_dataset, features_list)
#plt.scatter(data[:,1],data[:,3],c=data[:,0],cmap = cm.plasma_r)