import DecisionTree as dt
import RandomForest as rf
import DataUtils as utils
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

target = 'variety'
features = ['country','description','points','price','variety','winery']
dataset = pd.read_csv('WineReviews.csv', usecols=features)

dataset = utils.setUpWineDataset(dataset.copy())
train, test = train_test_split(dataset, test_size=0.2)

# dataset.dropna(inplace=True)
# dataset_labled = dataset.apply(LabelEncoder().fit_transform)

# y = dataset_labled['variety']
# X = dataset_labled.drop('variety', axis=1)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# n_trees = [1, 10, 30, 60, 100]
# for n in n_trees:
#     rf=RandomForestClassifier(n_estimators=n)
#     rf.fit(X_train,y_train)
#     y_pred=rf.predict(X_test)
#     acc=accuracy_score(y_test, y_pred)
#     print("[Randon forest algorithm (" + str(n) + " trees)] accuracy_score: {:.3f}.".format(acc))

# DECISION TREE #######################################################
# decisionTree = dt.DecisionTree(train.values, train.columns.get_loc(target), train.columns.values)
# decisionTree.join()
# print("[Decision Tree] Accuracy = " + str(decisionTree.getAccuracy(test)))

# decisionTree.printTree()
#######################################################################

# RANDOM FOREST ##########################################################
randomForest = rf.RandomForest(train, target, 10)
print("[Random Forest (10) trees)] Accuracy = " + str(randomForest.getAccuracy(test)))
# n_trees = [1, 10, 30, 60, 100]
# for n in n_trees:
#     randomForest = rf.RandomForest(train, target, n)
#     print("[Random Forest (" + str(n) + " trees)] Accuracy = " + str(randomForest.getAccuracy(test)))

# randomForest.printForest()
##########################################################################