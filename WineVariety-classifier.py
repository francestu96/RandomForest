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

# RANDOM FOREST ##########################################################
n_trees = [1, 10, 30, 60, 100]
for n in n_trees:
    randomForest = rf.RandomForest(train, target, n)
    print("[Random Forest (" + str(n) + " trees)] Accuracy = " + str(randomForest.getAccuracy(test)))

randomForest.printForest()
##########################################################################