import DecisionTree as dt
import RandomForest as rf
import DataUtils as utils
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

target = 'State'
features = ['State','Start_Time','Temperature(F)','Humidity(%)','Pressure(in)','Visibility(mi)','Weather_Condition','Crossing','Junction','Traffic_Signal']
dataset = pd.read_csv('US_Accidents_May19.csv', usecols=features)

dataset = utils.setUpUSAccidentDataset(features, dataset.copy())

# utils.setUpUSAccident(features, dataset)
# DATA ANALYSIS #######################
# graph = utils.top_accident_by_state(dataset) #
#######################################

train, test = train_test_split(dataset, test_size=0.2)

# DECISION TREE #######################################################
# decisionTree = dt.DecisionTree(train.values, train.columns.get_loc(target), train.columns.values)
# decisionTree.join()
# print("[Decision Tree] Accuracy = " + str(decisionTree.getAccuracy(test)))
# decisionTree.printTree()
#######################################################################

# RANDOM FOREST ##########################################################
randomForest = rf.RandomForest(train, target, 100)
print("[Random Forest] Accuracy = " + str(randomForest.getAccuracy(test)))
randomForest.printForest()
##########################################################################


# graph.show()