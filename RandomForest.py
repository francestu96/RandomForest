import DecisionTree as dt
import numpy as np
import math

from collections import Counter

class RandomForest():
    def __init__(self, dataset, target, n_trees):
        self.dataset = dataset
        self.target = target
        self.n_trees = n_trees

        # self.n_sample = int(dataset.shape[0] / (n_trees / 2)) 
        self.n_sample = int(dataset.shape[0] / (np.log2(n_trees) if n_trees > 1 else 1))       
        # self.n_features = int(np.log2(dataset.shape[1]))   
        self.n_features = dataset.shape[1]                                 
        if self.n_sample < 10:
            raise Exception('Too many trees according to the dataset size') 
        
        self.treeThreads = [self.create_tree() for i in range(n_trees)]
        [tree.start() for tree in self.treeThreads]

    def create_tree(self):
        targetIndex = self.dataset.columns.get_loc(self.target)
        samples_Idxs = np.random.permutation(self.dataset.shape[0])[:self.n_sample]
        features_idxs = list(set(np.append(np.random.permutation(self.dataset.shape[1])[:self.n_features], targetIndex)))
        train = self.dataset.iloc[samples_Idxs][self.dataset.columns[features_idxs]]
        
        return dt.DecisionTree(train.values, train.columns.get_loc(self.target), train.columns.values, features_idxs)

    def classify(self, row):
        [treeThread.join() for treeThread in self.treeThreads]            

        prediction = sum((Counter(dict(tree.classify(row))) for tree in self.treeThreads), Counter())
        return prediction.most_common(1)[0][0]

    def printForest(self):
        for treeThread in self.treeThreads:
            treeThread.join()
            treeThread.printTree()


    def getAccuracy(self, test):
        targetIndex = self.dataset.columns.get_loc(self.target)
        success = 0
        for row in test.values:
            actual = row[targetIndex]
            prediction = self.classify(row)
            if prediction == actual:
                success+=1

        return success / test.shape[0]
