from threading import Thread
import unidecode
import pandas as pd

class DecisionTree (Thread):
    def __init__(self, dataset, targetIndex, features, features_idxs=None):
        Thread.__init__(self)
        self.targetIndex = targetIndex
        self.features = features
        self.dataset = dataset
        self.features_idxs = features_idxs
        self.start()

    class Leaf:
        def __init__(self, decisionTree, dataset):
            self.predictions = decisionTree.class_counts(dataset)

    class DecisionNode:
        def __init__(self,
                    question,
                    true_branch,
                    false_branch):
            self.question = question
            self.true_branch = true_branch
            self.false_branch = false_branch

    class Question:
        def __init__(self, decisionTree, column, value):
            self.decisionTree = decisionTree
            self.column = column
            self.value = value

        def match(self, example):
            val = example[self.column]
            if (pd.isna(val) or pd.isna(self.value)):
                return False

            elif isinstance(val, int) or isinstance(val, float):
                return val >= self.value

            else:
                return val == self.value

        def __repr__(self):
            condition = "=="
            if isinstance(self.value, int) or isinstance(self.value, float):
                condition = ">="
            return "Is %s %s %s?" % (
                self.decisionTree.features[self.column], condition, str(self.value))

    def run(self):
        self.tree = self.build_tree()

    def build_tree(self):
        return self.build_tree_rec(self.dataset)

    def build_tree_rec(self, dataset):
        gain, question = self.find_best_split(dataset)

        if gain == 0:
            return self.Leaf(self, dataset)

        true_dataset, false_dataset = self.partition(dataset, question)

        true_branch = self.build_tree_rec(true_dataset)
        false_branch = self.build_tree_rec(false_dataset)

        return self.DecisionNode(question, true_branch, false_branch)

    def find_best_split(self, dataset):
        best_gain = 0 
        best_question = None
        current_uncertainty = self.gini(dataset)
        n_features = len(dataset[0])

        for col in range(n_features):
            if col == self.targetIndex:
                continue
            values = set([row[col] for row in dataset])
            for val in values:
                question = self.Question(self, col, val)
                true_dataset, false_dataset = self.partition(dataset, question)
                if len(true_dataset) == 0 or len(false_dataset) == 0:
                    continue

                gain = self.info_gain(true_dataset, false_dataset, current_uncertainty)
                if gain >= best_gain:
                    best_gain, best_question = gain, question

        return best_gain, best_question

    def info_gain(self, left, right, current_uncertainty):
        p = float(len(left)) / (len(left) + len(right))
        return current_uncertainty - p * self.gini(left) - (1 - p) * self.gini(right)

    def gini(self, dataset):
        counts = self.class_counts(dataset)
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / float(len(dataset))
            impurity -= prob_of_lbl**2
        return impurity

    def class_counts(self, dataset):
        counts = {}
        for row in dataset:
            label = row[self.targetIndex]
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        return counts

    def partition(self, dataset, question):
        true_dataset, false_dataset = [], []
        for row in dataset:
            if question.match(row):
                true_dataset.append(row)
            else:
                false_dataset.append(row)
        return true_dataset, false_dataset

    def printTree(self, spacing=""):
        file = open("trees/" + self.name.replace("Thread", "Tree"),"w")  
        self.printTreeRec(self.tree, file)
        file.close() 

    def printTreeRec(self, node, file, spacing=""):
        if isinstance(node, self.Leaf):
            file.write(spacing + "Predict" + unidecode.unidecode(str(node.predictions)) + "\n")
            return

        file.write(spacing + unidecode.unidecode(str(node.question)) + "\n") 
        file.write(spacing + '--> True:\n') 
        self.printTreeRec(node.true_branch, file, spacing + "  ")
        file.write(spacing + '--> False:\n') 
        self.printTreeRec(node.false_branch, file, spacing + "  ")


    def classify(self, row):
        if self.features_idxs != None:
            row = row[self.features_idxs]

        return self.classify_rec(row, self.tree)
    
    def classify_rec(self, row, node):
        if isinstance(node, self.Leaf):
            return node.predictions

        if node.question.match(row):
            return self.classify_rec(row, node.true_branch)
        
        else:
            return self.classify_rec(row, node.false_branch)
    
    
    def leaf_prob(self, counts):
        total = sum(counts.values()) * 1.0
        probs = {}
        for lbl in counts.keys():
            probs[lbl] = int(counts[lbl] / total)
        return probs
    
    def print_leaf(self, counts):
        total = sum(counts.values()) * 1.0
        probs = {}
        for lbl in counts.keys():
            probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
        return probs


    def getAccuracy(self, test):
        success = 0
        for row in test.values:
            actual = row[self.targetIndex]
            predicts = self.leaf_prob(self.classify(row))

            max_prob = 0
            for key in predicts:
                if predicts[key] > max_prob:
                    max_prob = predicts[key]
                    prediction = key

            if prediction == actual:
                success+=1

        return success / test.shape[0]