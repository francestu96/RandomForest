# RandomForest
Python RandomForest implementation from scratch.

In this repo there is a classifier for WineReviews dataset retrived on Kuggle (https://www.kaggle.com/zynicide/wine-reviews)

The goal of this classifier is to to classify correctly the wine variety based on wine reviews. 

I've found the best model which fits better this datasets is the Random Forest one.

This Random Forest is implemented by a bagging of Decision Tree models. Each tree is a thread in order to speed up the computation time.
There is also a "Tree" folder where there is stored a file for each tree. There it's shown the tree stucture and the questions each node corresponds to.
