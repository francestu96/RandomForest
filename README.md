# RandomForest
Python RandomForest implementation from scratch.

There are two classifiers for two differents dataset retrived on Kuggle.

In this repo there is just WineReviews.csv (https://www.kaggle.com/zynicide/wine-reviews)

The US_Accident dataset is too big to be uploaded (https://www.kaggle.com/sobhanmoosavi/us-accidents)

In the first dataset I try to classify wine variety whereas in the second one the US country where the accident appened.
In both cases, I've found the best model which fits better our datasets is the Random Forest one.

This Random Forest in implemented by a bagging of Decision Tree models. Each tree is a thread in order to speed up the computation time.
There is also a "Tree" folder where there is stored a file for each tree. There it's shown the tree stucture and the questions each node corresponds to.
