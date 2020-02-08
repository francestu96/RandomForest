from threading import Thread
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def setUpWineDataset(dataset):
    dataset.drop_duplicates('description', inplace=True)
    dataset.drop("description", axis=1, inplace=True)
    dataset = dataset[pd.notnull(dataset.price)]
    dataset = dataset.groupby('variety').filter(lambda x: len(x) > 100)
    dataset.loc['variety'] = dataset['variety'].replace(['Weissburgunder'], 'Chardonnay')
    dataset.loc['variety'] = dataset['variety'].replace(['Spatburgunder'], 'Pinot Noir')
    dataset.loc['variety'] = dataset['variety'].replace(['Grauburgunder'], 'Pinot Gris')
    dataset.loc['variety'] = dataset['variety'].replace(['Garnacha'], 'Grenache')
    dataset.loc['variety'] = dataset['variety'].replace(['Pinot Nero'], 'Pinot Noir')
    dataset.loc['variety'] = dataset['variety'].replace(['Alvarinho'], 'Albarino')
    dataset = dataset.sample(10000)
    return dataset