from threading import Thread
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def top_accident_by_state(dataset):
    fig,ax=plt.subplots(1,2,figsize=(15,8))
    clr = ("blue", "forestgreen", "gold", "red", "purple",'cadetblue','hotpink','orange','darksalmon','brown')
    dataset.State.value_counts().sort_values(ascending=False)[:10].sort_values().plot(kind='barh',color=clr,ax=ax[0])
    ax[0].set_title("Top 10 Acciedent Prone States",size=20)
    ax[0].set_xlabel('States',size=18)


    count=dataset['State'].value_counts()
    groups=list(dataset['State'].value_counts().index)[:10]
    counts=list(count[:10])
    counts.append(count.agg(sum)-count[:10].agg('sum'))
    groups.append('Other')
    type_dict=pd.DataFrame({"group":groups,"counts":counts})
    clr1=('brown','darksalmon','orange','hotpink','cadetblue','purple','red','gold','forestgreen','blue','plum')
    qx = type_dict.plot(kind='pie', y='counts', labels=groups,colors=clr1,autopct='%1.1f%%', pctdistance=0.9, radius=1.2,ax=ax[1])
    plt.legend(loc=0, bbox_to_anchor=(1.15,0.4)) 
    plt.subplots_adjust(wspace =0.5, hspace =0)
    plt.ylabel('')
    return plt

def setUpUSAccidentDataset(features, dataset):
    features.remove('Start_Time')
    features.append('Hour')
    dataset = dataset.sample(100000)
    dataset['Hour']=pd.to_datetime(dataset['Start_Time'], errors='coerce').dt.hour
    dataset.drop("Start_Time", axis=1, inplace=True)
    dataset = dataset.dropna()
    dataset.drop(dataset[(dataset['State'] == 'CA') | (dataset['State'] == 'TX')].index, inplace=True)
    return dataset

def setUpWineDataset(dataset):
    dataset = dataset.sample(10000)
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
    return dataset