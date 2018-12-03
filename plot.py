
import pandas as pd
import multiprocessing
import numpy as np
import random
import sys
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
df=pd.read_csv('1_total_fee_w2v.csv')
l=list(df['1_total_fee'].astype('str'))
name=list(df)

def plot_with_labels(low_dim_embs, labels, filename = 'tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize= (10, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy = (x, y), textcoords = 'offset points', ha = 'right', va = 'bottom')
    plt.savefig(filename) 

tsne = TSNE(perplexity = 30, n_components = 2, init = 'pca', n_iter = 5000)






plot_only = 300
low_dim_embs = tsne.fit_transform(df.iloc[:plot_only][name[1:]])
labels = [l[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels)

