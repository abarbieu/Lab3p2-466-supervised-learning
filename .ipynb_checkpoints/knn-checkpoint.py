#!/usr/bin/env python
# coding: utf-8

# In[104]:


import numpy as np
import pandas as pd
import sys
from collections import Counter
from classifier import initializeConfusion

# normalizes all the numeric columns
def normalizeNumeric(df, attrs):
    for a in attrs:
        if attrs[a] < 1:
            colMax = df[a].max()
            colMin = df[a].min()
            # probably no need to normalize if the values are very small. Might have to adjust the value
            if colMax < 5:
                continue
            df[a] = df[a].apply(lambda x: (x - colMin)/(colMax-colMin))
    return df


# In[87]:


def prepareData(df):
    attrs = {}
    for a in df.columns:
        attrs[a] = int(df[a][0])
    attrs.pop(df.columns[-1])
    
    aclass = df.iloc[1,0]
    
    # get attributes and whether or not they are numeric
    attrs = {}
    for a in df.columns:
        attrs[a] = int(df[a][0])
    
    # drop the numeric and class label rows
    df = df.drop([0, 1], axis=0)
    
    # clean data and convert to numeric
    for a in attrs:
        if attrs[a] == 0:
            df[a] = pd.to_numeric(df[a], errors='coerce')
    df = df.dropna()
    df = df[(df != '?').all(axis=1)]
    
    # getting list of categorical columns to pass to get_dummies to binarize categorical data
    catCols = [col for col in attrs if attrs[col] >= 1 and col != aclass]
    df = pd.get_dummies(df, columns = catCols)
    
    # append class labels to end of dataframe
    df = df[[c for c in df if c not in [aclass]] + [aclass]]
        
    # normalizing the numeric columns
    df = normalizeNumeric(df, attrs)
   
    return df, attrs


# In[99]:


def getStats(confusion, predictions):
    accuracy = 0
    numCorrect = 0
    
    # prediction = index 1, actual = index 2
    for pred in predictions:
        p = pred[1]
        actual = pred[2]
        if p == actual:
            numCorrect += 1
        
        # actual is vertical, predicted is horizontal
        confusion[actual][p] += 1
    
    accuracy = numCorrect / len(predictions)
    return accuracy

# returns the plurality class of the nearest neighbors
def mostCommonLabel(data, neighborIndices):
    labels = [data.loc[i][data.columns[-1]] for i in neighborIndices]
    common = Counter(labels)
    return max(labels, key=common.get)
    
def knn(data, dataMinusTag, k, x):
    # performs Euclidean distance on all elements in data (vectorized)
    dists = np.sqrt(np.sum((dataMinusTag - x) ** 2, axis=1))
    
    # creates a dataframe with distances corresponding to row indexes and sorts them
    distances = pd.DataFrame(dists, index=data.index, columns=["dists"])
    distances = distances.sort_values("dists")
            
    
    # get the row indicies of the k nearest neighbors. Ignore first item
    neighbors = [key.Index for i, key in enumerate(distances.itertuples()) if i <= k and i > 0]
    return mostCommonLabel(data, neighbors)

def knnPredictions(data, k):
    confusion = initializeConfusion(data)
    predictions = []
    
    # creates an np.array with the data minus the label column
    dataMinusTag = np.array(data.iloc[:, data.columns != data.columns[-1]])
                        
    # itertuples is faster than iterrows. Also d[1:-1] takes out the row index and class label
    for d in data.itertuples():
        pred = knn(data, dataMinusTag, k, d[1:-1])
        actual = d[-1]
        predictions.append((d.Index, pred, actual))
    
    
    accuracy = getStats(confusion, predictions)
    print(f"Accuracy of KNN with k = {k}: {accuracy:.4f}")
    print("Confusion Matrix")
    print("Predicted = Horizontal, Actual = Vertical")
    print(confusion)
    
# with heart.csv, k = 9 gives the best accuracy with 0.8704
if __name__ == "__main__":
    if len(sys.argv) == 3:
        _, datafile, k = sys.argv
    else:
        print("Usage: python3 knn.py <filename> <k>")
        exit(1)
    k = int(k)
        
    if k <= 1:
        print("K has to be greater than 1")
        exit(1)
        
    df = pd.read_csv(datafile)
    df, attrs = prepareData(df)

    knnPredictions(df, k)

