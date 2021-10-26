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


# In[118]:


# Euclidean distance = sqrt(sum((dist1 - dist2)^2)
# the binarized categorical data is also used in this calculation
def distance(d, x):
    res = 0
    
    # iterate all the way up until the class label
    for i in range(len(x)-1):
        res += ((x[i] - d[i])**2)
        
    return res**(1/2)

# returns the plurality class of the nearest neighbors
def mostCommonLabel(data, neighborIndices):
    labels = [data.loc[i][data.columns[-1]] for i in neighborIndices]
    common = Counter(labels)
    return max(labels, key=common.get)
    
def knn(data, k, x):
    distances = {row.Index: distance(row, x) for row in data.itertuples() if not row.Index == x.Index}
            
    # sort the distances lowest to highest
    distances = dict(sorted(distances.items(), key = lambda item: item[1]))
    
    # get the row indicies of the k nearest neighbors
    neighbors = [key for i, key in enumerate(distances) if i < k]     
    return mostCommonLabel(data, neighbors)

def knnPredictions(data, k):
    confusion = initializeConfusion(data)
    predictions = []
    
    # apparently itertuples does the same thing as iterrows but is much faster
    for d in data.itertuples():
        pred = knn(data, k, d)
        actual = d[-1]
        predictions.append((d.Index, pred, actual))
    
    
    accuracy = getStats(confusion, predictions)
    print(f"Accuracy of KNN with k = {k}: {accuracy:.4f}")
    print("Confusion Matrix")
    print("Predicted \u2193, Actual \u2192")
    print(confusion)
    
# with heart.csv, k = 2 gives the best accuracy with 0.7026
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

