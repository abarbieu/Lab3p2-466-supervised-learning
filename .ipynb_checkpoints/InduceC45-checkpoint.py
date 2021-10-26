#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import math
import json
import sys
import time


# In[6]:


# entropy of a series of data
def entropy(classcol):
    vals = classcol.value_counts()
    size = classcol.count()
    entropy=0
    for v in vals:
        entropy -= (v/size) * math.log(v/size,2)
    return entropy

# entropy of an attribute in a dataset, over each value of the attribute
def entropyAttr(data, attr):
    vals = data.pivot(columns=attr,values=data.columns[-1])
    entropyTot = 0
    for c in vals.columns:
        entropyTot += (vals[c].count()/len(data)) * entropy(vals[c])
    return entropyTot

# entropy of a series of data
def entropyValCounts(vals):
    size=0
    for v in vals:
        size += v
        
    entropy=0
    for v in vals:
        entropy -= (v/size) * math.log(v/size,2)
    return entropy

def splitEntropy(le, gt):
    sizeLe = 0
    for v in le:
        sizeLe += v
    sizeGt = 0
    
    for v in gt:
        sizeGt += v
    
    size = sizeLe+sizeGt
    
    return sizeLe/size * entropyValCounts(le) + sizeGt/size * entropyValCounts(gt)
    
def calcGainBetter(data,attr,p0):
    vals = data[attr].unique()
    
    bestSplit = None
    bestGain = -1
    for v in vals:
        le = data[data[attr] <= v].iloc[:,-1].value_counts()
        gt = data[data[attr] > v].iloc[:,-1].value_counts()
        
        splitGain = p0 - splitEntropy(le,gt)
        if splitGain > bestGain:
            bestGain = splitGain
            bestSplit = v
    
    return float(bestSplit), float(bestGain)

def findBestSplit(data, attr, p0):
    out=calcGainBetter(data, attr, p0)
    return out


# In[13]:


def selectSplittingAttr(attrs, data, threshold):
    p0 = entropy(data.iloc[:,-1])
    bestGain = 0
    alpha = None
    bestAttr = None
    
    for a in attrs:
        tmpAlpha=None
        tmpGain=0
        if attrs[a] < 1: # if attr is numeric
            tmpAlpha, tmpGain = findBestSplit(data, a, p0)
        else:
            tmpGain = p0 - entropyAttr(data, a)
        if tmpGain > bestGain:
            bestAttr = a
            bestGain = tmpGain
            alpha = tmpAlpha

    if bestGain > threshold:
        return bestAttr, alpha
    else:
        return None, None


# In[26]:


# class must be in last column
def c45(data, attrs, thresh, space=""):
    # base case 1
    classes = data.iloc[:,-1]
    firstclass = None
    allsame=True
    
    for c in classes:
        if firstclass == None:
            firstclass = c
        elif c != firstclass:
            allsame=False
            break
            
    if allsame:
        #create leaf node for perfect purity
        return {"leaf": {
            "decision": firstclass,
            "p": 1.0,
            "type": "allsame"
        }}
    
    pluralityClass = {
        "decision": classes.mode()[0],
        "p": classes.value_counts()[classes.mode()[0]]/len(classes)
    }
    
    # base case 2
    if len(attrs) == 0:
        pluralityClass.update({"type": "noAttrs"})
        return {"leaf": pluralityClass}                 # create leaf node with most frequent class
    
    # select splitting attr
    asplit, alpha = selectSplittingAttr(attrs, data, thresh)

    if asplit is None:
        pluralityClass.update({"type": "threshold"})
        return {"leaf": pluralityClass}
        
    elif alpha is None:
        
        attrs.pop(asplit)
        newNode = {"node": {"var": asplit, "plurality": pluralityClass, "edges": []}}
        possibleValues = data[asplit].unique()                # gets unique values in column
        
        
        for value in possibleValues:
            relatedData = data[(data == value).any(axis = 1)] # take rows that have that value
            
            if len(relatedData.columns) != 0:
                subtree = c45(relatedData, attrs, thresh, space + "  ") 
                edge = {"value": value}
                edge.update(subtree)
                newNode["node"]["edges"].append({"edge": edge})
                
        return newNode
    else:
        le = data[data[asplit] <= alpha]
        gt = data[data[asplit] > alpha]
        
        leTree = c45(le, attrs, thresh, space + "  ")
        gtTree = c45(gt, attrs, thresh, space + "  ")
        
        leEdge = {"value": alpha, "direction": "le"}
        gtEdge = {"value": alpha, "direction": "gt"}
        
        leEdge.update(leTree)
        gtEdge.update(gtTree)
        
        newNode = {"node": {"var": asplit, "edges": [
            {"edge": leEdge},
            {"edge": gtEdge},
        ]}}
        
        return newNode

# In[28]:


# Reads a training set csv file and a restrictions vector text file, returns arranged training set          
def readFiles(filename, restrictions=None):
    restr=None
    if restrictions != None:
        with open(restrictions) as r:
            lines = r.read().replace(', ', ' ')
            restr = [int(x) for x in lines.split(' ')]

    df = pd.read_csv(filename)
    aclass = df.iloc[1,0]
    
    attrs = {}
    for a in df.columns:
        attrs[a] = int(df[a][0])
    
    isLabeled = True
    if not isinstance(aclass, str):
        isLabeled = False
    df = df.drop([0,1], axis=0)
    if restr != None:
        for i,v in enumerate(df.columns):
            if restr[i] == 0:
                df = df.drop(columns=[v])
    if isLabeled:
        df = df[[c for c in df if c not in [aclass]] + [aclass]]
        
    attrs.pop(df.columns[-1])
    for a in attrs:
        if attrs[a] == 0:
            df[a] = pd.to_numeric(df[a], errors='coerce')
    df = df.dropna()
    df = df[(df != '?').all(axis=1)]
    return df, filename, isLabeled, attrs


if __name__ == "__main__":
    restrfile=None
    if len(sys.argv) == 3:
        _, datafile, thresh = sys.argv
    elif len(sys.argv) == 4:
        _, datafile, thresh, restrfile = sys.argv
    else:
        print("Usage: python3 InduceC45.py <datafile.csv> <threshold> [restrictions.txt]")
        exit(1)
    thresh=float(thresh)
    df, filename, tmp, attrs = readFiles(datafile, restrfile)
    
    tree={"dataset": filename}
    tree.update(c45(df, attrs, thresh))

    print(json.dumps(tree, sort_keys=False, indent=2))

