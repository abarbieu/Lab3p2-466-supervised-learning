#!/usr/bin/env python
# coding: utf-8

# ## Classifier

# Takes JSON input (from tree induction) and CSV file

# In[2]:


import numpy as np
import pandas as pd
import json
import sys
from InduceC45 import readFiles
    
def traverseTree(row, tree, nodeType):
    if nodeType == "leaf":
        return tree["decision"]        
        
    elif nodeType == "node":
        attrVal = row[tree["var"]]
        for obj in tree["edges"]:
            newType = "leaf" if "leaf" in obj["edge"].keys() else "node"
            
            if "direction" in obj["edge"].keys(): # edge is numeric
                
                if obj["edge"]["direction"] == "le" and attrVal <= obj["edge"]["value"]: # data is <= alpha
                    return traverseTree(row, obj["edge"][newType], newType)
                 
                elif obj["edge"]["direction"] == "gt" and attrVal > obj["edge"]["value"]: # data is > alpha
                    return traverseTree(row, obj["edge"][newType], newType)
                
            elif obj["edge"]["value"] == attrVal: # if attribute value matches edge
                return traverseTree(row, obj["edge"][newType], newType)
        return tree["plurality"]["decision"]

def initializeConfusion(df):
    labels = df.iloc[:, -1].unique() # labels are in last column (not using result df from classify)
    zeros = np.zeros(shape=(len(labels), len(labels)))
    confusion = pd.DataFrame(zeros, labels, labels)
    return confusion

def classifySimple(df, tree):
    predictions = []
    keys = list(tree)
    
    for i, row in df.iterrows():
        prediction = traverseTree(row, tree[keys[-1]], keys[-1])
      
        predictions.append([i, prediction])
    return pd.DataFrame(predictions, columns=['index', 'prediction']).set_index('index')

def evaluate(df, preds):
    confusion = initializeConfusion(df)
    numErrors, numCorrect, totalClassified = 0, 0, 0
    for i, row in df.iterrows():
        prediction = preds.loc[i,"prediction"]
        actual = row[df.columns[-1]]
        
        confusion[actual][prediction] += 1
        
        if prediction != actual:
            numErrors += 1
        else:
            numCorrect += 1

        totalClassified += 1
        
    results = df.join(preds)
    
    return {"accuracy": numCorrect / totalClassified,
          "errorRate": numErrors / totalClassified,
          "numClassified": totalClassified,
          "numCorrect": numCorrect,
          "numErrors": numErrors,
          "confusionLabel": "Actual \u2193, Predicted \u2192",
          "confusion": confusion,
          "results": results}    

if __name__ == "__main__":
    if len(sys.argv) == 3:
        _, datafile, treefile = sys.argv
    else:
        print("Usage: python3 classifier.py <datafile.csv> <tree.json>")
        exit(1)
        
    df, filename, isLabeled, attrs = readFiles(datafile)
    tree = None
    
    with open(treefile) as tf:
        tree = json.load(tf)
    
    preds = classifySimple(df, tree)
    results = evaluate(df, preds)
    
    for v in results:
        print(v, results[v])