import numpy as np
import pandas as pd
import math
import json
import sys
import time

def calcNumEntropy(data, attr):
    pass

def findBestSplit(data, attrs, x, ):
    p0 = entropy(data.iloc[:, -1])
    k = len(data.iloc[:, -1].unique()) # num unique labels 
    labelCol = data.columns[-1]
    gain = {}
    
    # initializing array counts
    arrayCounts = {}
    for j in range(len(k)):
        arrayCounts[j] = {}
        
#     for a in attrs:
#         for j in range(len(k)):
    pass
            
    

def selectSplittingAttr(attrs, data, threshold):
    p0 = entropy(data.iloc[:,-1])
    gain={}
    for a in attrs:
        p = 0
#         if isNumeric()
#             x = findBestSplit(data, a)
#             p = entropyAttr(data, a, x)
#         else:
#             p = entropyAttr(data, a)
        gain[a] = p0 - entropyAttr(data, a) # info gain
    
    bestAttr=max(gain,key=gain.get)
    if gain[bestAttr] > threshold:
        return bestAttr
    else:
        return None

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

    # class must be in last column
def c45(data, attrs, thresh):
    # base case 1
    #print(data)
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
        "p": classes.value_counts()[classes.mode()][0]/len(classes)
    }
    
    # base case 2
    if len(attrs) == 0:
        pluralityClass.update({"type": "noAttrs"})
        return {"leaf": pluralityClass}                 # create leaf node with most frequent class
    
    # select splitting attr
#     asplit = selectSplittingAttr(attrs, data, thresh)
    if asplit == None:
        pluralityClass.update({"type": "threshold"})
        return {"leaf": pluralityClass}
        
    else:
        attrs.remove(asplit)
        newNode = {"node": {"var": asplit, "plurality": pluralityClass, "edges": []}}
        possibleValues = data[asplit].unique()                # gets unique values in column
        
        for value in possibleValues:
            tic=time.clock()
            relatedData = data[(data == value).any(axis = 1)] # take rows that have that value
            
            if len(relatedData.columns) != 0:
                subtree = c45(relatedData, attrs, thresh) 
                edge = {"value": value}
                edge.update(subtree)
                newNode["node"]["edges"].append({"edge": edge})
        return newNode

# Reads a training set csv file and a restrictions vector text file, returns arranged training set          
def readFiles(filename=None, restrictions=None):
    if filename is None and restrictions is None:
        if len(sys.argv) < 2:
            print("Not enough arguments.")
            exit(1)
        elif len(sys.argv) == 3:
            restrictions = sys.argv[2]
        filename = sys.argv[1]

    restr=None
    if restrictions != None:
        with open(restrictions) as r:
            lines = r.read().replace(', ', ' ')
            restr = [int(x) for x in lines.split(' ')]

    df = pd.read_csv(filename)
    aclass = df.iloc[1,0]
    
    attrs = {}
    for a in df.columns:
        attrs[a] = df[a][1]
    
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
    return df, filename, isLabeled, attrs

# runs c45 with data from file of name training data with restrictions in filename restrictions
def induceC45(trainingData=None, restrictions=None, threshold=0.2):
    df,filename,tmp, attrs = readFiles(trainingData, restrictions)
    print(attrs)
#     tree={"dataset": filename}
#     tree.update(c45(df, df.columns[:-1].tolist(), threshold))
#     return tree


# prints a decision tree
def printTree(tree):
    with open("tree.json", 'w') as f:
        json.dump(tree, f)
    print(json.dumps(tree, sort_keys=False, indent=2))
    

if __name__ == "__main__":
    printTree(induceC45(threshold=0))