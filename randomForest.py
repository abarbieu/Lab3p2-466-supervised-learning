import numpy as np
import pandas as pd
import math
import json
import sys
import time
from InduceC45 import c45, readFiles
from classifier import classify, readArrange, initializeConfusion

def getArgs():
    restr=None
    if len(sys.argv) != 6:
        print("Usage: python3 randomForest.py <datafile.csv> <m> <k> <N> <outputFileName.csv>")
        exit(1)
    else:
        _, datafile, m, k, N, outputfile = sys.argv
    
    print(outputfile)

getArgs()