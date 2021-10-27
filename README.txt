CSC 466
Fall 2021
Lab 3 Part 2

Eric Inman (eainman@calpoly.edu)
Aidan Barbieux (abarbieu@calpoly.edu)

Instructions on Running Classifiers:

Validation.py - used for testing C45 and tree construction
    1. run "python3 validation.py <filename> <k number of folds> <threshold>" without the parentheses
    
    2. Validation.py will create decision trees from the data in the filename you specified
       and test the accuracy of the tree with cross-validation
       
randomForest.py - used for creating and testing Random Forests
    1. run "python3 randomForest.py <filename> <numberOfAttributes> <percentageOfDataPoints> 
       <numberOfTrees> <outfile>" without the parentheses
    
    2. You have to specify an outfile for the program to dump the output into. 
    
knn.py - used for classifying entire datasets using k nearest neighbors
    1. run "python3 knn.py <filename> <k>"
    
    2. k has to be greater than 1. 
    
    3. Sit back and let the program do the work.
    
    
Output Files:

C45:            Parameters
    irisC.out - 20 , 0.2
    letterC.out - 10 , 0.1
    redC.out - 10 , 0.1
    whiteC.out - 5 , 0.2
    crxC.out - 10 , 0.15
    heart.out - 5 , 0.2
    
Random Forest:
    irisForest.out - 3, .7, 15
    letterForest.out - 8, .2, 10
    redForest.out - 10, .3, 10
    whiteForest.out - 8, .2, 15
    crxForest.out - 8, .5, 15
    heartForest.out - 8, .3, 10
    
KNN:
    irisK.out - 9
    letterK.out - 4
    redK.out - 2
    whiteK.out - 2
    crxK.out - 8
    heartK.out - 9
    