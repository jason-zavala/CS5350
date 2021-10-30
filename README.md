# CS5350

This is a machine learning library developed by Jason Zavala for CS5350/6350 in University of Utah


Decision Tree commands:
To run the decision tree you need 2 parameters (between gini index, majority error, and entropy). The second argument is the depth of the tree. 

EXAMPLE:

python3 decisionTree.py entropy 3
python3 decisionTree.py gini 2
python3 decisionTree.py me 56


Update: 

passing "ada" for the first argument will choose the gini index for purity, and run the adaboost algo.

Example:

python3 decisionTree.py ada 3

Update: 

Passing "random" or "bag" for the third argument will run using the bagged trees algorithm / random forests method respectively. 


update: 
Ensemble Learning: 

to run my decisionTree_bankHeist.py use:

python3 decisionTree_bankHeist.py ada 1 bag


the first argument tells it to use the adaboost algorithm
the second argument tells the depth of the tree
the third argument is the type of ensemble method (random, boost, bag)

Examples:
python3 decisionTree_bankHeist.py ada 1 random
python3 decisionTree_bankHeist.py ada 2 bag
python3 decisionTree_bankHeist.py ada 122 boost



