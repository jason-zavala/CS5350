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



Update: 

To run the stochastic_gradient_descent.py use:

python3 stochastic_gradient_descent.py

To run normal gradient_descent.py use:
python3 gradient_descent.py

Update for Perceptron: 

to run perceptron algorithm, the first argument needs to be the method for perceptron.
The options are: standard, average, voted

Example usage: 

python3 Perceptron.py standard
python3 Perceptron.py average
python3 Perceptron.py voted


#Update for SVM: 

To run the SVM algorithm you need 4 arguments: 

1 - The hyperparameter *C*
2 - The schedule version (*0* or *1*)
3 - Which version *primal* or *dual* version of SVM
4 - *linear* or *gaussian* kernel 

**EXAMPLE USAGE**
python3 svm.py 0.802 0 dual linear
python3 svm.py 0.573 0 dual gaussian

#Neural Netowrks

To run the NN algorithm you need 1 argument:

**EXAMPLE USAGE**
*The following will run the back propogation algorithm:*
python3 nn.py part_a
*The following will run the stoch. gradient decscent verison:*
python3 nn.py part_b
*The following will initialize all the weights to 0:*
python3 nn.py part_c





