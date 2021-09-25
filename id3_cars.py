"""
Jason Zavala | Shandian Zhe
09/23/2021
Machine Learning | CS5350
Univeristy of Utah
Fall 2021
"""
from collections import Counter
import math, re



"""
This represents the data structure for the actual tree
to start of we have:

root

each node has:

value    - decision split
label    - leafs
children - connected nodes
"""
class Node: 
    # constructur => give default values to parameters
    def __init__(self, value = None, label = None, children = {}) -> None:
        self.value      = value
        self.label      = label
        self.children   = children

    # override to string for representing obj
    def __str__ (self, level=0):
        return "hello world!"

def calculate_entropy(data, label_values):
    entropy = 0
    size = len(data)
    label_ratio = []

    for value in label_values:
        label_ratio.append(sum(x["label"] == value for x in data)/size)
    
    # sum using H(s) = -p * log_2(p)
    for ratio in label_ratio:
        if ratio != 0:
            entropy += -ratio * math.log(ratio, 2)
    
    return entropy

# read CSV and format it into a usable structure
def read_csv(CSVfile, attr):
    data = []
    with open(CSVfile, 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            val   = {}
            count = 0

            for attributes in attr:
                val[attributes] = terms[count]
                count = count + 1
                # get the last col
            val["label"] = terms[count]
            data.append(val)
    f.close()
    return data

# class that represents our actual decision tree
class DecisionTree:
    # member variables that represent different data streams build from our CSV
    training_set = []
    labels = []
    attributes = []
    attribute_values = {} 

    def __init__(self) -> None:
        pass

def main():
    desc_file = "data-desc.txt"
    l = []

    with open(desc_file, 'r') as f:
        l = f.readlines()
    f.close()

    label_val = l[2].strip().replace(" ", "").split(',')
    attributes = l[len(l) - 1].strip().split(',')[:-1]
    attr_val = {}

    for i in attributes:
        for lines in l:
            if i in lines:
                values = re.split(',|:|\.', lines.strip().replace(" ", ""))[1:-1]
                attr_val[i] = values
                break
            
    # Get the training data
    CSVfile = "train.csv"
    data = read_csv(CSVfile, attributes)           
    
    # find entropy
    curr_entropy = calculate_entropy(data, label_val)

    size = len(data)
    attr_information_gain = {}
    
    for a in attributes:
        expected_attribute_entropy = []
        for value in attr_val[a]: 
            data_attribute_value = list(filter(lambda arr: arr[a] == value, data))
            attribute_value_rat  = len(data_attribute_value)/size
            expected_attribute_entropy.append(calculate_entropy(data_attribute_value, label_val)* attribute_value_rat)
        expected_attribute_entropy = sum(expected_attribute_entropy)
        attr_information_gain[a] = curr_entropy - expected_attribute_entropy
    
    # split on max info gain 
    best_attribute = max(attr_information_gain, key=attr_information_gain.get)
    print(attr_information_gain)
    print(best_attribute)

if __name__ == "__main__":
    main()