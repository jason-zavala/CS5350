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

    def __init__(self, data_description_file, training_file) -> None:
        # open file and save every file into the list and parse on indexes
        lines = []
        with open(data_description_file, 'r') as f:
            lines = f.readlines()
        f.close()

        # grab labels from index 2 and sanitize the data
        self.labels = lines[2].strip().replace(" ", "").split(',')

        # the attributes are in the last column, we reach into the last index and then sanitize the data there
        self.attributes = lines[-1].strip().split(',')[:-1]

        # basically what we need to do is loop through the lines and check for the attributes, then if we find it - we strip/sanitize the parts we dont need and store it in atr val
        for attribute in self.attributes:
            for line in lines: 
                if attribute in line:
                    self.attribute_values[attribute] = re.split(',|:|\.', line.strip().replace(" ", ""))[1:-1]
                    # once we have the values we need we actually dont need to keep looping so we just break
                    break
        # now we worry about the actual CSVfile
        with open(training_file, 'r') as f:
            for line in f:
                terms = line.strip().split(',')
                val   = {}

                for index, attribute in enumerate(self.attributes):
                    val[attribute] = terms[index]
                    # get the last col
                val["label"] = terms[index]
                self.training_set.append(val)
                
        f.close()
    
    # generically calculate entropy
    def calculate_entropy(self, training_set):
        label_proportion = []
        # iterate over our list labels are store the p values
        for label in self.labels:
            # TODO: come back and try  to optimize this... I feel like there is a simpler way to do this..
            label_proportion.append(sum(split["label"] == label for split in training_set) / len(training_set))
        
        # next use -p * log(p)
        entropy = 0
        for proportion in label_proportion:
            # [on_true] if [expression] else [on_false] 
            entropy += (-proportion * math.log(proportion, 2) if proportion != 0 else 0)
        return entropy

        
def main():
    print('hello world')

if __name__ == "__main__":
    main()