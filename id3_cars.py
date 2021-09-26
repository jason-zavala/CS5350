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
    def __init__(self, value = None, label = None, children = None) -> None:
        # [on_true] if [expression] else [on_false]
        self.attribute = "" if value is None else value
        self.label = "" if label is None else label
        self.children = {} if children is None else children

    # override to string for representing obj
    def __str__(self, level = 0) -> str:
        ret = ""
        if self.attribute != "":
            ret += "[" + repr(self.attribute) + "]\n"
        else:
            ret += "= " + repr(self.label) + "\n"
        
        for key, value in self.children.items():
            ret += "\t"*level+repr(key)
            ret += value.__str__(level+1)
        return ret

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
                val["label"] = terms[index + 1]
                self.training_set.append(val)
                
        f.close()
    
    # generically calculate entropy
    def calculate_entropy(self, training_set):
        if len(training_set) == 0:
            return 0
        
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


    def calculate_majority_error(self, data):
        if len(data) == 0:
            return 0
        
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

    def best_attribute(self, data, attributes):
        current_entropy = self.calculate_entropy(data)
        # for each attribute in our attributes list, calc the entropy & information gain for each set of values in our list
        data_size = len(data)
        attribute_information_gain = {}

        for attribute in attributes:
            expected_attribute_entropy = []
            for value in self.attribute_values[attribute]:
                data_attribute_values = list(filter(lambda x: x[attribute] == value, data))
                attribute_value_p = len(data_attribute_values)/data_size
                expected_attribute_entropy.append(self.calculate_entropy(data_attribute_values) * attribute_value_p)
            expected_attribute_entropy = sum(expected_attribute_entropy)
            attribute_information_gain[attribute] = current_entropy - expected_attribute_entropy

        return max(attribute_information_gain, key=attribute_information_gain.get)
    
    def id3_algorithm(self, training_set, attributes):
        # base/edge cases
        data_labels = []
        for data in training_set:
            data_labels.append(data["label"]) 
        if data_labels.count(data_labels[0]) == len(data_labels):
            return Node("", data_labels[0], {})
        
        # If attributes is empty, return a leaf node with the most common label
        data_label_count = {}
        if len(attributes) == 0:
            for label in self.labels:
                data_label_count[label] = data_labels.count(label)
            #just return most common
            return Node("", max(data_label_count, key=data_label_count.get), {})
                 
        # otherwise create a root node for decision tree
        root = Node()

        # A = aattribute in Attributes that best split S
        root.attribute = self.best_attribute(training_set,attributes)

        # 
        for value in self.attribute_values[root.attribute]:
            data_best_attribute_value = list(filter(lambda x: x[root.attribute] == value, training_set))

            # empty? just add a lead node with most common label 
            if len(data_best_attribute_value) == 0:
                data_labels = []
                for data in training_set:
                    data_labels.append(data["label"])

                data_labels_count = {}
                for label in self.labels:
                    data_labels_count[label] = data_labels.count(label)

                return Node(None, max(data_labels_count, key=data_labels_count.get), None)
            else:
                attr_copy = attributes.copy()
                attr_copy.remove(root.attribute)
                root.children[value] = self.id3_algorithm(data_best_attribute_value, attr_copy)

        return root
        
def main():
    data_desc_file = "data-desc.txt"
    training_file  = "train.csv"
    decision_tree  = DecisionTree(data_desc_file, training_file)
    print(decision_tree.training_set)
    root = decision_tree.id3_algorithm(decision_tree.training_set, decision_tree.attributes)
    #print(root)

if __name__ == "__main__":
    main()