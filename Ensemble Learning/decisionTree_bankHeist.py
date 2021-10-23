"""
Jason Zavala | Shandian Zhe
09/23/2021
Machine Learning | CS5350
Univeristy of Utah
Fall 2021
"""
from collections import Counter
import math, re, sys, os, random
import statistics as st
# added this here for testing

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
    weight = []

    def __init__(self, training_file, information_gain_method, maximus_deptheus, ensemble) -> None:

        self.information_gain_method = information_gain_method
        self.max_depth   = maximus_deptheus
        self.forest_size = maximus_deptheus
        self.ensemble = ensemble
        # manually set because this is already 2 days late
        self.labels = ["yes", "no"]
        self.attributes = ["age", "job", "marital", "education", "default", "balance",\
           "housing", "loan", "contact", "day", "month", "duration", "campaign",\
           "pdays", "previous", "poutcome"] 

        self.attribute_values["age"] = ["numeric"]
        self.attribute_values["job"] = ["admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student", "blue-collar", "self-employed", "retired", "technician", "services"]
        self.attribute_values["marital"] = ["married", "divorced", "single"]
        self.attribute_values["education"] = ["unknown", "secondary", "primary", "tertiary"]
        self.attribute_values["default"] = ["yes", "no"]
        self.attribute_values["balance"] = ["numeric"]
        self.attribute_values["housing"] = ["yes", "no"]
        self.attribute_values["loan"] = ["yes", "no"]
        self.attribute_values["contact"] = ["unknown", "telephone", "cellular"]
        self.attribute_values["day"] = ["numeric"]
        self.attribute_values["month"] = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
        self.attribute_values["duration"] = ["numeric"]
        self.attribute_values["campaign"] = ["numeric"]
        self.attribute_values["pdays"] = ["numeric"]
        self.attribute_values["previous"] = ["numeric"]
        self.attribute_values["poutcome"] = ["unknown", "other", "failure", "success"]

        # now we worry about the actual CSVfile
        self.training_set = read_data(training_file, self.attributes)
        w = 1/len(self.training_set)
        
        #fill our arr with appropriate weights
        for _ in range(len(self.training_set)):
            self.weight.append(w)
        """
        most_common_value = {}

        for attribute in self.attributes:
            value_counts = {}
            for value in self.attribute_values[attribute]:
                value_counts[value] = len(list(filter(lambda x: x[attribute] == value, self.training_set)))
            value_counts.pop("unknown", None)
            most_common_value[attribute] = max(value_counts)
        
        # replace unknowns
        for data in self.training_set:
            for attribute in self.attributes:
                if data[attribute] == "unknown":
                    data[attribute] = most_common_value[attribute] 

                    """
    
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

    # gini index
    def calculate_gini_index(self, data):
        size = len(data)

        if size == 0:
            return 0
        
        label_p = []
        
        
        for label in self.labels:
            weighted_sum = 0
            for index, data in enumerate(self.training_set):
                if data["label"] == label:
                    weighted_sum += self.weight[index]
            label_p.append(weighted_sum)
            #label_p.append(sum(split["label"] == label for split in data) / len(data))
        gini = 0

        for p in label_p:
            gini += p**2
        
        return 1 - gini

    def calculate_majority_error(self, data):
        if len(data) == 0:
            return 0
        label_p = []

        for label in self.labels:
            label_p.append(sum(split["label"] == label for split in data) / len(data))
        
        return min(label_p)

    # this is where the magic happens
    def calculate_information_gain(self, data):

        if self.information_gain_method == "entropy": 
            return self.calculate_entropy(data)
        elif self.information_gain_method == "gini":
            return self.calculate_gini_index(data)
        return self.calculate_majority_error(data)
    
    def best_attribute(self, data, attributes):
        current_entropy = self.calculate_information_gain(data)
        # for each attribute in our attributes list, calc the entropy & information gain for each set of values in our list
        data_size = len(data)
        attribute_information_gain = {} 

        for attribute in attributes:
            
            expected_attribute_entropy = []

            if self.attribute_values[attribute] == ["numeric"]:
                num_val = []
                for d in data:
                    num_val.append(int(d[attribute]))
                median = st.median(num_val)

                #filter on less than median
                data_attribute_values = list(filter(lambda x: int(x[attribute]) < median, data))
                attribute_value_p = len(data_attribute_values)/data_size
                expected_attribute_entropy.append(self.calculate_information_gain(data_attribute_values) * attribute_value_p)

                #filter on >= median. basically we're turning the number into yes or no's
                data_attribute_values = list(filter(lambda x: int(x[attribute]) >= median, data))
                attribute_value_p = len(data_attribute_values)/data_size
                expected_attribute_entropy.append(self.calculate_information_gain(data_attribute_values) * attribute_value_p)

                expected_attribute_entropy = sum(expected_attribute_entropy)
                attribute_information_gain[attribute] = current_entropy - expected_attribute_entropy
            else:
                for value in self.attribute_values[attribute]:
                    data_attribute_values = list(filter(lambda x: x[attribute] == value, data))
                    attribute_value_p = len(data_attribute_values)/data_size
                    expected_attribute_entropy.append(self.calculate_information_gain(data_attribute_values) * attribute_value_p)
                expected_attribute_entropy = sum(expected_attribute_entropy)
                attribute_information_gain[attribute] = current_entropy - expected_attribute_entropy

        return max(attribute_information_gain, key=attribute_information_gain.get)
    
    def calculate_weight(self, total_error):
        total_error_remaining = 1 - total_error
        return (1/2) * math.log(total_error_remaining/total_error)

    def calculate_new_weight(self, stump, scale):
        total_weight = 0
        for index, data in enumerate(self.training_set):
            current_node = stump
            decision = current_node.attribute
            decision_value = data[decision]

            if self.attribute_values[decision] == ["numeric"]:
                num_median = int(list(current_node.children.keys())[0][2:])

                if int(decision_value) < num_median:
                    current_node = current_node.children["< " + str(num_median)]
                else:
                    current_node = current_node.children[">= " + str(num_median)]
            else:
                current_node = current_node.children[decision_value]
            
            label_prediction = current_node.label

            if data["label"] == label_prediction:
                self.weight[index] *= math.exp(-scale)
            else:
                self.weight[index] *= math.exp(scale)
            total_weight += self.weight[index]

        # last step is to normalize

        for index, _ in enumerate(self.weight):
            self.weight[index] = self.weight[index]/total_weight
        return total_weight

    def calculate_error(self, stump):
        total_error = 0
        for index, data in enumerate(self.training_set):
            current_node = stump
            decision = current_node.attribute
            decision_value = data[decision]

            if self.attribute_values[decision] == ["numeric"]:
                num_median = int(list(current_node.children.keys())[0][2:])

                if int(decision_value) < num_median:
                    current_node = current_node.children["< " + str(num_median)]
                else:
                    current_node = current_node.children[">= " + str(num_median)]
            else:
                current_node = current_node.children[decision_value]
            
            label_prediction = current_node.label

            if data["label"] != label_prediction:
                total_error += self.weight[index]

        return total_error

    def adaboost(self):       
        # stumpify
        forest_gump = []
        

        for _ in range(self.forest_size):
            data = []
            m = 1000
            if self.ensemble == "bag":
                for _ in range(m):
                    random_index = random.randrange(len(self.training_set))
                    data.append(self.training_set[random_index])
            else: 
                data = self.training_set
            
            stump = self.id3_algorithm(data, self.attributes)
            
            stump_error = self.calculate_error(stump)
            
            new_weight = self.calculate_weight(stump_error)
            self.calculate_new_weight(stump, new_weight)
            forest_gump.append((stump, self.weight))
        return forest_gump
    
    def id3_algorithm(self, training_set, attributes, depth=0):
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

        #this is where the level magic happens
        if depth >= self.max_depth:
            data_label_count = {}
            for label in self.labels:
                data_label_count[label] = data_labels.count(label)

            root.label = max(data_label_count, key = data_label_count.get)
            return root

        # A = aattribute in Attributes that best split S
        random_attributes = []
        if self.ensemble == "random":
            for _ in range(int(len(attributes)/2)):
                random_i = random.randrange(1,len(attributes))
                random_attributes.append(attributes[random_i])
        else:
            random_attributes = attributes

        root.attribute = self.best_attribute(training_set,random_attributes)

        if self.attribute_values[root.attribute] == ["numeric"] :
            num_val = []
            for data in training_set:
                num_val.append(int(data[root.attribute]))
            median = int(st.median(num_val))
            
            
            data_best_attribute_value = list(filter(lambda x: int(x[root.attribute]) < median, training_set))

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
                attr_copy = random_attributes.copy()
                attr_copy.remove(root.attribute)
                root.children["< " + str(median)] = self.id3_algorithm(data_best_attribute_value, attr_copy, depth + 1)


            # same thing but for greater than
            data_best_attribute_value = list(filter(lambda x: int(x[root.attribute]) >= median, training_set))

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
                attr_copy = random_attributes.copy()
                attr_copy.remove(root.attribute)
                root.children[">= " + str(median)] = self.id3_algorithm(data_best_attribute_value, attr_copy, depth + 1)

        else:
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
                    attr_copy = random_attributes.copy()
                    attr_copy.remove(root.attribute)
                    root.children[value] = self.id3_algorithm(data_best_attribute_value, attr_copy, depth + 1)

        return root
    
    def adaboost_prediction(self, forest, data):
        label_predictions = {}
        for label in self.labels:
            label_predictions[label] = 0
        
        for index, stump in enumerate(forest):
            current_node = stump[0]
            decision = current_node.attribute
            decision_value = data[decision]

            if self.attribute_values[decision] == ["numeric"]:
                num_median = int(list(current_node.children.keys())[0][2:])

                if int(decision_value) < num_median:
                    current_node = current_node.children["< " + str(num_median)]
                else:
                    current_node = current_node.children[">= " + str(num_median)]
            else:
                current_node = current_node.children[decision_value]
            
            label_prediction = current_node.label 
            weight = stump[1]
            label_predictions[label_prediction] += weight[index]

        label = ""
        max_sum = 0 
        for l, sum in label_predictions.items():
            if (sum > max_sum):
                max_sum = sum
                label = l

        return label


def read_data(csv, attributes):

    data = []
    with open(csv, 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            val   = {}

            for index, attribute in enumerate(attributes):
                val[attribute] = terms[index]
                # get the last col
            val["label"] = terms[index + 1]
            data.append(val)
    f.close()
    return data
    
def calculate_error_percentage(dataset, forest, decision_tree):
    predictions_errors = 0 
    for data in dataset:
        label_prediction = decision_tree.adaboost_prediction(forest, data)

        #keep track of errors
        if data["label"] != label_prediction:
            predictions_errors += 1
    error_percentage = predictions_errors/len(dataset)
    return error_percentage

def main():
    #data_desc_file = os.path.join("bank", "data-desc.txt")
    training_file = os.path.join("bank", "train.csv")
    test_file = os.path.join("bank", "test.csv")

    purity         = sys.argv[1]
    max_depth      = int(sys.argv[2])
    ensemble_learning = sys.argv[3]

    for i in range(100):
        decision_tree  = DecisionTree(training_file,purity, i + 1, ensemble_learning)

        if purity == "ada":
            #decision_tree.forest_size = 1
            decision_tree.max_depth = 1
            
            decision_tree.information_gain_method = "gini"    
            forest = decision_tree.adaboost()
            training_error_percentage = calculate_error_percentage(decision_tree.training_set, forest, decision_tree)
            test_error_percentage = calculate_error_percentage(read_data(test_file, decision_tree.attributes), forest, decision_tree)
            print(str(i) + "," + str(training_error_percentage) + "," + str(test_error_percentage))
    
        else: 
            root = decision_tree.id3_algorithm(decision_tree.training_set, decision_tree.attributes, 0)
            print(root)
    

        
if __name__ == "__main__":
    main()