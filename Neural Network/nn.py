import os, sys, random, math, decimal, copy


def read_file(csv):
    data = []
    with open (csv, 'r') as f:
        for line in f:
            data.append(list(map(float, line.strip().split(','))))
    
    return data

def sub(a, b):
    res = []

    for index in range(len(a)):
            res.append(a[index] - b[index])
    return res

def add(a, b):
    res = []

    for index in range(len(a)):
            res.append(a[index] + b[index])
    return res

def magnitude(a):
        squared_sum = decimal.Decimal(0)
        for val in a:
                val = decimal.Decimal(val)
                squared_sum += val ** 2
        return math.sqrt(squared_sum)

def scalar_multiplication(vec, s):
    return [s * index for index in vec]

#######################################
def dot(a, b):
    return sum([a * b for a,b in zip(a,b)])

def sigmoid(a):
    return 1/(1 + math.exp(-a))

def gaussian_dist(a):
    return (1 / math.sqrt(2 * math.pi)) * (math.e ** ( (-0.5) * a**2))

def forward_pass(weights, initial_values):
    width = 2
    z_values = []
    input_width = len(initial_values)
    z_values.append(initial_values)


    for layer in range(1,3):
        z_values.append([1]) #this represents the bias

        for node in range((input_width - 1)):
            #look at the previous layer
            prev_layer = layer - 1
            input = z_values[prev_layer]
            node_weight = []
            for wght in range(input_width):
                index = (prev_layer * width * input_width) + (width * wght) + node
                node_weight.append(weights[index][3])
            z_values[layer].append(sigmoid(dot(input, node_weight)))
    
    prev_layer = 2
    input = z_values[prev_layer]
    node_weight = []
    for wght in range(input_width):
        index = (prev_layer * width * input_width) + wght
        node_weight.append(weights[index][3])
    y = dot(input, node_weight)
    return (z_values, y)

def backpropagation(weights, z_values, y, label, input_width, width, part_a):
    #first calc the first partial deriviate
    d_ly = y - label
    partial_derivatives = []

    # we want to go backwards (hence *back* prop)

    for layer in reversed(range(1, 4)):
        #top layer
        if layer == 3: 
            for i in range(input_width):
                d_yw = z_values[2][i]
                partial_derivatives.append(d_ly * d_yw)
            #no need to check each one
            continue
        if layer == 2:
            for node in range(width):
                n_layer  = layer - 1
                for i in range(input_width):
                    index = (2 * width * input_width) + node + 1

                    d_yz = weights[index][3]
                    to = z_values[2][node + 1]
                    frm = z_values[n_layer][i]

                    d_zw = to * (1 - to) * (frm)
                    partial_derivatives.append(d_yz * d_ly * d_zw)
            continue

        #last layer
        if layer == 1: 
            for node in range(width):
                n_layer = layer - 1
                for i in range(input_width):
                    sum = 0

                    for path in range(width):
                        index = (2 * width * input_width) + path + 1
                        d_yz = weights[index][3]

                        index = (layer * width * input_width) + (width * (i + 1)) + node + path
                        d_zz_top = weights[index][3]
                        to = z_values[layer][node + 1]
                        frm = z_values[n_layer][i]
                        d_zw = to * (1 - to) * frm
                        sum += (d_ly * d_zz_top * d_yz * d_zw)
                    partial_derivatives.append(sum)
        if part_a == "part_a":
            print("Partial Derivative:", partial_derivatives)


    return partial_derivatives

def q3_backProp(part_a):
    weights = []    

    #data from HW
    # layer 1
    weights.append( [ 1, 0, 1, -1] )
    weights.append( [ 1, 0, 2, 1 ] )
    weights.append( [ 1, 1, 1, -2 ] )
    weights.append( [ 1, 1, 2, 2 ] )
    weights.append( [ 1, 2, 1, -3 ] )
    weights.append( [ 1, 2, 2, 3 ] )
    # layer 2
    weights.append( [ 2, 0, 1, -1] )
    weights.append( [ 2, 0, 2, 1 ] )
    weights.append( [ 2, 1, 1, -2 ] )
    weights.append( [ 2, 1, 2, 2 ] )
    weights.append( [ 2, 2, 1, -3 ] )
    weights.append( [ 2, 2, 2, 3 ] )
    # layer 3 - output
    weights.append( [ 3, 0, 1, -1 ] )
    weights.append( [ 3, 1, 1, 2 ] )
    weights.append( [ 3, 2, 1, -1.5 ] )

    init_val = [1, 1, 1]
    input_width = len(init_val)
    z_values, y = forward_pass(weights,init_val )
    return backpropagation(weights, z_values, y, 1, input_width, 2, part_a)

def neural_net(data, weights, lr, t, a, width, input_width, part_a):
    weight_value = 3

    for epoch in range(t):
        #insure convergence
        r = lr / (1 + (lr * epoch / a))

        #next step is to shuffle the data
        random.shuffle(data)

        #track previous weights
        prev_weights = list(val[weight_value] for val in weights)

        for sample in copy.deepcopy(data):

            label = sample[-1]

            #add bias
            sample = [1] + sample[:-1]

            z_values, y = forward_pass(weights, sample)
            gradient = backpropagation(weights, z_values, y, label, input_width, width, "")

            #update weights
            weight_values = list(val[weight_value] for val in weights)
            updated = sub(weight_values, scalar_multiplication(gradient, r))
            for i in range(len(weights)):
                weights[i][weight_value] = updated[i]
            
        weight_delta = sub(prev_weights, list(val[weight_value] for val in weights))
        tol = 10e-3

        if magnitude(weight_delta) < tol:
            print("Converged at:", epoch)
            return weights
        
    return weights

def get_error(data, learned_weight):
    # Next pt
    error = 0
    #swap out all the 0's for -1's
    for d in data: 
        if d[-1] == 0:
            d[-1] = -1

    for d in copy.deepcopy(data): 
        gen_or_forg = d[-1]

        d = [1] + d[:-1]

        _, prediction = forward_pass(learned_weight, d)

        if gen_or_forg * prediction <= 0:
            error+=1
    return error/len(data)
   

def initialize_weights(num_of_weights, input_width, width, part_a):
    weights = []
    for i in range(num_of_weights):
        layer = int(i / (input_width * width))
        to = i % width + 1
        frm = int(i/2) % width
        if part_a == "part_c":
            weight_values = 0
        else:
            weight_values = gaussian_dist(random.uniform(0, 1))
        weights.append([layer, frm, to, weight_values])
    
    for i in range(input_width):
        layer = 3
        frm = i
        to = 1
        if part_a == "part_c":
            weight_values = 0
        else:   
            weight_values = gaussian_dist(random.uniform(0, 1))
        weights.append([layer, frm, to, weight_values])

    return weights

def main():

    part_a = "part_a" if len(sys.argv) <= 1 else sys.argv[1]
    #question part_a3 back propogation
    if part_a == "part_a":
        q3_backProp(part_a)
    else:
        #collecting our data
        train_file = os.path.join("bank-note","train.csv")
        test_file  = os.path.join("bank-note","test.csv")

        training_data = read_file(train_file)
        testing_data = read_file(test_file)

        #set initial values (lr, epoch, etc)
        lr = 0.001
        t = 100
        a = 0.001

        layers_count = 3

        for width in [5, 10, 25, 50, 100]:
            print("For width:", width)
            width = 2
            input_width = len(training_data[0])
            num_of_weights = input_width * width * (layers_count - 1)
            weights = initialize_weights(num_of_weights, input_width, width, part_a)
            learned_weight = neural_net(training_data, weights, lr, t, a, width, input_width, part_a)
            learned_w_val  = list(val[3] for val in learned_weight)

            print("Learned weight vector:", [round(num, 3) for num in learned_w_val])
            print("Training error:", get_error(training_data, learned_weight))
            print("Test error:", get_error(testing_data, learned_weight))




if __name__ == "__main__":
    main()