import math

def dot(a, b):
    return sum([a * b for a,b in zip(a,b)])

def sigmoid(a):
    return 1/(1 + math.exp(-a))

def forward_pass(weights, initial_values):
    width = 2
    z_values = []
    input_width = len(initial_values)
    z_values.append(initial_values)


    for layer in range(1,3):
        z_values.append([1]) #this represents the bias

        for node in range(width):
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
    print(y)

def main():
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

    z_values = forward_pass(weights, [1, 1, 1])

if __name__ == "__main__":
    main()