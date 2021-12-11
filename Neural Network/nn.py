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
    return (z_values, y)

def backpropagation(weights, z_values, y, label, input_width, width):
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
        print("partial der:", partial_derivatives)


    return 0

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

    init_val = [1, 1, 1]
    input_width = len(init_val)
    z_values, y = forward_pass(weights,init_val )
    backpropagation(weights, z_values, y, 1, input_width, 2)

if __name__ == "__main__":
    main()