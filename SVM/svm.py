import os, sys, random, math, decimal, copy
import numpy as np
import scipy.optimize as opt


def read_file(csv):
    data = []
    with open (csv, 'r') as f:
        for line in f:
            data.append(list(map(float, line.strip().split(','))))
    
    return data

def dot(a, b):
        res = 0
        for index in range(len(a)):
                res += a[index] * b[index]
        return res

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

def get_error(data, learned_weight):
    # Next pt
    error = 0
    #swap out all the 0's for -1's
    for d in data: 
        if d[-1] == 0:
            d[-1] = -1

    for d in copy.deepcopy(data): 
        gen_or_forg = d[-1]

        d[-1] = 1

        prediction = dot(learned_weight, d)

        if gen_or_forg * prediction <= 0:
            error+=1
    return error/len(data)
    
def svm(data, w, a, c, learning_rate, t, schedule):
    # swap all 0's in the final colum for -1's 
    SIZE = len(data)
    for d in data: 
        if d[-1] == 0:
            d[-1] = -1


    for epoch in range(t):
        # update our learning rate, the whole point is that we update over each itr to get it converge
        if schedule == 0:
            rate_learned = learning_rate / (1 + (learning_rate * epoch / a))
        else:
            rate_learned = learning_rate/(epoch + 1)
        previous_weight = w # keep track so we can compare it to track delta_weight

        #Don’t forget to shuffle the training examples at the start of each epoch 
        random.shuffle(data)

        for d in copy.deepcopy(data): 
            w_0 = w[:-1] # w without the bias
            genuine_or_forged = d[-1] # capturing the last colum, per the data desc this is the genuine or forged value
            d[-1] = 1 # fold b into x
            prediction = dot(w, d)  
            #just check if it's misclassified, and if it is we update the weight_vector
            if genuine_or_forged * prediction <= 1: 
                add_left_side = sub(w, scalar_multiplication((w_0 + [0]), rate_learned))
                add_right_side = scalar_multiplication(d, rate_learned * c * SIZE * genuine_or_forged)
                w = add(add_left_side, add_right_side)
            else:
                w[:-1] = scalar_multiplication(w[:-1], 1 - rate_learned) 
        # calculate delta weight
        delta_weight = sub(previous_weight, w)
        if magnitude(delta_weight) < 10e-3:
            print("converged at:", epoch)
            return w

    return w

def quad_convex_equation(alpha, *args):
    x = args[0]
    y = args[1]

    xx = (x*x.T)
    yy = (y*y.T)

    yyaaxx = alpha.T.dot((xx*yy)[0, 0] * alpha)
    
    return (0.5) *  yyaaxx - np.sum(alpha)

def constraint(alpha, *args):
    y = args[1]
    return np.sum(alpha * y)

def dual_svm(data, w, a, c, learning_rate, t, schedule):

    SIZE = len(data)
    x = np.matrix([data[d][:-1] for d in range(SIZE)])
    y = np.matrix([data[d][-1]  for d in range(SIZE)]).T
    y = np.where(y == 0, -1, y) # condition, true, else
    
    init_guess = np.random.rand(SIZE)
    args = (x, y)
    method = "SLSQP"
    bounds = [(0, c)] * SIZE # we want to define our bounds from 0 -> c for every size
    constraints = [{'type' : 'eq', 'fun': constraint, 'args':args}]

    optimal_alpha = opt.minimize(quad_convex_equation, init_guess, args, method, bounds = bounds, constraints = constraints)
    print(optimal_alpha.x)
    return w



def main():

    # SETUP RUNTIME ARGS
    c = float(100/873) if len(sys.argv) == 1 else float(sys.argv[1])
    schedule = 0 if len(sys.argv) <= 2 else float(sys.argv[2])
    domain = "primal" if len(sys.argv) <= 3 else "dual"

    #################################################################

    #collecting our data
    train_file = os.path.join("bank-note","train.csv")
    test_file  = os.path.join("bank-note","test.csv")

    data_training = read_file(train_file)
    data_testing  = read_file(test_file)

    w  = [0] * len(data_training[0]) # initializt an array of length data_training full of just 0's
    lr = 0.0001 # learning rate
    t  = 100 # t, although this is a bit confusing maybe I should name this something more pacific
    a  = 0.0001

    if domain == "primal":
        learned_weight = svm(data_training, w, a, c, lr, t, schedule)
        print("Learned weight vector: ", [round(num, 3) for num in learned_weight])
        # get error percentage
        print("Test error    :", get_error(data_testing, learned_weight) )
        print("Training error:", get_error(data_training, learned_weight), "\n")
    else:
        learned_weight = dual_svm(data_training, w, a, c, lr, t, schedule)
        sys.exit(0)
    


    #################################################################
    #                       QUESTION 3                              #
    #################################################################

    
    

if __name__ == "__main__":
    main()