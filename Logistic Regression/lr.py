import os, sys, random, math, decimal, copy

def read_file(csv):
    data = []
    with open (csv, 'r') as f:
        for line in f:
            data.append(list(map(float, line.strip().split(','))))
    
    return data

def avg(data):
    sum = [0] * (len(data[0]) - 1)

    for i in range(len(data)):
        res = data[i][:-1]
        sum = add(sum, res)
    return scalar_multiplication(sum, 1/len(data))

def var(data):
    sum = [0] * ((len(data[0])) - 1)

    res = data[0][:-1]
    diff = sub(res, avg(data))

    for j in range(len(diff)):
        diff[j] = diff[j] ** 2
    sum = add(sum, res)
    return scalar_multiplication(sum, 1)

def map_est(x, y, w, m, var):
    x[-1] = 1

    frac = (math.e ** (-y * dot(w,x)))/ (1 + math.e ** (-y * dot(w, x)))
    s = scalar_multiplication(x, -y)
    gradient = add(scalar_multiplication(s, frac), scalar_multiplication(w, 2/sum(var)))
    return gradient

def ml_est(x, y, w, m):
    x[-1] = 1
    ml = math.log(1 + math.e  ** (-y * dot(w, x)))
    return scalar_multiplication([1] * len(x), ml)

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
    
def logistic_regression(data, w, a, c, learning_rate, t, est):
    # swap all 0's in the final colum for -1's 
    SIZE = len(data)
    for d in data: 
        if d[-1] == 0:
            d[-1] = -1


    for epoch in range(t):
        # update our learning rate, the whole point is that we update over each itr to get it converge
        rate_learned = learning_rate / (1 + (learning_rate * epoch / a))
        
        previous_weight = w # keep track so we can compare it to track delta_weight

        #Don’t forget to shuffle the training examples at the start of each epoch 
        random.shuffle(data)

        for d in copy.deepcopy(data): 
            w_0 = w[:-1] # w without the bias
            genuine_or_forged = d[-1] # capturing the last colum, per the data desc this is the genuine or forged value
            d[-1] = 1 # fold b into x
            if est == "map":
                proton_gradient = map_est(d, genuine_or_forged, w, len(data), var(data))
            else: 
                proton_gradient = ml_est(d, genuine_or_forged, w, len(data))
            
            prediction = dot(proton_gradient, d)  
            #just check if it's misclassified, and if it is we update the weight_vector
            if genuine_or_forged * prediction <= 1: 
                add_left_side = sub(w, scalar_multiplication((w_0 + [0]), rate_learned))
                add_right_side = scalar_multiplication(d, rate_learned * c * SIZE * genuine_or_forged)
                w = add(add_left_side, add_right_side)
            else:
                w[:-1] = scalar_multiplication(w[:-1], 1 - rate_learned) 
        # calculate delta weight
        delta_weight = sub(previous_weight, w)
        if magnitude(delta_weight) < 10e-1:
            print("converged at:", epoch)
            return w

    return w

def main():
    estimation = "map" if len(sys.argv) == 1 else sys.argv[1]

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
    c = 100/873
    learned_weight = logistic_regression(data_training, w, a, c, lr, t, estimation)
    print("Learned weight vector: ", [round(num, 3) for num in learned_weight])
    # get error percentage
    print("Test error    :", get_error(data_testing, learned_weight) )
    print("Training error:", get_error(data_training, learned_weight), "\n")

    

if __name__ == "__main__":
    main()