import os, sys, random


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

def add(a, b):
        res = []

        for index in range(len(a)):
                res.append(a[index] + b[index])
        return res

def scalar_multiplication(s, vec):
    for i in range(len(vec)):
            vec[i] = s * vec[i]
    return vec

def perceptron(data, w, learning_rate, epoch):
    print("running standard...")
    # swap all 0's in the final colum for -1's 
    for d in data: 
        if d[-1] == 0:
            d[-1] = -1
    
    for _ in range(epoch):
        random.shuffle(data)
        for d in data: 
            genuine_or_forged = d[-1] # capturing the last colum, per the data desc this is the genuine or forged value
            d[-1] = 1 # fold b into x
            prediction = dot(w, d)  

            #just check if it's misclassified, and if it is we update the weight_vector
            if genuine_or_forged * prediction <= 0: 
                w = add(w, scalar_multiplication(learning_rate * genuine_or_forged, d))
    return w

def perceptron_voted(data, w, learning_rate, epoch):
    print("running voted...")
    m = 0
    C_m = 0
    weights = []
    weights.append((w, C_m))
    


    # swap all 0's in the final colum for -1's 
    for d in data: 
        if d[-1] == 0:
            d[-1] = -1
    
    for _ in range(epoch):
        random.shuffle(data)
        for d in data: 
            genuine_or_forged = d[-1] # capturing the last colum, per the data desc this is the genuine or forged value
            d[-1] = 1 # fold b into x
            prediction = dot(weights[m][0], d)  

            
            if genuine_or_forged * prediction <= 0: 

                if len(weights) == 1 : 
                    weights[0] =  (w, C_m)

                weights.append((add(weights[m][0], scalar_multiplication(learning_rate * genuine_or_forged, d)), C_m))
                m+=1 # incr out counter
                C_m = 1 # reset C-m
            else: 
                #C_m ++
                C_m += 1
    return weights

def perceptron_average(data, w, learning_rate, epoch):
    print("running average...")
    a = w
    # swap all 0's in the final colum for -1's 
    for d in data: 
        if d[-1] == 0:
            d[-1] = -1
    
    for _ in range(epoch):
        random.shuffle(data)
        for d in data: 
            genuine_or_forged = d[-1] # capturing the last colum, per the data desc this is the genuine or forged value
            d[-1] = 1 # fold b into x
            prediction = dot(w, d)  

            if genuine_or_forged * prediction <= 0: 
                w = add(w, scalar_multiplication(learning_rate * genuine_or_forged, d))
            a = add(a, w)

    return a

def main():
    perceptron_method = "standard" if len(sys.argv) == 1 else sys.argv[1]

    train_file = os.path.join("bank-note","train.csv")
    test_file = os.path.join("bank-note","test.csv")

    data_training = read_file(train_file)
    data_testing = read_file(test_file)
    
    w = [0] * len(data_training[0]) # fold b into w 
    lr = 0.1
    epoch = 10 # t

    if perceptron_method == "standard":
        #HW 2a: 
        learned_weight = perceptron(data_testing, w, lr, epoch)
        print("Learning weight vector: ", [round(num, 3) for num in learned_weight] , "\n")

        # Next pt
        error = 0
        #swap out all the 0's for -1's
        for d in data_testing: 
            if d[-1] == 0:
                d[-1] = -1

        for d in data_testing: 
            gen_or_forg = d[-1]

            d[-1] = 1

            prediction = dot(learned_weight, d)

            if gen_or_forg * prediction <= 0:
                error+=1

        print("Average prediction error:", round(error/len(data_testing) * 100), "%\n")
    
    elif perceptron_method == "voted":
        #Voted method for HW: 
        learned_weight = perceptron_voted(data_testing, w, lr, epoch)
        for i, weight in enumerate(learned_weight):
            # this was left here as documentation of how I logged it. For grading porpoises I left it in and will just print the learning weight vectors 

            #f = open("voted_weights.txt", "a")
            #f.write("Iteration: " + str( 1 + i) + "\n")
            #f.write("Weight vectors: " + str([round(num, 3) for num in weight[0]]) + "\n")
            #f.write("C_m: " + str(weight[1]) +  "\n\n")
            #f.close()

            #f = open("voted_weights.txt", "a")
            #f.write("\item Weight vectors: " + str([round(num, 3) for num in weight[0]]) + "\n")
            #f.write("C_m: " + str(weight[1]) +  "\n\n")
            #f.close()

            print("Learning weight vector: ", [round(num, 3) for num in weight[0]])
            print("C_M value:", weight[1], "\n")

        # Next pt
        error = 0
        #swap out all the 0's for -1's
        for d in data_testing: 
            if d[-1] == 0:
                d[-1] = -1

        for d in data_testing: 
            gen_or_forg = d[-1]

            d[-1] = 1
            final_prediction = 0
            for weight in learned_weight: 
                prediction = dot(weight[0], d)
                sgn = 1 if prediction > 0 else -1
                final_prediction += sgn * weight[1] # note this is the scaled weight

            if gen_or_forg * final_prediction <= 0:
                error+=1

        print("Average prediction error:", round(error/len(data_testing) * 100), "%\n")

    elif perceptron_method == "average":
     #HW 2a: 
        
        learned_weight = perceptron_average(data_testing, w, lr, epoch)
        print("Learning weight vector: ", [round(num, 3) for num in learned_weight] , "\n")

        # Next pt
        error = 0
        #swap out all the 0's for -1's
        for d in data_testing: 
            if d[-1] == 0:
                d[-1] = -1

        for d in data_testing: 
            gen_or_forg = d[-1]

            d[-1] = 1

            prediction = dot(learned_weight, d)

            if gen_or_forg * prediction <= 0:
                error+=1

        print("Average prediction error:", round(error/len(data_testing) * 100), "%\n")
    else:
        print("invalid command")
        sys.exit(0)

if __name__ == "__main__":
    main()