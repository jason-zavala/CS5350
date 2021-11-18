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

def scalar_multiplication(s, vec):
    for i in range(len(vec)):
            vec[i] = s * vec[i]
    return vec

def svm(data, w, a, c, learning_rate, epoch):
    print("Running SVM")
    # swap all 0's in the final colum for -1's 
    for d in data: 
        if d[-1] == 0:
            d[-1] = -1

    w_0 = w[:-1] # w without the bias

    for _ in range(epoch):
        random.shuffle(data)
        for d in data: 
            genuine_or_forged = d[-1] # capturing the last colum, per the data desc this is the genuine or forged value
            d[-1] = 1 # fold b into x
            prediction = dot(w, d)  

            #just check if it's misclassified, and if it is we update the weight_vector
            if genuine_or_forged * prediction <= 1: 
                w = add(sub(w, scalar_multiplication(learning_rate, (w_0 + [0]))), scalar_multiplication(learning_rate*c*len(data)*genuine_or_forged, d))
            else:
                w[:-1] = scalar_multiplication(1- learning_rate, w[:-1]) 

    return w

def main():

    train_file = os.path.join("bank-note","train.csv")
    test_file = os.path.join("bank-note","test.csv")

    data_training = read_file(train_file)
    data_testing = read_file(test_file)
    
    w = [0] * len(data_training[0]) # fold b into w 
    lr = 0.1
    epoch = 10 # t
    c = 100/873 #hyper parameter
    a = 1 

    learned_weight = svm(data_testing, w, a, c, lr, epoch)
    print("Learning weight vector: ", [round(num, 3) for num in learned_weight] , "\n")

    # # Next pt
    # error = 0
    # #swap out all the 0's for -1's
    # for d in data_testing: 
    #     if d[-1] == 0:
    #         d[-1] = -1

    # for d in data_testing: 
    #     gen_or_forg = d[-1]

    #     d[-1] = 1

    #     prediction = dot(learned_weight, d)

    #     if gen_or_forg * prediction <= 0:
    #         error+=1

    # print("Average prediction error:", round(error/len(data_testing) * 100), "%\n")
    

if __name__ == "__main__":
    main()