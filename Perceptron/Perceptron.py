import os


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

    # swap all 0's in the final colum for -1's 
    for d in data: 
        if d[-1] == 0:
            d[-1] = -1
    
    for _ in range(epoch):

        for d in data: 
            genuine_or_forged = d[-1] # capturing the last colum, per the data desc this is the genuine or forged value
            d[-1] = 1 # fold b into x
            prediction = dot(w, d)  

            if genuine_or_forged * prediction <= 0: 
                w = add(w, scalar_multiplication(learning_rate * genuine_or_forged, d))
    return w

def main():
    train_file = os.path.join("bank-note","train.csv")
    test_file = os.path.join("bank-note","test.csv")

    data_training = read_file(train_file)
    data_testing = read_file(test_file)

    fsize = len(data_training[0])

    w = [0] * fsize
    lr = 0.1
    epoch = 10

    optimal_weight = perceptron(data_training, w, lr, epoch)
    print(optimal_weight)


if __name__ == "__main__":
    main()