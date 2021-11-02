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

def perceptron(data, w, learning_rate, epoch):

    # swap all 0's in the final colum for -1's 
    for d in data: 
        if d[-1] == 0:
            d[-1] = -1
    
    for i in range(epoch):

        for d in data: 
            genuine_or_forged = d[-1] # capturing the last colum, per the data desc this is the genuine or forged value
            d[-1] = 1 # fold b into x
            prediction = dot(w, d)  
            print(prediction)
            print(genuine_or_forged)

            if genuine_or_forged * prediction <= 0: 
                print("nm")
            else:
                print("m")

def main():
    train_file = os.path.join("bank-note","train.csv")
    test_file = os.path.join("bank-note","test.csv")

    data_training = read_file(train_file)
    data_testing = read_file(test_file)

    fsize = len(data_training[0]) - 1

    w = [-1] * fsize
    lr = 1
    epoch = 10

    optimal_weight = perceptron(data_testing, w, lr, epoch)
    print()


if __name__ == "__main__":
    main()