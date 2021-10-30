import math, os, decimal,random


def dot(a, b):
        res = 0
        for index in range(len(a)):
                res += a[index] * b[index]
        return res

def subtract(a, b):
        res = []

        for index in range(len(a)):
                res.append(a[index] - b[index])
        return res

def magnitude(a):
        squared_sum = decimal.Decimal(0)
        for val in a:
                val = decimal.Decimal(val)
                squared_sum += val ** 2
        return math.sqrt(squared_sum)

def gradient_descent(data, w, b , lr):
        fsize = len(data[0]) - 1
        max_steps = 10000
        


        for step in range(max_steps):
                random_index = random.randrange(len(data))
                random_data_sample_uwu = [data[random_index]]
                gradient = [0] * fsize
                for j in range(fsize):
                        for x in random_data_sample_uwu:
                                y = x[-1]
                                wx = dot(w,x)
                                gradient[j] += -(y - wx -b) * x[j]
                b_slope = 0
                for x in random_data_sample_uwu:
                        y = x[-1]
                        wx = dot(w, x)
                        b_slope += -(y - wx -b)

                
                ss = [lr * g for g in gradient]
                prev_weight = w 
                w = subtract(w, ss)
                b -= b_slope * lr


                cost_function = decimal.Decimal(0)
                for x in data:
                        y = x[-1]
                        wx = dot(w, x)
                        cost_function += decimal.Decimal((y - wx -b)) ** 2

                cost_function /= 2
                #f = open("alt_j_cost_function.csv", "a")
                #f.write(str(cost_function) + "\n")
                #f.close()

                #print(w, b)
                weight_delta = subtract(prev_weight, w)
                threshold = 10e-6

                if magnitude(weight_delta) < threshold:
                        return w
                
        return step

       
def read_data(csv):
        data = []
        with open(csv, 'r') as f:
                for line in f:
                        val = list(map(float, line.strip().split(',')))
                        data.append(val)
        f.close()
        return data

def main():
        #sample_file = os.path.join("concrete","sample.csv")
        train_file = os.path.join("concrete","train.csv")
        test_file = os.path.join("concrete","test.csv")

        data = read_data(train_file)
        print(data)
        w = [0] * 7
        b = 0
        lr = 0.003125

        weight = gradient_descent(data, w, b, lr)
        print(weight)

        weight = 9999
        while weight == 9999:
                print(lr)
                weight = gradient_descent(data, w, b, lr)
                lr /=2
     

        weight = [-0.005295317011470618, 0.004810033398940123, -0.02090702863439017, 0.10627116810099517, -0.00822561103711401, -0.032258014129808533, -0.01217284177512961]
        cost_function = 0
        test_data = read_data(test_file)
        for x in test_data:
                y = x[-1]
                wx = dot(weight, x)
                cost_function += (y - wx -b) ** 2
        cost_function /= 2
        print(cost_function)
        

if __name__ == "__main__":
    main()