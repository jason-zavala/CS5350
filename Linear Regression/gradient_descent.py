import math, os


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
        squared_sum = 0
        for val in a:
                squared_sum += val ** 2
        return math.sqrt(squared_sum)

def gradient_descent(data, w, b , lr):
        fsize = len(data[0]) - 1
        max_steps = 100

        for step in range(max_steps):
                gradient = [0] * fsize
                for j in range(fsize):
                        for x in data:
                                y = x[-1]
                                wx = dot(w,x)
                                gradient[j] += -(y - wx -b) * x[j]
                b_slope = 0
                for x in data:
                        y = x[-1]
                        wx = dot(w, x)
                        b_slope += -(y - wx -b)
                
                ss = [lr * g for g in gradient]
                w = subtract(w, ss)
                b -= b_slope * lr
                print(w, b)

                if magnitude(ss) < lr:
                        print(step)
                        break
       
def read_data(csv):
        data = []
        with open(csv, 'r') as f:
                for line in f:
                        val = list(map(int, line.strip().split(',')))
                        data.append(val)
        f.close()
        return data

def main():
        sample_file = os.path.join("concrete","sample.csv")
        train_file = os.path.join("concrete","train.csv")
        test_file = os.path.join("concrete","test.csv")

        data = read_data(sample_file)

        w = [-1, 1, -1]
        b = -1
        lr = 0.01

        gradient_descent(data, w, b, lr)

if __name__ == "__main__":
    main()