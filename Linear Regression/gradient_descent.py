import math



def subtract(a, b):
        res = []

        for index in range(len(a)):
                res.append[a[index] - b[index]]
        return res

def less_than(a, b):
        for val in a:
                if abs(val) >= x:
                        return False

def magnitude(a):
        squared_sum = 0
        for val in a:
                squared_sum += val ** 2
        return math.sqrt(squared_sum)

def gradient_descent(data, w, b , lr):
        print()
        
# Gradient Descent
data = [[1, -1, 2],
        [1, 1, 3],
        [-1, 1, 0],
        [1, 2, -4],
        [3, -1, -1]]
        
y = [1, 4, -1, -2, 0]

w = [-1, 1, -1]
b = -1
lr = 0.01

ss = [0, 0, 0]
max_steps = 100
for step in range(0, max_steps):
        gradient = [0, 0, 0]
        b_slope = 0
        for j in range(0, len(data[0])):
                for i, x in enumerate(data):
                        gradient[j] += -(y[i] - (w[0]*x[0] + w[1]*x[1] + w[2]*x[2]) - b) * x[j]
        for i, x in enumerate(data):
                b_slope += -(y[i] - (w[0]*x[0] + w[1]*x[1] + w[2]*x[2]) - b)
        #print(gradient)
        #print(b_slope)

        # new weight & bias
        ss = [lr * g for g in gradient]
        w = [w[0] - ss[0], w[1] - ss[1], w[2] - ss[2]]
        b -= b_slope * lr
        print(w, b)
        
        #print(ss)
        if abs(ss[0]) < lr and abs(ss[1]) < lr and abs(ss[2]) < lr:
                print(step)
                break
