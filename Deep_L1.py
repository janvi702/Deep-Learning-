import numpy as np
X = np.array([
    [1,1],
    [3,3]
])
y = np.array([0,1])

weights = np.zeros(X.shape[1])
bias = 0
learning_rate = 1
epochs = 2
print("Initial weights : ",weights) 
print("Initial bias : ",bias)

for epoch in range(epochs):
    print("\n========Epoch", epoch + 1,"========")

    for i in range(len(X)):
        x = X[i]
        print(x)
        actual = y[i]
        print(y)

        output = np.dot(x, weights) + bias
        print("x",x)
        print("output",output)

        if output >= 0:
            prediction = 1
        else:
            prediction = 0
        error = actual - prediction

        weights = weights + learning_rate * error * x
        print("weight = ",weights)
        bias = bias + learning_rate * error
        print("Bias = ",bias)

        print("\nInput :",x)
        print("\nOutput :",output)
        print("Prediction :",prediction)
        print("Actual :",actual)
        print("Error :",error)
        print("Weights :",weights)
        print("Bias :",bias)

print("\nTraining Completed")
print("Final weights :",weights)
print("Final Bias :",bias)
