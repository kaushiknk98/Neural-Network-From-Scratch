import numpy as np
import pandas as pd


class NeuralNet:
    def __init__(self, train, header = True, h1 = 4, h2 = 4):
        np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h1 and h2 represent the number of nodes in 1st and 2nd hidden layers
        """columns = "sepal length,sepal width,petal length, petal width,class".split(",") 
        raw_input = pd.read_csv(train, names="columns")"""
        # TODO: Remember to implement the preprocess method
        train_dataset = self.preprocess(train)
        ncols = len(train_dataset.columns)
        nrows = 120
        self.X = train_dataset.iloc[:, 0:(ncols-3)].values.reshape(nrows, ncols-3)
        self.y = train_dataset.iloc[:, (ncols-3):].values.reshape(nrows, 3)
        #
        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(self.y[0])

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X[0:4]
        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X), h2))
        self.delta23 = np.zeros((h2, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))
    #
    # TODO: I have coded the sigmoid activation function, you need to do the same for tanh and ReLu
    #

    def __activation(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            self.__sigmoid(self, x)
        elif activation == "tanh":
            self.__tanh(self,x)
        elif activation == "relu":
            self.__relu(self,x)

    #
    # TODO: Define the function for tanh, ReLu and their derivatives
    #

    def __activation_derivative(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            self.__sigmoid_derivative(self, x)
        if activation == "tanh":
            self.__tanh_derivative(self, x)
        if activation == "relu":
            self.__relu_derivative(self, x)        

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # derivative of sigmoid function, indicates confidence about existing weight

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def __tanh(self,x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    def __tanh_derivative(self,x):
        return 1-(x**2)
    def __relu(self,x):
        return np.maximum(0,x)
    def __relu_derivative(self,x):
        if x>0:
            return 1
        else :
            return 0
    #
    # TODO: Write code for pre-processing the dataset, which would include standardization, normalization,
    #   categorical to numerical, etc
    #
    
    def preprocess(self, X):
        X=pd.get_dummies(X);
        print(X.shape)
        return X

    # Below is the training function

    def train(self, max_iterations = 1000, learning_rate = 0.05):
        for iteration in range(max_iterations):
            out = self.forward_pass()
            error = 0.5 * np.power((out - self.y), 2)
            self.backward_pass(out, activation="sigmoid")
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)

            self.w23 += update_layer2
            self.w12 += update_layer1
            self.w01 += update_input

        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))
        print("The final weight vectors are (starting from input to output layers)")
        print(self.w01)
        print(self.w12)
        print(self.w23)

    def forward_pass(self):
        # pass our inputs through our neural network
        in1 = np.dot(self.X, self.w01 )
        self.X12 = self.__sigmoid(in1)
        in2 = np.dot(self.X12, self.w12)
        self.X23 = self.__sigmoid(in2)
        in3 = np.dot(self.X23, self.w23)
        out = self.__sigmoid(in3)
        return out



    def backward_pass(self, out, activation):
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation)
        self.compute_hidden_layer2_delta(activation)
        self.compute_hidden_layer1_delta(activation)

    # TODO: Implement other activation functions

    def compute_output_delta(self, out, activation="sigmoid"):
        if activation == "sigmoid":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))
        if activation == "tanh":
            delta_output = (self.y - out) * (self.__tanh_derivative(out))
        if activation == "relu":
            delta_output = (self.y - out) * (self.__relu_derivative(out))
        self.deltaOut = delta_output

    # TODO: Implement other activation functions

    def compute_hidden_layer2_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__sigmoid_derivative(self.X23))
        if activation == "tanh":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__tanh_derivative(self.X23))
        if activation == "relu":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__relu_derivative(self.X23))
        self.delta23 = delta_hidden_layer2

    # TODO: Implement other activation functions

    def compute_hidden_layer1_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__sigmoid_derivative(self.X12))
        if activation == "tanh":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__tanh_derivative(self.X12))
        if activation == "relu":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__relu_derivative(self.X12))

            self.delta12 = delta_hidden_layer1


    # TODO: Implement the predict function for applying the trained model on the  test dataset.
    # You can assume that the test dataset has the same format as the training dataset
    # You have to output the test error from this function

    def predict(self, test, header = True):
        """columns = "sepal length,sepal width,petal length, petal width,class".split(",") """
        df=pd.read_csv(test)
        df=self.preprocess(df)
        ncols = len(df.columns)
        nrows = 30
        self.X = df.iloc[:, 0:(ncols-3)].values.reshape(nrows, ncols-3)
        self.y = df.iloc[:, (ncols-3):].values.reshape(nrows, 3)
        ct=0
        print(self.y)
        y1=self.forward_pass()
        return (0.5 * np.power((y1 - self.y), 2))


if __name__ == "__main__":
    columns = "sepal length,sepal width,petal length, petal width,class".split(",") 
    raw_input = pd.read_csv("train.csv", names=columns)
    neural_network = NeuralNet(raw_input)
    neural_network.train()
    testError = neural_network.predict("test.csv")
    print("Test Error : "+str(np.sum(testError))+"%")
    print("Test Accuracy : "+str(100-np.sum(testError))+"%")

