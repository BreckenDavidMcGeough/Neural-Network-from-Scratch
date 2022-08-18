from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error



class NeuralNetwork:
    def __init__(self,df):
        self.df = df
        self.X, self.x_test, self.y, self.y_test = self.preprocessing()
        self.epochs = 10000
        self.alpha = 10
        self.hiddenNeurons = 10
        self.W1 = np.asmatrix([[np.random.randn() for _ in range(self.hiddenNeurons)] for _ in range(self.X.shape[1])])
        self.W2 = np.asmatrix([[np.random.randn() for _ in range(self.hiddenNeurons)] for _ in range(self.hiddenNeurons)])
        self.W3 = np.asmatrix([[np.random.randn()] for _ in range(self.hiddenNeurons)])
        
    def best_features(self):
        corr_matrix = self.df.corr()["Outcome"]
        best_matrix = []
        columns = self.df.columns
        for col in columns:
            if corr_matrix[col] > .2 and col != "Outcome":
                best_matrix.append(col)
        return best_matrix
    
    def preprocessing(self):
        best_features = self.best_features()
        self.df["Bias"] = [1 for _ in range(len(self.df))]
        best_features.insert(0,"Bias")
        X = self.df[best_features]
        X = scale(X)
        for i in range(X.shape[0]):
            X[i][0] = 1
        y = self.df[["Outcome"]]
        x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state = 5)
        x_train, x_test, y_train, y_test = np.asmatrix(x_train), np.asmatrix(x_test), np.asmatrix(y_train), np.asmatrix(y_test)
        return x_train,x_test,y_train,y_test
    
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def sigmoid_derivative(self,z):
        return np.exp(-z)/(np.square((1+np.exp(-z))))
    
    def forward_propagation(self,x):
        self.z1 = np.dot(x,self.W1)
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1,self.W2)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2,self.W3)
        yHat = self.sigmoid(self.z3)
        return yHat
    
    def backpropagation(self):
        for _ in range(self.epochs):
            yHat = self.forward_propagation(self.X)
            
            #dJdyHat = -(self.y - yHat) regression gradient loss function
            
            #This is the gradient of the cross entropy loss function for logistic binary classificiation 
            dJdyHat = ((-1 * self.y) / yHat) + ((1 - self.y) / (1 - yHat))
            
            delta3 = np.multiply(dJdyHat,self.sigmoid_derivative(self.z3)) #hadamard product
            delta2 = np.multiply(np.dot(delta3,self.W3.transpose()),self.sigmoid_derivative(self.z2))
            delta1 = np.multiply(np.dot(delta2,self.W2.transpose()),self.sigmoid_derivative(self.z1))
            
            dJdW3 = np.dot(self.a2.transpose(),delta3)
            dJdW2 = np.dot(self.a1.transpose(),delta2)
            dJdW1 = np.dot(self.X.transpose(),delta1)
            
            self.W3 = self.W3 - self.alpha * dJdW3
            self.W2 = self.W2 - self.alpha * dJdW2
            self.W1 = self.W1 - self.alpha * dJdW1
            
    def normalize_results(self):
        self.backpropagation()
        predictions = self.forward_propagation(self.x_test)
        norm_predictions = []
        for pred in predictions:
            if pred >= .5:
                norm_predictions.append(1)
            else:
                norm_predictions.append(0)
        return norm_predictions
    
    def metrics(self):
        predictions = self.normalize_results()
        y = self.y_test
        num_wrong = 0
        for i in range(len(predictions)):
            if predictions[i] != y[i]:
                num_wrong += 1
        return ((len(predictions) - num_wrong)/len(predictions)) * 100
    
    def shapes(self):
        print("shape of x_train: " + str(self.X.shape))
        print("shape of y_train: " + str(self.y.shape))
        print("shape of x_test: " + str(self.x_test.shape))
        print("shape of y_test: " + str(self.y_test.shape))
        print("shape of W1: " + str(self.W1.shape))
        print("shape of W2: " + str(self.W2.shape))
        print("shape of W3: " + str(self.W3.shape))


df = pd.read_csv("diabetes.csv")
nn = NeuralNetwork(df)
print(nn.metrics())