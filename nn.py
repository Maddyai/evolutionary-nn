from sklearn.model_selection import train_test_split
import random

import pickle
import pandas as pd
import numpy


def sigmoid(x, derivative=False, lambda_rate=1):  
    if derivative:
        return sigmoid(x, lambda_rate=lambda_rate)*sigmoid(1-(x), lambda_rate=lambda_rate)
    else:
        return 1/(1+numpy.exp(-lambda_rate*x))


def clear_up(file_name="new_data.csv"):

    data = pd.read_csv(file_name, header=None)
    data.columns = ['input_1', 'input_2', 'output_1', 'output_2']
    data = data.drop_duplicates()
    data = (data - data.min()) / (data.max() - data.min())
    # data.plot(kind='density', subplots=True, layout=(2,2), sharex=False)
    # plt.show()
    return data

class NeuralNetwork():

    def __init__(self, parameters):
        self.prev_loss = 10
        self.count = 0 
        self.loss = 10
        self.parameters = parameters
        self.network = {}

    def create_random(self):
        for key in self.parameters:
            self.network[key] = random.choice(self.parameters[key])

    def new_model(self, network):
        self.network = network

    def save_model(self):
        with open('model_best.pickle', 'wb') as handle:
            pickle.dump(self.nn_weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def build_model(self, parameters):
        inumpyut_dimension = x_train.shape[0]
        output_dimension = y_train.shape[0]

        hidden_dimension = parameters["no_neurons"]
        self.learning_rate = parameters["learning_rate"]
        self.momentum_rate = parameters["momentum_rate"]
        self.lambda_rate = parameters["lambda_rate"]

        weight_1 = numpy.random.rand(hidden_dimension,inumpyut_dimension) 
        bias_1 = numpy.zeros((hidden_dimension,1))

        weight_2 = numpy.random.rand(output_dimension,hidden_dimension) 
        bias_2 = numpy.zeros((output_dimension,1))
        
        self.prev_delta_weight_1 = numpy.zeros((hidden_dimension,output_dimension))
        self.prev_delta_weight_2 = numpy.zeros((output_dimension, hidden_dimension))

        self.nn_weights = { 'weight_1': weight_1, 'bias_1': bias_1, 'weight_2': weight_2, 'bias_2': bias_2}


        self.epochs = 1000
        self.training_loss= []
        self.validation_loss = []


    def evaluate(self, x_train, y_train, x_cv, y_cv):

        weight_1 = self.nn_weights['weight_1']
        weight_2 = self.nn_weights['weight_2']
        bias_1 = self.nn_weights['bias_1']
        bias_2 = self.nn_weights['bias_2']

        V1 = numpy.dot(weight_1, x_train) + bias_1
        activation_1 = sigmoid(V1, lambda_rate=self.lambda_rate)
        
        V2 = numpy.dot(weight_2, activation_1) + bias_2
        predicted = sigmoid(V2, lambda_rate=self.lambda_rate) 
        
        rms_error = numpy.mean(numpy.square(y_train - predicted))

        V1 = numpy.dot(weight_1, x_cv) + bias_1
        activation_1 = sigmoid(V1, lambda_rate=self.lambda_rate)
        V2 = numpy.dot(weight_2, activation_1) + bias_2
        validation_loss_predicted = sigmoid(V2, lambda_rate=self.lambda_rate)
        
        rmse_cross_validation = numpy.mean(numpy.square(y_cv - validation_loss_predicted))
        # print (rmse_cross_validation)

        return rms_error, rmse_cross_validation

    def fit(self):

        self.build_model(self.network)

        for _ in range(self.epochs):
            weight_1 = self.nn_weights['weight_1']
            weight_2 = self.nn_weights['weight_2']
            bias_1 = self.nn_weights['bias_1']
            bias_2 = self.nn_weights['bias_2']

            #------------------------------------------------- Forward Pass

            V1 = numpy.dot(weight_1, x_train) +bias_1
            activation_1 = sigmoid(V1, lambda_rate=self.lambda_rate)
            V2 = numpy.dot(weight_2, activation_1) + bias_2
            predicted = sigmoid(V2, lambda_rate=self.lambda_rate)

            #------------------------------------------------- Backward Pass

            error = y_train - predicted
            
            gradient_2 = self.lambda_rate *(error * sigmoid(V2, derivative=True, lambda_rate=self.lambda_rate))
            gradient_1 = self.lambda_rate * numpy.dot(weight_2.T, gradient_2)*sigmoid(V1, derivative=True, lambda_rate=self.lambda_rate)
            
            delta_weight_1 = numpy.dot(gradient_1, x_train.T)
            delta_weight_2 = numpy.dot(gradient_2, activation_1.T)
            delta_bias_1 = numpy.sum(gradient_1, axis=1, keepdims=True)
            delta_bias_2 = numpy.sum(gradient_2, axis=1, keepdims=True)
            
            #----------------------------------------------------- Update the weights

            weight_1 += self.learning_rate*delta_weight_1 + self.momentum_rate * self.prev_delta_weight_1
            weight_2 += self.learning_rate*delta_weight_2 + self.momentum_rate * self.prev_delta_weight_2

            bias_1 += self.learning_rate*delta_bias_1
            bias_2 += self.learning_rate*delta_bias_2
            
            #------------------------------------------------------Save the weights of NN
            self.nn_weights = {
                'weight_1': weight_1, 'weight_2': weight_2,
                'bias_1': bias_1, 'bias_2': bias_2
                }
            #--------------------------------------------------------Previous weights for momentum
            self.prev_delta_weight_1 = delta_weight_1
            self.prev_delta_weight_2 = delta_weight_2
            
            rms_error, rmse_cross_validation = self.evaluate(x_train, y_train, x_cv, y_cv)
            self.training_loss.append(rms_error)
            self.validation_loss.append(rmse_cross_validation)

            self.count += 1
            if self.count > 100:
                return self.loss

            if rmse_cross_validation < self.loss:
                self.count = 0
                self.loss = rmse_cross_validation

        return self.loss

# -----------------------------------------Data handlelling 

data = clear_up("data.csv")
data_min = data.min()
data_max = data.max()

# print (data_min)
# print (data_max)

train , cv = train_test_split(data, test_size = 0.15)

x_train = numpy.array(train[['input_1', 'input_2']]).T
y_train = numpy.array(train[['output_1', 'output_2']]).T

x_cv = numpy.array(cv[['input_1', 'input_2']]).T
y_cv = numpy.array(cv[['output_1', 'output_2']]).T

# -----------------------------------------Data handlelling 
