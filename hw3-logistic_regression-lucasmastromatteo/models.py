#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
   This file contains the Logistic Regression classifier

   Brown CS142, Spring 2020
'''
import random
import numpy as np


def softmax(x):
    '''
    Apply softmax to an array

    @params:
        x: the original array
    @return:
        an array with softmax applied elementwise.
    '''
    e = np.exp(x - np.max(x))
    return (e + 1e-6) / (np.sum(e) + 1e-6)

class LogisticRegression:
    '''
    Multiclass Logistic Regression that learns weights using 
    stochastic gradient descent.
    '''
    def __init__(self, n_features, n_classes, batch_size, conv_threshold):
        '''
        Initializes a LogisticRegression classifer.

        @attrs:
            n_features: the number of features in the classification problem
            n_classes: the number of classes in the classification problem
            weights: The weights of the Logistic Regression model
            alpha: The learning rate used in stochastic gradient descent
        '''
        self.n_classes = n_classes
        self.n_features = n_features
        self.weights = np.zeros((n_classes, n_features + 1))  # An extra row added for the bias
        self.alpha = 0.03  # DO NOT TUNE THIS PARAMETER
        self.batch_size = batch_size
        self.conv_threshold = conv_threshold

    def train(self, X, Y):
        '''
        Trains the model using stochastic gradient descent

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            num_epochs: integer representing the number of epochs taken to reach convergence
        '''
        # TODO
        num_epochs = 0
        converge = False
        lastloss = None
        while not converge:
            num_epochs += 1
            # Shuffle training examples
            np.random.seed(12)
            shuffler = np.random.permutation(len(Y))
            shuffleX = X[shuffler,:]
            shuffleY = Y[shuffler]

            for i in range(int(len(Y)/self.batch_size)):
                Xbatch = shuffleX[i*self.batch_size : (i+1)*self.batch_size, :]
                Ybatch = shuffleY[i*self.batch_size : (i+1)*self.batch_size]
                DLw = np.zeros((self.n_classes,self.n_features + 1))
                #sftmx = np.apply_along_axis(softmax,0,np.matmul(self.weights,np.transpose(Xbatch)))
                #for k in range(len(Ybatch)):
                for x,y in zip(Xbatch,Ybatch):
                    sftmx = softmax(np.dot(self.weights,x))
                    for j in range(self.n_classes):
                        # calculate partial derivative
                        if y == j:
                            DLw[j,:] += (sftmx[j] - 1)*x
                        else:
                            DLw[j,:] += (sftmx[j])*x
                        # if Ybatch[k] == j:
                        #     DLw[j,:] += np.matmul((sftmx[j,:]-1),Xbatch)
                        # else:
                        #     DLw[j,:] += np.matmul(sftmx[j,:],Xbatch)
                #update weights
                self.weights = self.weights - (self.alpha * DLw)/len(Xbatch)

            # Check for convergence
            if lastloss is not None:
                print(self.loss(X,Y))
                print(lastloss)
                if abs(self.loss(X,Y) - lastloss) < self.conv_threshold:
                    converge = True
            lastloss = self.loss(X,Y)

        return num_epochs

    def loss(self, X, Y):
        '''
        Returns the total log loss on some dataset (X, Y), divided by the number of examples.
        @params:
            X: 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: 1D Numpy array containing the corresponding labels for each example
        @return:
            A float number which is the average loss of the model on the dataset
        '''
        # TODO
        W_X_IP = np.apply_along_axis(softmax,0,np.matmul(self.weights,np.transpose(X)))
        total_loss = 0
        for i in range(len(Y)):
            total_loss += -1 * np.log(W_X_IP[Y[i]][i])

        return (1/len(Y)) * total_loss

    def predict(self, X):
        '''
        Compute predictions based on the learned weights and examples X

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
        @return:
            A 1D Numpy array with one element for each row in X containing the predicted class.
        '''
        # TODO
        #w_dot_X = np.matmul(self.weights,np.transpose(X))
        w_dot_X = np.apply_along_axis(softmax,0,np.matmul(self.weights,np.transpose(X)))
        predictions = []
        for i in range(len(w_dot_X[0,:])):
           # raw_predicts = w_dot_X[:,i]
           # softmax_predicts = softmax(raw_predicts)
            predictions.append(np.argmax(w_dot_X[:,i]))
            #predictions.append(np.argmax(softmax_predicts))
        return predictions

    def accuracy(self, X, Y):
        '''
        Outputs the accuracy of the trained model on a given testing dataset X and labels Y.

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            a float number indicating accuracy (between 0 and 1)
        '''
        # TODO
        predicts = np.array(self.predict(X))
        num_correct = 0

        for i in range(len(predicts)):
            if predicts[i] == Y[i]:
                num_correct += 1
        return num_correct/len(Y)
