import numpy as np
import matplotlib.pyplot as plt

def sigmoid_function(x):
    return 1.0 / (1.0 + np.exp(-x))

class RegularizedLogisticRegression(object):
    '''
    Implement regularized logistic regression for binary classification.

    The weight vector w should be learned by minimizing the regularized loss
    \l(h, (x,y)) = log(1 + exp(-y <w, x>)) + \lambda \|w\|_2^2. In other words, the objective
    function that we are trying to minimize is the log loss for binary logistic regression 
    plus Tikhonov regularization with a coefficient of \lambda.
    '''
    def __init__(self):
        self.learningRate = 0.00001 # Feel free to play around with this if you'd like, though this value will do
        self.num_epochs = 10000 # Feel free to play around with this if you'd like, though this value will do
        self.batch_size = 15 # Feel free to play around with this if you'd like, though this value will do
        self.weights = None

        #####################################################################
        #                                                                    #
        #    MAKE SURE TO SET THIS TO THE OPTIMAL LAMBDA BEFORE SUBMITTING    #
        #                                                                    #
        #####################################################################

        self.lmbda = 1 # tune this parameter

    def train(self, X, Y):
        '''
        Train the model, using batch stochastic gradient descent
        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            None
        '''
        #[TODO]
        num_examples,num_features = np.shape(X)
        self.weights = np.zeros((1,num_features))
        for epoch in range(self.num_epochs):
            # shuffle training examples
            shuffler = np.random.permutation(len(Y))
            shuffleX = X[shuffler,:]
            shuffleY = Y[shuffler]
            DLs_raw = np.zeros((1,num_features))
            for i in range(int(len(Y)/self.batch_size)):
                Xbatch = shuffleX[i*self.batch_size : (i+1)*self.batch_size, :]
                Ybatch = shuffleY[i*self.batch_size : (i+1)*self.batch_size]
                for x,y in zip(Xbatch,Ybatch):
                    DLs_raw += (sigmoid_function(np.dot(self.weights,x)) - y)*x
                # add regularization factor
                DLs = (DLs_raw/len(Ybatch)) + 2 * self.lmbda * self.weights
                # update weights
                self.weights = self.weights - (self.learningRate*DLs)
            if epoch % 1000 == 0:
                print(epoch)

    def predict(self, X):
        '''
        Compute predictions based on the learned parameters and examples X
        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
        @return:
            A 1D Numpy array with one element for each row in X containing the predicted class.
        '''
        #[TODO]
        raw_predicts = np.matmul(self.weights,np.transpose(X))
        prediction_percents = sigmoid_function(raw_predicts)
        percent_lst = list(prediction_percents[0,:])
        predictions = [1 if val >= .5 else 0 for val in percent_lst]
        return predictions

    def accuracy(self,X, Y):
        '''
        Output the accuracy of the trained model on a given testing dataset X and labels Y.
        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            a float number indicating accuracy (between 0 and 1)
        '''
        #[TODO]
        predictions = np.array(self.predict(X))
        correct = 0
        for i in range(len(Y)):
            if Y[i] == predictions[i]:
                correct += 1
        return correct/len(Y)

    def runTrainTestValSplit(self, lambda_list, X_train, Y_train, X_val, Y_val):
        '''
        Given the training and validation data, fit the model with training data and test it with
        respect to each lambda. Record the training error and validation error, which are equivalent 
        to (1 - accuracy).

        @params:
            lambda_list: a list of lambdas
            X_train: a 2D Numpy array for trainig where each row contains an example,
            padded by 1 column for the bias
            Y_train: a 1D Numpy array for training containing the corresponding labels for each example
            X_val: a 2D Numpy array for validation where each row contains an example,
            padded by 1 column for the bias
            Y_val: a 1D Numpy array for validation containing the corresponding labels for each example
        @returns:
            train_errors: a list of training errors with respect to the lambda_list
            val_errors: a list of validation errors with respect to the lambda_list
        '''
        train_errors = []
        val_errors = []
        #[TODO] train model and calculate train and validation errors here for each lambda
        for lam in lambda_list:
            self.lmbda = lam
            self.train(X_train, Y_train)
            train_error = 1 - self.accuracy(X_train,Y_train)
            val_error = 1 - self.accuracy(X_val,Y_val)
            train_errors.append(train_error)
            val_errors.append(val_error)
        return train_errors, val_errors

    def _kFoldSplitIndices(self, dataset, k):
        '''
        Helper function for k-fold cross validation. Evenly split the indices of a
        dataset into k groups.

        For example, indices = [0, 1, 2, 3] with k = 2 may have an output
        indices_split = [[1, 3], [2, 0]].
        
        Please don't change this.
        @params:
            dataset: a Numpy array where each row contains an example
            k: an integer, which is the number of folds
        @return:
            indices_split: a list containing k groups of indices
        '''
        num_data = dataset.shape[0]
        fold_size = int(num_data / k)
        indices = np.arange(num_data)
        np.random.shuffle(indices)
        indices_split = np.split(indices[:fold_size*k], k)
        return indices_split

    def runKFold(self, lambda_list, X, Y, k = 3):
        '''
        Run k-fold cross validation on X and Y with respect to each lambda. Return all k-fold
        errors.
        
        Each run of k-fold involves k iterations. For an arbitrary iteration i, the i-th fold is
        used as testing data while the rest k-1 folds are combined as one set of training data. The k results are
        averaged as the cross validation error.

        @params:
            lambda_list: a list of lambdas
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
            k: an integer, which is the number of folds, k is 3 by default
        @return:
            k_fold_errors: a list of k-fold errors with respect to the lambda_list
        '''
        k_fold_errors = []
        for lmbda in lambda_list:
            self.lmbda = lmbda
            #[TODO] call _kFoldSplitIndices to split indices into k groups randomly
            indices = self._kFoldSplitIndices(X,k)
            #[TODO] for each iteration i = 1...k, train the model using lmbda
            # on k???1 folds of data. Then test with the i-th fold.
            i_fold_errors = []
            for i in range(k):
                train_folds_inds = np.delete(indices,i,0).flatten()
                test_fold_inds = indices[i].flatten()

                train_folds_X = X[train_folds_inds,:]
                train_folds_Y = Y[train_folds_inds]

                test_fold_X = X[test_fold_inds,:]
                test_fold_Y = Y[test_fold_inds]
                self.train(train_folds_X,train_folds_Y)
                i_fold_error = 1 - self.accuracy(test_fold_X,test_fold_Y)
                i_fold_errors.append(i_fold_error)
            #[TODO] calculate and record the cross validation error by averaging total errors
            k_fold_error = sum(i_fold_errors)/len(i_fold_errors)
            k_fold_errors.append(k_fold_error)
        return k_fold_errors

    def plotError(self, lambda_list, train_errors, val_errors, k_fold_errors):
        '''
        Produce a plot of the cost function on the training and validation sets, and the
        cost function of k-fold with respect to the regularization parameter lambda. Use this plot
        to determine a valid lambda.
        @params:
            lambda_list: a list of lambdas
            train_errors: a list of training errors with respect to the lambda_list
            val_errors: a list of validation errors with respect to the lambda_list
            k_fold_errors: a list of k-fold errors with respect to the lambda_list
        @return:
            None
        '''
        plt.figure()
        plt.semilogx(lambda_list, train_errors, label = 'training error')
        plt.semilogx(lambda_list, val_errors, label = 'validation error')
        plt.semilogx(lambda_list, k_fold_errors, label = 'k-fold error')
        plt.xlabel('lambda')
        plt.ylabel('error')
        plt.legend()
        plt.show()
