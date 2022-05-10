import numpy as np
from qp import solve_QP


def linear_kernel(xi, xj):
    """
    Kernel Function, linear kernel (ie: regular dot product)

    :param xi: an input sample (1D np array)
    :param xj: an input sample (1D np array)
    :return: float64
    """
    #TODO
    return np.dot(xi,xj)


def rbf_kernel(xi, xj, gamma=0.1):
    """
    Kernel Function, radial basis function kernel

    :param xi: an input sample (1D np array)
    :param xj: an input sample (1D np array)
    :param gamma: parameter of the RBF kernel (scalar)
    :return: float64
    """
    # TODO
    vector_diff = xi - xj
    norm2 = np.linalg.norm(vector_diff)**2
    return np.exp(-1*gamma*norm2)

def polynomial_kernel(xi, xj, c=2, d=2):
    """
    Kernel Function, polynomial kernel

    :param xi: an input sample (1D np array)
    :param xj: an input sample (1D np array)
    :param c: mean of the polynomial kernel (scalar)
    :param d: exponent of the polynomial (scalar)
    :return: float64
    """
    #TODO
    dot_pdt = np.dot(xi,xj)
    return (dot_pdt + c)**d


class SVM(object):

    def __init__(self, kernel_func=linear_kernel, lambda_param=.1):
        self.kernel_func = kernel_func
        self.lambda_param = lambda_param

    def train(self, inputs, labels):
        """
        train the model with the input data (inputs and labels),
        find the coefficients and constaints for the quadratic program and
        calculate the alphas

        :param inputs: inputs of data, a numpy array
        :param labels: labels of data, a numpy array
        :return: None
        """
        self.train_inputs = inputs
        self.train_labels = labels

        # constructing QP variables
        G = self._get_gram_matrix()
        Q, c = self._objective_function(G)
        A, b = self._inequality_constraint(G)

        # TODO: Uncomment the next line when you have implemented _get_gram_matrix(),
        # _inequality_constraints() and _objective_function().
        self.alpha = solve_QP(Q, c, A, b)[:self.train_inputs.shape[0]]

    def _get_gram_matrix(self):
        """
        Generate the Gram matrix for the training data stored in self.train_inputs.

        Recall that element i, j of the matrix is K(x_i, x_j), where K is the
        kernel function.

        :return: the Gram matrix for the training data, a numpy array
        """

        # TODO 
        num_examples = len(self.train_inputs)
        gram_matrix = np.zeros((num_examples,num_examples))
        for i in range(num_examples):
            for j in range(num_examples):
                gram_matrix[i,j] = self.kernel_func(self.train_inputs[i],self.train_inputs[j])
        return gram_matrix

    def _objective_function(self, G):
        """
        Generate the coefficients on the variables in the objective function for the
        SVM quadratic program.

        Recall the objective function is:
        minimize (1/2)x^T Q x + c^T x

        :param G: the Gram matrix for the training data, a numpy array
        :return: two numpy arrays, Q and c which fully specify the objective function
        """

        # TODO
        # Ga = np.matmul(G,self.alpha)
        # # create x vector
        # xi = np.zeros(m)
        # for i in range(m):
        #     val = self.train_labels[i]*Ga[i]
        #     xi[i] = max(0,val)
        # x = np.concatenate((self.alpha,xi))

        #create Q matrix
        m = len(self.train_labels)
        Qa = 2*self.lambda_param*G
        Qb = np.concatenate((Qa,np.zeros((m,m))),axis=1)
        Q = np.concatenate((Qb,np.zeros((m,2*m))),axis=0)

        #create c vector
        c1 = np.zeros(m)
        c2 = (1/m) * np.ones(m)
        c = np.concatenate((c1,c2))
        return Q,c

    def _inequality_constraint(self, G):
        """
        Generate the inequality constraints for the SVM quadratic program. The
        constraints will be enforced so that Ax <= b.
        :param G: the Gram matrix for the training data, a numpy array
        :return: two numpy arrays, A and b which fully specify the constraints
        """

        # TODO (hint: you can think of x as the concatenation of all the alphas and
        # all the all the xi's; think about what this implies for what A should look like.)
        #construct b
        m = len(self.train_labels)
        b1 = -1*np.ones(m)
        b2 = np.zeros(m)
        b = np.concatenate((b1,b2))

        #construct A
        yG = np.copy(G)
        for i in range(len(self.train_labels)):
            yG[i,:] *= -1*self.train_labels[i]
        diag = np.ones(m)
        diag_matrix= -1*np.diag(diag)
        A1 = np.concatenate((yG,diag_matrix),axis=1)
        A2 = np.zeros((m,m))
        A3 = np.concatenate((A2,diag_matrix),axis=1)
        A = np.concatenate((A1,A3),axis=0)

        # A = np.zeros((2*m,2*m))
        # for i in range(m):
        #     A[2*m-i-1,i] = -1
        #     A[2*m-i-1,m+i] = -1
        # for i in range(m):
        #     for j in range(m):
        #         A[m-i-1,j] = G[i,j]*-1*self.train_labels[i]

        return A,b

    def predict(self, inputs):
        """
        Generate predictions given input.

        :param input: 2D Numpy array. Each row is a vector for which we output a prediction.
        :return: A 1D numpy array of predictions.
        """

        #TODO
        predictions = np.zeros(len(inputs))
        for i in range(len(inputs)):
            sum = 0
            for j in range(len(self.train_labels)):
                sum += self.alpha[j] * self.kernel_func(self.train_inputs[j],inputs[i])
            if sum >= 0:
                predictions[i] = 1
            elif sum < 0:
                predictions[i] = -1
        return predictions

    def accuracy(self, inputs, labels):
        """
        Calculate the accuracy of the classifer given inputs and their true labels.

        :param inputs: 2D Numpy array which we are testing calculating the accuracy of.
        :param labels: 1D Numpy array with the inputs corresponding true labels.
        :return: A float indicating the accuracy (between 0.0 and 1.0)
        """

        #TODO
        predicts = self.predict(inputs)
        num_correct = 0
        for p,l in zip(predicts,labels):
            if p == l:
                num_correct += 1
        accuracy = num_correct/len(labels)
        return accuracy
