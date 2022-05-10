import numpy as np
import random
import copy
import math

def node_score_error(prob):
    '''
        TODO:
        Calculate the node score using the train error of the subdataset and return it.
        For a dataset with two classes, C(p) = min{p, 1-p}
    '''
    C_vals = [prob,1-prob]
    return min(C_vals)


def node_score_entropy(prob):
    '''
        TODO:
        Calculate the node score using the entropy of the subdataset and return it.
        For a dataset with 2 classes, C(p) = -p * log(p) - (1-p) * log(1-p)
        For the purposes of this calculation, assume 0*log0 = 0.
        HINT: remember to consider the range of values that p can take!
    '''
    if prob == 0:
        C1 = 0
    else:
        C1 = -1 * prob * np.log(prob)
    if prob == 1:
        C2 = 0
    else:
        C2 = -1 * (1-prob) * np.log(1-prob)
    cp = C1 + C2
    return cp


def node_score_gini(prob):
    '''
        TODO:
        Calculate the node score using the gini index of the subdataset and return it.
        For dataset with 2 classes, C(p) = 2 * p * (1-p)
    '''
    cp = 2 * prob * (1-prob)
    return cp



class Node:
    '''
    Helper to construct the tree structure.
    '''
    def __init__(self, left=None, right=None, depth=0, index_split_on=0, isleaf=False, label=1):
        self.left = left
        self.right = right
        self.depth = depth
        self.index_split_on = index_split_on
        self.isleaf = isleaf
        self.label = label
        self.info = {} # used for visualization


    def _set_info(self, gain, num_samples):
        '''
        Helper function to add to info attribute.
        You do not need to modify this. 
        '''

        self.info['gain'] = gain
        self.info['num_samples'] = num_samples


class DecisionTree:

    def __init__(self, data, validation_data=None, gain_function=node_score_entropy, max_depth=40):
        self.max_depth = max_depth
        self.root = Node()
        self.gain_function = gain_function

        indices = list(range(1, len(data[0])))

        self._split_recurs(self.root, data, indices)

        # Pruning
        if validation_data is not None:
            self._prune_recurs(self.root, validation_data)


    def predict(self, features):
        '''
        Helper function to predict the label given a row of features.
        You do not need to modify this.
        '''
        return self._predict_recurs(self.root, features)


    def accuracy(self, data):
        '''
        Helper function to calculate the accuracy on the given data.
        You do not need to modify this.
        '''
        return 1 - self.loss(data)


    def loss(self, data):
        '''
        Helper function to calculate the loss on the given data.
        You do not need to modify this.
        '''
        cnt = 0.0
        test_Y = [row[0] for row in data]
        for i in range(len(data)):
            prediction = self.predict(data[i])
            if (prediction != test_Y[i]):
                cnt += 1.0
        return cnt/len(data)


    def _predict_recurs(self, node, row):
        '''
        Helper function to predict the label given a row of features.
        Traverse the tree until leaves to get the label.
        You do not need to modify this.
        '''
        if node.isleaf or node.index_split_on == 0:
            return node.label
        split_index = node.index_split_on
        if not row[split_index]:
            return self._predict_recurs(node.left, row)
        else:
            return self._predict_recurs(node.right, row)


    def _prune_recurs(self, node, validation_data):
        '''
        TODO:
        Prune the tree bottom up recursively. Nothing needs to be returned.
        Do not prune if the node is a leaf.
        Do not prune if the node is non-leaf and has at least one non-leaf child.
        Prune if deleting the node could reduce loss on the validation data.
        NOTE:
        This might be slightly different from the pruning described in lecture.
        Here we won't consider pruning a node's parent if we don't prune the node 
        itself (i.e. we will only prune nodes that have two leaves as children.)
        HINT: Think about what variables need to be set when pruning a node!
        '''
        if not node.isleaf:
            # if ((node.left.isleaf) and (not node.right.isleaf)):
            #     self._prune_recurs(node.right,validation_data)
            # if ((not node.left.isleaf) and (node.right.isleaf)):
            #     self._prune_recurs(node.left,validation_data)
            # if (not node.left.isleaf) and (not node.right.isleaf):
            #     self._prune_recurs(node.left,validation_data)
            #     self._prune_recurs(node.right,validation_data)
            if node.left is not None:
                self._prune_recurs(node.left,validation_data)
            if node.right is not None:
                self._prune_recurs(node.right,validation_data)
            if (not node.isleaf) and node.left.isleaf and node.right.isleaf:
                # calculate current loss
                current_loss = self.loss(validation_data)
                left = node.left
                right = node.right
                node.left = None
                node.right = None
                node.isleaf = True
                # calculate loss after removal of node, set left and right node to none, save
                if self.loss(validation_data) > current_loss:
                    node.isleaf = False
                    node.left = left
                    node.right = right

    def _is_terminal(self, node, data, indices):
        '''
        TODO:
        Helper function to determine whether the node should stop splitting.
        Stop the recursion if:
            1. The dataset is empty.
            2. There are no more indices to split on.
            3. All the instances in this dataset belong to the same class
            4. The depth of the node reaches the maximum depth.
        Return:
            - A boolean, True indicating the current node should be a leaf and 
              False if the node is not a leaf.
            - A label, indicating the label of the leaf (or the label the node would 
              be if we were to terminate at that node). If there is no data left, you
              can return either label at random.
        '''
        if len(data) == 0:
            return True, random.randint(0,1)
        elif len(indices) == 0:
            if sum(data[:,0]) > (len(data[:,0])/2):
                label = 1
            else:
                label = 0
            return True,label
        elif np.max(data[:,0]) == np.min(data[:,0]):
            return True,np.max(data[:,0])
        elif node.depth == self.max_depth:
            if sum(data[:,0]) > (len(data[:,0])/2):
                label = 1
            else:
                label = 0
            return True,label
        else:
            if sum(data[:,0]) > (len(data[:,0])/2):
                label = 1
            else:
                label = 0
            return False,label

    def _split_recurs(self, node, data, indices):
        '''
        TODO:
        Recursively split the node based on the rows and indices given.
        Nothing needs to be returned.

        First use _is_terminal() to check if the node needs to be split.
        If so, select the column that has the maximum infomation gain to split on.
        Store the label predicted for this node, the split column, and use _set_info()
        to keep track of the gain and the number of datapoints at the split.
        Then, split the data based on its value in the selected column.
        The data should be recursively passed to the children.
        '''
        is_leaf, label = self._is_terminal(node,data,indices)
        if is_leaf:
            node.isleaf = is_leaf
            node.label = label
            return
        else:
            best_index = 0
            best_gain = float('-inf')
            for index in indices:
                gain = self._calc_gain(data,index,self.gain_function)
                if (gain > best_gain): # and (gain > 0):
                    best_index = index
                    best_gain = gain

            node.isleaf = is_leaf
            node.label = label
            node.index_split_on = best_index
            node._set_info(best_gain,len(data[:,0]))

            data_left = data[data[:,best_index] == 0]
            data_right = data[data[:,best_index] == 1]
            indices.remove(best_index)

            node.left = Node(depth=node.depth+1)
            node.right = Node(depth=node.depth+1)

            self._split_recurs(node.left,data_left,copy.copy(indices))
            self._split_recurs(node.right,data_right,copy.copy(indices))

    def _calc_gain(self, data, split_index, gain_function):
        '''
        TODO:
        Calculate the gain of the proposed splitting and return it.
        Gain = C(P[y=1]) - P[x_i=True] * C(P[y=1|x_i=True]) - P[x_i=False] * C(P[y=0|x_i=False])
        Here the C(p) is the gain_function. For example, if C(p) = min(p, 1-p), this would be
        considering training error gain. Other alternatives are entropy and gini functions.
        '''
        p_y1 = sum(data[:,0])/len(data[:,0])
        c_a = gain_function(p_y1)

        split_col = data[:,split_index]
        data_true = data[data[:,split_index] == 1]
        data_false = data[data[:,split_index] == 0]

        p_xi_true = len(split_col[split_col[:]==1])/len(split_col)
        p_xi_false = 1 - p_xi_true
        try:
            p_y1_xi_true = sum(data_true[:,0])/len(data_true[:,0])
        except ZeroDivisionError:
            p_y1_xi_true = 0
        try:
            p_y0_xi_false = 1 - (sum(data_false[:,0])/len(data_false[:,0]))
        except ZeroDivisionError:
            p_y0_xi_false = 1

        c_b = gain_function(p_y1_xi_true)
        c_c = gain_function(p_y0_xi_false)
        # print(c_a,p_xi_true*c_b,p_xi_false*c_c)
        gain = c_a - p_xi_true*c_b - p_xi_false*c_c
        return gain
    

    def print_tree(self):
        '''
        Helper function for tree_visualization.
        Only effective with very shallow trees.
        You do not need to modify this.
        '''
        print('---START PRINT TREE---')
        def print_subtree(node, indent=''):
            if node is None:
                return str("None")
            if node.isleaf:
                return str(node.label)
            else:
                decision = 'split attribute = {:d}; gain = {:f}; number of samples = {:d}'.format(node.index_split_on, node.info['gain'], node.info['num_samples'])
            left = indent + '0 -> '+ print_subtree(node.left, indent + '\t\t')
            right = indent + '1 -> '+ print_subtree(node.right, indent + '\t\t')
            return (decision + '\n' + left + '\n' + right)

        print(print_subtree(self.root))
        print('----END PRINT TREE---')


    def loss_plot_vec(self, data):
        '''
        Helper function to visualize the loss when the tree expands.
        You do not need to modify this.
        '''
        self._loss_plot_recurs(self.root, data, 0)
        loss_vec = []
        q = [self.root]
        num_correct = 0
        while len(q) > 0:
            node = q.pop(0)
            num_correct = num_correct + node.info['curr_num_correct']
            loss_vec.append(num_correct)
            if node.left != None:
                q.append(node.left)
            if node.right != None:
                q.append(node.right)

        return 1 - np.array(loss_vec)/len(data)


    def _loss_plot_recurs(self, node, rows, prev_num_correct):
        '''
        Helper function to visualize the loss when the tree expands.
        You do not need to modify this.
        '''
        labels = [row[0] for row in rows]
        curr_num_correct = labels.count(node.label) - prev_num_correct
        node.info['curr_num_correct'] = curr_num_correct

        if not node.isleaf:
            left_data, right_data = [], []
            left_num_correct, right_num_correct = 0, 0
            for row in rows:
                if not row[node.index_split_on]:
                    left_data.append(row)
                else:
                    right_data.append(row)

            left_labels = [row[0] for row in left_data]
            left_num_correct = left_labels.count(node.label)
            right_labels = [row[0] for row in right_data]
            right_num_correct = right_labels.count(node.label)

            if node.left != None:
                self._loss_plot_recurs(node.left, left_data, left_num_correct)
            if node.right != None:
                self._loss_plot_recurs(node.right, right_data, right_num_correct)
