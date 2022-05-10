import random
import numpy as np
import matplotlib.pyplot as plt

from get_data import get_data
from models import DecisionTree, node_score_error, node_score_entropy, node_score_gini


def loss_plot(ax, title, tree, pruned_tree, train_data, test_data):
    '''
        Example plotting code. This plots four curves: the training and testing
        average loss using tree and pruned tree.
        You do not need to change this code!
        Arguments:
            - ax: A matplotlib Axes instance.
            - title: A title for the graph (string)
            - tree: An unpruned DecisionTree instance
            - pruned_tree: A pruned DecisionTree instance
            - train_data: Training dataset returned from get_data
            - test_data: Test dataset returned from get_data
    '''
    fontsize=8
    ax.plot(tree.loss_plot_vec(train_data), label='train non-pruned')
    ax.plot(tree.loss_plot_vec(test_data), label='test non-pruned')
    ax.plot(pruned_tree.loss_plot_vec(train_data), label='train pruned')
    ax.plot(pruned_tree.loss_plot_vec(test_data), label='test pruned')


    ax.locator_params(nbins=3)
    ax.set_xlabel('number of nodes', fontsize=fontsize)
    ax.set_ylabel('loss', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    legend = ax.legend(loc='upper center', shadow=True, fontsize=fontsize-2)

def explore_dataset(filename, class_name):
    train_data, validation_data, test_data = get_data(filename, class_name)

    # TODO: Print 12 loss values associated with the dataset.
    # For each measure of gain (training error, entropy, gini):
    #      (a) Print average training loss (not-pruned)
    tree1 = DecisionTree(train_data,validation_data=None,gain_function=node_score_error)
    print('Unpruned training error training loss: ' + str(tree1.loss(train_data)))

    tree2 = DecisionTree(train_data,validation_data=None,gain_function=node_score_entropy)
    print('Unpruned entropy training loss: ' + str(tree2.loss(train_data)))

    tree3 = DecisionTree(train_data,validation_data=None,gain_function=node_score_gini)
    print('Unpruned gini training loss: ' + str(tree3.loss(train_data)))

    #      (b) Print average test loss (not-pruned)
    tree1 = DecisionTree(train_data,validation_data=None,gain_function=node_score_error)
    print('Unpruned training error test loss: ' + str(tree1.loss(test_data)))

    tree2 = DecisionTree(train_data,validation_data=None,gain_function=node_score_entropy)
    print('Unpruned entropy test loss: ' + str(tree2.loss(test_data)))

    tree3 = DecisionTree(train_data,validation_data=None,gain_function=node_score_gini)
    print('Unpruned gini test loss: ' + str(tree3.loss(test_data)))

    #      (c) Print average training loss (pruned)
    tree1 = DecisionTree(train_data,validation_data=validation_data,gain_function=node_score_error)
    print('pruned training error training loss: ' + str(tree1.loss(train_data)))

    tree2 = DecisionTree(train_data,validation_data=validation_data,gain_function=node_score_entropy)
    print('pruned entropy training loss: ' + str(tree2.loss(train_data)))

    tree3 = DecisionTree(train_data,validation_data=validation_data,gain_function=node_score_gini)
    print('pruned gini training loss: ' + str(tree3.loss(train_data)))

    #      (d) Print average test loss (pruned)
    tree1 = DecisionTree(train_data,validation_data=validation_data,gain_function=node_score_error)
    print('pruned training error test loss: ' + str(tree1.loss(test_data)))

    tree2 = DecisionTree(train_data,validation_data=validation_data,gain_function=node_score_entropy)
    print('pruned entropy test loss: ' + str(tree2.loss(test_data)))

    tree3 = DecisionTree(train_data,validation_data=validation_data,gain_function=node_score_gini)
    print('pruned gini test loss: ' + str(tree3.loss(test_data)))
    # TODO: Feel free to print or plot anything you like here. Just comment
    # make sure to comment it out, or put it in a function that isn't called
    # by default when you hand in your code!
    # fake_data = np.array([[1,1,1],[1,0,0],[0,1,0],[0,0,1]])
    # tree = DecisionTree(fake_data,validation_data=fake_data)
    # tree.print_tree()
    # accuracy = tree.accuracy(fake_data)
    # print(accuracy)
def create_loss_plots(filename,class_name):
    train_data, validation_data, test_data = get_data(filename, class_name)
    treeD1 = DecisionTree(train_data,gain_function=node_score_entropy,max_depth=1)
    treeD2 = DecisionTree(train_data,gain_function=node_score_entropy,max_depth=2)
    treeD3 = DecisionTree(train_data,gain_function=node_score_entropy,max_depth=3)
    treeD4 = DecisionTree(train_data,gain_function=node_score_entropy,max_depth=4)
    treeD5 = DecisionTree(train_data,gain_function=node_score_entropy,max_depth=5)
    treeD6 = DecisionTree(train_data,gain_function=node_score_entropy,max_depth=6)
    treeD7 = DecisionTree(train_data,gain_function=node_score_entropy,max_depth=7)
    treeD8 = DecisionTree(train_data,gain_function=node_score_entropy,max_depth=8)
    treeD9 = DecisionTree(train_data,gain_function=node_score_entropy,max_depth=9)
    treeD10 = DecisionTree(train_data,gain_function=node_score_entropy,max_depth=10)
    treeD11 = DecisionTree(train_data,gain_function=node_score_entropy,max_depth=11)
    treeD12 = DecisionTree(train_data,gain_function=node_score_entropy,max_depth=12)
    treeD13 = DecisionTree(train_data,gain_function=node_score_entropy,max_depth=13)
    treeD14 = DecisionTree(train_data,gain_function=node_score_entropy,max_depth=14)
    treeD15 = DecisionTree(train_data,gain_function=node_score_entropy,max_depth=15)

    losses = [treeD1.loss(train_data),treeD2.loss(train_data),treeD3.loss(train_data),treeD4.loss(train_data),
            treeD5.loss(train_data),treeD6.loss(train_data),treeD7.loss(train_data),treeD8.loss(train_data),
            treeD9.loss(train_data),treeD10.loss(train_data),treeD11.loss(train_data),treeD12.loss(train_data),
            treeD13.loss(train_data),treeD14.loss(train_data),treeD15.loss(train_data)]
    depths = [x for x in range(1,16)]
    plt.figure()
    plt.plot(depths,losses)
    plt.title('Training Loss vs Maximum Depth')
    plt.xlabel('Maximum Depth')
    plt.ylabel('Training Loss')
    plt.show()
def main():
    ########### PLEASE DO NOT CHANGE THESE LINES OF CODE! ###################
    random.seed(1)
    np.random.seed(1)
    #########################################################################

    # explore_dataset('data/chess.csv', 'won')
    # explore_dataset('data/spam.csv', '1')
    create_loss_plots('data/spam.csv', '1')

main()
