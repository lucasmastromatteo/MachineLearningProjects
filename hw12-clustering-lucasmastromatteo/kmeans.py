"""
    This is a file you will have to fill in.
    It contains helper functions required by K-means method via iterative improvement
"""
import numpy as np
from random import sample

def init_centroids(k, inputs):
    """
    Selects k random rows from inputs and returns them as the chosen centroids
    Hint: use random.sample (it is already imported for you!)
    :param k: number of cluster centroids
    :param inputs: a 2D Numpy array, each row of which is one input
    :return: a Numpy array of k cluster centroids, one per row
    """
    # TODO
    # number_of_rows = inputs.shape[0]
    # random_indices = np.random.choice(number_of_rows, size=k, replace=False)
    # centroids = inputs[random_indices, :]
    indices = sample(list(range(len(inputs))),k)
    centroids = inputs[indices]
    return centroids

def assign_step(inputs, centroids):
    """
    Determines a centroid index for every row of the inputs using Euclidean Distance
    :param inputs: inputs of data, a 2D Numpy array
    :param centroids: a Numpy array of k current centroids
    :return: a Numpy array of centroid indices, one for each row of the inputs
    """
    # TODO
    def get_centroid(row):
        '''gets centroid value for one row'''
        dist = float('inf')
        centroid_id = None
        for i in range(len(centroids)):
            new_dist = np.linalg.norm(centroids[i] - row)
            if new_dist < dist:
                dist = new_dist
                centroid_id = i
        return centroid_id

    centroid_indices = np.apply_along_axis(get_centroid,1,inputs)
    return centroid_indices

def update_step(inputs, indices, k):
    """
    Computes the centroid for each cluster
    :param inputs: inputs of data, a 2D Numpy array
    :param indices: a Numpy array of centroid indices, one for each row of the inputs
    :param k: number of cluster centroids, an int
    :return: a Numpy array of k cluster centroids, one per row
    """
    # TODO
    centroids = np.zeros((k,len(inputs[0])))
    for i in range(k):
        cluster_inds = np.where(indices == i)[0]
        cluster = inputs[cluster_inds]
        centroid = np.mean(cluster,axis=0)
        centroids[i] = centroid
    return centroids

def kmeans(inputs, k, max_iter, tol):
    """
    Runs the K-means algorithm on n rows of inputs using k clusters via iterative improvement
    :param inputs: inputs of data, a 2D Numpy array
    :param k: number of cluster centroids, an int
    :param max_iter: the maximum number of times the algorithm can iterate trying to optimize the centroid values, an int
    :param tol: the tolerance we determine convergence with when compared to the ratio as stated on handout
    :return: a Numpy array of k cluster centroids, one per row
    """
    # TODO
    centroids = init_centroids(k,inputs)
    for i in range(max_iter):
        indices = assign_step(inputs,centroids)
        new_centroids = update_step(inputs,indices,k)
        norm_diff = np.apply_along_axis(np.linalg.norm,1,new_centroids-centroids)
        norms = norm_diff/(np.apply_along_axis(np.linalg.norm,1,centroids))
        if not np.any(norms > tol):
            return new_centroids
        else:
            centroids = new_centroids

    return centroids
