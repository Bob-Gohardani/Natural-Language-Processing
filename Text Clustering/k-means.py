import numpy as np
import pandas

# dataset has two features: how many packets sent per second / size of the packet
def load_dataset(name):
    return np.loadtxt(name)

# euclidian distance between two points
def euclidian(a, b):
    return np.linalg.norm(a-b)

# epsilon : max error that we can accept
def  kmeans(k, epsilon=0.01, distance="euclidian"):
    if distance == "euclidian":
        dist_method = euclidian
    # load dataset
    dataset = load_dataset('durudataset.txt')
    # get number of rows and columns(features) from dataset
    num_instances, num_features = dataset.shape
    # these are the initial centroids of the clusters k (each of them is an element that exists in our dataset)
    prototypes = dataset[np.random.randint(0, num_instances-1, size = k)]
    # save centroids of each iteration
    prototypes_old = np.zeros(prototypes.shape)
    # store cluster and which cluster each item belongs tp
    belongs_to = np.zeros((num_instances, 1))
    # get euclidian distance between current centroids and older version (if it doesnt change then we are at last iteration of k-means algorithm)
    norm = dist_method(prototypes, prototypes_old)
    iteration = 0

    while norm > epsilon: # until distance between 2 generation of centroids is bigger than margin error
        iteration += 1
        # get euclidian distance between current centroids and older version
        norm = dist_method(prototypes, prototypes_old)
        # loop through each element of the dataset
        for index_instance, instance in enumerate(dataset):
            # create a distance vector with K rows (columns are dstance between centroid and each element)
            dist_vec = np.zeros((k,1))
            # loop through all centroids and calculate distance between them and the element
            for index_prototype, prototype in enumerate(prototypes):
                dist_vec[index_prototype] = dist_method(prototype, instance)
            # then save which cluster the element belongs to
            belongs_to[index_instance, 0] = np.argmin(dist_vec)
        # this is just a buffer matrix to save the data before setting it to final centroids
        tmp_prototypes = np.zeros((k, num_features))

        # loop through all centroids
        for index in range(len(prototypes)):
            # find the index of all elements that belong to this cluster
            belong_to_this = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
            # get the averge position for all elements of this cluster and set it as new centroid
            prototype = np.mean(dataset[belong_to_this], axis=0)
            tmp_prototypes[index, :] = prototype

        prototypes = tmp_prototypes

    # at the end return final centroid and belongs_to matrices
    return prototypes, belongs_to

def execute():
    #train our model on the data and find the results
    centroids, belongs_to = kmeans(2)
    print(centroids)
    print("-----------------")
    print(belongs_to)


execute()
