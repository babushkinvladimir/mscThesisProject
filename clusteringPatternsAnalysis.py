""" module analyzing seasonal patterns in clusters
    author: Vladimir Babushkin

    Parameters
    ----------
    trainYear  : which year use for analyse
    region: NSW ot NYC
    isPrice: if False predicts load
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from loadData import loadData

# import errors
########################################################################################################################
# define the region NSW or NYC
region = 'NSW'

########################################################################################################################
# define what to predict -- price or load
isPrice = True

########################################################################################################################
# define results directory
directory = os.getcwd()

if region == 'NYC':
    if isPrice:
        datapathLoad = '/DATA/NYISO_PRICE'
    else:
        datapathLoad = '/DATA/NYISO'

if region == 'NSW':
    datapathLoad = '/DATA/NSW'

if isPrice:
    resultsDirectory = directory + "/results/clusteringPatternsPrice" + region + "/"
else:
    resultsDirectory = directory + "/results/clusteringPatternsLoad" + region + "/"

if not os.path.exists(resultsDirectory):
    os.makedirs(resultsDirectory)
########################################################################################################################
# load data

(datasetLoad, datesLoad) = loadData(datapathLoad, region, isPrice)
########################################################################################################################

trainYear = 2019

# figure parameters
w = 6
h = 8

trainDates = [d for d in datesLoad if d.year == trainYear]
trainDataset = [datasetLoad[i] for i in range(len(datesLoad)) if datesLoad[i].year == trainYear]

# # normalizing does not make huge difference in the clustering
# trainDataset = (trainDataset - np.min(trainDataset)) / (np.max(trainDataset) - np.min(trainDataset))

# perform kMeans clustering
from sklearn.cluster import KMeans

for numClusters in range(2, 8):
    kmeans = KMeans(n_clusters=numClusters, random_state=0).fit(trainDataset)
    # plot a heatmap shownig how many days fall for each cluster per month
    labels = kmeans.labels_
    heatmapData = np.zeros((12, numClusters))
    for k in range(len(trainDates)):
        for m in range(1, 13):
            if trainDates[k].month == m:
                for l in range(numClusters):
                    if labels[k] == l:
                        heatmapData[m - 1, l] = heatmapData[m - 1, l] + 1



    fig = plt.figure(figsize=(w, h))
    ax = fig.add_subplot(111)
    plt.imshow(heatmapData, cmap='hot', interpolation='nearest')
    yLabels = ['', 'Jan', '', 'Feb', '', 'Mar', '', 'Apr', '', 'May', '', 'Jun', '', 'Jul', '', 'Aug', '', 'Sep', '',
               'Oct', '', 'Nov', '', 'Dec']
    xLabels = [str(x + 1) for x in range(numClusters)]
    plt.yticks(np.arange(-0.5, 12, 0.5), yLabels, fontsize=15)
    plt.xticks(np.arange(0, numClusters, 1), xLabels, fontsize=15)
    plt.xlabel('cluster labels', fontsize=15)
    plt.title(" K-means, k = " + str(numClusters) + ", for " + str(trainYear), fontsize=16, y=1.03)
    plt.colorbar()
    plt.show()
    plt.savefig(resultsDirectory + "kmeans" + str(numClusters) + ".pdf",
                bbox_inches='tight',
                transparent=True,
                pad_inches=0)
########################################################################################################################
# perform kMedoids clustering
from _k_medoids import KMedoids

for numClusters in range(2, 8):
    kmedoids = KMedoids(n_clusters=numClusters, random_state=0).fit(trainDataset)

    # plot a heatmap shownig how many days fall for each cluster per month
    labels = kmedoids.labels_
    heatmapData = np.zeros((12, numClusters))
    for k in range(len(trainDates)):
        for m in range(1, 13):
            if trainDates[k].month == m:
                for l in range(numClusters):
                    if labels[k] == l:
                        heatmapData[m - 1, l] = heatmapData[m - 1, l] + 1



    fig = plt.figure(figsize=(w, h))
    ax = fig.add_subplot(111)
    plt.imshow(heatmapData, cmap='hot', interpolation='nearest')
    yLabels = ['', 'Jan', '', 'Feb', '', 'Mar', '', 'Apr', '', 'May', '', 'Jun', '', 'Jul', '', 'Aug', '', 'Sep', '',
               'Oct', '', 'Nov', '', 'Dec']
    xLabels = [str(x + 1) for x in range(numClusters)]
    plt.yticks(np.arange(-0.5, 12, 0.5), yLabels, fontsize=15)
    plt.xticks(np.arange(0, numClusters, 1), xLabels, fontsize=15)
    plt.xlabel('cluster labels', fontsize=15)
    plt.title(" K-medoids, k = " + str(numClusters) + ", for " + str(trainYear), fontsize=16, y=1.03)
    plt.colorbar()
    plt.show()
    plt.savefig(resultsDirectory + "kmedoids" + str(numClusters) + ".pdf",
                bbox_inches='tight',
                transparent=True,
                pad_inches=0)

########################################################################################################################
# perform Fuzzy C-means clustering
import skfuzzy as fuzz

for numClusters in range(2, 8):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(np.transpose(trainDataset), numClusters, 2, error=0.005,
                                                     maxiter=1000,
                                                     init=None)
    labels = np.argmax(u, axis=0)
    # plot a heatmap shownig how many days fall for each cluster per month
    heatmapData = np.zeros((12, numClusters))
    for k in range(len(trainDates)):
        for m in range(1, 13):
            if trainDates[k].month == m:
                for l in range(numClusters):
                    if labels[k] == l:
                        heatmapData[m - 1, l] = heatmapData[m - 1, l] + 1



    fig = plt.figure(figsize=(w, h))
    ax = fig.add_subplot(111)
    plt.imshow(heatmapData, cmap='hot', interpolation='nearest')
    yLabels = ['', 'Jan', '', 'Feb', '', 'Mar', '', 'Apr', '', 'May', '', 'Jun', '', 'Jul', '', 'Aug', '', 'Sep', '',
               'Oct', '', 'Nov', '', 'Dec']
    xLabels = [str(x + 1) for x in range(numClusters)]
    plt.yticks(np.arange(-0.5, 12, 0.5), yLabels, fontsize=15)
    plt.xticks(np.arange(0, numClusters, 1), xLabels, fontsize=15)
    plt.xlabel('cluster labels', fontsize=15)
    plt.title(" Fuzzy C-means, k = " + str(numClusters) + ", for " + str(trainYear), fontsize=16, y=1.03)
    plt.colorbar()
    plt.show()
    plt.savefig(resultsDirectory + "fuzzy" + str(numClusters) + ".pdf",
                bbox_inches='tight',
                transparent=True,
                pad_inches=0)
########################################################################################################################
# perform Hierarchical clustering
import numpy as np
from sklearn.cluster import AgglomerativeClustering

for numClusters in range(2, 8):

    # plot a heatmap shownig how many days fall for each cluster per month
    labels = AgglomerativeClustering(n_clusters=numClusters, affinity='euclidean', linkage='ward').fit_predict(
        trainDataset)

    heatmapData = np.zeros((12, numClusters))
    for k in range(len(trainDates)):
        for m in range(1, 13):
            if trainDates[k].month == m:
                for l in range(numClusters):
                    if labels[k] == l:
                        heatmapData[m - 1, l] = heatmapData[m - 1, l] + 1



    fig = plt.figure(figsize=(w, h))
    ax = fig.add_subplot(111)
    plt.imshow(heatmapData, cmap='hot', interpolation='nearest')
    yLabels = ['', 'Jan', '', 'Feb', '', 'Mar', '', 'Apr', '', 'May', '', 'Jun', '', 'Jul', '', 'Aug', '', 'Sep', '',
               'Oct', '', 'Nov', '', 'Dec']
    xLabels = [str(x + 1) for x in range(numClusters)]
    plt.yticks(np.arange(-0.5, 12, 0.5), yLabels, fontsize=15)
    plt.xticks(np.arange(0, numClusters, 1), xLabels, fontsize=15)
    plt.xlabel('cluster labels', fontsize=15)
    plt.title(" Hierarchical, k = " + str(numClusters) + ", for " + str(trainYear), fontsize=16, y=1.03)
    plt.colorbar()
    plt.show()
    plt.savefig(resultsDirectory + "hierarchical" + str(numClusters) + ".pdf",
                bbox_inches='tight',
                transparent=True,
                pad_inches=0)

########################################################################################################################
# perform SOM clustering
import numpy as np
from minisom import MiniSom

for numClusters in range(2, 8):

    som_shape = (1, numClusters)
    som = MiniSom(som_shape[0], som_shape[1], 24, sigma=.8, learning_rate=.5,
                  neighborhood_function='gaussian', random_seed=10)

    som.pca_weights_init(trainDataset)
    som.train_batch(trainDataset, 3000, verbose=True)

    # each neuron represents a cluster
    winner_coordinates = np.array([som.winner(x) for x in trainDataset]).T

    # get labels
    # with np.ravel_multi_index we convert the bidimensional
    # coordinates to a monodimensional index
    labels = np.ravel_multi_index(winner_coordinates, som_shape)

    heatmapData = np.zeros((12, numClusters))
    for k in range(len(trainDates)):
        for m in range(1, 13):
            if trainDates[k].month == m:
                for l in range(numClusters):
                    if labels[k] == l:
                        heatmapData[m - 1, l] = heatmapData[m - 1, l] + 1



    fig = plt.figure(figsize=(w, h))
    ax = fig.add_subplot(111)
    plt.imshow(heatmapData, cmap='hot', interpolation='nearest')
    yLabels = ['', 'Jan', '', 'Feb', '', 'Mar', '', 'Apr', '', 'May', '', 'Jun', '', 'Jul', '', 'Aug', '', 'Sep',
               '', 'Oct', '', 'Nov', '', 'Dec']
    xLabels = [str(x + 1) for x in range(numClusters)]
    plt.yticks(np.arange(-0.5, 12, 0.5), yLabels, fontsize=15)
    plt.xticks(np.arange(0, numClusters, 1), xLabels, fontsize=15)
    plt.xlabel('cluster labels', fontsize=15)
    plt.title(" SOM, k = " + str(numClusters) + ", for " + str(trainYear), fontsize=16, y=1.03)
    plt.colorbar()
    plt.show()
    plt.savefig(resultsDirectory + "som" + str(numClusters) + ".pdf",
                bbox_inches='tight',
                transparent=True,
                pad_inches=0)
