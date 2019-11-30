import pandas as pd
import numpy as np
from kmodes.kmodes import KModes    #https://github.com/nicodv/kmodes/
from sklearn.cluster import KMeans  #https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

pd.set_option('mode.chained_assignment', None)  # disable pandas' SettingWithCopy warning
COL_NAMES = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship",
             "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "target"]

def importData():
    """
    import data from files and prepare them for analyse
    :return: training and testing DataFrame with only categorical data
    """
    train = pd.read_csv("adult.data", names=COL_NAMES, skipinitialspace=True)
    train = prepareData(train)

    test = pd.read_csv("adult.test", names=COL_NAMES, skipinitialspace=True, header=1)
    test = prepareData(test)

    return train, test

def prepareData(data):
    """
    prepare data for categorical clustering
    remove undesired variables (i.e. variable with a lot of unknown value
    :param data: the DataFrame to prepare
    :return: the DataFrame edited and ready for categorical clustering
    """
    #remove "empty" columns
    data.drop(columns=["capital-gain", "capital-loss"], inplace=True)

    #categorize numerical columns
    categorizedData = categorizeData(data)

    return categorizedData

def categorizeData(data):
    """
    Categorize the numerical variables
    :param data: a DataFrame with numerical variables
    :return: a DataFrame with only categorical variables
    """
    data["age"] = categorizeVariable(data["age"], 10)
    data["fnlwgt"] = categorizeVariable(data["fnlwgt"], 4)
    data["education-num"] = categorizeVariable(data["education-num"], 6)
    data["hours-per-week"] = categorizeVariable(data["hours-per-week"], 5)

    return data

def categorizeVariable(variable, k_cluster):
    """
    Categorize a numerical variable with the K-mean algorithm and the given cluster number
    :param variable: the numerical variable
    :param k_cluster: the number of wanted cluster for the K-mean algorithm
    :return: the categorized variable
    """
    kmean = KMeans(n_clusters=k_cluster, verbose=False, n_jobs=-1)

    reshapedVariable = variable.values.reshape(-1, 1)
    categorizedVariable = kmean.fit_predict(reshapedVariable)
    return categorizedVariable

def initStatsList(kmodes):
    """
    Initialize a list for computing stats with zeros except for the first column containing the
    number of training elements for each cluster
    :param kmodes: the K-modes object used for the clustering
    :return: a list of size (cluster number, 9) with only the first column computed (the others are zeros)
    """
    clusterIDs, stats = np.unique(kmodes.labels_, return_counts=True)

    # stat for each cluster = [trainCount, trainCount<=50K, trainCount>50K, clusterClass, trainingError, testCount, testCount<=50K, testCount>50K, testError]
    #                              0              1               2               3             4            5            6               7            8
    stats = np.hstack((np.reshape(stats, (-1, 1)), np.zeros((len(clusterIDs), 8)))).tolist()
    clustersStats = dict(zip(clusterIDs, stats))

    return clustersStats

def countTrainingData(clustersStats, trainData, trainLabels):
    """
    Count the number of training data of each target class for each cluster
    and save the result in the given clustersStats matrix
    :param clustersStats: the stats matrix where to save the result
    :param trainData: the training DataFrame
    :param trainLabels: the training labels (or predictions for the clustering)
    """
    for i, data in enumerate(trainData.values):
        if data[-1] == "<=50K":
            clustersStats[trainLabels[i]][1] += 1
        else:
            clustersStats[trainLabels[i]][2] += 1

def countTestingData(clustersStats, testData):
    """
    Count the number of testing data of each target class for each cluster
    and save the result in the given clustersStats matrix
    The given testData is supposed to be a matrix of size (2, number of testing data)
    and with the first column being the targets and the second the prediction of the clustering
    :param clustersStats: the stats matrix where to save the result
    :param testData: the test data matrix with the target and the predictions
    """
    for data in testData:
        clustersStats[data[1]][5] += 1

        if data[0] == "<=50K.":
            clustersStats[data[1]][6] += 1
        else:
            clustersStats[data[1]][7] += 1

def computeErrors(clustersStats, trainSize, testSize):
    """
    Compute the training and testing errors for each clusters and save it in the given stats matrix
    Returns the global training and testing errors
    :param clustersStats: the stats matrix where to save the result
    :param trainSize: the number of training data
    :param testSize: the number of testing data
    :return: the global training and testing errors
    """
    globalTrainError = 0
    globalTestError = 0
    for stats in clustersStats.values():
        stats[3] = "<=50K" if np.argmax(stats[1:3]) == 0 else ">50K"

        if stats[3] == "<=50K":
            stats[4] = stats[2] / stats[0]
            stats[8] = stats[7] / stats[5] if stats[5] > 0 else 0
        else:
            stats[4] = stats[1] / stats[0]
            stats[8] = stats[6] / stats[5] if stats[5] > 0 else 0

        # print("Cluster stats : ", stats)
        globalTrainError += stats[4] * (stats[0] / trainSize)
        globalTestError += stats[8] * (stats[5] / testSize)

    return globalTrainError, globalTestError

def computeSats(kmodes, trainData, testData):
    """
    Compute and returns the global training and testing errors
    :param kmodes: the K-modes object used for the clustering
    :param trainData: the training DataFrame
    :param testData: the testing DataFrame
    :return: the global training and testing errors
    """
    # testing
    testData["prediction"] = kmodes.predict(testData.drop(columns="target"))

    # compute statistics
    clustersStats = initStatsList(kmodes)
    countTrainingData(clustersStats, trainData, kmodes.labels_)
    countTestingData(clustersStats, testData[["target", "prediction"]].values)
    globalTrainError, globalTestError = computeErrors(clustersStats, trainData.shape[0], testData.shape[0])

    # dropping predictions after use to not influence next iteration
    testData.drop(columns="prediction", inplace=True)

    return globalTrainError, globalTestError

def findGoodK(trainData, testData):
    """
    Search the best K (number of cluster) for the K-modes algorithm between 2 and 75
    The search is done with a 20-long sample logarithmically distributed
    :param trainData: the training DataFrame
    :param testData: the testing DataFrame used to compare the K values
    :return: the best found K between 2 and 75
    """
    bestK = None
    bestError = float("inf")

    for k in np.unique(np.geomspace(2, 75, 20, dtype=int)):
        print("--- Computing clustering with {} clusters ---".format(k))

        # training
        kmodes = KModes(n_clusters=k, n_jobs=-1) #https://github.com/nicodv/kmodes/blob/master/kmodes/kmodes.py#L294
        kmodes.fit(trainData.drop(columns="target"))

        # Compute stats
        globalTrainError, globalTestError = computeSats(kmodes, trainData, testData)

        print("Global training error : {0:.2f} %".format(globalTrainError*100))
        print("Global testing  error : {0:.2f} %\n".format(globalTestError*100))

        kerror = (globalTrainError + globalTestError) / 2
        if kerror < bestError:
            bestError = kerror
            bestK = k

    print("Best K found : {} with test error of {} %".format(bestK, bestError*100))
    return bestK

def score(data, predictions):
    """
    The scoring function used to compare different clustering based on the importance of each cluster's elements
    This method does not use the targets of the data
    :param data: the DataFrame containing the data for which to compute the score
    :param predictions: the predictions for the data of the given DataFrame of the clustering to score
    :return: the score of the given data clustered according to the given predictions
    """
    CF = np.zeros(predictions.max() +1)
    CfIdx = 0

    dataItemCounts = [ dict(zip(*np.unique(data[col], return_counts=True))) for col in data.columns ]

    for _, cluster in data.groupby(predictions):
        clusterItems = [ zip(*np.unique(cluster[col], return_counts=True)) for col in cluster.columns ]

        Z = lambda item, columnIndex: data.shape[0] - dataItemCounts[columnIndex][item] + 1
        clusterItemWeights = [ count**3 * Z(item, colIdx) for colIdx, col in enumerate(clusterItems) for item, count in col ]

        CF[CfIdx] = 1/cluster.shape[0] * np.sum(clusterItemWeights)
        CfIdx += 1

    return 1/(data.shape[0]**2) * np.sum( CF )

def greedyAlgorithm(k, trainData, testData, currentVariables=None, score_by_target=True):
    """
    Implementation of the greedy algorithm. The best variables are found using either the test error
    or the score according to the score_by_target parameter and considering the given k (number of cluster).
    :param k: the number of cluster for the K-modes algorithm
    :param trainData: the training DataFrame
    :param testData: the testing DataFrame
    :param currentVariables: the currently selected variables. Set it to None for the first iteration
    :param score_by_target: if True the best variable will be find via global test error, if False, by score
    :return: The list of variables in order of finding throughout the algorithm
    """
    if currentVariables is None:
        currentVariables = []
    unwantedVariables = currentVariables + ["capital-gain", "capital-loss", "target"]
    variablesToTest = [var for var in COL_NAMES if var not in unwantedVariables]

    # stop condition
    if not variablesToTest:
        return currentVariables

    bestVariable = None
    bestVarErrorTrain = float("inf")
    bestVarErrorTest = float("inf")
    bestVarScore = float("inf")
    for variable in variablesToTest:
        trainingSet = currentVariables + [variable]
        testingSet = trainingSet + ["target"]

        kmodes = KModes(n_clusters=k, n_jobs=-1)
        kmodes.fit(trainData[trainingSet])

        # Compute stats
        globalTrainError, globalTestError = computeSats(kmodes, trainData, testData[testingSet])

        # Compute score
        varScore = score(trainData.drop(columns="target"), kmodes.labels_)

        varError = (globalTrainError + globalTestError) / 2
        bestError = (bestVarErrorTrain + bestVarErrorTest) / 2
        if score_by_target and varError < bestError:  # use the target to find the best variable
            bestVariable = variable
            bestVarErrorTrain = globalTrainError
            bestVarErrorTest = globalTestError
            bestVarScore = varScore
        elif not score_by_target and varScore < bestVarScore:   # use the score to find the best variable
            bestVariable = variable
            bestVarErrorTrain = globalTrainError
            bestVarErrorTest = globalTestError
            bestVarScore = varScore

    currentVariables.append(bestVariable)
    print("Best set found : {} with test error of {:.3f} %, train error of {:.3f} and score of {:.3f}".format(
        currentVariables, bestVarErrorTest*100, bestVarErrorTrain*100, bestVarScore)
    )

    return greedyAlgorithm(k, trainData, testData, currentVariables, score_by_target)

def main():
    trainData, testData = importData()

    k = findGoodK(trainData, testData)
    # k = 51

    greedyAlgorithm(k, trainData, testData, score_by_target=False)

if __name__ == '__main__':
    main()
