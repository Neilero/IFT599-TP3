import pandas as pd
import numpy as np
from kmodes.kmodes import KModes    #https://github.com/nicodv/kmodes/
from sklearn.cluster import KMeans  #https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

pd.set_option('mode.chained_assignment', None)  # disable pandas' SettingWithCopy warning
COL_NAMES = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship",
             "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "target"]

def importData():

    train = pd.read_csv("adult.data", names=COL_NAMES, skipinitialspace=True)
    train = prepareData(train)

    test = pd.read_csv("adult.test", names=COL_NAMES, skipinitialspace=True, header=1)
    test = prepareData(test)

    return train, test

def prepareData(data):
    #remove "empty" columns
    data.drop(columns=["capital-gain", "capital-loss", "hours-per-week"], inplace=True)

    #categorize numerical columns
    categorizedData = categorizeData(data)

    return categorizedData

def categorizeData(data):
    data["age"] = categorizeVariable(data["age"], 10)
    data["fnlwgt"] = categorizeVariable(data["fnlwgt"], 4)
    data["education-num"] = categorizeVariable(data["education-num"], 6)

    return data

def categorizeVariable(variable, k_cluster):
    kmean = KMeans(n_clusters=k_cluster, verbose=False, n_jobs=-1)

    reshapedVariable = variable.values.reshape(-1, 1)
    categorizedVariable = kmean.fit_predict(reshapedVariable)
    return categorizedVariable

def initStatsList(kmodes):
    clusterIDs, stats = np.unique(kmodes.labels_, return_counts=True)

    # stat for each cluster = [trainCount, trainCount<=50K, trainCount>50K, clusterClass, trainingError, testCount, testCount<=50K, testCount>50K, testError]
    #                              0              1               2               3             4            5            6               7            8
    stats = np.hstack((np.reshape(stats, (-1, 1)), np.zeros((len(clusterIDs), 8)))).tolist()
    clustersStats = dict(zip(clusterIDs, stats))

    return clustersStats

def countTrainingData(clustersStats, trainData, trainLabels):
    for i, data in enumerate(trainData.values):
        if data[-1] == "<=50K":
            clustersStats[trainLabels[i]][1] += 1
        else:
            clustersStats[trainLabels[i]][2] += 1

def countTestingData(clustersStats, testData):
    for data in testData:
        clustersStats[data[1]][5] += 1

        if data[0] == "<=50K.":
            clustersStats[data[1]][6] += 1
        else:
            clustersStats[data[1]][7] += 1

def computeErrors(clustersStats, trainSize, testSize):
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

        if globalTestError < bestError:
            bestError = globalTestError
            bestK = k

    print("Best K found : {} with test error of {} %".format(bestK, bestError*100))
    return bestK

def greedyAlgorithm(k, currentVariables, trainData, testData, score_by_target = True):
    unwantedVariables = currentVariables + ["capital-gain", "capital-loss", "hours-per-week", "target"]
    variablesToTest = [var for var in COL_NAMES if var not in unwantedVariables]

    # stop condition
    if not variablesToTest:
        return currentVariables

    bestVariable = None
    bestVarError = float("inf")
    bestVarScore = float("inf")
    for variable in variablesToTest:
        trainingSet = currentVariables + [variable]
        testingSet = trainingSet + ["target"]

        kmodes = KModes(n_clusters=k, n_jobs=-1)
        kmodes.fit(trainData[trainingSet])

        # Compute stats
        globalTrainError, globalTestError = computeSats(kmodes, trainData, testData[testingSet])

        # Compute score
        varScore = score(trainData, kmodes.labels_)

        if score_by_target and globalTestError < bestVarError:  # use the target to find the best variable
            bestVariable = variable
            bestVarError = globalTestError
            bestVarScore = varScore
        elif not score_by_target and varScore < bestVarScore:   # use the score to find the best variable
            bestVariable = variable
            bestVarError = globalTestError
            bestVarScore = varScore

    currentVariables.append(bestVariable)
    print("Best set found : {} with test error of {:.3f} % and score of {}".format(currentVariables, bestVarError*100, bestVarScore))

    return greedyAlgorithm(k, currentVariables, trainData, testData)

def score(data, predictions):
    CF = np.zeros(predictions.max() +1)
    CfIdx = 0
    for _, cluster in data.groupby(predictions):
        clusterItems = [ zip(*np.unique(cluster[col], return_counts=True)) for col in cluster.columns ]
        dataItemCounts = [ dict(zip(*np.unique(data[col], return_counts=True))) for col in data.columns ]

        Z = lambda item, columnIndex: data.shape[0] - dataItemCounts[columnIndex][item] + 1
        clusterItemWeights = [ count**3 * Z(item, colIdx) for colIdx, col in enumerate(clusterItems) for item, count in col ]

        CF[CfIdx] = 1/cluster.shape[0] * np.sum(clusterItemWeights)
        CfIdx += 1

    return 1/(data.shape[0]**2) * np.sum( CF )

def main():
    trainData, testData = importData()

    # k = findGoodK(trainData, testData)
    k = 16

    greedyAlgorithm(k, [], trainData, testData, score_by_target=False)

if __name__ == '__main__':
    main()