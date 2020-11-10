import matplotlib.pyplot as plt
import numpy as np

def main():
    process('dataset/ionosphere/ionosphere.data', 35, 'g', 'b', 8)
    process('dataset/connectionistBench/sonar.all-data', 61, 'R', 'M', 16)

def process(filename, col, a, b, batch):

    train, trainResult, test, testResult = prepare(filename, col, a, b)

    testAcc, trainAcc = [], []
    iteration = range(100)
    for i in iteration:
        betas = np.zeros((train.shape[1],1))
        betas, cost_history = iterateGradientDescent(train, trainResult, betas, i, batch)
        testAcc.append(calculateAccuracy(betas, test, testResult))
        trainAcc.append(calculateAccuracy(betas, train, trainResult))

    plt.figure(figsize=(20,10))
    plt.suptitle(filename)
    plt.subplot(2, 1, 1)
    plt.title('Cost-Iteration')
    plt.plot(cost_history, label='cost')
    plt.subplot(2, 1, 2)
    plt.title('Accuracy-Iteration')
    plt.plot(iteration, trainAcc, label='acc')
    plt.legend(loc='lower right')
    plt.show()

def prepare(filename, col, a, b):

    data = np.genfromtxt(filename, delimiter=',', dtype=float, encoding='utf8', converters={col-1: lambda x: float(1) if x == a else float(0)})
    data = np.concatenate((np.ones((data.shape[0],1)), data), axis=1)

    np.random.shuffle(data)
    trainLen = int(len(data) * 0.8)
    train, test = data[:trainLen], data[trainLen:]
    
    trainResult = train[:, col]
    train = train[:, :col]
    trainResult = np.reshape(trainResult,(trainResult.shape[0],1))

    testResult = test[:, col]
    test = test[:, :col]
    testResult = np.reshape(testResult,(testResult.shape[0],1))

    return train, trainResult, test, testResult

def splitBatches(data, dataResults, batchSize):
    split, results = [], []
    count = data.shape[0] // batchSize
    for i in range(count):
        split.append(data[(i) * batchSize : (i+1) * batchSize, :])
        results.append(dataResults[(i) * batchSize : (i+1) * batchSize, :])
    return np.asarray(split), np.asarray(results), count

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-1.0 * x))

def hypothesis(data, betas):
    return sigmoid(np.dot(data,betas))

def calculateCost(data, dataResults, betas):
    m = data.shape[0]
    predicts = hypothesis(data, betas)
    cost = dataResults * np.log(predicts) + (1-dataResults) * np.log(1-predicts)
    cost = cost.sum() / (-1*m)
    return cost

def updateBetas(data, dataResults, betas):
    m = data.shape[0]
    predicts = hypothesis(data, betas)
    gradient = np.dot(np.transpose(data), predicts-dataResults)
    betas = betas-gradient*( 0.1 / m ) # learningRate is 0.1
    return betas

def iterateGradientDescent(data, dataResults, betas, maxIterationNo, batch):
    cost_history = []
    dataPartition, results, batchCount= splitBatches(data, dataResults, batch)
    for i in range(maxIterationNo):
        for j in range(batchCount):
            betas = updateBetas(dataPartition[j], results[j], betas)
        cost_history.append(calculateCost(data,dataResults,betas))
        if len(cost_history)>2 and ((cost_history[-2]-cost_history[-1]) < 0.001): # treshold is 0.001
            break
    return betas, cost_history

def calculateAccuracy(beta, data, dataResults):
    results=(hypothesis(data, beta) > 0.5).astype(int)
    return np.sum(results==dataResults)/ dataResults.shape[0]

if __name__ == "__main__":
    main()
