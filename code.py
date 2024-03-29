import matplotlib.pyplot as plt
import numpy as np

def main():
    process('dataset/ionosphere/ionosphere.data', 35, 'g', 'b', 16)
    process('dataset/connectionistBench/sonar.all-data', 61, 'R', 'M', 16)

def process(filename, col, a, b, batch):

    train, trainResult, test, testResult = prepare(filename, col, a, b)

    testAcc, trainAcc = [], []
    iteration = range(100)
    for i in iteration:
        betas = np.zeros((train.shape[1],1))
        betas, cost_history = gradientDescent(train, trainResult, betas, i, batch)
        trainAcc.append(accuracy(betas, train, trainResult))
        testAcc.append(accuracy(betas, test, testResult))

    plt.figure(figsize=(20,10))
    plt.suptitle(filename)
    plt.subplot(2, 1, 1)
    plt.ylabel("Accuracy")
    plt.title('Accuracy - Iteration')
    #plt.plot(iteration, trainAcc, label='train')
    plt.plot(iteration, testAcc, label='test')
    plt.legend(loc='lower right')
    plt.subplot(2, 1, 2)
    plt.ylabel("Cost")
    plt.xlabel("Iterations")
    plt.title('Cost - Iteration')
    plt.plot(cost_history, label='cost')
    plt.legend(loc='upper right')
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

def cost(data, dataResults, betas):
    predicts = hypothesis(data, betas)
    cost = dataResults * np.log(predicts) + (1-dataResults) * np.log(1-predicts)
    return cost.sum() / (-data.shape[0])

def updateBetas(data, dataResults, betas):
    predicts = hypothesis(data, betas)
    gradient = np.dot(np.transpose(data), predicts-dataResults)
    return betas-gradient*( 0.1 / data.shape[0] ) # learningRate is 0.1

def batchSplit(data, dataResults, batchSize):
    split, results = [], []
    count = data.shape[0] // batchSize
    for i in range(count):
        split.append(data[(i) * batchSize : (i+1) * batchSize, :])
        results.append(dataResults[(i) * batchSize : (i+1) * batchSize, :])
    return np.asarray(split), np.asarray(results), count

def gradientDescent(data, dataResults, betas, maxIterationNo, batch):
    cost_history = []
    dataPartition, results, batchCount = batchSplit(data, dataResults, batch)
    for i in range(maxIterationNo):
        for j in range(batchCount):
            betas = updateBetas(dataPartition[j], results[j], betas)
        cost_history.append(cost(data,dataResults, betas))
        if len(cost_history) > 2 and ((cost_history[-2] - cost_history[-1]) < 0.001): # treshold is 0.001
            break
    return betas, cost_history

sigmoid = lambda x : 1 / (1 + np.exp(-1 * x))
hypothesis = lambda data, betas : sigmoid(np.dot(data,betas))
accuracy = lambda beta, data, dataResults : np.sum((hypothesis(data, beta) > 0.5).astype(int) == dataResults) / dataResults.shape[0]

if __name__ == "__main__":
    main()
