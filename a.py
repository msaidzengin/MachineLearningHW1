import matplotlib.pyplot as plt
import numpy as np

learningRate=0.1
iterationNo=500
batchSize=16
epsilon=0.1

def main():
    startProcessingData('dataset/ionosphere/ionosphere.data')
    plt.show()

def startProcessingData(filename):

    data = np.genfromtxt(filename, delimiter=',', dtype=float, encoding='utf8', converters={34: lambda x: float(1) if x == 'g' else float(0)})
    data = np.concatenate((np.ones((data.shape[0],1)), data), axis=1)
    trainingData, trainingDataResults, testData, testDataResults = shuffleAndSplitData(data, 0.2)

    accTest=np.zeros((iterationNo,1))
    iteration=np.zeros((iterationNo,1))
    accTraining=np.zeros((iterationNo,1))
    plt.figure(figsize=(20,10))
    plt.suptitle("Dataset: "+filename)
    plt.subplot(2, 1, 1)
    plt.title('Cost-Iteration')
    for i in range(1,iterationNo+1):
        betas = np.zeros((trainingData.shape[1],1))
        betas, cost_history = iterateGradientDescent(trainingData, trainingDataResults, betas, learningRate, i, batchSize, epsilon)
        accTest[i-1]=calculateAccuracy(betas, testData, testDataResults)
        accTraining[i-1]=calculateAccuracy(betas, trainingData, trainingDataResults)
        iteration[i-1]=i
    plt.plot(cost_history, label='cost-'+str(i))
    plt.subplot(2, 1, 2)
    plt.title('Accuracy-Iteration')
    plt.plot(iteration, accTest, label='Test Data')
    plt.plot(iteration, accTraining, label='Training Data')
    plt.legend(loc='lower right')

def shuffleAndSplitData(data, testDataRate):
    np.random.shuffle(data)

    testDataIndexStartNo=int(round(data.shape[0]*testDataRate))
    resultsColumnIndexNo=data.shape[1]-1

    testDataRaw=data[-testDataIndexStartNo:]
    testData=testDataRaw[:, 0:resultsColumnIndexNo]
    testDataResults=testDataRaw[:, resultsColumnIndexNo]
    testDataResults=np.reshape(testDataResults,(testDataResults.shape[0],1))
    
    trainingDataRaw=data[:data.shape[0]-testDataIndexStartNo]
    trainingData=trainingDataRaw[:, 0:resultsColumnIndexNo]
    trainingDataResults=trainingDataRaw[:, resultsColumnIndexNo]
    trainingDataResults=np.reshape(trainingDataResults,(trainingDataResults.shape[0],1))

    return trainingData, trainingDataResults, testData, testDataResults

def splitBatches(data, dataResults, batchSize):
    splitData=[]
    splitDataResults=[]
    batchCount=data.shape[0] // batchSize
    for i in range(batchCount):
        splitData.append(data[(i) * batchSize : (i+1) * batchSize, :])
        splitDataResults.append(dataResults[(i) * batchSize : (i+1) * batchSize, :])
    splitData=np.asarray(splitData)
    splitDataResults=np.asarray(splitDataResults)
    return splitData, splitDataResults, batchCount

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-1.0 * x))

def hypothesis(data, betas):
    return sigmoid(np.dot(data,betas))

def calculateCost(data, dataResults, betas):
    m=data.shape[0]
    predicts=hypothesis(data, betas)
    cost=dataResults*np.log(predicts) + (1-dataResults)*np.log(1-predicts)
    cost=cost.sum()/(-1*m)
    return cost

def updateBetas(data, dataResults, betas, learningRate):
    m=data.shape[0]
    predicts=hypothesis(data, betas)
    gradient=np.dot(np.transpose(data), predicts-dataResults)
    betas=betas-gradient*(learningRate/m)
    return betas

def iterateGradientDescent(data, dataResults, betas, learningRate, maxIterationNo, batchSize, epsilon):
    cost_history = []
    dataPartition, dataPartitionResults, batchCount= splitBatches(data, dataResults, batchSize)
    for i in range(maxIterationNo):
        for j in range(batchCount):
            betas = updateBetas(dataPartition[j], dataPartitionResults[j], betas, learningRate)
        cost=calculateCost(data,dataResults,betas)
        cost_history.append(cost)
        if len(cost_history)>2 and ((cost_history[-2]-cost_history[-1])<epsilon):
            break
    return betas, cost_history

def calculateAccuracy(beta, data, dataResults):
    results=(hypothesis(data, beta) > 0.5).astype(int)
    return np.sum(results==dataResults)/ dataResults.shape[0]

if __name__ == "__main__":
    main()