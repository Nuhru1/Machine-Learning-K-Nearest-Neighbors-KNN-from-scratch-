# import necessary library
import csv
import operator
import random
import math


# this function will read the csv file containing the datast and split it into training and test set
# according to the percentage we want
def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open('iris.data', 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        
        for x in range(len(dataset)):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
                
                
#  This function compute the distance between 2 elements (As we know KNN use distance between elements for classification)                 
def euclideanDistance(instance1, instance2, length):
    
    distance = 0
    
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
        
    return math.sqrt(distance)
        
    
# This function will return the k neighbors based on the distance   
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance,trainingSet[x], len(testInstance)-1)
        distances.append((trainingSet[x] ,dist))
    distances.sort(key = operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
        
    return neighbors



# This function will take the k neighbors we got and return the class the new elements belongs to 
# it will assign the new element to class which have more elements near the new element.
def getResponse(neighbors):
    classVotes = {}
    
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] +=1
        else:
            classVotes[response] = 1
            
    sortedVotes = sorted(classVotes.items(), key = operator.itemgetter(1), reverse = True)
    return sortedVotes[0][0]
            
# This function the accuracy of our prediction     
def getAccuracy(testSet, predictions):
    count = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            count+=1
    return (count/len(testSet))*100.0



#-------------------------------------------------------------------
def main():
    trainingSet =[]
    testSet = []
    split = 0.75
    
    loadDataset('iris.data', split, trainingSet, testSet)
    
    predictions = []
    k = 3
    
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print("predicted : ", result, "   actual : ", testSet[x][-1])
        
    accuracy = getAccuracy(testSet, predictions)
    print("Accuracy", accuracy)
        
    
if __name__ == "__main__":
    main() 
