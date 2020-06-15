import operator
import treePlotter
from math import log

#计算信息熵
def calcShannonEnt(dataSet):
    '''
    dataSet: 数据集
    '''
    numEntries = len(dataSet) # 数据集中实例的总数；
    labelCounts = {} # 每个标签对应的出现次数；

    # 计算数据集中各种标签各出现了多少次
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0 # 香农熵
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries # 某个标签出现的概率
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

def createDataSet():
    dataSet = [
        [1, 1, 'maybe'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]

    labels = ['no surfacing', 'flippers']
    return dataSet, labels

#按照给定特征划分数据集
def splitDataSet(dataSet, axis, value):
    '''
    dataSet: 待划分的数据集;
    axis: 划分数据集的特征;
    value: 特征的返回值;
    '''
    resDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # 在示例中除去当前用来被划分的特征
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            resDataSet.append(reduceFeatVec)

    return resDataSet

#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    '''
    dataSet: 待划分的数据集，必须是一种由列表元素组成的列表，而且所有
    的列表元素都要具有相同的数据长度；
    数据的最后一列或者每个实例的最后一个元素是当前实例的类别标签。无需
    限定list中的数据类型，它们既可以是数字也可以是字符串，并不影响实际
    计算。
    '''
    numFeatures = len(dataSet[0]) - 1 # 数据集总的特征数
    baseEntropy = calcShannonEnt(dataSet) # 划分之前的信息熵
    bestInfoGain = 0.0
    bestFeature = -1
    
    # 计算用不同特征来划分数据集的信息增益情况
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList) # 创建唯一的分类标签
        newEntropy = 0.0

        for value in uniqueVals:
            # 计算每种划分方式的信息熵
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / len(dataSet)
            newEntropy += prob * calcShannonEnt(subDataSet)
        
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    
    return bestFeature

def majorityCnt(classList):
    '''
    Function uses the list of class name，and create a data dictionary whose key value is the unique value
    in the classList. The dict object stores the appearance frequency of every class label. At last, using 
    operator module to sort the dictionary and return the most frequent class name.
    '''
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse = True)
        return sortedClassCount[0][0]

def createTree(dataSet, labels):
    '''
    Function to create the tree.
    '''
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        # Stop to split the dataset if all class are the same.
        return classList[0]
    if len(dataSet[0]) == 1:
        # 'len == 1' represents that all features has been tested.
        # Return the most appearance class if has traversed all features.
        return majorityCnt(classList)
    
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree, filename):
    import pickle
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)

def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)

if __name__ == "__main__":
    # test using lenses dataset.
    with open('lenses.txt') as fr:
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    print(lensesTree)
    treePlotter.createPlot(lensesTree)
    storeTree(lensesTree, 'classifierStorage.txt')

    # load tree
    # lensesTree = grabTree('classifierStorage.txt')
    # treePlotter.createPlot(lensesTree)


            
