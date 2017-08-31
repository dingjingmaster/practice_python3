# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 17:34:33 2017

@author: DingJing
"""
import sys
#reload(sys)
#sys.setdefaultencoding("utf8")

import numpy as np

def file2matrix(fileName):
    
    # 打开文件
    fRead = open(fileName)
    
    # 读取所有文件内容
    arrayLines = fRead.readlines()
    
    # 获取文件行数 1000 行
    lineNum = len(arrayLines)
    
    # 返回 numpy 矩阵， lineNum 行 3 列
    matArray = np.zeros((lineNum, 3))
    
    # 返回分类标签向量
    classLabelVector = []
    
    # 行的索引值
    index = 0
    
    for line in arrayLines:
        line = line.strip()
        
        # 以 '\t' 切割
        lineArray = line.split('\t')
        
        # 前三维是特征
        matArray[index,:] = lineArray[0:3]
        
        # 根据喜欢程度进行分类 1代表不喜欢， 2代表一般， 3代表非常喜欢
        if lineArray[-1] == 'didntLike':
            classLabelVector.append(1)
        elif lineArray[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif lineArray[-1] == 'largeDoses':
            classLabelVector.append(3)
        
        index += 1
    
    return (matArray, classLabelVector)
    

# 训练集归一化
def autoNorm(dataSet):
    
    # 获取数据的最值
    minVal = dataSet.min(0)
    maxVal = dataSet.max(0)
    
    # 获取最大值和最小值的范围
    ranges = maxVal - minVal
    
    # 返回 dataSet 的矩阵行列数
    normDataSet = np.zeros(np.shape(dataSet))
    
    # 返回 dataSet 的行数
    m = dataSet.shape[0]
    
    # 原是指减去最小值 --- 归一化
    normDataSet = dataSet - np.tile(minVal, (m, 1))
    
    # 除以最大值与最小值的差 --- 归一化
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    
    # 返回 归一化数据结果， 数据范围， 最小值
    return (normDataSet, ranges, minVal)

# KNN 算法
def classify(input, dataSet, label, k):
    
    dataSize = dataSet.shape[0]
    
    # 计算欧氏距离
    diff = np.tile(input, (dataSize, 1)) - dataSet
    sqDiff = diff ** 2
    
    # 行向量分别相加
    squareDist = np.sum(sqDiff, axis = 1)
    dist = squareDist ** 0.5
    
    # 距离排序
    sortDisIndex = np.argsort(dist)
    
    classCount = {}
    
    for i in range(k):
        voteLabel = label[sortDisIndex[i]]
        # 对选取的 K 个样本所属类别个数进行统计
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
        
    # 选取出现类别次数最多的类别
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            classes = key
            
    return classes





















