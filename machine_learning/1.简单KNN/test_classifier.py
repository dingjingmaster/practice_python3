# -*- coding: UTF-8 -*-

import numpy as np
import operator

'''
    创建数据集 二维， 以 打斗镜头的数量为 x 轴、 以接吻镜头为 y 轴
    判断属于哪种类型的电影
    思路:
        1. 特征 打斗镜头 和 接吻镜头
        2. 需要判断的结果 爱情片 还是 动作片
'''
# 创建数据集
def createDataSet():

    group = np.array([[1,101], [5,89], [108,5], [115,8]])

    labels = ['爱情片', '爱情片', '动作片', '动作片']

    return group, labels


# k-邻近算法
# inX 用于分类的数据(测试集)
# dataSet 用于训练的数据(训练集)
# labes 分类标签
# k 选择距离最小的 k 个点
# 分类结果: sortedClassCount[0][0]
def classify0(inX, dataSet, labels, k):
    '''
        inX [101, 20]
    '''
    # numpy 函数 shape[0] 返回 dataSet 的行数
    dataSetSize = dataSet.shape[0]                      #   dataSet

    print ("dataSetSize")
    print (dataSetSize)

    '''
        dataSetSize
        4
    '''

    # 在列向量方向上重复 inX 共 1 次（横向），行向量方向上重复 inX 共 dataSetSize 次（纵向）
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet

    '''
        行重复 dataSetSize 次， 列重复 1 次

        [   [101, 20]
            [101, 20]
            [101, 20]
            [101, 20]
        ]

        再与 dataSet 相减
        
    '''


    print ("diffMat")
    print (diffMat)

    '''
        diffMat 结果
            [ [100 -81]
              [ 96 -69]
              [ -7  15]
              [-14  12]
            ]
    '''

    # 二维特征相减后平方
    sqDiffMat = diffMat ** 2

    print ("sqDiffMat")
    print (sqDiffMat)

    '''
    [   [10000  6561]
        [ 9216  4761]
        [   49   225]
        [  196   144]
    ]

    '''

    # sum()所有元素相加， sum(0)列相加， sum(1)列相加
    sqDistance = sqDiffMat.sum(axis = 1)

    print ("sqDistance")
    print (sqDistance)
    '''
        sqDistance
        [16561 13977   274   340]
    '''

    # 开方，计算出距离
    distances = sqDistance ** 0.5

    # 返回 distances 中元素从小到大排序后的索引值
    sortedDistIndices = distances.argsort()

    # 定义一个记录类别次数的字典
    classCount = {}

    for i in range(k):
        # 取出前 k 个元素的类别
        voteIlabel = labels[sortedDistIndices[i]]

        # dict.get(key, default = None),字典的get()方法，返回指定键值，值不存在则返回默认值
        # 计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)

    # 返回次数最多的类别
    return sortedClassCount[0][0]

    pass

if __name__ == '__main__':

    # 创建数据集
    group, labels = createDataSet()

    # 打印数据集
    print (group)
    print (labels)

    # 测试集
    test = [101, 20]

    # knn 分类
    test_classify = classify0(test, group, labels, 3)

    # 打印分类结果
    print (test_classify)
