# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys
#reload(sys)
#sys.setdefaultencoding("utf8")

from function import file2matrix    # 读取训练集中的数据
from function import autoNorm       # 归一化处理
from function import classify       # 分类
from numpy import array



if __name__ == '__main__':
    
    # 结果
    resultList = ['讨厌', '有些喜欢', '非常喜欢']
    
    # 三维用户特征输入
    percentTats = float(input("玩视频游戏所耗时间比: "))
    ffMiles = float(input("每年获得的飞行常客里程数: "))
    iceCream = float(input("每周消费的冰激凌公升数: "))
    
    # 打开的文件名
    fileName = "./data_set.txt"
    
    # 打开并处理数据
    dataMat, dataLabel = file2matrix(fileName)
    
    # 归一化
    normMat, ranges, minVals = autoNorm(dataMat)
    
    # 获取输入数据
    inArr = array([ffMiles, percentTats, iceCream])
    
    # 获取结果
    res = classify((inArr - minVals)/ranges, normMat, dataLabel, 3)
    
    print ("你对这个人的喜欢程度: ", resultList[res - 1])
    
    exit(0)


