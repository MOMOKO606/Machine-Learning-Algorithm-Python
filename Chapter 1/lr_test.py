# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 11:54:38 2017

@author: bianl
"""

import numpy as np
from Logistic_Regression_train import sig


def load_weight(w):
    '''
    导入LR模型
    input:w(string):权重文件
    output:np.mat(w)(mat):权重矩阵
    '''
    f=open(w)
    w=[]
    for line in f.readlines():
        lines=line.strip().split("\t")
        w_tmp=[]
        for x in lines:
            w_tmp.append(float(x))
        w.append(w_tmp)
    f.close()
    return np.mat(w)


def load_data(file_name,n):
    '''
    导入测试数据
    input:file_name(string):测试集文件
          n(int):# of features
    output:np.mat(feature_data)(mat):测试集的特征
    '''
    f=open(file_name)
    feature_data=[]
    for line in f.readlines():
        feature_tmp=[]
        lines=line.strip().split("\t")
        #  丢弃不符合要求的数据
        if len(lines) <> n-1:  
            continue
        feature_tmp.append(1)
        for x in lines:
            feature_tmp.append(float(x))
        feature_data.append(feature_tmp)
    f.close()
    return np.mat(feature_data)
        

def predict(data,w):
    '''
    对测试数据进行预测
    input:data(mat):测试数据的特征
          w(mat):模型的参数
    output:h(mat)最终的预测结果
    '''
    h=sig(data*w.T)  # 取得sigmoid值
    m=np.shape(h)[0]
    for i in xrange(m):
        if h[i,0]<0.5:
            h[i,0]=0.0
        else:
            h[i,0]=1.0
    return h
                         

def save_result(file_name,result):
    '''
    保存最终的预测结果
    input:file_name(string):结果文件名
          result(mat):预测的结果
    '''
    m=np.shape(result)[0]
    #  输出预测结果到文件
    tmp=[]
    for i in xrange(m):
        tmp.append(str(h[i,0]))
    f_result=open(file_name,"w")
    f_result.write("\t".join(tmp))
    f_result.close()


if __name__ == "__main__":
    #  1.导入LR模型
    print "---------1.load model------"
    w=load_weight("weights")
    n=np.shape(w)[1]
    #  2.导入测试数据
    print "---------1.load data------"
    testData=load_data("test_data",n)
    #  3.对测试数据进行预测
    print "---------3.get prediction------"
    h=predict(testData,w)
    #  4.保存最终的预测结果
    print "---------4.save prediction------"
    save_result("result",h)
    
    