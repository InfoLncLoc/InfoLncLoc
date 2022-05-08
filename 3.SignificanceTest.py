import scipy.stats as stats
import pandas as pd
import numpy as np
import os

from scipy.stats.stats import F_onewayBadInputSizesWarning

all_test = []

def mannWhitney(filename):
    cell = []
    print(filename)
    # 读取原始特征向量，正负样本分开
    n_feature = pd.read_csv("./5.NPSample/nSample/" + filename + "_nf.csv", header=0, low_memory=False, sep=',', index_col=0)
    p_feature = pd.read_csv("./5.NPSample/pSample/" + filename + "_pf.csv",header=0, low_memory=False, sep=',', index_col=0)
    # 划分离散型特征和连续型特征
    n_f_1 = n_feature.iloc[:, 0:61]
    n_f_2 = n_feature.iloc[:, 61:]
    p_f_1 = p_feature.iloc[:, 0:61]
    p_f_2 = p_feature.iloc[:, 61:]
    # 连续型特征进行mann-Whitney秩和检验
    coln_1 = []
    for i in n_f_1.columns:
        coln_1.append(i)
    for i in coln_1:
        n_arr = []
        p_arr = []
        for j in n_f_1[i]:
            n_arr.append(j)
        for j in p_f_1[i]:
            p_arr.append(j)
        _, pnorm = stats.mannwhitneyu(n_arr,p_arr,alternative='two-sided')
        cell.append(pnorm)
    # 离散型特征进行Fisher检验
    coln_2 = []
    for i in n_f_2.columns:
        coln_2.append(i)
    for i in coln_2:
        n_arr_2 = []
        p_arr_2 = []
        for j in n_f_2[i]:
            n_arr_2.append(j)
        for j in p_f_2[i]:
            p_arr_2.append(j)
        _, pnorm = stats.fisher_exact([[sum(n_arr_2), len(n_arr_2)-sum(n_arr_2)], [sum(p_arr_2), len(p_arr_2)-sum(p_arr_2)]])
        cell.append(pnorm)
    all_test.append(cell)
    
def testValue():
    testvalue = pd.read_csv("./5.NPSample/mannWhitneytest.csv", header=0, low_memory=False, sep=',', index_col=0)
    sample_name = []
    for i in testvalue.columns:
        sample_name.append(i)
    for i in sample_name:
        # 读取原始特征向量
        feature = pd.read_csv("./3.CellFeature/" + i + ".csv", header=0, low_memory=False, sep=',', index_col=0)
        col_name_1 = []
        for h in feature.columns:
            col_name_1.append(h)
        # 筛选p < 0.05的行
        t_0 = pd.DataFrame(testvalue[i].iloc[:len(col_name_1)])
        
        t_0.index = col_name_1
        t = t_0.loc[t_0[i]<0.05]
        col_name = []
        for j in t.index:
            col_name.append(j)
        

        feature_test = feature.loc[:,col_name]
        feature_test.to_csv('./6.endFeature/' + i + ".csv", index=True, header=True)



if __name__ == "__main__":
   
    filelist = []
    row_n = []
    col_n = []
    for file in os.listdir('./5.NPSample/nSample/'):
        if os.path.splitext(file)[1] == '.csv':
            filelist.append(file)
    print(filelist)
    for i in filelist:
        row_n.append(i.split("_")[0])
        mannWhitney(i.split("_")[0])
    all_test = pd.DataFrame(all_test)
    all_test.index = row_n
    
    z = pd.read_csv("./5.NPSample/nSample/A549_nf.csv", header=0, low_memory=False, sep=',', index_col=0)
    for i in z.columns:
        col_n.append(i)
   

    pd.DataFrame(all_test.T).to_csv('./5.NPSample/mannWhitneytest.csv', index=True, header=True)
    testValue()

        