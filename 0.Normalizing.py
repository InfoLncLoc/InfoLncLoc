import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.impute import SimpleImputer


def delNone(filename):
    feature = (pd.read_csv(filename, header=0, low_memory=False, sep=',', index_col=0))
    row_n = feature.index
    row_c = feature.columns
    #  min-max标准化
    min_max_scaler = preprocessing.MinMaxScaler()
    x_minmax = min_max_scaler.fit_transform(np.array(feature))
    imp_zero = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)  # 实例化一个填充0的处理器
    x_minmax = imp_zero.fit_transform(np.array(x_minmax))
    x_minmax = pd.DataFrame(x_minmax)
    x_minmax.columns = row_c
    x_minmax.index = row_n
    x_minmax.to_csv('./1.DataProcessing/result_02_oridata_Standardization.csv', index=True, header=True)
    


if __name__ == "__main__":
    delNone('./1.DataProcessing/0.GENCODE_OriData.csv')
