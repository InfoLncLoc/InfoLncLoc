from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from pylab import mpl
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import os
import matplotlib.pyplot as plt
from numpy import mat
from sklearn import datasets

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from pylab import mpl
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix

from sklearn.datasets import make_gaussian_quantiles
from imblearn.over_sampling import SMOTE

def getOriIndepend(filename):
    feature = pd.read_csv('./3.CellFeature/' + filename, header=0, low_memory=False, sep=',', index_col=0)
    label = pd.read_csv('./4.CellLabel/' + filename.split('.')[0] + '_label.csv', header=0, low_memory=False, sep=',', index_col=0).iloc[:,0:1]
    print(label)
    X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.17, random_state=50, shuffle=True, stratify=label)
    print(X_train)
    print(y_train)
    pd.DataFrame(X_train).to_csv('./8.SelfIndependTest/IndTrain/' + filename, index=True, header=True)
    pd.DataFrame(X_test).to_csv('./8.SelfIndependTest/IndTest/' + filename, index=True,header=True)
    pd.DataFrame(y_train).to_csv('./8.SelfIndependTest/IndTrainLabel/' + filename.split('.')[0] + '_label.csv', index=True, header=True)
    pd.DataFrame(y_test).to_csv('./8.SelfIndependTest/IndTestLabel/' + filename.split('.')[0] + '_label.csv', index=True,header=True)

   
if __name__ == "__main__":
    filelist = []
    for file in os.listdir('./6.endFeature/'):
        if os.path.splitext(file)[1] == '.csv':
            filelist.append(file)
    for i in filelist:
        getOriIndepend(i)


    
