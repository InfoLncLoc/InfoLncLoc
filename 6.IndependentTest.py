
import numpy as np
import pandas as pd
import os
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn import metrics
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier


def tencross(filename):
    data = np.array(pd.read_csv('./8.SelfIndependTest/IndTrain/' + filename, header=None, low_memory=False, sep=',').iloc[1:, 1:])
    label = pd.read_csv('./8.SelfIndependTest/IndTrainLabel/' + filename.split('.')[0] + '_label.csv', header=None, low_memory=False, sep=',').iloc[1:, 1:2]
    imp_zero = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)  # 实例化一个填充0的处理器
    data = imp_zero.fit_transform(data)
    #  SMOTE平衡样本
    smo = SMOTE(random_state=42)
    data, label = smo.fit_resample(data, label)
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    for train_index, test_index in kf.split(data, label):
        X_train, X_test = data[train_index], data[test_index]
        Y_train, Y_test = np.array(label)[train_index], np.array(label)[test_index]
        model_gbdt = lgb.LGBMClassifier(objective='binary', learning_rate=0.1, n_estimators=50, subsample=0.8, max_depth=5, num_leaves=20, colsample_bytree=0.8, min_child_samples=20, min_child_weight=0.0005, feature_fraction=0.5, bagging_fraction=0.8)
        model_gbdt.fit(X_train, Y_train)
        y_pred_proba_gbdt = model_gbdt.predict_proba(X_test)[:, 1]
        for i in Y_test:
            y_label_gbdt.append(int(i))
        for i in y_pred_proba_gbdt:
            y_pred_gbdt.append(i)


def oriData_modle(filename):
    X_train = np.array(pd.read_csv('./8.SelfIndependTest/IndTrain/' + filename, header=None, low_memory=False, sep=',').iloc[1:, 1:])
    train_label = pd.read_csv('./8.SelfIndependTest/IndTrainLabel/' + filename.split('.')[0] + '_label.csv', header=None, low_memory=False, sep=',').iloc[1:, 1:2]
    y_train = []
    imp_zero = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)  # 实例化一个填充0的处理器
    data = imp_zero.fit_transform(X_train)
    for i in train_label[1]:
        y_train.append(int(i))
    smo = SMOTE(random_state=42)
    X_smo, y_smo = smo.fit_resample(data, train_label)
    X_test = np.array(pd.read_csv('./8.SelfIndependTest/IndTest/' + filename, header=None, low_memory=False, sep=',').iloc[1:, 1:])
    X_test = imp_zero.fit_transform(X_test)
    test_label = pd.read_csv('./8.SelfIndependTest/IndTestLabel/' + filename.split('.')[0] + '_label.csv', header=None, low_memory=False, sep=',').iloc[1:, 1:2]
    y_test = []
    for i in test_label[1]:
        y_test.append(int(i))
    model = lgb.LGBMClassifier(objective='binary', learning_rate=0.1, n_estimators=50, subsample=0.8, max_depth=5, num_leaves=20, colsample_bytree=0.8, min_child_samples=20, min_child_weight=0.0005, feature_fraction=0.5, bagging_fraction=0.8)
    model.fit(X_smo, y_smo)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    for i in y_pred_proba:
        y_pred_ori.append(i)
    for i in y_test:
        y_label_ori.append(int(i))




def plotRoc():
    plt.figure(dpi=300)
    plt.plot([0, 1], [0, 1], 'k--')
    fpr_ori, tpr_ori, thresholds_ori = metrics.roc_curve(y_label_ori, y_pred_ori)
    roc_auc_ori = metrics.roc_auc_score(y_label_ori, y_pred_ori)

    fpr_gbdt, tpr_gbdt, thresholds_gbdt = metrics.roc_curve(y_label_gbdt, y_pred_gbdt)
    roc_auc_gbdt = metrics.roc_auc_score(y_label_gbdt, y_pred_gbdt)

    plt.plot(fpr_ori, tpr_ori, label='Independent test dataset (AUC = %0.3f)' % roc_auc_ori)
    #plt.plot(fpr_ind, tpr_ind, label='IndependData2 (AUC = %0.3f)' % roc_auc_ind)
    plt.plot(fpr_gbdt, tpr_gbdt, label='Training dataset (AUC = 0.935)')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig("Datacompare.tiff")
    plt.savefig("Datacompare.pdf")
    plt.savefig("Datacompare.png")
    plt.show()


if __name__ == "__main__":
    y_pred_ori = []
    y_label_ori = []
    y_pred_independ = []
    y_label_independ = []
    y_pred_gbdt = []
    y_label_gbdt = []

    filelist1 = []
    for file in os.listdir('./8.SelfIndependTest/IndTrain'):
        if os.path.splitext(file)[1] == '.csv':
            filelist1.append(file)
    for item in filelist1:
        oriData_modle(item)

   
    
    filelist3 = []
    for file in os.listdir('./6.endFeature/'):
        if os.path.splitext(file)[1] == '.csv':
            filelist3.append(file)
    for i in filelist3:
        tencross(i)

    plotRoc()