import numpy as np
import pandas as pd
import os
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc
from scipy import interp
from numpy import mat
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix


def gbdt_modle(filename):
    feature = np.array(pd.read_csv('../6.endFeature/' + filename, header=None, low_memory=False, sep=',').iloc[1:, 1:])
    label = pd.read_csv('../4.CellLabel/' + filename.split('.')[0] + '_label.csv', header=None, low_memory=False, sep=',').iloc[1:, 1:2]
    y = []
    for i in label[1]:
        y.append(int(i))

    #  缺失值填充为 0
    imp_zero = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)  # 实例化一个填充0的处理器
    data = imp_zero.fit_transform(feature)
    #  SMOTE平衡样本
    smo = SMOTE(random_state=42)
    X_smo, y_smo = smo.fit_resample(data, y)

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    acc = []
    sp_c = []
    sn_c = []
    recall = []
    F1score = []
    pression = []
    tprs = []
    aucs = []
    mcc = []
    mean_fpr = np.linspace(0, 1, 100)
    data = X_smo
    label = y_smo

    for train_index, test_index in kf.split(data, label):
        # 建立模型，并对训练集进行测试，求出预测得分
        # 划分训练集和测试集
        X_train, X_test = data[train_index], data[test_index]
        Y_train, Y_test = np.array(label)[train_index], np.array(label)[test_index]
        model = XGBClassifier(learning_rate =0.001,
                                n_estimators=50, 
                                max_depth=6,
                                min_child_weight=5,
                                gamma=3,
                                subsample=0.7,
                                colsample_bytree=0.7,
                                objective= 'binary:logistic',
                                nthread=4,
                                seed=0,
                                eg_alpha=0.01, reg_lambda=0.01)
        model.fit(X_train, Y_train)

        # 利用model.predict获取测试集的预测值
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        matr = confusion_matrix(Y_test, y_pred)
        matr = mat(matr)
        tp = matr[0, 0]
        fn = matr[1, 0]
        fp = matr[0, 1]
        tn = matr[1, 1]
        sn = tp / (tp + fn)
        sp = tn / (fp + tn)
        if tp == 0 and fn == 0:
            sn = 0
        if fp == 0 and tn == 0:
            sp = 0
        
        # 计算fpr(假阳性率),tpr(真阳性率),thresholds(阈值)[绘制ROC曲线要用到这几个值]
        fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_pred_proba)
        # interp:插值, 把结果添加到tprs列表中
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = metrics.roc_auc_score(Y_test, y_pred_proba)
        aucs.append(roc_auc)
        sp_c.append(sp)
        sn_c.append(sn)
        acc.append(metrics.accuracy_score(Y_test, y_pred))
        recall.append(metrics.recall_score(Y_test, y_pred))
        F1score.append(metrics.f1_score(Y_test, y_pred))
        pression.append(metrics.precision_score(Y_test, y_pred))
        mcc.append(matthews_corrcoef(Y_test, y_pred))
        
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    # mean_fpr_save = pd.DataFrame(mean_fpr)
    # mean_tpr_save = pd.DataFrame(mean_tpr)
    # mean_fpr_save.to_csv("fpr.csv",sep = ",")
    # mean_tpr_save.to_csv("tpr.csv",sep = ",")
    mean_auc = auc(mean_fpr, mean_tpr)  # 计算平均AUC值
    mean_acc = np.mean(acc)
    mean_recall = np.mean(recall)
    mean_f1 = np.mean(F1score)
    mean_precesion = np.mean(pression)
    mean_mcc = np.mean(mcc)
    mean_sp = np.mean(sp_c)
    mean_sn = np.mean(sn_c)

    auc_all.append(mean_auc)
    acc_all.append(mean_acc)
    recall_all.append(mean_recall)
    F1score_all.append(mean_f1)
    pression_all.append(mean_precesion)
    mcc_all.append(mean_mcc)
    sp_all.append(mean_sp)
    sn_all.append(mean_sn)


if __name__ == '__main__':
    filelist = []
    for file in os.listdir('../6.endFeature/'):
        if os.path.splitext(file)[1] == '.csv':
            filelist.append(file)
    rowname = []
    auc_all = []
    acc_all = []
    recall_all = []
    F1score_all = []
    pression_all = []
    mcc_all = []
    sp_all = []
    sn_all = []
    for i in filelist:
        rowname.append(i.split(".")[0])
        gbdt_modle(i)
    asso_1 = pd.DataFrame()
    asso_1['name'] = rowname
    asso_1['auc'] = auc_all
    asso_1['acc'] = acc_all
    asso_1['Recall'] = recall_all
    asso_1['F1-score'] = F1score_all
    asso_1['precesion'] = pression_all
    asso_1['mcc'] = mcc_all
    asso_1['sp'] = sp_all
    asso_1['sn'] = sn_all
    asso_1.to_csv('./xgbscore.csv', index=False)
