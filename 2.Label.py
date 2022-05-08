import pandas as pd
import os


def dividingCellLines(filename):
    # 获取特征
    feature = pd.read_csv("H://end_lncLocation//3.CellFeature//" + filename.split("_")[0] + ".csv", header=0, low_memory=False, sep=',')
    label = pd.read_csv("H://end_lncLocation//4.CellLabel//" + filename, header=0, low_memory=False, sep=',')
    negative_label = label.loc[label['label'] == 0]
    positive_label = label.loc[label['label'] == 1]
    n_lnc = []
    p_lnc = []
    for i in negative_label["ENSEMBL ID"]:
        n_lnc.append(i)
    for i in positive_label["ENSEMBL ID"]:
        p_lnc.append(i)
    n_feature = feature.loc[feature["ID"].isin(n_lnc)]
    p_feature = feature.loc[feature["ID"].isin(p_lnc)]

    n_feature.to_csv('./5.NPSample/nSample/' + filename.split("_")[0] + '_nf.csv', index=False, header=True)
    p_feature.to_csv('./5.NPSample/pSample/' + filename.split("_")[0] + '_pf.csv', index=False, header=True)
    negative_label.to_csv('./5.NPSample/nLabel/' + filename.split("_")[0] + '_nl.csv', index=False, header=True)
    positive_label.to_csv('./5.NPSample/pLabel/' + filename.split("_")[0] + '_pl.csv', index=False, header=True)


if __name__ == "__main__":
   
    filelist = []
    for file in os.listdir('./4.CellLabel'):
        if os.path.splitext(file)[1] == '.csv':
            filelist.append(file)
    for i in filelist:
        dividingCellLines(i)