import pandas as pd
import os


def dividingCellLines(filename1, filename2):
    # 获取特征
    feature = filename1
    lncRNA_name = pd.read_csv("H://end_lncLocation//2.lncATLAs_label//" + filename2, header=0, low_memory=False, sep=',')
    lnc = []
    for i in lncRNA_name["ENSEMBL ID"]:
        lnc.append(i)
    cell_feature = feature.loc[feature['ID'].isin(lnc)]
    cell_feature = cell_feature.sort_values(by="ID", ascending=False)
    cell_feature.to_csv('./3.CellFeature/' + filename2.split("_")[0] + '.csv', index=False, header=True)
    label = []
    for i in cell_feature["ID"]:
        label.append(i)
    cell_label = lncRNA_name.loc[lncRNA_name["ENSEMBL ID"].isin(label)]
    cell_label = cell_label.sort_values(by="ENSEMBL ID", ascending=False)
    cell_label.to_csv('./4.CellLabel/' + filename2.split("_")[0] + '_label.csv', index=False, header=True)


if __name__ == "__main__":
    data = pd.read_csv("H://end_lncLocation//1.DataProcessing//result_02_oridata_Standardization.csv", header=0, low_memory=False, sep=',')
    filelist = []
    for file in os.listdir('./2.lncATLAs_label'):
        if os.path.splitext(file)[1] == '.csv':
            filelist.append(file)
    for i in filelist:
        dividingCellLines(data, i)