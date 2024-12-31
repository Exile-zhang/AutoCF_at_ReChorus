import logging
import numpy as np
import pickle
from scipy.sparse import csr_matrix, coo_matrix
import pandas as pd
import time

# 时间转换函数，转换时间戳为时间并记录最小和最大年份
minn = 2022
maxx = 0

class DataProcessor:
    def __init__(self, train_file='train.csv', test_file='test.csv', dev_file='dev.csv'):
        self.train_file = train_file
        self.test_file = test_file
        self.dev_file = dev_file
        self.usrnum = 0
        self.itmnum = 0


    def combined_data(self):
        train_df = pd.read_csv(self.train_file, sep='\t')
        test_df = pd.read_csv(self.test_file, sep='\t')
        dev_df = pd.read_csv(self.dev_file, sep='\t')

        test_df = test_df.drop(columns=['neg_items'])
        dev_df = dev_df.drop(columns=['neg_items'])

        train_df = train_df[['user_id', 'item_id', 'time']]
        test_df = test_df[['user_id', 'item_id', 'time']]
        dev_df = dev_df[['user_id', 'item_id', 'time']]
        combined_df = pd.concat([train_df, test_df, dev_df], ignore_index=True)

        return combined_df

    def transTime(self, timeStamp):
        timeArr = time.localtime(timeStamp)
        year = timeArr.tm_year
        global minn
        global maxx
        minn = min(minn, year)
        maxx = max(maxx, year)
        return time.mktime(timeArr)

    def mapping(self, df):
        usrId = dict()
        itmId = dict()
        usrid, itmid = [0, 0]
        interaction = list()
        for idx, row in df.iterrows():
            row_val = row['user_id']
            col_val = row['item_id']
            timeStamp = self.transTime(int(row['time']))
            if timeStamp is None:
                continue
            if row_val not in usrId:
                usrId[row_val] = usrid
                interaction.append(dict())
                usrid += 1
            if col_val not in itmId:
                itmId[col_val] = itmid
                itmid += 1
            usr = usrId[row_val]
            itm = itmId[col_val]
            interaction[usr][itm] = timeStamp
        return interaction, usrid, itmid, usrId, itmId

    def remap_ids(self, interaction, usrnum, itmnum):
        usrIdMap = {old_id: new_id for new_id, old_id in enumerate(range(usrnum))}
        itmIdMap = {old_id: new_id for new_id, old_id in enumerate(range(itmnum))}
        remapped_interaction = [{} for _ in range(usrnum)]
        for old_usr, data in enumerate(interaction):
            new_usr = usrIdMap.get(old_usr, None)
            if new_usr is not None:
                for old_itm, value in data.items():
                    new_itm = itmIdMap.get(old_itm, None)
                    if new_itm is not None:
                        remapped_interaction[new_usr][new_itm] = value
        return remapped_interaction, usrIdMap, itmIdMap

    def split(self, interaction, usrnum, itmnum):
        pickNum = 10000
        usrPerm = np.random.permutation(usrnum)
        pickUsr = usrPerm[:pickNum]

        tstInt = [None] * usrnum
        exception = 0
        for usr in pickUsr:
            temp = list()
            data = interaction[usr]
            for itm in data:
                temp.append((itm, data[itm]))
            if len(temp) == 0:
                exception += 1
                continue
            temp.sort(key=lambda x: x[1])
            tstInt[usr] = temp[-1][0]
            interaction[usr][tstInt[usr]] = None
        return interaction, tstInt

    def trans(self, interaction, usrnum, itmnum):
        r, c, d = [list(), list(), list()]
        for usr in range(usrnum):
            if interaction[usr] is None:
                continue
            data = interaction[usr]
            for col in data:
                if data[col] is not None:
                    r.append(usr)
                    c.append(col)
                    d.append(data[col])
        intMat = csr_matrix((d, (r, c)), shape=(usrnum, itmnum))
        return intMat

    def prepare_interaction_matrix(self):

        combined_df = self.combined_data()
        interaction, usrnum, itmnum, usrId, itmId = self.mapping(combined_df)
        interaction, usrIdMap, itmIdMap = self.remap_ids(interaction, usrnum, itmnum)
        trnInt, tstInt = self.split(interaction, usrnum, itmnum)
        trn_mat = self.trans(trnInt, usrnum, itmnum)
        self.usrnum = usrnum
        self.itmnum = itmnum
        return trn_mat, tstInt

    def prepare_mat(self):
        trnMat, tstLst = self.prepare_interaction_matrix()
        trnMat = coo_matrix(trnMat)
        row = list(trnMat.row)
        col = list(trnMat.col)
        data = list(trnMat.data)

        for i in range(len(tstLst)):
            if tstLst[i] is not None:
                row.append(i)
                col.append(tstLst[i])
                data.append(1)

        row = np.array(row)
        col = np.array(col)
        data = np.array(data)

        leng = len(row)
        indices = np.random.permutation(leng)
        trn = int(leng * 0.7)
        val = int(leng * 0.75)

        trnIndices = indices[:trn]
        trnMat = coo_matrix((data[trnIndices], (row[trnIndices], col[trnIndices])), shape=[self.usrnum, self.itmnum])

        valIndices = indices[trn:val]
        valMat = coo_matrix((data[valIndices], (row[valIndices], col[valIndices])), shape=[self.usrnum, self.itmnum])

        tstIndices = indices[val:]
        tstMat = coo_matrix((data[tstIndices], (row[tstIndices], col[tstIndices])), shape=[self.usrnum, self.itmnum])

        with open('./trnMat.pkl', 'wb') as fs:
            pickle.dump(trnMat, fs)
        
        with open('./valMat.pkl', 'wb') as fs:
            pickle.dump(valMat, fs)
        
        with open('./tstMat.pkl', 'wb') as fs:
            pickle.dump(tstMat, fs)


processor = DataProcessor()

processor.prepare_mat()
