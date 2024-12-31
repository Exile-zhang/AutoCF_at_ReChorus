import pickle
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import scipy.sparse as sp
import torch as t
import torch.utils.data as data
import torch.utils.data as dataloader
import logging
import os
import pandas as pd
from utils import utils
from helpers.BaseReader import BaseReader

class AutoCFReader(BaseReader):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        # Initialize file paths based on dataset
        if args.dataset == 'Grocery_and_Gourmet_Food':
            predir = 'data/Grocery_and_Gourmet_Food/'
        elif args.dataset == 'MINDTOPK':
            predir = 'data/MINDTOPK/'
        elif args.dataset == 'MovieLens-1M':
            predir = 'data/MovieLens-1M/'
        self.predir = predir
        self.trnfile = predir + 'trnMat.pkl'
        self.tstfile = predir + 'tstMat.pkl'
        self.trnMat = self.loadOneFile(self.trnfile)
        self.tstMat = self.loadOneFile(self.tstfile)
        # Get user and item counts
        self.n_users, self.n_items = self.trnMat.shape
        args.user, args.item = self.trnMat.shape

    def loadOneFile(self, filename):
        with open(filename, 'rb') as fs:
            ret = (pickle.load(fs) != 0).astype(np.float32)
        if type(ret) != coo_matrix:
            ret = sp.coo_matrix(ret)
        return ret

    def normalizeAdj(self, mat):
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def makeTorchAdj(self, mat):
        a = sp.csr_matrix((self.args.user, self.args.user))
        b = sp.csr_matrix((self.args.item, self.args.item))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self.normalizeAdj(mat)

        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape)

    def makeAllOne(self, torchAdj):
        idxs = torchAdj._indices()
        vals = t.ones_like(torchAdj._values())
        shape = torchAdj.shape
        return t.sparse.FloatTensor(idxs, vals, shape)

    def LoadData(self, args):
        trnMat = self.trnMat
        tstMat = self.tstMat
        args.user, args.item = trnMat.shape
        self.torchBiAdj = self.makeTorchAdj(trnMat)
        self.allOneAdj = self.makeAllOne(self.torchBiAdj)

        trnData = TrnData(trnMat)
        self.trnLoader = dataloader.DataLoader(trnData, batch_size=4096, shuffle=True, num_workers=0)
        tstData = TstData(tstMat, trnMat)
        self.tstLoader = dataloader.DataLoader(tstData, batch_size=256, shuffle=False, num_workers=0)


class TrnData(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)

    def negSampling(self, item_num):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                iNeg = np.random.randint(item_num)
                if (u, iNeg) not in self.dokmat:
                    break
            self.negs[i] = iNeg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]

class TstData(data.Dataset):
    def __init__(self, coomat, trnMat):
        coomat = coomat.tocoo()
        self.csrmat = (trnMat.tocsr() != 0) * 1.0

        tstLocs = [None] * coomat.shape[0]
        tstUsrs = set()
        for i in range(len(coomat.data)):
            row = coomat.row[i]
            col = coomat.col[i]
            if tstLocs[row] is None:
                tstLocs[row] = list()
            tstLocs[row].append(col)
            tstUsrs.add(row)
        tstUsrs = np.array(list(tstUsrs))
        self.tstUsrs = tstUsrs
        self.tstLocs = tstLocs

    def __len__(self):
        return len(self.tstUsrs)

    def __getitem__(self, idx):
        return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])
