import os
import gc
import torch
import torch.nn as nn
import logging
import numpy as np
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch as t
import pickle
from utils import utils
from utils.utils import calcRegLoss, contrast
from models.BaseModel import BaseModel
from models.general.AutoCF import AutoCF
from helpers.BaseRunner import BaseRunner
from helpers.AutoCFReader import AutoCFReader
from typing import Dict, List



class AutoCFRunner(BaseRunner):
    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument('--epoch', type=int, default=100,
                            help='Number of epochs.')
        parser.add_argument('--check_epoch', type=int, default=1,
                            help='Check some tensors every check_epoch.')
        parser.add_argument('--test_epoch', type=int, default=-1,
                            help='Print test results every test_epoch (-1 means no print).')
        parser.add_argument('--early_stop', type=int, default=10,
                            help='The number of epochs when dev results drop continuously.')
        parser.add_argument('--lr', type=float, default=1e-3,
                            help='Learning rate.')
        parser.add_argument('--l2', type=float, default=0,
                            help='Weight decay in optimizer.')
        parser.add_argument('--batch_size', type=int, default=4096,
                            help='Batch size during training.')
        parser.add_argument('--eval_batch_size', type=int, default=256,
                            help='Batch size during testing.')
        parser.add_argument('--optimizer', type=str, default='Adam',
                            help='optimizer: SGD, Adam, Adagrad, Adadelta')
        parser.add_argument('--num_workers', type=int, default=0,
                            help='Number of processors when prepare batches in DataLoader')
        parser.add_argument('--pin_memory', type=int, default=0,
                            help='pin_memory in DataLoader')
        parser.add_argument('--topk', type=str, default='5,10,20,50',
                            help='The number of items recommended to each user.')
        parser.add_argument('--metric', type=str, default='NDCG,HR',
                            help='metrics: NDCG, HR')
        parser.add_argument('--main_metric', type=str, default='',
                            help='Main metric to determine the best model.')
        parser.add_argument('--fixSteps', default=10, type=int, help='steps to train on the same sampled graph')
        parser.add_argument('--ssl_reg', default=1, type=float, help='contrastive regularizer')
        parser.add_argument('--tstBat', default=256, type=int, help='number of users in a testing batch')
        parser.add_argument('--save_path', default='tem', help='file name to save model and training record')
        return parser



    def __init__(self, args, handler):
        self.args = args
        self.handler = handler
        self.epoch = args.epoch
        self.check_epoch = args.check_epoch
        self.test_epoch = args.test_epoch
        self.early_stop = args.early_stop
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.l2 = args.l2
        self.optimizer_name = args.optimizer
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.topk = [int(x) for x in args.topk.split(',')]
        self.metrics = [m.strip().upper() for m in args.metric.split(',')]
        self.main_metric = '{}@{}'.format(self.metrics[0], self.topk[0]) if not len(args.main_metric) else args.main_metric  # early stop based on main_metric
        self.main_topk = int(self.main_metric.split("@")[1])
        self.fixSteps = args.fixSteps
        self.ssl_reg = args.ssl_reg
        self.time = None
        self.model = None
        self.opt = None
        self.masker = None
        self.sampler = None



    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, self.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret



    def trainEpoch(self, corpus):
        trnLoader = self.handler.trnLoader
        trnLoader.dataset.negSampling(corpus.n_items)
        epLoss, epPreLoss = 0, 0
        steps = trnLoader.dataset.__len__() // self.batch_size
        for i, tem in enumerate(trnLoader):
            if i % self.fixSteps == 0:
                sampScores, seeds = self.sampler(self.handler.allOneAdj, self.model.getEgoEmbeds())
                encoderAdj, decoderAdj = self.masker(self.handler.torchBiAdj, seeds)
            ancs, poss, _ = tem
            ancs = ancs.long()
            poss = poss.long()
            usrEmbeds, itmEmbeds = self.model(encoderAdj, decoderAdj)
            ancEmbeds = usrEmbeds[ancs]
            posEmbeds = itmEmbeds[poss]

            bprLoss = (-torch.sum(ancEmbeds * posEmbeds, dim=-1)).mean()
            regLoss = calcRegLoss(self.model) * self.l2

            contrastLoss = (contrast(ancs, usrEmbeds) + contrast(poss, itmEmbeds)) * self.ssl_reg + contrast(ancs, usrEmbeds, itmEmbeds)
            loss = bprLoss + regLoss + contrastLoss
            if i % self.fixSteps == 0:
                localGlobalLoss = -sampScores.mean()
                loss += localGlobalLoss
            epLoss += loss.item()
            epPreLoss += bprLoss.item()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        return {"Loss": epLoss / steps, "preLoss": epPreLoss / steps}



    def testEpoch(self):
        tstLoader = self.handler.tstLoader
        epHR = {k: 0 for k in self.topk}
        epNdcg = {k: 0 for k in self.topk}
        i = 0
        num = tstLoader.dataset.__len__()

        for usr, trnMask in tstLoader:
            i += 1
            usr = usr.long()
            trnMask = trnMask
            usrEmbeds, itmEmbeds = self.model(self.handler.torchBiAdj, self.handler.torchBiAdj)

            allPreds = torch.mm(usrEmbeds[usr], torch.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
    
            for k in self.topk:
                _, topLocs = torch.topk(allPreds, k)
                hr_at_k, ndcg_at_k = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
                
                epHR[k] += hr_at_k[k]
                epNdcg[k] += ndcg_at_k[k]

        ret = dict()
        for k in self.topk:
            ret[f'HR@{k}'] = epHR[k] / num
            ret[f'NDCG@{k}'] = epNdcg[k] / num

        return ret
    


    def calcRes(self, topLocs, tstLocs, batIds):
        assert topLocs.shape[0] == len(batIds)
    
        allHrAtK = {k: 0 for k in self.topk}
        allNdcgAtK = {k: 0 for k in self.topk}

        for i in range(len(batIds)):
            temTstLocs = tstLocs[batIds[i]]

            for k in self.topk:
                temTopLocs = list(topLocs[i][:k])
                tstNum = len(temTstLocs)

                maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, k))])
                dcg = 0

                hr_at_k = any(val in temTopLocs for val in temTstLocs)
                hr_at_k = 1.0 if hr_at_k else 0.0 

                for val in temTstLocs:
                    if val in temTopLocs:
                        dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))

                ndcg_at_k = dcg / maxDcg if maxDcg > 0 else 0.0

                allHrAtK[k] += hr_at_k
                allNdcgAtK[k] += ndcg_at_k

        return allHrAtK, allNdcgAtK



    def train(self, data_dict: Dict[str, BaseModel.Dataset]):
        self.model = AutoCF(self.args, data_dict['train'].corpus) 
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0)
        self.masker = RandomMaskSubgraphs(self.args)
        self.sampler = LocalGraph(self.args)
        stloc = 0
        logging.info('Model Initialized')
        self._check_time(start=True)
        bestRes = None
        try:
            for ep in range(stloc, self.epoch):
                tstFlag = (ep % self.test_epoch == 0)
                reses = self.trainEpoch(data_dict['train'].corpus)
                logging.info(self.makePrint('Train', ep, reses, tstFlag))

                if tstFlag:
                    reses = self.testEpoch()
                    logging.info(self.makePrint('Test', ep, reses, tstFlag))
                    bestRes = reses if bestRes is None or reses['HR@5'] > bestRes['HR@5'] else bestRes

                print()
        
        except KeyboardInterrupt:
            logging.info("Early stop manually")
            exit_here = input("Exit completely without evaluation? (y/n) (default n):")
            if exit_here.lower().startswith('y'):
                logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
                exit(1)

        reses = self.testEpoch()
        logging.info(self.makePrint('Test', self.epoch, reses, True))
        logging.info(self.makePrint('Best Result', self.epoch, bestRes, True))
        


    def predict(self, dataset: BaseModel.Dataset, save_prediction: bool = False) -> np.ndarray:
        dataset.model.eval()
        predictions = list()
        dl = DataLoader(dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers,
						collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)

        for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc='Predict'):
            device_batch = utils.batch_to_gpu(batch, dataset.model.device)
        
            if hasattr(dataset.model, 'inference'):
                prediction = dataset.model.inference(device_batch)[0]
            else:
                usrEmbeds, itmEmbeds = dataset.model(device_batch)
                prediction = usrEmbeds

            predictions.extend(prediction.cpu().numpy())

        predictions = np.array(predictions)

        if dataset.model.test_all:
            rows, cols = [], []
            for i, user_id in enumerate(dataset.data['user_id']):
                clicked_items = list(dataset.corpus.train_clicked_set[user_id] | dataset.corpus.residual_clicked_set[user_id])
                idx = [i] * len(clicked_items)
                rows.extend(idx)
                cols.extend(clicked_items)
            predictions[rows, cols] = -np.inf
        return predictions


    
class LocalGraph(nn.Module):
    def __init__(self, args):
        super(LocalGraph, self).__init__()
        self.args = args

    def makeNoise(self, scores):
        noise = torch.rand(scores.shape)
        noise = -torch.log(-torch.log(noise))
        return torch.log(scores) + noise

    def forward(self, allOneAdj, embeds):
        args = self.args
        
        order = torch.sparse.sum(allOneAdj, dim=-1).to_dense().view([-1, 1])
        fstEmbeds = torch.spmm(allOneAdj, embeds) - embeds
        fstNum = order
        scdEmbeds = (torch.spmm(allOneAdj, fstEmbeds) - fstEmbeds) - order * embeds
        scdNum = (torch.spmm(allOneAdj, fstNum) - fstNum) - order
        subgraphEmbeds = (fstEmbeds + scdEmbeds) / (fstNum + scdNum + 1e-8)
        subgraphEmbeds = F.normalize(subgraphEmbeds, p=2)
        embeds = F.normalize(embeds, p=2)
        scores = torch.sigmoid(torch.sum(subgraphEmbeds * embeds, dim=-1))
        scores = self.makeNoise(scores)
        _, seeds = torch.topk(scores, args.seedNum)
        return scores, seeds



class RandomMaskSubgraphs(nn.Module):
    def __init__(self, args):
        super(RandomMaskSubgraphs, self).__init__()
        self.args = args
        self.flag = False

    def normalizeAdj(self, adj):
        degree = t.pow(t.sparse.sum(adj, dim=1).to_dense() + 1e-12, -0.5)
        newRows, newCols = adj._indices()[0, :], adj._indices()[1, :]
        rowNorm, colNorm = degree[newRows], degree[newCols]
        newVals = adj._values() * rowNorm * colNorm
        return t.sparse.FloatTensor(adj._indices(), newVals, adj.shape)

    def forward(self, adj, seeds):
        args = self.args
        rows = adj._indices()[0, :]
        cols = adj._indices()[1, :]

        maskNodes = [seeds]

        for i in range(args.maskDepth):
            curSeeds = seeds if i == 0 else nxtSeeds
            nxtSeeds = list()
            for seed in curSeeds:
                rowIdct = (rows == seed)
                colIdct = (cols == seed)
                idct = t.logical_or(rowIdct, colIdct)

                if i != args.maskDepth - 1:
                    mskRows = rows[idct]
                    mskCols = cols[idct]
                    nxtSeeds.append(mskRows)
                    nxtSeeds.append(mskCols)

                rows = rows[t.logical_not(idct)]
                cols = cols[t.logical_not(idct)]
            if len(nxtSeeds) > 0:
                nxtSeeds = t.unique(t.concat(nxtSeeds))
                maskNodes.append(nxtSeeds)

        sampNum = int((args.user + args.item) * args.keepRate)
        sampedNodes = t.randint(args.user + args.item, size=[sampNum])

        if self.flag == False:
            l1 = adj._values().shape[0]
            l2 = rows.shape[0]
            print('-----')
            print('LENGTH CHANGE', '%.2f' % (l2 / l1), l2, l1)
            tem = t.unique(t.concat(maskNodes))
            print('Original SAMPLED NODES', '%.2f' % (tem.shape[0] / (args.user + args.item)), tem.shape[0],
                  (args.user + args.item))
            
        maskNodes.append(sampedNodes)
        maskNodes = t.unique(t.concat(maskNodes))
        if self.flag == False:
            print('AUGMENTED SAMPLED NODES', '%.2f' % (maskNodes.shape[0] / (args.user + args.item)),
                  maskNodes.shape[0], (args.user + args.item))
            self.flag = True
            print('-----')

        encoderAdj = self.normalizeAdj(
            t.sparse.FloatTensor(t.stack([rows, cols], dim=0), t.ones_like(rows), adj.shape))

        temNum = maskNodes.shape[0]
        temRows = maskNodes[t.randint(temNum, size=[adj._values().shape[0]])]
        temCols = maskNodes[t.randint(temNum, size=[adj._values().shape[0]])]

        newRows = t.concat([temRows, temCols, t.arange(args.user + args.item), rows])
        newCols = t.concat([temCols, temRows, t.arange(args.user + args.item), cols])

        # filter duplicated
        hashVal = newRows * (args.user + args.item) + newCols
        hashVal = t.unique(hashVal)
        newCols = hashVal % (args.user + args.item)
        newRows = ((hashVal - newCols) / (args.user + args.item)).long()

        decoderAdj = t.sparse.FloatTensor(t.stack([newRows, newCols], dim=0), t.ones_like(newRows).float(),adj.shape)

        return encoderAdj, decoderAdj
