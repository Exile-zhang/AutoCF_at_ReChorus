from statistics import mean
import torch as t
from torch import nn
import torch.nn.functional as F
import torch
import torch.nn as nn

from models.BaseModel import GeneralModel

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class AutoCF(GeneralModel):
    reader = 'AutoCFReader'
    runner = 'AutoCFRunner'
    extra_log_args = ['latdim', 'gcn_layers', 'gt_layers','seedNum','maskDepth','keepRate']
	
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--latdim', default=32, type=int, help='embedding size')
        parser.add_argument('--head', default=4, type=int, help='number of heads in attention')
        parser.add_argument('--gcn_layers', default=2, type=int, help='number of gcn layers')
        parser.add_argument('--gt_layers', default=1, type=int, help='number of graph transformer layers')
        parser.add_argument('--seedNum', default=500, type=int, help='number of seeds in patch masking')
        parser.add_argument('--maskDepth', default=2, type=int, help='depth to mask')
        parser.add_argument('--keepRate', default=0.2, type=float, help='ratio of nodes to keep')
        return GeneralModel.parse_model_args(parser)
    
    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.args = args
        self.latdim = args.latdim
        self.gcn_layers = args.gcn_layers
        self.gt_layers = args.gt_layers
        self.head = args.head
        self._define_params()
        self.apply(self.init_weights)
		
    def _define_params(self):
        self.uEmbeds = nn.Parameter(init(t.empty(self.user_num, self.latdim)))
        self.iEmbeds = nn.Parameter(init(t.empty(self.item_num, self.latdim)))
        self.gcnLayers = nn.Sequential(*[GCNLayer(self.args) for i in range(self.gcn_layers)])
        self.gtLayers = nn.Sequential(*[GTLayer(self.args) for i in range(self.gt_layers)])
	
    def getEgoEmbeds(self):
        return t.concat([self.uEmbeds, self.iEmbeds], axis=0)

    def forward(self, encoderAdj, decoderAdj=None):
        if isinstance(encoderAdj, dict):
            user_ids = encoderAdj['user_id'].long()
            item_ids = encoderAdj['item_id'].long()

            item_ids = item_ids.flatten()
            user_ids = user_ids.repeat_interleave(item_ids.size(0) // user_ids.size(0))

            # 检查形状匹配
            assert user_ids.size(0) == item_ids.size(0), \
                f"user_ids shape {user_ids.size()}, item_ids shape {item_ids.size()}"

            # 构建稀疏矩阵的索引和值
            indices = torch.stack([user_ids, item_ids])  # (2, N)
            values = torch.ones(indices.size(1))        # 默认值全为1

            # 构建稀疏邻接矩阵
            encoderAdj = torch.sparse.FloatTensor(indices,values,(self.args.user + self.args.item, self.args.user + self.args.item))

        # 处理解码器邻接矩阵，若为字典则转为稀疏矩阵
        if decoderAdj is not None and isinstance(decoderAdj, dict):
            indices = torch.tensor(list(decoderAdj.keys())).t()
            values = torch.tensor(list(decoderAdj.values()))

            decoderAdj = torch.sparse.FloatTensor(indices,values,(self.args.user + self.args.item, self.args.user + self.args.item))

        # 初始化嵌入，拼接用户和物品的初始嵌入
        embeddings = torch.concat([self.uEmbeds, self.iEmbeds], dim=0)
        embedding_list = [embeddings]

        # 逐层进行编码器的图卷积操作
        for layer in self.gcnLayers:
            embeddings = layer(encoderAdj, embedding_list[-1])
            embedding_list.append(embeddings)

        # 如果有解码器邻接矩阵，逐层进行解码器的图卷积操作
        if decoderAdj is not None:
            for layer in self.gtLayers:
                embeddings = layer(decoderAdj, embedding_list[-1])
                embedding_list.append(embeddings)

        # 将所有层的嵌入加和，生成最终嵌入
        embeddings = sum(embedding_list)

        # 分割嵌入，返回用户和物品的嵌入
        return embeddings[:self.user_num], embeddings[self.user_num:]


class GCNLayer(nn.Module):
	def __init__(self, args):
		super(GCNLayer, self).__init__()
		self.args =args

	def forward(self, adj, embeds):
		return t.spmm(adj, embeds)

class GTLayer(nn.Module):
    def __init__(self, args):
        super(GTLayer, self).__init__()
        self.args = args
        self.qTrans = nn.Parameter(init(torch.empty(args.latdim, args.latdim)))
        self.kTrans = nn.Parameter(init(torch.empty(args.latdim, args.latdim)))
        self.vTrans = nn.Parameter(init(torch.empty(args.latdim, args.latdim)))

    def forward(self, adj, embeds):
        args = self.args  
        
        indices = adj._indices()
        rows, cols = indices[0, :], indices[1, :]
        rowEmbeds = embeds[rows]
        colEmbeds = embeds[cols]

        qEmbeds = (rowEmbeds @ self.qTrans).view([-1, args.head, args.latdim // args.head])
        kEmbeds = (colEmbeds @ self.kTrans).view([-1, args.head, args.latdim // args.head])
        vEmbeds = (colEmbeds @ self.vTrans).view([-1, args.head, args.latdim // args.head])

        att = torch.einsum('ehd, ehd -> eh', qEmbeds, kEmbeds)
        att = torch.clamp(att, -10.0, 10.0)
        expAtt = torch.exp(att)
        tem = torch.zeros([adj.shape[0], args.head])
        attNorm = (tem.index_add_(0, rows, expAtt))[rows]
        att = expAtt / (attNorm + 1e-8)

        resEmbeds = torch.einsum('eh, ehd -> ehd', att, vEmbeds).view([-1, args.latdim])
        tem = torch.zeros([adj.shape[0], args.latdim])
        resEmbeds = tem.index_add_(0, rows, resEmbeds)  # nd

        return resEmbeds


