# AutoCF_at_ReChorus

## 项目简介
这是中山大学人工智能专业的大三机器学习大作业项目

AutoCF 旨在提高在有限标注数据上的表示质量。对比学习在推荐系统中已经引起了广泛关注，并且在基于图的协同过滤（CF）模型中取得了显著的进展。然而，大多数对比学习方法依赖于手动生成有效的对比视图，这通常是通过启发式数据增强来实现的。尽管这种方法在特定任务和数据集上可能表现良好，但它不具备良好的泛化能力，无法适应不同的数据集和下游推荐任务，且对噪声扰动较为敏感。

AutoCF 提出了一个新的解决方案，通过自动化生成对比视图，解决了现有方法在噪声和长尾数据分布下性能严重下降的问题。实验结果表明，与当前主流的自监督学习（SSL）方法（如 NCL 和 SimGCL）相比，AutoCF 在高噪声和长尾数据分布下能显著提高性能。



本项目是基于Rechorus框架下复现AutoCF模型。
本项目开发参考了以下两个项目：
- [ReChorus](https://github.com/THUwangcy/ReChorus)
- [AutoCF](https://github.com/HKUDS/AutoCF)

## 环境要求

 见 `requirements.txt`



## 数据集准备

1. 下载并解压数据集到`data/` 目录下
2. 运行Rechorus模型准备的相应.iypnb进行数据预处理，生成3个.csv文件
3. 运行DataProcess获得3个.pkl文件


## 结构


- `data/`: 存放各种数据集
- `docs/`: 存放模型指导
- `log/`: 存放训练日志以及预测结果
- `model/`: 存放训练后的模型权重
- `src`/: 存放代码
  - `AutoCF_main.py`: 基于main.py实现的只服务于实现的AutoCF模型
  - `helpers/`: 存放各种reader.py和runner.py文件 
    - `AutoCFRunner.py`: 基于AutoCF模型和Rechorus的BaseRunner.py实现
    - `AutoCFReader.py`: 基于AutoCF模型和Rechorus的BaseReader.py实现
  - `models/`: 存放各种模型代码
    - `AutoCF.py`: 基于AutoCF模型和Rechorus的Basemodel.py实现
    

## 使用

运行下面的命令：
```
python src/AutoCF_main.py --model_name AutoCF --emb_size 32 --lr 1e-3 --l2 1e-6 --dataset （data文件夹内子文件夹的名称）
```

- `--emb_size`: 批处理大小
- `--dataset`: 数据集名称
- `--lr`: 学习率
- `--l2`: 优化器的权重衰减
- `--seed`: 随机种子（推荐设置为500）
- `--dataset` 目前data文件夹内已有Grocery_and_Gourmet_Food，MINDTOPK，MovieLens_1M

因为一些技术原因，AutoCF完成训练后并不能生成测试集和模型权重，可以在日志上查看HR和NDCG指标。

## 结果

| Data                     | AutoCF                          | NeuMF                          | BPRMF                          |
|:-------------------------|----------------------------------|--------------------------------|--------------------------------|
| Grocery_and_Gourmet_Food | HR@5 = 0.1614</br>NDCG@5 = 0.0668</br>HR@10 = 0.2234</br>NDCG@10 = 0.0839</br>HR@20 = 0.2986</br>NDCG@20 = 0.1034</br>HR@50 = 0.4099</br>NDCG@50 = 0.1281 | HR@5 = 0.2891</br>NDCG@5 = 0.1947</br>HR@10 = 0.3992</br>NDCG@10 = 0.2303</br>HR@20 = 0.5085</br>NDCG@20 = 0.2578</br>HR@50 = 0.7287</br>NDCG@50 = 0.3012 | HR@5 = 0.3460</br>NDCG@5 = 0.2393</br>HR@10 = 0.4545</br>NDCG@10 = 0.2746</br>HR@20 = 0.5638</br>NDCG@20 = 0.3021</br>HR@50 = 0.7667</br>NDCG@50 = 0.3420 |
| MINDTOPK                 | HR@5 = 0.2537</br>NDCG@5 = 0.0807</br>HR@10 = 0.3594</br>NDCG@10 = 0.0954</br>HR@20 = 0.4891</br>NDCG@20 = 0.1189</br>HR@50 = 0.6691</br>NDCG@50 = 0.1544 | HR@5 = 0.0847</br>NDCG@5 = 0.0549</br>HR@10 = 0.1328</br>NDCG@10 = 0.0703</br>HR@20 = 0.1960</br>NDCG@20 = 0.0862</br>HR@50 = 0.3646</br>NDCG@50 = 0.1189 | HR@5 = 0.0955</br>NDCG@5 = 0.0609</br>HR@10 = 0.1531</br>NDCG@10 = 0.0794</br>HR@20 = 0.2314</br>NDCG@20 = 0.0991</br>HR@50 = 0.3522</br>NDCG@50 = 0.1233 |
| MovieLens-1M             | HR@5 = 0.6763</br>NDCG@5 = 0.2832</br>HR@10 = 0.7693</br>NDCG@10 = 0.2695</br>HR@20 = 0.8410</br>NDCG@20 = 0.2752</br>HR@50 = 0.8947</br>NDCG@50 = 0.3212 | HR@5 = 0.4919</br>NDCG@5 = 0.3388</br>HR@10 = 0.6611</br>NDCG@10 = 0.3938</br>HR@20 = 0.8227</br>NDCG@20 = 0.4347</br>HR@50 = 0.9618</br>NDCG@50 = 0.4628 | HR@5 = 0.5068</br>NDCG@5 = 0.3485</br>HR@10 = 0.6886</br>NDCG@10 = 0.4073</br>HR@20 = 0.8364</br>NDCG@20 = 0.4448</br>HR@50 = 0.9732</br>NDCG@50 = 0.4726 |



## 许可协议

本项目基于

- [Rechorus](https://github.com/THUwangcy/ReChorus)
- [AutoCF](https://github.com/HKUDS/AutoCF)

沿用MIT协议
