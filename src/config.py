
"""
配置文件模块
定义模型训练、数据路径等所有配置参数
"""

import torch

class Config:
    """
    配置类，包含模型训练和数据处理的所有参数
    """
    
    # 词汇表文件路径
    vocab_path = 'data/processed/vocabs.json'
    
    # 数据文件路径配置字典
    data_path = {
        'train': 'data/processed/train.csv',  # 训练数据路径
        'val': 'data/processed/val.csv',      # 验证数据路径
        'test1': 'data/processed/test.csv',   # 测试数据路径
        'pos': 'data/processed/pos.csv'       # 港口位置数据路径
    }
    
    # 模型保存路径
    model_save_path = 'checkpoints/model.pth'

    # 模型超参数配置
    embed_dim = 512    # 嵌入层维度
    hidden_dim = 1024  # 隐藏层维度
    
    # 训练参数配置
    batch_size = 64      # 每批数据量大小
    num_epochs = 200     # 最大训练轮数
    learning_rate = 3e-6 # 学习率
    weight_decay = 1e-5  # 权重衰减系数
    patience = 60        # 早停机制等待周期数

    @property
    def device(self):
        """
        自动检测并返回可用设备(cuda/cpu)
        
        返回:
            str: 'cuda' 如果GPU可用，否则 'cpu'
        """
        return 'cuda' if torch.cuda.is_available() else 'cpu'