
"""
模型定义模块
包含港口预测模型的核心网络结构
"""

import torch
import torch.nn as nn
import torch.nn.init as init

class PortPredictionModel(nn.Module):
    """
    港口预测模型类
    基于船舶和起始港口信息预测目的港口
    """
    
    def __init__(self, num_ships, num_ports, embed_dim, hidden_dim):
        """
        初始化模型
        
        参数:
            num_ships: 船舶词汇表大小
            num_ports: 港口词汇表大小
            embed_dim: 嵌入层维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        # 船舶嵌入层
        self.ship_embed = nn.Embedding(num_ships, embed_dim)
        # 港口嵌入层
        self.port_embed = nn.Embedding(num_ports, embed_dim)

        # 全连接网络结构
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),  # 输入是船舶和港口嵌入的拼接
            nn.ReLU(),  # 激活函数
            nn.Linear(hidden_dim, hidden_dim),  # 隐藏层
            nn.ReLU(),  # 激活函数
            nn.Linear(hidden_dim, num_ports)  # 输出层，输出港口数量维度
        )

        # 初始化权重
        self._init_weights()

    def forward(self, ships, start_ports):
        """
        前向传播
        
        参数:
            ships: 船舶索引张量
            start_ports: 起始港口索引张量
            
        返回:
            torch.Tensor: 预测logits
        """
        # 获取船舶和港口的嵌入表示
        ship_emb = self.ship_embed(ships)
        port_emb = self.port_embed(start_ports)
        
        # 拼接特征
        combined = torch.cat([ship_emb, port_emb], dim=1)
        
        # 通过全连接网络
        return self.fc(combined)

    def _init_weights(self):
        """
        权重初始化方法
        使用Kaiming正态分布初始化线性层，正态分布初始化嵌入层
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用ReLU作为非线性函数初始化线性层
                init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                # 使用正态分布初始化嵌入层
                init.normal_(module.weight, mean=0, std=0.02)