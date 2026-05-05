
"""
数据加载和处理模块
包含数据预处理、数据集类和数据加载器创建功能
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import numpy as np

def preprocess_data(config):
    """
    数据预处理函数，构建词汇表和有效港口映射
    
    参数:
        config: 配置对象，包含数据路径等信息
        
    返回:
        tuple: (port2idx, ship2idx, ship_to_valid_ports)
            - port2idx: 港口到索引的映射字典
            - ship2idx: 船舶到索引的映射字典
            - ship_to_valid_ports: 船舶到有效港口的映射字典
    """
    
    # 加载原始数据文件
    pos_df = pd.read_csv(config.data_path['pos'])  # 港口位置数据
    train_df = pd.read_csv(config.data_path['train'])  # 训练数据

    # 构建港口词汇表，从所有出现过的港口创建映射
    all_ports = pd.concat([pos_df['start_port'], pos_df['end_port']]).unique()
    port2idx = {port: idx + 1 for idx, port in enumerate(all_ports)}  # 索引从1开始
    port2idx['<unk>'] = 0  # 保留0给未知港口

    # 构建船舶词汇表
    ship2idx = {ship: idx + 1 for idx, ship in enumerate(train_df['ship_mmsi'].unique())}
    ship2idx['<unk>'] = 0  # 保留0给未知船舶

    # 构建船舶到有效港口的映射关系
    ship_to_valid_ports = defaultdict(set)
    for _, row in pos_df.iterrows():
        ship_to_valid_ports[row['ship_mmsi']].add(row['start_port'])
        ship_to_valid_ports[row['ship_mmsi']].add(row['end_port'])

    # 数据验证：确保每艘船至少有一个有效港口
    for ship in ship_to_valid_ports:
        if len(ship_to_valid_ports[ship]) == 0:
            raise ValueError(f"Ship {ship} has no valid ports!")

    return port2idx, ship2idx, ship_to_valid_ports

class PortDataset(Dataset):
    """
    自定义PyTorch数据集类，用于加载和处理港口预测数据
    """
    
    def __init__(self, df, ship2idx, port2idx, ship_to_valid_ports):
        """
        初始化数据集
        
        参数:
            df: 包含原始数据的DataFrame
            ship2idx: 船舶到索引的映射字典
            port2idx: 港口到索引的映射字典
            ship_to_valid_ports: 船舶到有效港口的映射字典
        """
        self.df = df.copy()
        # 数据清洗：去除字符串两端的空格
        self.df['ship_mmsi'] = self.df['ship_mmsi'].astype(str).str.strip()
        self.df['start_port_code'] = self.df['start_port_code'].astype(str).str.strip()
        self.df['end_port_code'] = self.df['end_port_code'].astype(str).str.strip()

        self.ship2idx = ship2idx  # 船舶词汇表
        self.port2idx = port2idx  # 港口词汇表

        # 预处理有效港口映射：将港口代码转换为索引
        self.ship_to_valid_ports = {}
        for ship_code, ports in ship_to_valid_ports.items():
            ship_idx = ship2idx.get(ship_code, 0)
            if ship_idx != 0:  # 只处理已知船舶
                # 转换港口代码为索引，并过滤未知港口
                port_indices = [port2idx[p] for p in ports if p in port2idx]
                self.ship_to_valid_ports[ship_idx] = port_indices

        # 将原始数据映射为索引
        self.ships = self.df['ship_mmsi'].map(lambda x: ship2idx.get(x, 0))
        self.start_ports = self.df['start_port_code'].map(lambda x: port2idx.get(x, 0))
        self.end_ports = self.df['end_port_code'].map(lambda x: port2idx.get(x, 0))

    def __len__(self):
        """返回数据集大小"""
        return len(self.df)

    def __getitem__(self, idx):
        """
        获取单个数据样本
        
        参数:
            idx: 数据索引
            
        返回:
            dict: 包含以下键的字典:
                - ships: 船舶索引张量
                - start_ports: 起始港口索引张量
                - end_ports: 目的港口索引张量
                - valid_masks: 有效港口掩码张量
        """
        row = self.df.iloc[idx]

        # 获取船舶和港口索引，未知则使用0
        ship = self.ship2idx.get(row['ship_mmsi'], 0)
        start_port = self.port2idx.get(row['start_port'], 0)
        end_port = self.port2idx.get(row['end_port'], 0)

        # 生成有效港口mask：1表示有效，0表示无效
        valid_ports = self.ship_to_valid_ports.get(row['ship_mmsi'], set())
        valid_mask = [1 if port in valid_ports else 0 for port in self.port2idx]

        return {
            'ships': torch.tensor(ship, dtype=torch.long),
            'start_ports': torch.tensor(start_port, dtype=torch.long),
            'end_ports': torch.tensor(end_port, dtype=torch.long),
            'valid_masks': torch.tensor(valid_mask, dtype=torch.float)
        }

def collate_fn(batch):
    """
    批处理函数，用于DataLoader
    
    参数:
        batch: 批数据列表
        
    返回:
        dict: 批处理后的数据字典，包含:
            - ships: 船舶索引张量堆叠
            - start_ports: 起始港口索引张量堆叠
            - end_ports: 目的港口索引张量堆叠
            - valid_masks: 有效港口掩码张量堆叠
    """
    return {
        'ships': torch.stack([item['ships'] for item in batch]),
        'start_ports': torch.stack([item['start_ports'] for item in batch]),
        'end_ports': torch.stack([item['end_ports'] for item in batch]),
        'valid_masks': torch.stack([item['valid_masks'] for item in batch])
    }