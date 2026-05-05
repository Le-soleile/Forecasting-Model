
"""
预测脚本
加载训练好的模型并对测试数据进行预测
"""

import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from model import PortPredictionModel
from config import Config

class PortDataset(Dataset):
    """
    预测用数据集类
    与训练数据集类略有不同，专注于预测功能
    """
    
    def __init__(self, df, ship2idx, port2idx, ship_to_valid_ports):
        """
        初始化预测数据集
        
        参数:
            df: 测试数据DataFrame
            ship2idx: 船舶到索引的映射
            port2idx: 港口到索引的映射
            ship_to_valid_ports: 船舶到有效港口的映射
        """
        self.df = df.copy()
        # 数据清洗
        self.df['ship_mmsi'] = self.df['ship_mmsi'].astype(str).str.strip()
        self.df['start_port_code'] = self.df['start_port_code'].astype(str).str.strip()
        self.df['end_port_code'] = self.df['end_port_code'].astype(str).str.strip()

        self.ship2idx = ship2idx  # 船舶词汇表
        self.port2idx = port2idx  # 港口词汇表
        self.idx2port = {v: k for k, v in port2idx.items()}  # 反向港口映射

        # 预处理有效港口映射
        self.ship_to_valid_ports = {}
        for ship_code, ports in ship_to_valid_ports.items():
            ship_idx = ship2idx.get(ship_code, 0)
            if ship_idx != 0:  # 只处理已知船舶
                port_indices = [port2idx[p] for p in ports if p in port2idx]
                self.ship_to_valid_ports[ship_idx] = port_indices

        # 数据映射
        self.ships = self.df['ship_mmsi'].map(lambda x: ship2idx.get(x, 0))
        self.start_ports = self.df['start_port_code'].map(lambda x: port2idx.get(x, 0))
        self.end_ports = self.df['end_port_code'].map(lambda x: port2idx.get(x, 0))

    def __len__(self):
        """返回数据集大小"""
        return len(self.df)

    def __getitem__(self, idx):
        """
        获取单个样本
        
        返回:
            dict: 包含:
                - ship: 船舶索引
                - start_port: 起始港口索引
                - end_port: 目的港口索引
                - valid_ports: 有效港口列表
        """
        ship = self.ships.iloc[idx]
        start_port = self.start_ports.iloc[idx]
        end_port = self.end_ports.iloc[idx]

        # 对于未知船舶，使用所有港口作为候选
        valid_ports = self.ship_to_valid_ports.get(ship, list(self.port2idx.values()))

        return {
            'ship': torch.tensor(ship, dtype=torch.long),
            'start_port': torch.tensor(start_port, dtype=torch.long),
            'end_port': torch.tensor(end_port, dtype=torch.long),
            'valid_ports': valid_ports
        }

def collate_fn(batch):
    """
    预测用批处理函数
    
    返回:
        dict: 包含:
            - ships: 船舶索引张量
            - start_ports: 起始港口索引张量
            - end_ports: 目的港口索引张量
            - valid_masks: 有效港口掩码张量
    """
    ships = torch.stack([item['ship'] for item in batch])
    start_ports = torch.stack([item['start_port'] for item in batch])
    end_ports = torch.stack([item['end_port'] for item in batch])
    valid_ports_list = [item['valid_ports'] for item in batch]

    batch_size = len(batch)
    num_classes = len(port2idx)
    valid_masks = torch.zeros(batch_size, num_classes, dtype=torch.float32)

    # 生成有效港口掩码
    for i, valid_ports in enumerate(valid_ports_list):
        valid_masks[i, valid_ports] = 1.0  # 允许预测所有有效港口

    return {
        'ships': ships,
        'start_ports': start_ports,
        'end_ports': end_ports,
        'valid_masks': valid_masks
    }

if __name__ == "__main__":
    # 加载配置
    config = Config()
    
    # 加载模型检查点
    checkpoint = torch.load(config.model_save_path, map_location=config.device)

    # 恢复训练配置
    train_config = Config()
    train_config.__dict__.update(checkpoint['config'])

    # 加载词汇表
    with open(train_config.vocab_path, 'r') as f:
        vocabs = json.load(f)
        port2idx = vocabs['port2idx']
        ship2idx = vocabs['ship2idx']
        ship_to_valid_ports = vocabs['ship_to_valid_ports']

    # 初始化模型
    model = PortPredictionModel(
        num_ships=len(ship2idx),
        num_ports=len(port2idx),
        embed_dim=train_config.embed_dim,
        hidden_dim=train_config.hidden_dim
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.device)
    model.eval()

    # 加载测试数据
    test_df = pd.read_csv(config.data_path['test'])
    test_dataset = PortDataset(test_df, ship2idx, port2idx, ship_to_valid_ports)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=collate_fn)

    idx2port = {v: k for k, v in port2idx.items()}  # 创建反向港口映射
    predictions = []  # 存储预测结果

    # 预测过程
    with torch.no_grad():
        for batch in test_loader:
            # 数据移动到设备
            ships = batch['ships'].to(config.device)
            start_ports = batch['start_ports'].to(config.device)
            valid_masks = batch['valid_masks'].to(config.device)

            # 前向传播
            logits = model(ships, start_ports)
            # 应用掩码
            logits_masked = logits + (1 - valid_masks) * -1e9
            # 获取预测结果
            preds = torch.argmax(logits_masked, dim=1).cpu().numpy()

            # 将索引转换为港口名称
            batch_preds = []
            for idx in preds:
                port = idx2port.get(idx, "UNKNOWN_PORT")
                batch_preds.append(port)
            predictions.extend(batch_preds)

    # 保存预测结果
    test_df['predicted_end_port'] = predictions
    test_df.to_csv('test_predictions1.csv', index=False)

    # 打印统计信息
    unknown_ship = sum(test_dataset.ships == 0)
    unknown_start_port = sum(test_dataset.start_ports == 0)
    print(f"[数据统计] 未知船舶数: {unknown_ship} | 未知起始港口数: {unknown_start_port}")
    print("预测结果已保存至 test_predictions.csv")