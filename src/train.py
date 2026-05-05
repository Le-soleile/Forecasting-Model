
"""
模型训练脚本
包含完整的训练流程、验证和模型保存功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import PortDataset, collate_fn, preprocess_data, DataLoader, Dataset
from model import PortPredictionModel
from config import Config
import pandas as pd
import json
import ast
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),  # 文件日志
        logging.StreamHandler()  # 控制台日志
    ]
)
logger = logging.getLogger(__name__)

def main():
    """主训练函数"""
    
    # 加载配置
    config = Config()

    # 数据预处理
    port2idx, ship2idx, ship_to_valid_ports = preprocess_data(config)

    # 保存词汇表
    with open(config.vocab_path, 'w') as f:
        json.dump({
            "port2idx": port2idx,
            "ship2idx": ship2idx,
            "ship_to_valid_ports": {k: list(v) for k, v in ship_to_valid_ports.items()}
        }, f, indent=2)

    # 加载数据集
    train_df = pd.read_csv(config.data_path['train'])
    val_df = pd.read_csv(config.data_path['val'])

    # 创建数据集和数据加载器
    train_dataset = PortDataset(train_df, ship2idx, port2idx, ship_to_valid_ports)
    val_dataset = PortDataset(val_df, ship2idx, port2idx, ship_to_valid_ports)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )

    # 初始化模型
    model = PortPredictionModel(
        num_ships=len(ship2idx),
        num_ports=len(port2idx),
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim
    ).to(config.device)

    # 设置优化器和学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    criterion = nn.CrossEntropyLoss()

    # 训练状态变量
    best_acc = 0.0
    early_stop_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    logger.info("开始模型训练...")
    logger.info(f"训练设备: {config.device}")
    logger.info(f"训练参数: {config.__dict__}")

    # 训练循环
    for epoch in range(config.num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            # 数据移动到设备
            ships = batch['ships'].to(config.device)
            start_ports = batch['start_ports'].to(config.device)
            end_ports = batch['end_ports'].to(config.device)
            valid_masks = batch['valid_masks'].to(config.device)

            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            logits = model(ships, start_ports)
            # 应用有效港口掩码
            logits_masked = logits + (1 - valid_masks) * -1e4
            # 计算损失
            loss = criterion(logits_masked, end_ports)

            # 跳过NaN损失
            if torch.isnan(loss):
                logger.warning("检测到NaN损失，跳过当前批次")
                continue

            # 反向传播和优化
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            train_loss += loss.item()

        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                ships = batch['ships'].to(config.device)
                start_ports = batch['start_ports'].to(config.device)
                end_ports = batch['end_ports'].to(config.device)
                valid_masks = batch['valid_masks'].to(config.device)

                # 前向传播
                logits = model(ships, start_ports)
                logits_masked = logits + (1 - valid_masks) * -1e9
                loss = criterion(logits_masked, end_ports)
                val_loss += loss.item()

                # 计算准确率
                preds = torch.argmax(logits_masked, dim=1)
                correct += (preds == end_ports).sum().item()
                total += end_ports.size(0)

        # 计算验证指标
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        # 更新学习率
        scheduler.step(avg_val_loss)

        # 记录日志
        logger.info(
            f"Epoch {epoch+1:03d}/{config.num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # 早停机制和模型保存
        if val_acc > best_acc:
            best_acc = val_acc
            early_stop_counter = 0
            
            # 保存模型检查点
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config.__dict__
            }, config.model_save_path)
            
            logger.info(f"发现新的最佳模型，准确率: {val_acc:.2f}%")
        else:
            early_stop_counter += 1
            if early_stop_counter >= config.early_stop_patience:
                logger.info(f"早停触发，连续{config.early_stop_patience}个周期未提升")
                break

    logger.info(f"训练完成，最佳验证准确率: {best_acc:.2f}%")

if __name__ == '__main__':
    main()