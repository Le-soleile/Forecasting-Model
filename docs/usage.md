港口预测模型使用文档

1.本模型只对抵达港口进行预测，时间根据港口及船舶信息进行计算

2. 文件结构说明

project/
├── config.py        # 配置文件
├── data_loader.py   # 数据加载与预处理
├── model.py         # 模型定义
├── train.py         # 训练脚本
└── predict.py       # 预测脚本（文档3）

3. 配置文件 (config.py)

集中管理所有配置参数
自动检测运行设备（CPU/GPU）

4. 数据加载 (data_loader.py)

数据预处理和清洗
构建词汇表
生成有效港口掩码
创建PyTorch数据集和数据加载器
主要函数和类

5. 模型定义 (model.py)

6. 训练脚本 (train.py)

实现早停和学习率调度
保存最佳模型

7. 预测脚本 (predict.py)

加载训练好的模型
对测试数据进行预测
保存预测结果


8. 使用流程

准备数据文件并放入正确路径
运行训练脚本:
bash
python train.py
训练完成后，运行预测脚本:
bash
python predict.py
查看预测结果 test_predictions1.csv

9. 注意事项
pos.csv文件是通过数据处理得到船舶可以到达的港口历史选项，让模型搜索范围更小，耗时短精度高
确保所有数据文件使用UTF-8编码
训练前检查GPU是否可用
可根据需要调整config.py中的超参数
未知船舶和港口会被特殊处理，但可能影响预测精度
