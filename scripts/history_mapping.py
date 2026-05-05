import pandas as pd

# 读取两个文件（根据实际文件类型修改read_excel为read_csv等）
time_df = pd.read_excel('1.xlsx', header=None)  # 时间文件
target_df = pd.read_excel('target.xlsx', header=None)  # 目标文件

# 创建港口组合到时间的映射字典（顺序敏感）
port_time_map = {
    (row[0], row[1]): row[2]
    for _, row in time_df.iterrows()
}

# 定义处理函数
def update_time(row):
    key = (row[0], row[1])
    # 如果找到匹配项返回时间值，否则返回1
    return port_time_map.get(key, 1)

# 应用更新（直接操作第三列）
target_df[2] = target_df.apply(update_time, axis=1)

# 保存结果（保留其他列数据）
target_df.to_excel('updated_target.xlsx', index=False, header=False)

print("处理完成，未匹配项已补1")