import pandas as pd
#计算时间差值
# 读取Excel文件
df = pd.read_excel('time.xlsx', sheet_name='Sheet1')

# 处理时间列（支持带毫秒的时间格式）
time_columns = ['leg_start_postime', 'leg_end_postime']
for col in time_columns:
    # 移除时区信息并保留毫秒
    clean_series = df[col].str.replace(r'\+\d+$', '', regex=True)
    # 转换支持多种时间格式（包含/不包含毫秒）
    df[col] = pd.to_datetime(
        clean_series,
        format='mixed',  # 自动识别格式
        errors='coerce'
    )

# 验证是否有无法解析的时间
if df[time_columns].isna().any().any():
    print("警告：发现无法解析的时间格式，请检查原始数据！")

# 计算时间差值
df['time_diff'] = df['leg_end_postime'] - df['leg_start_postime']

# 将结果保存到新文件
df.to_excel('time_with_differences.xlsx', index=False)

print("处理完成，结果已保存至 time_with_differences.xlsx")