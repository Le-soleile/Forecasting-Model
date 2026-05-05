import pandas as pd
from datetime import datetime, timedelta


def process_time_data(input_file, output_file):
    # 读取Excel文件
    df = pd.read_excel(input_file)

    # 定义时间计算函数（重命名为row_time_calculator避免冲突）
    def row_time_calculator(row):
        try:
            # 解析原始时间（带时区）
            base_time_str = str(row[1]).split('+')[0]
            base_time = datetime.strptime(base_time_str, "%Y/%m/%d %H:%M:%S")

            # 计算时间增量（天转秒）
            days_to_add = float(row[0])
            delta = timedelta(days=days_to_add)

            # 计算新时间
            new_time = base_time + delta

            # 格式化输出（保持原始时区）
            return new_time.strftime("%Y/%m/%d %H:%M:%S") + "+08"
        except Exception as e:
            print(f"处理行时出错: {e}")
            return "计算错误"

    # 应用计算函数
    df['计算结果'] = df.apply(row_time_calculator, axis=1)

    # 保存结果
    df.to_excel(output_file, index=False)
    print(f"处理完成，结果已保存至 {output_file}")


# 使用示例
if __name__ == "__main__":
    process_time_data('工作簿1.xlsx', 'output.xlsx')