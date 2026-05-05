import pandas as pd


def process_port_statistics():
    # 读取船舶数据
    vessels_df = pd.read_excel('E:\File_E\pycharm\ship_analysis\ship_1.xlsx')
    vessel_info = dict(zip(
        vessels_df['ship_mmsi'],
        vessels_df[['dead_weight', 'length', 'width', 'height']].values
    ))

    # 初始化港口统计字典
    port_stats = {}

    # 处理航行记录
    voyages_df = pd.read_excel(r'E:\File_E\pycharm\ship_analysis\train_6_test.xlsx')
    for _, row in voyages_df.iterrows():
        vessel_code = row['ship_mmsi']
        departure = row['start_port_code']
        arrival = row['end_port_code']

        if vessel_code not in vessel_info:
            continue  # 跳过无效船舶编码

        tonnage, length, width, height = vessel_info[vessel_code]

        # 更新出发港数据
        for port in [departure, arrival]:
            if port not in port_stats:
                port_stats[port] = {
                    'max_t': -float('inf'), 'min_t': float('inf'),
                    'max_l': -float('inf'), 'min_l': float('inf'),
                    'max_w': -float('inf'), 'min_w': float('inf'),
                    'max_h': -float('inf'), 'min_h': float('inf')
                }

            stats = port_stats[port]
            stats.update({
                'max_t': max(stats['max_t'], tonnage),
                'min_t': min(stats['min_t'], tonnage),
                'max_l': max(stats['max_l'], length),
                'min_l': min(stats['min_l'], length),
                'max_w': max(stats['max_w'], width),
                'min_w': min(stats['min_w'], width),
                'max_h': max(stats['max_h'], height),
                'min_h': min(stats['min_h'], height)
            })

    # 处理港口列表
    ports_df = pd.read_excel('E:\File_E\pycharm\ship_analysis\port_3.xlsx')
    results = []

    for port in ports_df['port_code']:
        if port in port_stats:
            stats = port_stats[port]
            results.append([
                stats['max_t'], stats['min_t'],
                stats['max_l'], stats['min_l'],
                stats['max_w'], stats['min_w'],
                stats['max_h'], stats['min_h']
            ])
        else:
            results.append([0] * 8)  # 无数据时填充0

    # 生成结果文件
    pd.DataFrame(results, columns=[
        '最大吨位', '最小吨位',
        '最大长度', '最小长度',
        '最大宽度', '最小宽度',
        '最大高度', '最小高度'
    ]).to_excel('E:\File_E\pycharm\ship_analysis\port_statistics.xlsx', index=False)


if __name__ == "__main__":
    process_port_statistics()