import baostock as bs
import pandas as pd
from datetime import datetime

# 登录系统
lg = bs.login()
print('login success!')
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)

stock_code = "sh.600150"  # 要查询的股票代码

# 查询股票历史数据
# 获取最近的数据
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = '2024-01-01'  # 获取最近2-3年的数据

print(f"正在下载 {stock_code} 从 {start_date} 到 {end_date} 的数据...")

# 查询日K线数据
rs = bs.query_history_k_data_plus(stock_code,
    "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
    start_date=start_date, 
    end_date=end_date,
    frequency="d", adjustflag="2") # adjustflag 1 后复权； 2 前复权； 3 不复权

print('query_history_k_data_plus respond error_code:'+rs.error_code)
print('query_history_k_data_plus respond  error_msg:'+rs.error_msg)

# 打印结果集
data_list = []
while (rs.error_code == '0') & rs.next():
    # 获取一条记录，将记录合并在一起
    data_list.append(rs.get_row_data())
    
# 转换为DataFrame
result = pd.DataFrame(data_list, columns=rs.fields)

print(f"\n原始数据行数: {len(result)}")
print("\n查看前5行原始数据:")
print(result.head())

# 数据清洗和类型转换
# 先替换空字符串为NaN
result = result.replace('', pd.NA)

# 转换数据类型，使用 pd.to_numeric 来处理错误
result['date'] = pd.to_datetime(result['date'])
result['close'] = pd.to_numeric(result['close'], errors='coerce')
result['open'] = pd.to_numeric(result['open'], errors='coerce')
result['high'] = pd.to_numeric(result['high'], errors='coerce')
result['low'] = pd.to_numeric(result['low'], errors='coerce')
result['volume'] = pd.to_numeric(result['volume'], errors='coerce')
result['amount'] = pd.to_numeric(result['amount'], errors='coerce')
result['pctChg'] = pd.to_numeric(result['pctChg'], errors='coerce')
result['turn'] = pd.to_numeric(result['turn'], errors='coerce')

# 删除包含空值的行（特别是价格数据）
print(f"\n转换前数据行数: {len(result)}")
result = result.dropna(subset=['date', 'close', 'open', 'high', 'low'])
print(f"删除空值后数据行数: {len(result)}")

# 按日期排序
result = result.sort_values('date')
result = result.reset_index(drop=True)

# 保存为Excel文件
output_file = f'{stock_code}.xlsx'
result.to_excel(output_file, index=False)
print(f"\n数据已保存到: {output_file}")
print(f"共 {len(result)} 条有效记录")

# 显示数据摘要
print("\n数据预览:")
print(result.head())
print("\n最新5条数据:")
print(result.tail())

# 显示数据统计信息
print("\n数据统计信息:")
print(f"日期范围: {result['date'].min()} 至 {result['date'].max()}")
print(f"价格范围: {result['close'].min():.2f} - {result['close'].max():.2f}")
print(f"平均成交量: {result['volume'].mean():.0f}")

# 登出系统
bs.logout()
print("\n登出成功！")