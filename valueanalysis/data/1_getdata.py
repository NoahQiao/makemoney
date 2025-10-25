import baostock as bs
import pandas as pd
from datetime import datetime, timedelta
import os
from tqdm import tqdm
import time
import json

class StockDataCollector:
    def __init__(self, start_date='2025-01-01', end_date='2025-09-12'):
        self.start_date = start_date
        self.end_date = end_date
        self.min_market_cap = 100  # 100亿
        self.failed_stocks = []
        
    def login(self):
        """登录baostock"""
        max_retries = 3
        for i in range(max_retries):
            try:
                lg = bs.login()
                if lg.error_code == '0':
                    print("Baostock登录成功!")
                    return
                else:
                    print(f"登录失败，尝试 {i+1}/{max_retries}: {lg.error_msg}")
                    time.sleep(2)
            except Exception as e:
                print(f"登录异常，尝试 {i+1}/{max_retries}: {str(e)}")
                time.sleep(2)
        
        raise Exception("登录失败，请检查网络连接")
        
    def logout(self):
        """登出baostock"""
        try:
            bs.logout()
            print("Baostock登出成功!")
        except:
            pass
    
    def check_existing_data(self):
        """检查已有数据的进度"""
        if os.path.exists('data/daily_data_temp.parquet'):
            print("\n检查已有数据...")
            df_existing = pd.read_parquet('data/daily_data_temp.parquet')
            
            # 统计信息
            print(f"已有数据记录数: {len(df_existing)}")
            print(f"日期范围: {df_existing['date'].min()} 至 {df_existing['date'].max()}")
            
            # 检查每个日期的数据完整性
            date_stats = df_existing.groupby('date').agg({
                'code': 'count',
                'amount': 'sum'
            }).rename(columns={'code': 'stock_count', 'amount': 'total_amount'})
            
            print("\n每日数据统计:")
            print(date_stats.tail(10))  # 显示最后10天
            
            # 获取最后一个日期
            last_date = df_existing['date'].max()
            last_date_count = date_stats.loc[last_date, 'stock_count']
            
            # 如果有股票清单，检查最后一天是否完整
            if os.path.exists('data/stock_list.parquet'):
                stock_list = pd.read_parquet('data/stock_list.parquet')
                expected_count = len(stock_list)
                print(f"\n最后交易日 {last_date}: {last_date_count}/{expected_count} 只股票")
                
                if last_date_count < expected_count * 0.95:  # 如果少于95%，认为不完整
                    print(f"最后一天数据可能不完整，建议重新下载")
                    return df_existing, last_date, False
                else:
                    return df_existing, last_date, True
            else:
                # 没有股票清单，通过与前几天比较判断
                avg_count = date_stats['stock_count'].iloc[-5:-1].mean()  # 倒数2-5天的平均
                if last_date_count < avg_count * 0.9:
                    print(f"最后一天数据可能不完整（{last_date_count} vs 平均{avg_count:.0f}），建议重新下载")
                    return df_existing, last_date, False
                else:
                    return df_existing, last_date, True
        
        return None, None, None
    
    def get_stock_list_from_file(self):
        """从文件加载股票列表"""
        if os.path.exists('data/stock_list.parquet'):
            df = pd.read_parquet('data/stock_list.parquet')
            print(f"从文件加载股票列表，共 {len(df)} 只股票")
            return df
        return None
        
    def get_stock_list(self, date):
        """获取指定日期的股票列表（市值>100亿，非ST）"""
        print(f"获取 {date} 的股票列表...")
        
        # 获取所有股票列表
        try:
            rs = bs.query_all_stock(day=date)
        except Exception as e:
            print(f"获取股票列表失败: {str(e)}")
            return pd.DataFrame()
            
        stock_list = []
        
        while (rs.error_code == '0') & rs.next():
            row = rs.get_row_data()
            code = row[0]
            # 只保留沪深股票
            if code.startswith(('sh.60', 'sh.68', 'sz.00', 'sz.30')):
                stock_list.append({
                    'code': code,
                    'code_name': row[2] if len(row) > 2 else '',
                    'type': row[1] if len(row) > 1 else ''
                })
        
        df_stocks = pd.DataFrame(stock_list)
        print(f"获取到 {len(df_stocks)} 只股票，开始筛选...")
        
        # 获取股票基本信息（市值、是否ST等）
        valid_stocks = []
        
        for idx, stock in tqdm(df_stocks.iterrows(), total=len(df_stocks), desc="筛选股票"):
            code = stock['code']
            
            try:
                # 获取股票基本信息
                rs_basic = bs.query_stock_basic(code=code)
                if rs_basic.error_code != '0':
                    continue
                    
                basic_data = []
                while (rs_basic.error_code == '0') & rs_basic.next():
                    basic_data.append(rs_basic.get_row_data())
                
                if not basic_data:
                    continue
                    
                # 检查是否ST（通过股票名称）
                stock_name = basic_data[0][1] if len(basic_data[0]) > 1 else ''
                if 'ST' in stock_name or '*' in stock_name:
                    continue
                
                # 获取最新交易日的数据来计算市值
                rs_daily = bs.query_history_k_data_plus(
                    code,
                    "date,close,volume,amount,turn",
                    start_date=date,
                    end_date=date,
                    frequency="d",
                    adjustflag="2"
                )
                
                daily_data = []
                while (rs_daily.error_code == '0') & rs_daily.next():
                    daily_data.append(rs_daily.get_row_data())
                
                if daily_data and len(daily_data[0]) >= 5:
                    # 计算市值
                    try:
                        amount = float(daily_data[0][3]) if daily_data[0][3] else 0
                        turn = float(daily_data[0][4]) if daily_data[0][4] else 0
                        
                        if turn > 0 and amount > 0:
                            # 市值 ≈ 成交额 / 换手率 * 100
                            market_cap = amount / turn * 100 / 1e8  # 转换为亿元
                            
                            if market_cap >= self.min_market_cap:
                                valid_stocks.append({
                                    'code': code,
                                    'name': stock_name,
                                    'market_cap': market_cap
                                })
                    except:
                        pass
                
                # 添加延迟
                time.sleep(0.1)
                
            except Exception as e:
                continue
        
        print(f"筛选完成，符合条件的股票: {len(valid_stocks)} 只")
        return pd.DataFrame(valid_stocks)
    
    def get_single_stock_data(self, code, date, stock_info, max_retries=3):
        """获取单个股票的数据，带重试机制"""
        for retry in range(max_retries):
            try:
                rs = bs.query_history_k_data_plus(
                    code,
                    "date,code,open,high,low,close,volume,amount,turn,pctChg",
                    start_date=date,
                    end_date=date,
                    frequency="d",
                    adjustflag="2"
                )
                
                if rs.error_code != '0':
                    if retry < max_retries - 1:
                        time.sleep(1)
                        continue
                    else:
                        return None
                
                data = []
                while (rs.error_code == '0') & rs.next():
                    row_data = rs.get_row_data()
                    if row_data[0]:  # 确保有数据
                        data.append({
                            'date': row_data[0],
                            'code': row_data[1],
                            'name': stock_info['name'],
                            'open': float(row_data[2]) if row_data[2] else 0,
                            'high': float(row_data[3]) if row_data[3] else 0,
                            'low': float(row_data[4]) if row_data[4] else 0,
                            'close': float(row_data[5]) if row_data[5] else 0,
                            'volume': float(row_data[6]) if row_data[6] else 0,
                            'amount': float(row_data[7]) if row_data[7] else 0,
                            'turn': float(row_data[8]) if row_data[8] else 0,
                            'pct_chg': float(row_data[9]) if row_data[9] else 0,
                            'market_cap': stock_info['market_cap']
                        })
                
                return data
                
            except Exception as e:
                if retry < max_retries - 1:
                    time.sleep(2)
                else:
                    self.failed_stocks.append({'code': code, 'date': date, 'error': str(e)})
                    return None
    
    def get_daily_data(self, date, stock_list):
        """获取指定日期的所有股票成交数据"""
        print(f"\n获取 {date} 的成交数据...")
        
        daily_data = []
        failed_count = 0
        
        pbar = tqdm(stock_list.iterrows(), total=len(stock_list), desc=f"获取{date}数据")
        
        for idx, stock in pbar:
            code = stock['code']
            
            # 获取单个股票数据
            stock_data = self.get_single_stock_data(code, date, stock)
            
            if stock_data:
                daily_data.extend(stock_data)
            else:
                failed_count += 1
            
            pbar.set_postfix({'成功': len(daily_data), '失败': failed_count})
            
            # 添加延迟
            time.sleep(0.15)
        
        print(f"完成 {date}: 成功 {len(daily_data)} 条，失败 {failed_count} 个股票")
        return pd.DataFrame(daily_data)
    
    def get_trading_dates(self):
        """获取交易日列表"""
        print("获取交易日列表...")
        
        try:
            rs = bs.query_trade_dates(start_date=self.start_date, end_date=self.end_date)
            
            dates = []
            while (rs.error_code == '0') & rs.next():
                row = rs.get_row_data()
                if row[1] == '1':  # is_trading_day
                    dates.append(row[0])
            
            print(f"获取到 {len(dates)} 个交易日")
            return dates
        except Exception as e:
            print(f"获取交易日列表失败: {str(e)}")
            return []
    
    def collect_all_data(self, use_existing_stock_list=None):
        """
        收集所有数据
        
        参数:
        use_existing_stock_list: 
            - None: 自动检测，如果有stock_list.parquet则使用
            - True: 强制使用已有的股票清单
            - False: 重新获取股票清单
        """
        self.login()
        
        try:
            # 创建数据目录
            os.makedirs('data', exist_ok=True)
            
            # 检查已有数据
            existing_data, last_complete_date, last_date_complete = self.check_existing_data()
            
            # 获取交易日列表
            trading_dates = self.get_trading_dates()
            if not trading_dates:
                print("未获取到交易日数据")
                return None
            
            # 决定从哪个日期开始
            if existing_data is not None:
                if last_date_complete:
                    # 最后一天完整，从下一个交易日开始
                    start_from_date = None
                    for i, date in enumerate(trading_dates):
                        if date > last_complete_date:
                            start_from_date = date
                            trading_dates = trading_dates[i:]
                            break
                    
                    if start_from_date is None:
                        print("已下载所有数据！")
                        return existing_data
                else:
                    # 最后一天不完整，删除最后一天的数据并重新下载
                    print(f"\n删除 {last_complete_date} 的不完整数据...")
                    existing_data = existing_data[existing_data['date'] < last_complete_date]
                    
                    # 从最后一天开始
                    start_from_date = last_complete_date
                    start_idx = trading_dates.index(last_complete_date)
                    trading_dates = trading_dates[start_idx:]
                
                print(f"\n将从 {start_from_date} 开始下载，剩余 {len(trading_dates)} 个交易日")
            else:
                existing_data = pd.DataFrame()
                print(f"\n开始全新下载，共 {len(trading_dates)} 个交易日")
            
            # 获取股票列表
            if use_existing_stock_list is None:
                # 自动检测
                stock_list = self.get_stock_list_from_file()
                if stock_list is None:
                    use_existing_stock_list = False
                else:
                    use_existing_stock_list = True
            
            if use_existing_stock_list and os.path.exists('data/stock_list.parquet'):
                stock_list = self.get_stock_list_from_file()
            else:
                # 使用第一个交易日获取股票列表
                first_date = trading_dates[0] if trading_dates else self.start_date
                stock_list = self.get_stock_list(first_date)
                if stock_list.empty:
                    print("未能获取股票列表")
                    return None
                
                # 保存股票列表
                stock_list.to_parquet('data/stock_list.parquet', index=False)
                print(f"股票列表已保存: {len(stock_list)} 只股票")
            
            # 准备数据容器
            if not existing_data.empty:
                all_data = [existing_data]
            else:
                all_data = []
            
            # 收集每日数据
            for i, date in enumerate(trading_dates):
                daily_data = self.get_daily_data(date, stock_list)
                
                if not daily_data.empty:
                    all_data.append(daily_data)
                    
                    # 合并并保存
                    df_all = pd.concat(all_data, ignore_index=True)
                    df_all.to_parquet('data/daily_data_temp.parquet', index=False)
                    
                    # 显示进度
                    total_amount = daily_data['amount'].sum() / 1e8
                    print(f"保存进度: {i+1}/{len(trading_dates)}, {date} 成交额: {total_amount:.2f}亿元")
            
            # 最终数据
            if all_data:
                df_final = pd.concat(all_data, ignore_index=True)
                
                # 保存最终数据
                df_final.to_parquet('data/daily_data.parquet', index=False)
                
                # 保存失败记录
                if self.failed_stocks:
                    pd.DataFrame(self.failed_stocks).to_csv('data/failed_stocks.csv', index=False)
                    print(f"\n有 {len(self.failed_stocks)} 条失败记录，详见 failed_stocks.csv")
                
                # 生成摘要
                print("\n" + "="*50)
                print("数据收集完成！")
                print("="*50)
                print(f"日期范围: {df_final['date'].min()} 至 {df_final['date'].max()}")
                print(f"股票数量: {len(stock_list)}")
                print(f"总记录数: {len(df_final)}")
                print(f"总成交额: {df_final['amount'].sum() / 1e8:.2f} 亿元")
                
                # 每日统计
                daily_stats = df_final.groupby('date').agg({
                    'amount': 'sum',
                    'code': 'count'
                }).rename(columns={'amount': 'total_amount', 'code': 'stock_count'})
                daily_stats['total_amount'] = daily_stats['total_amount'] / 1e8
                
                print("\n最后5个交易日统计:")
                print(daily_stats.tail())
                
                return df_final
            
            return None
            
        except KeyboardInterrupt:
            print("\n程序被中断！数据已自动保存到 daily_data_temp.parquet")
            
        except Exception as e:
            print(f"\n发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            
        finally:
            self.logout()

# 使用示例
if __name__ == "__main__":
    # 创建收集器实例
    collector = StockDataCollector(
        start_date='2025-01-01',  # 根据实际需要调整
        end_date='2025-09-12'
    )
    
    print("="*60)
    print("股票数据收集工具")
    print("="*60)
    
    # 检查是否有已存在的股票清单
    if os.path.exists('data/stock_list.parquet'):
        print("\n检测到已有股票清单文件")
        print("1. 使用现有股票清单继续下载")
        print("2. 重新获取股票清单")
        choice = input("\n请选择 (1/2，默认1): ").strip() or '1'
        use_existing = (choice == '1')
    else:
        use_existing = None
    
    # 开始收集数据
    df_daily = collector.collect_all_data(use_existing_stock_list=use_existing)
    
    if df_daily is not None:
        print("\n数据文件已保存:")
        print("- data/daily_data.parquet (最终数据)")
        print("- data/stock_list.parquet (股票清单)")