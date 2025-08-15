
# MACD 回测代码使用说明
# # 基础使用
# backtester = MACDBacktester('your_stock_data.xlsx')
# backtester.calculate_macd(12, 26, 9)
# backtester.generate_signals()
# backtester.backtest()
# backtester.plot_results()

# # 参数优化
# results = backtester.optimize_parameters(
#     fast_range=(5, 20),
#     slow_range=(20, 40),
#     signal_range=(5, 15)
# )

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class MACDBacktester:
    """MACD策略回测系统"""
    
    def __init__(self, data_file=None):
        """
        初始化回测系统
        
        Parameters:
        -----------
        data_file : str, 股票数据Excel文件路径
        """
        self.data = None
        self.signals = []
        self.trades = []
        self.performance = {}
        
        if data_file:
            self.load_data(data_file)
    
    def load_data(self, file_path):
        """加载股票数据"""
        try:
            # 读取Excel文件
            self.data = pd.read_excel(file_path)
            
            # 确保日期列是datetime类型
            self.data['date'] = pd.to_datetime(self.data['date'])
            
            # 按日期排序
            self.data = self.data.sort_values('date').reset_index(drop=True)
            
            # 确保数值列是float类型
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'pctChg']
            for col in numeric_columns:
                if col in self.data.columns:
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            
            # 删除包含空值的行
            self.data = self.data.dropna(subset=['close'])
            
            print(f"成功加载数据: {len(self.data)} 条记录")
            print(f"日期范围: {self.data['date'].min()} 至 {self.data['date'].max()}")
            print(f"价格范围: ¥{self.data['close'].min():.2f} - ¥{self.data['close'].max():.2f}")
            
        except Exception as e:
            print(f"加载数据失败: {e}")
            raise
    
    def calculate_ema(self, prices, period):
        """
        计算指数移动平均线(EMA)
        
        Parameters:
        -----------
        prices : array-like, 价格序列
        period : int, EMA周期
        
        Returns:
        --------
        array, EMA值序列
        """
        prices = np.array(prices)
        ema = np.zeros_like(prices)
        ema[:] = np.nan
        
        if len(prices) < period:
            return ema
        
        # 计算第一个EMA值（使用简单移动平均）
        ema[period-1] = np.mean(prices[:period])
        
        # 计算后续EMA值
        multiplier = 2 / (period + 1)
        for i in range(period, len(prices)):
            ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
        
        return ema
    
    def calculate_macd(self, fast_period=12, slow_period=26, signal_period=9):
        """
        计算MACD指标
        
        Parameters:
        -----------
        fast_period : int, 快速EMA周期，默认12
        slow_period : int, 慢速EMA周期，默认26
        signal_period : int, 信号线EMA周期，默认9
        
        Returns:
        --------
        None, 结果存储在self.data中
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        prices = self.data['close'].values
        
        # 计算快速和慢速EMA
        ema_fast = self.calculate_ema(prices, fast_period)
        ema_slow = self.calculate_ema(prices, slow_period)
        
        # 计算DIF (MACD线)
        dif = ema_fast - ema_slow
        
        # 计算DEA (信号线) - 对DIF进行EMA平滑
        # 需要过滤掉NaN值
        valid_dif = dif[~np.isnan(dif)]
        dea_temp = self.calculate_ema(valid_dif, signal_period)
        
        # 将DEA对齐到原始数据
        dea = np.zeros_like(dif)
        dea[:] = np.nan
        valid_idx = np.where(~np.isnan(dif))[0]
        for i, idx in enumerate(valid_idx):
            if i < len(dea_temp):
                dea[idx] = dea_temp[i]
        
        # 计算MACD柱状图
        macd_hist = (dif - dea) * 2
        
        # 存储到数据框
        self.data['ema_fast'] = ema_fast
        self.data['ema_slow'] = ema_slow
        self.data['dif'] = dif
        self.data['dea'] = dea
        self.data['macd_hist'] = macd_hist
        
        print(f"MACD计算完成 (快线:{fast_period}, 慢线:{slow_period}, 信号线:{signal_period})")
    
    def generate_signals(self):
        """
        生成交易信号
        
        金叉（DIF上穿DEA）: 买入信号
        死叉（DIF下穿DEA）: 卖出信号
        """
        if 'dif' not in self.data.columns or 'dea' not in self.data.columns:
            raise ValueError("请先计算MACD指标")
        
        self.signals = []
        self.data['signal'] = 0  # 0: 无信号, 1: 买入, -1: 卖出
        
        # 寻找交叉点
        for i in range(1, len(self.data)):
            if pd.isna(self.data.loc[i, 'dif']) or pd.isna(self.data.loc[i, 'dea']):
                continue
            
            curr_diff = self.data.loc[i, 'dif'] - self.data.loc[i, 'dea']
            prev_diff = self.data.loc[i-1, 'dif'] - self.data.loc[i-1, 'dea']
            
            # 金叉 - 买入信号
            if prev_diff <= 0 and curr_diff > 0:
                self.data.loc[i, 'signal'] = 1
                self.signals.append({
                    'date': self.data.loc[i, 'date'],
                    'type': 'buy',
                    'price': self.data.loc[i, 'close'],
                    'index': i
                })
            
            # 死叉 - 卖出信号
            elif prev_diff >= 0 and curr_diff < 0:
                self.data.loc[i, 'signal'] = -1
                self.signals.append({
                    'date': self.data.loc[i, 'date'],
                    'type': 'sell',
                    'price': self.data.loc[i, 'close'],
                    'index': i
                })
        
        print(f"生成信号: {len([s for s in self.signals if s['type']=='buy'])} 个买入信号, "
              f"{len([s for s in self.signals if s['type']=='sell'])} 个卖出信号")
    
    def backtest(self, initial_capital=100000, commission=0.0003):
        """
        执行回测
        
        Parameters:
        -----------
        initial_capital : float, 初始资金，默认10万
        commission : float, 手续费率，默认0.03%
        """
        if not self.signals:
            print("没有交易信号，无法进行回测")
            return
        
        self.trades = []
        capital = initial_capital
        position = 0  # 持仓数量
        total_commission = 0
        
        for signal in self.signals:
            if signal['type'] == 'buy' and position == 0:
                # 买入 - 使用所有资金
                shares = int(capital / signal['price'] / (1 + commission))
                if shares > 0:
                    cost = shares * signal['price'] * (1 + commission)
                    capital -= cost
                    position = shares
                    total_commission += shares * signal['price'] * commission
                    
                    self.trades.append({
                        'buy_date': signal['date'],
                        'buy_price': signal['price'],
                        'shares': shares,
                        'buy_cost': cost
                    })
            
            elif signal['type'] == 'sell' and position > 0:
                # 卖出 - 卖出所有持仓
                revenue = position * signal['price'] * (1 - commission)
                capital += revenue
                total_commission += position * signal['price'] * commission
                
                # 更新最后一笔买入交易
                if self.trades and 'sell_date' not in self.trades[-1]:
                    self.trades[-1].update({
                        'sell_date': signal['date'],
                        'sell_price': signal['price'],
                        'sell_revenue': revenue,
                        'profit': revenue - self.trades[-1]['buy_cost'],
                        'return_pct': ((revenue - self.trades[-1]['buy_cost']) / 
                                     self.trades[-1]['buy_cost']) * 100
                    })
                
                position = 0
        
        # 如果还有持仓，按最后收盘价计算
        if position > 0:
            last_price = self.data.iloc[-1]['close']
            revenue = position * last_price * (1 - commission)
            if self.trades and 'sell_date' not in self.trades[-1]:
                self.trades[-1].update({
                    'sell_date': self.data.iloc[-1]['date'],
                    'sell_price': last_price,
                    'sell_revenue': revenue,
                    'profit': revenue - self.trades[-1]['buy_cost'],
                    'return_pct': ((revenue - self.trades[-1]['buy_cost']) / 
                                 self.trades[-1]['buy_cost']) * 100,
                    'is_open': True  # 标记为未平仓
                })
            capital += revenue
            position = 0
        
        # 计算性能指标
        final_capital = capital
        total_return = ((final_capital - initial_capital) / initial_capital) * 100
        
        completed_trades = [t for t in self.trades if 'return_pct' in t]
        winning_trades = [t for t in completed_trades if t['profit'] > 0]
        losing_trades = [t for t in completed_trades if t['profit'] <= 0]
        
        win_rate = (len(winning_trades) / len(completed_trades) * 100) if completed_trades else 0
        
        avg_win = np.mean([t['return_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['return_pct'] for t in losing_trades]) if losing_trades else 0
        
        max_win = max([t['return_pct'] for t in winning_trades]) if winning_trades else 0
        max_loss = min([t['return_pct'] for t in losing_trades]) if losing_trades else 0
        
        # 计算最大回撤
        cumulative_returns = []
        temp_capital = initial_capital
        for trade in completed_trades:
            temp_capital *= (1 + trade['return_pct']/100)
            cumulative_returns.append(temp_capital)
        
        if cumulative_returns:
            peak = cumulative_returns[0]
            max_drawdown = 0
            for value in cumulative_returns:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)
        else:
            max_drawdown = 0
        
        self.performance = {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'total_trades': len(completed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_win': max_win,
            'max_loss': max_loss,
            'max_drawdown': max_drawdown,
            'total_commission': total_commission,
            'profit_factor': abs(sum([t['profit'] for t in winning_trades]) / 
                                sum([t['profit'] for t in losing_trades])) if losing_trades else float('inf')
        }
        
        print("\n" + "="*60)
        print("回测结果汇总")
        print("="*60)
        print(f"初始资金: ¥{initial_capital:,.2f}")
        print(f"最终资金: ¥{final_capital:,.2f}")
        print(f"总收益率: {total_return:.2f}%")
        print(f"总交易次数: {len(completed_trades)}")
        print(f"胜率: {win_rate:.1f}%")
        print(f"盈利交易: {len(winning_trades)} 笔")
        print(f"亏损交易: {len(losing_trades)} 笔")
        print(f"平均盈利: {avg_win:.2f}%")
        print(f"平均亏损: {avg_loss:.2f}%")
        print(f"最大盈利: {max_win:.2f}%")
        print(f"最大亏损: {max_loss:.2f}%")
        print(f"最大回撤: {max_drawdown:.2f}%")
        print(f"手续费总计: ¥{total_commission:,.2f}")
        print("="*60)
    
    def plot_results(self, figsize=(15, 10)):
        """绘制回测结果图表"""
        if self.data is None:
            print("没有数据可以绘制")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=figsize, height_ratios=[2, 1, 1])
        
        # 1. 价格走势和交易信号
        ax1 = axes[0]
        ax1.plot(self.data['date'], self.data['close'], label='收盘价', color='black', linewidth=1)
        
        # 标记买卖点
        buy_signals = [s for s in self.signals if s['type'] == 'buy']
        sell_signals = [s for s in self.signals if s['type'] == 'sell']
        
        if buy_signals:
            buy_dates = [s['date'] for s in buy_signals]
            buy_prices = [s['price'] for s in buy_signals]
            ax1.scatter(buy_dates, buy_prices, color='green', marker='^', 
                       s=100, label='买入信号', zorder=5)
        
        if sell_signals:
            sell_dates = [s['date'] for s in sell_signals]
            sell_prices = [s['price'] for s in sell_signals]
            ax1.scatter(sell_dates, sell_prices, color='red', marker='v', 
                       s=100, label='卖出信号', zorder=5)
        
        ax1.set_title('股价走势及交易信号', fontsize=14, fontweight='bold')
        ax1.set_ylabel('价格 (¥)')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 2. MACD指标
        ax2 = axes[1]
        valid_data = self.data.dropna(subset=['dif', 'dea'])
        ax2.plot(valid_data['date'], valid_data['dif'], label='DIF', color='red', linewidth=1)
        ax2.plot(valid_data['date'], valid_data['dea'], label='DEA', color='blue', linewidth=1)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # MACD柱状图
        colors = ['green' if x > 0 else 'red' for x in valid_data['macd_hist']]
        ax2.bar(valid_data['date'], valid_data['macd_hist'], color=colors, alpha=0.3, label='MACD')
        
        ax2.set_title('MACD指标', fontsize=14, fontweight='bold')
        ax2.set_ylabel('MACD值')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # 3. 累计收益曲线
        ax3 = axes[2]
        
        # 计算累计收益
        cumulative_returns = [0]  # 从0开始
        cumulative_dates = [self.data['date'].iloc[0]]
        
        for trade in self.trades:
            if 'return_pct' in trade:
                if cumulative_returns:
                    cumulative_returns.append(cumulative_returns[-1] + trade['return_pct'])
                else:
                    cumulative_returns.append(trade['return_pct'])
                cumulative_dates.append(trade['sell_date'])
        
        if len(cumulative_returns) > 1:
            ax3.plot(cumulative_dates, cumulative_returns, marker='o', 
                    color='blue', linewidth=2, markersize=6)
            ax3.fill_between(cumulative_dates, 0, cumulative_returns, 
                            where=[r >= 0 for r in cumulative_returns], 
                            color='green', alpha=0.3, label='盈利')
            ax3.fill_between(cumulative_dates, 0, cumulative_returns, 
                            where=[r < 0 for r in cumulative_returns], 
                            color='red', alpha=0.3, label='亏损')
        
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_title('累计收益率', fontsize=14, fontweight='bold')
        ax3.set_xlabel('日期')
        ax3.set_ylabel('收益率 (%)')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        
        # 设置日期格式
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def print_trades(self, last_n=None):
        """
        打印交易记录
        
        Parameters:
        -----------
        last_n : int, 只显示最后n笔交易，None表示显示全部
        """
        if not self.trades:
            print("没有交易记录")
            return
        
        print("\n" + "="*80)
        print("详细交易记录")
        print("="*80)
        
        trades_to_show = self.trades[-last_n:] if last_n else self.trades
        
        for i, trade in enumerate(trades_to_show, 1):
            print(f"\n交易 #{i}:")
            print(f"  买入: {trade['buy_date'].strftime('%Y-%m-%d')} @ ¥{trade['buy_price']:.2f}")
            print(f"  数量: {trade['shares']} 股")
            
            if 'sell_date' in trade:
                print(f"  卖出: {trade['sell_date'].strftime('%Y-%m-%d')} @ ¥{trade['sell_price']:.2f}")
                print(f"  收益: ¥{trade['profit']:.2f} ({trade['return_pct']:.2f}%)")
                
                if trade.get('is_open'):
                    print(f"  状态: 未平仓（按最后收盘价计算）")
            else:
                print(f"  状态: 持仓中")
        
        print("="*80)
    
    def optimize_parameters(self, fast_range=(5, 20), slow_range=(20, 40), 
                           signal_range=(5, 15), step=1):
        """
        网格搜索优化MACD参数
        
        Parameters:
        -----------
        fast_range : tuple, 快线周期范围
        slow_range : tuple, 慢线周期范围
        signal_range : tuple, 信号线周期范围
        step : int, 步长
        
        Returns:
        --------
        dict, 最优参数和结果
        """
        print("开始参数优化...")
        print(f"快线范围: {fast_range}, 慢线范围: {slow_range}, 信号线范围: {signal_range}")
        
        best_params = None
        best_return = -float('inf')
        results = []
        
        total_combinations = (
            len(range(fast_range[0], fast_range[1]+1, step)) *
            len(range(slow_range[0], slow_range[1]+1, step)) *
            len(range(signal_range[0], signal_range[1]+1, step))
        )
        
        count = 0
        for fast in range(fast_range[0], fast_range[1]+1, step):
            for slow in range(slow_range[0], slow_range[1]+1, step):
                if fast >= slow:  # 快线周期必须小于慢线周期
                    continue
                    
                for signal in range(signal_range[0], signal_range[1]+1, step):
                    count += 1
                    
                    # 计算MACD
                    self.calculate_macd(fast, slow, signal)
                    self.generate_signals()
                    
                    # 回测
                    self.backtest(initial_capital=100000, commission=0.0003)
                    
                    # 记录结果
                    if self.performance:
                        total_return = self.performance['total_return']
                        win_rate = self.performance['win_rate']
                        
                        results.append({
                            'fast': fast,
                            'slow': slow,
                            'signal': signal,
                            'return': total_return,
                            'win_rate': win_rate,
                            'trades': self.performance['total_trades']
                        })
                        
                        # 更新最优参数
                        if total_return > best_return:
                            best_return = total_return
                            best_params = {
                                'fast': fast,
                                'slow': slow,
                                'signal': signal,
                                'return': total_return,
                                'win_rate': win_rate,
                                'trades': self.performance['total_trades']
                            }
                    
                    # 显示进度
                    if count % 10 == 0:
                        print(f"进度: {count}/{total_combinations} "
                              f"({count/total_combinations*100:.1f}%)")
        
        # 显示最优结果
        print("\n" + "="*60)
        print("参数优化结果")
        print("="*60)
        
        if best_params:
            print(f"最优参数组合:")
            print(f"  快线: {best_params['fast']}")
            print(f"  慢线: {best_params['slow']}")
            print(f"  信号线: {best_params['signal']}")
            print(f"最优收益率: {best_params['return']:.2f}%")
            print(f"胜率: {best_params['win_rate']:.1f}%")
            print(f"交易次数: {best_params['trades']}")
            
            # 用最优参数重新计算
            self.calculate_macd(best_params['fast'], best_params['slow'], best_params['signal'])
            self.generate_signals()
            self.backtest()
        else:
            print("未找到有效的参数组合")
        
        # 将结果转换为DataFrame便于分析
        results_df = pd.DataFrame(results)
        
        return {
            'best_params': best_params,
            'all_results': results_df
        }
    
    def filter_date_range(self, start_date=None, end_date=None):
        """
        筛选指定日期范围的数据
        
        Parameters:
        -----------
        start_date : str or datetime, 开始日期
        end_date : str or datetime, 结束日期
        """
        if self.data is None:
            print("请先加载数据")
            return
        
        # 保存原始数据
        if not hasattr(self, 'original_data'):
            self.original_data = self.data.copy()
        
        # 转换日期格式
        if start_date:
            start_date = pd.to_datetime(start_date)
        else:
            start_date = self.data['date'].min()
        
        if end_date:
            end_date = pd.to_datetime(end_date)
        else:
            end_date = self.data['date'].max()
        
        # 筛选数据
        mask = (self.data['date'] >= start_date) & (self.data['date'] <= end_date)
        self.data = self.data[mask].reset_index(drop=True)
        
        print(f"数据已筛选: {start_date.date()} 至 {end_date.date()}")
        print(f"共 {len(self.data)} 条记录")
    
    def reset_data(self):
        """恢复原始数据"""
        if hasattr(self, 'original_data'):
            self.data = self.original_data.copy()
            print("数据已恢复到原始状态")


# 使用示例
def main():
    """主函数 - 演示如何使用MACD回测系统"""
    
    # 1. 创建回测器实例并加载数据
    backtester = MACDBacktester()
    
    # 假设你已经有了从baostock下载的Excel文件
    excel_file = 'sh.601127.xlsx'  # 替换为你的文件路径
    
    try:
        backtester.load_data(excel_file)
    except:
        print("请确保Excel文件存在并包含正确的数据格式")
        return
    
    # 2. 使用默认参数进行回测
    print("\n" + "="*60)
    print("使用默认参数(12, 26, 9)进行回测")
    print("="*60)
    
    backtester.calculate_macd(fast_period=12, slow_period=26, signal_period=9)
    backtester.generate_signals()
    backtester.backtest(initial_capital=100000, commission=0.0003)
    
    # 3. 显示交易记录
    backtester.print_trades(last_n=5)  # 显示最后5笔交易
    
    # 4. 绘制图表
    backtester.plot_results(figsize=(15, 10))
    
    # 5. 筛选特定时间段
    print("\n" + "="*60)
    print("筛选最近6个月的数据")
    print("="*60)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    backtester.filter_date_range(start_date, end_date)
    
    # 重新计算和回测
    backtester.calculate_macd(fast_period=12, slow_period=26, signal_period=9)
    backtester.generate_signals()
    backtester.backtest(initial_capital=100000)
    
    # 6. 参数优化（可选，比较耗时）
    optimize = input("\n是否进行参数优化？(y/n): ")
    if optimize.lower() == 'y':
        print("\n" + "="*60)
        print("开始参数优化（可能需要几分钟）")
        print("="*60)
        
        # 恢复完整数据
        backtester.reset_data()
        
        # 优化参数
        optimization_results = backtester.optimize_parameters(
            fast_range=(8, 15),
            slow_range=(20, 30),
            signal_range=(7, 11),
            step=1
        )
        
        # 使用最优参数绘制结果
        if optimization_results['best_params']:
            backtester.plot_results(figsize=(15, 10))
            
            # 保存优化结果
            results_df = optimization_results['all_results']
            results_df.to_excel('optimization_results.xlsx', index=False)
            print("\n优化结果已保存到 optimization_results.xlsx")


if __name__ == "__main__":
    main()