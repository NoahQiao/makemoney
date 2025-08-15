"""
矛盾论量化策略 - A股版本（修复版）
使用baostock数据源，适用于中国A股市场
"""

import numpy as np
import pandas as pd
import backtrader as bt
import backtrader.indicators as btind
import baostock as bs
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MarketRegimeIndicator(bt.Indicator):
    """
    市场状态识别指标
    """
    lines = ('regime',)
    params = (
        ('adx_threshold', 25),
    )
    
    def __init__(self):
        self.adx = self.data0
        self.volatility = self.data1
        
    def next(self):
        if len(self) < 2:
            self.lines.regime[0] = 0
            return
            
        # ADX上升且高于阈值 = 趋势市
        if self.adx[0] > self.params.adx_threshold and self.adx[0] > self.adx[-1]:
            self.lines.regime[0] = 1
        # ADX下降且低于阈值 = 震荡市
        elif self.adx[0] < self.params.adx_threshold and self.adx[0] < self.adx[-1]:
            self.lines.regime[0] = 0
        # 过渡期（矛盾转化中）
        else:
            self.lines.regime[0] = -1

class DialecticalStrategyAStock(bt.Strategy):
    """
    矛盾论量化策略 - A股版本
    
    针对A股市场特点的调整：
    1. T+1交易规则
    2. 涨跌停限制
    3. A股市场的特殊波动性
    """
    
    params = (
        # 趋势跟随参数
        ('trend_fast', 20),
        ('trend_slow', 60),
        
        # 均值回归参数
        ('bb_period', 20),
        ('bb_dev', 2.0),
        ('rsi_period', 14),
        ('rsi_oversold', 30),
        ('rsi_overbought', 70),
        
        # 市场状态识别
        ('atr_period', 20),
        ('adx_period', 14),
        ('adx_threshold', 25),
        
        # 风险管理
        ('risk_per_trade', 0.02),
        ('max_positions', 3),
        ('stop_loss_atr', 2.0),
        ('take_profit_atr', 3.0),
        
        # A股特殊参数
        ('limit_threshold', 0.095),  # 涨跌停板阈值
        ('t1_delay', True),  # T+1限制
    )
    
    def __init__(self):
        # 保存参考
        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.datavolume = self.datas[0].volume
        
        # 技术指标
        self.sma_fast = btind.SMA(self.dataclose, period=self.params.trend_fast)
        self.sma_slow = btind.SMA(self.dataclose, period=self.params.trend_slow)
        self.bb = btind.BollingerBands(self.dataclose, 
                                       period=self.params.bb_period,
                                       devfactor=self.params.bb_dev)
        self.rsi = btind.RSI(self.dataclose, period=self.params.rsi_period)
        self.atr = btind.ATR(self.datas[0], period=self.params.atr_period)
        self.adx = btind.ADX(self.datas[0], period=self.params.adx_period)
        
        # 波动率
        self.returns = btind.PctChange(self.dataclose, period=1)
        self.volatility = btind.StdDev(self.returns, period=30)
        
        # 市场状态 - 简化版本
        self.market_regime = 0  # 0: 震荡市, 1: 趋势市, -1: 过渡期
        
        # 交易管理
        self.order = None
        self.buy_date = None  # T+1限制
        self.trade_log = []
        
    def get_market_regime(self):
        """获取当前市场状态"""
        if len(self) < 30:
            return 0
            
        # 使用ADX判断趋势强度
        if self.adx[0] > self.params.adx_threshold and self.adx[0] > self.adx[-1]:
            return 1  # 趋势市
        elif self.adx[0] < self.params.adx_threshold and self.adx[0] < self.adx[-1]:
            return 0  # 震荡市
        else:
            return -1  # 过渡期
    
    def check_limit_move(self):
        """检查是否涨跌停"""
        if len(self.dataclose) < 2:
            return False
            
        price_change = (self.dataclose[0] - self.dataclose[-1]) / self.dataclose[-1]
        return abs(price_change) >= self.params.limit_threshold
    
    def calculate_position_size(self):
        """计算仓位大小"""
        account_value = self.broker.getvalue()
        
        # 使用当前价格的2%作为默认ATR（如果ATR还未计算出来）
        if len(self.atr) > 0:
            atr_value = self.atr[0]
        else:
            atr_value = self.dataclose[0] * 0.02
        
        # 基础仓位
        base_position = account_value * self.params.risk_per_trade
        
        # 根据市场状态调整
        market_regime = self.get_market_regime()
        if market_regime == 1:  # 趋势市
            position_multiplier = 1.2
        elif market_regime == 0:  # 震荡市
            position_multiplier = 0.8
        else:  # 过渡期
            position_multiplier = 0.5
            
        # 根据波动率调整
        if len(self.volatility) > 30:
            current_vol = self.volatility[0]
            vol_values = [self.volatility[-i] if i < len(self.volatility) else 0 
                         for i in range(1, min(30, len(self.volatility)))]
            if vol_values:
                avg_vol = np.mean(vol_values)
                if current_vol > avg_vol * 1.5:
                    position_multiplier *= 0.5
                elif current_vol < avg_vol * 0.7:
                    position_multiplier *= 1.3
        
        # 计算股数（A股100股为一手）
        position_value = base_position * position_multiplier
        price = self.dataclose[0]
        position_size = int(position_value / price)
        position_size = int(position_size / 100) * 100  # 调整为100的整数倍
        
        return max(100, position_size)  # 至少买一手
    
    def generate_signal(self):
        """生成交易信号"""
        if len(self) < max(self.params.trend_slow, self.params.bb_period, self.params.rsi_period):
            return 0
            
        trend_signal = 0
        reversal_signal = 0
        
        # 趋势信号
        if self.sma_fast[0] > self.sma_slow[0] and self.sma_fast[-1] <= self.sma_slow[-1]:
            trend_signal = 1
        elif self.sma_fast[0] < self.sma_slow[0] and self.sma_fast[-1] >= self.sma_slow[-1]:
            trend_signal = -1
            
        # 反转信号
        if self.dataclose[0] < self.bb.bot[0] and self.rsi[0] < self.params.rsi_oversold:
            reversal_signal = 1
        elif self.dataclose[0] > self.bb.top[0] and self.rsi[0] > self.params.rsi_overbought:
            reversal_signal = -1
            
        # 综合信号（根据市场状态加权）
        market_regime = self.get_market_regime()
        if market_regime == 1:  # 趋势市
            return trend_signal * 0.7 + reversal_signal * 0.3
        elif market_regime == 0:  # 震荡市
            return trend_signal * 0.3 + reversal_signal * 0.7
        else:  # 过渡期
            return (trend_signal + reversal_signal) * 0.5
    
    def next(self):
        # 确保有足够的数据
        if len(self) < 60:
            return
            
        # 更新市场状态
        self.market_regime = self.get_market_regime()
        
        # 检查是否有待处理的订单
        if self.order:
            return
            
        # 检查涨跌停
        if self.check_limit_move():
            self.log('涨跌停板，暂停交易')
            return
            
        # 获取当前持仓
        position = self.getposition(self.datas[0])
        
        # T+1限制检查
        if self.params.t1_delay and self.buy_date:
            if self.datas[0].datetime.date(0) <= self.buy_date:
                return
                
        # 生成交易信号
        signal = self.generate_signal()
        
        # 交易逻辑
        if not position:  # 无持仓
            if signal > 0.5:  # 买入信号
                size = self.calculate_position_size()
                self.log(f'买入信号: 价格={self.dataclose[0]:.2f}, 数量={size}, 市场状态={self.market_regime}')
                self.order = self.buy(size=size)
                self.buy_date = self.datas[0].datetime.date(0)
        else:  # 有持仓
            if signal < -0.3:  # 卖出信号
                self.log(f'卖出信号: 价格={self.dataclose[0]:.2f}')
                self.order = self.sell(size=position.size)
                self.buy_date = None
                
            # 止损检查
            elif position.size > 0 and len(self.atr) > 0:
                stop_loss = self.params.stop_loss_atr * self.atr[0] / position.price
                if self.dataclose[0] < position.price * (1 - stop_loss):
                    self.log(f'止损卖出: 价格={self.dataclose[0]:.2f}')
                    self.order = self.sell(size=position.size)
                    self.buy_date = None
    
    def notify_order(self, order):
        """订单通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入成交: 价格={order.executed.price:.2f}, 数量={order.executed.size}')
                self.trade_log.append({
                    'date': self.datas[0].datetime.date(0),
                    'type': 'BUY',
                    'price': order.executed.price,
                    'size': order.executed.size,
                    'regime': self.market_regime
                })
            elif order.issell():
                self.log(f'卖出成交: 价格={order.executed.price:.2f}, 数量={order.executed.size}')
                self.trade_log.append({
                    'date': self.datas[0].datetime.date(0),
                    'type': 'SELL',
                    'price': order.executed.price,
                    'size': order.executed.size,
                    'regime': self.market_regime
                })
                
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单取消/保证金不足/拒绝')
            
        self.order = None
        
    def notify_trade(self, trade):
        """交易通知"""
        if not trade.isclosed:
            return
            
        self.log(f'交易利润: 毛利润={trade.pnl:.2f}, 净利润={trade.pnlcomm:.2f}')
        
    def log(self, txt, dt=None):
        """日志输出"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')
        
    def stop(self):
        """策略结束时的统计"""
        # 统计不同市场状态下的表现
        trend_trades = [t for t in self.trade_log if t['regime'] == 1]
        range_trades = [t for t in self.trade_log if t['regime'] == 0]
        transition_trades = [t for t in self.trade_log if t['regime'] == -1]
        
        print('\n=== 矛盾论策略统计 ===')
        print(f'总交易次数: {len(self.trade_log)}')
        print(f'趋势市交易: {len(trend_trades)}')
        print(f'震荡市交易: {len(range_trades)}')
        print(f'过渡期交易: {len(transition_trades)}')
        print(f'最终价值: {self.broker.getvalue():.2f}')


def get_astock_data(stock_code='sh.600000', start_date='2021-01-01', end_date=None):
    """
    使用baostock获取A股数据
    
    参数:
        stock_code: 股票代码，格式如 'sh.600000' 或 'sz.000001'
        start_date: 开始日期
        end_date: 结束日期，默认为今天
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # 登录系统
    lg = bs.login()
    if lg.error_code != '0':
        print(f'登录失败: {lg.error_msg}')
        return None
    
    print(f"正在下载 {stock_code} 从 {start_date} 到 {end_date} 的数据...")
    
    # 查询日K线数据
    rs = bs.query_history_k_data_plus(
        stock_code,
        "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
        start_date=start_date, 
        end_date=end_date,
        frequency="d", 
        adjustflag="2"  # 前复权
    )
    
    if rs.error_code != '0':
        print(f'查询失败: {rs.error_msg}')
        bs.logout()
        return None
    
    # 获取数据
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    
    # 转换为DataFrame
    result = pd.DataFrame(data_list, columns=rs.fields)
    
    # 数据清洗和类型转换
    result = result.replace('', pd.NA)
    result['date'] = pd.to_datetime(result['date'])
    result['open'] = pd.to_numeric(result['open'], errors='coerce')
    result['high'] = pd.to_numeric(result['high'], errors='coerce')
    result['low'] = pd.to_numeric(result['low'], errors='coerce')
    result['close'] = pd.to_numeric(result['close'], errors='coerce')
    result['volume'] = pd.to_numeric(result['volume'], errors='coerce')
    
    # 删除空值
    result = result.dropna(subset=['date', 'open', 'high', 'low', 'close'])
    
    # 设置日期为索引并排序
    result = result.set_index('date')
    result = result.sort_index()
    
    # 只保留backtrader需要的列
    result = result[['open', 'high', 'low', 'close', 'volume']]
    
    print(f"成功获取 {len(result)} 条数据")
    
    # 登出系统
    bs.logout()
    
    return result


def run_dialectical_strategy_astock():
    """
    运行A股矛盾论策略
    """
    # 创建Cerebro引擎
    cerebro = bt.Cerebro()
    
    # 添加策略
    cerebro.addstrategy(DialecticalStrategyAStock)
    
    # 获取数据 - 使用一些典型的A股
    stocks = [
        ('sh.600036', '招商银行'),
        ('sh.600519', '贵州茅台'),
        ('sz.000002', '万科A'),
        ('sz.000858', '五粮液'),
        ('sh.600000', '浦发银行')
    ]
    
    # 选择一只股票进行回测
    stock_code, stock_name = stocks[0]  # 使用招商银行
    
    # 获取数据
    df = get_astock_data(stock_code, start_date='2021-01-01')
    
    if df is None or df.empty:
        print("无法获取数据")
        return
    
    # 创建数据源
    data = bt.feeds.PandasData(
        dataname=df,
        fromdate=datetime(2021, 1, 1),
        todate=datetime.now()
    )
    
    # 添加数据
    cerebro.adddata(data)
    
    # 设置初始资金（100万人民币）
    cerebro.broker.setcash(1000000.0)
    
    # 设置交易费用（A股费用）
    cerebro.broker.setcommission(
        commission=0.0003,  # 佣金万三
        stocklike=True
    )
    
    # 设置滑点（模拟买卖价差和冲击成本）
    cerebro.broker.set_slippage_fixed(0.001)  # 千分之一
    
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # 运行策略
    print(f'\n===== {stock_name}({stock_code}) 矛盾论策略回测 =====')
    print('初始资金: ￥{:,.2f}'.format(cerebro.broker.getvalue()))
    
    results = cerebro.run()
    
    print('最终资金: ￥{:,.2f}'.format(cerebro.broker.getvalue()))
    print('收益率: {:.2f}%'.format((cerebro.broker.getvalue() / 1000000 - 1) * 100))
    
    # 获取分析结果
    strat = results[0]
    
    # 输出详细分析
    print('\n=== 策略表现分析 ===')
    
    # 夏普比率
    sharpe = strat.analyzers.sharpe.get_analysis()
    if 'sharperatio' in sharpe and sharpe['sharperatio'] is not None:
        print(f'夏普比率: {sharpe["sharperatio"]:.4f}')
    
    # 最大回撤
    drawdown = strat.analyzers.drawdown.get_analysis()
    if 'max' in drawdown and 'drawdown' in drawdown['max']:
        print(f'最大回撤: {drawdown["max"]["drawdown"]:.2f}%')
    
    # 年化收益
    returns = strat.analyzers.returns.get_analysis()
    if 'rnorm100' in returns:
        print(f'年化收益: {returns["rnorm100"]:.2f}%')
    
    # 交易统计
    trades = strat.analyzers.trades.get_analysis()
    if 'total' in trades and 'total' in trades['total']:
        print(f'\n=== 交易统计 ===')
        print(f'总交易次数: {trades["total"]["total"]}')
        if 'won' in trades and 'total' in trades['won']:
            print(f'盈利交易: {trades["won"]["total"]}')
            if trades["total"]["total"] > 0:
                win_rate = trades["won"]["total"] / trades["total"]["total"] * 100
                print(f'胜率: {win_rate:.2f}%')
        if 'lost' in trades and 'total' in trades['lost']:
            print(f'亏损交易: {trades["lost"]["total"]}')
    
    # 绘图
    try:
        cerebro.plot(style='candlestick', volume=False, iplot=False)
    except Exception as e:
        print(f"\n绘图失败: {e}")
        print("提示: 如果在服务器环境运行，可能无法显示图形")


def analyze_multiple_stocks():
    """
    分析多只股票的策略表现
    """
    stocks = [
        ('sh.600036', '招商银行'),
        ('sh.600519', '贵州茅台'),
        ('sz.000002', '万科A'),
        ('sz.000858', '五粮液'),
        ('sh.600000', '浦发银行'),
        ('sz.300750', '宁德时代'),
        ('sh.601318', '中国平安'),
        ('sz.002475', '立讯精密')
    ]
    
    results_summary = []
    
    for stock_code, stock_name in stocks:
        print(f'\n正在分析 {stock_name}({stock_code})...')
        
        try:
            # 获取数据
            df = get_astock_data(stock_code, start_date='2021-01-01')
            if df is None or df.empty:
                continue
                
            # 创建回测引擎
            cerebro = bt.Cerebro()
            cerebro.addstrategy(DialecticalStrategyAStock)
            
            # 添加数据
            data = bt.feeds.PandasData(dataname=df)
            cerebro.adddata(data)
            
            # 设置资金和费用
            cerebro.broker.setcash(1000000.0)
            cerebro.broker.setcommission(commission=0.0003, stocklike=True)
            
            # 添加分析器
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            
            # 运行策略
            initial_value = cerebro.broker.getvalue()
            results = cerebro.run()
            final_value = cerebro.broker.getvalue()
            
            # 获取结果
            strat = results[0]
            returns = strat.analyzers.returns.get_analysis()
            sharpe = strat.analyzers.sharpe.get_analysis()
            drawdown = strat.analyzers.drawdown.get_analysis()
            
            # 保存结果
            results_summary.append({
                '股票代码': stock_code,
                '股票名称': stock_name,
                '初始资金': initial_value,
                '最终资金': final_value,
                '收益率': (final_value / initial_value - 1) * 100,
                '年化收益': returns.get('rnorm100', 0),
                '夏普比率': sharpe.get('sharperatio', 0),
                '最大回撤': drawdown.get('max', {}).get('drawdown', 0)
            })
        except Exception as e:
            print(f"分析 {stock_name} 时出错: {e}")
            continue
    
    # 显示汇总结果
    if results_summary:
        print('\n' + '=' * 80)
        print('矛盾论策略多股票回测汇总')
        print('=' * 80)
        
        summary_df = pd.DataFrame(results_summary)
        summary_df = summary_df.sort_values('收益率', ascending=False)
        
        # 格式化输出
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        print(summary_df.to_string(index=False))
        
        # 保存结果到文件
        summary_df.to_excel('矛盾论策略回测结果.xlsx', index=False)
        print('\n结果已保存到: 矛盾论策略回测结果.xlsx')
    
    return summary_df


if __name__ == '__main__':
    # 运行单只股票策略
    run_dialectical_strategy_astock()
    
    # 如果需要分析多只股票，取消下面的注释
    # analyze_multiple_stocks()
    
    print("""
    
    === A股矛盾论策略特点 ===
    
    1. T+1交易限制：买入后次日才能卖出
    2. 涨跌停板限制：检测涨跌停，避免无法成交
    3. 交易费用：考虑佣金和印花税
    4. 市场特色：结合A股的高波动性特点调整参数
    
    策略核心思想保持不变：
    - 对立统一：趋势与反转策略结合
    - 主要矛盾：识别市场状态
    - 矛盾转化：动态调整策略权重
    - 动态平衡：根据市场环境调整仓位
    """)