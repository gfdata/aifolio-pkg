# -*- coding: utf-8 -*-
# @time: 2022/2/14 15:10
# @Author：lhf
# ----------------------
import sys
import pandas as pd


def parse_rqalpha(sys_analyser_dict: dict, time_zone='UTC'):
    """
    解析rqalpha框架的运行结果为pyfolio可使用的数据格式。(mod_sys_analyzer中存储的交易记录)
    :return:
    """
    this_name = sys._getframe().f_code.co_name

    # returns : pd.Series；使用单位净值
    try:
        cumrets = sys_analyser_dict['portfolio']['unit_net_value']
    except Exception as e:
        raise ValueError(f'{this_name}:unit_net_value数据缺失 {e}')
    cumrets = cumrets.resample('1D').last().fillna(method='ffill')  # 采样日频率
    pf_returns = cumrets.pct_change().fillna(0)
    pf_returns.index = pd.to_datetime(pf_returns.index)

    # positions : pd.DataFrame, optional
    pf_stock = pd.DataFrame()
    if 'stock_positions' in sys_analyser_dict.keys():
        temp = sys_analyser_dict['stock_positions']
        pf_stock['symbol'] = temp['order_book_id']
        pf_stock['datetime'] = temp.index
        pf_stock['values'] = temp['last_price'] * temp['quantity']

    pf_future = pd.DataFrame()  # 来自future的，要乘上价格乘数
    if 'future_positions' in sys_analyser_dict.keys():
        temp = sys_analyser_dict['future_positions']
        pf_future['symbol'] = temp['order_book_id']
        pf_future['datetime'] = temp.index

        def _net_values(se):
            se = se.fillna(0)  # 填充nan否则相减为nan
            return se['contract_multiplier'] * se['last_price'] * (se['LONG_quantity'] - se['SHORT_quantity'])

        pf_future['values'] = temp.apply(_net_values, axis=1)
    # cash现金
    pf_po = pd.concat([pf_stock, pf_future]).pivot_table(index='datetime', columns='symbol', values='values')
    pf_po.index = pd.to_datetime(pf_po.index)
    pf_po = pf_po.join(sys_analyser_dict['portfolio']['cash'].to_frame(name='cash'))

    # transactions : pd.DataFrame, optional
    pf_trans = pd.DataFrame()
    trades = sys_analyser_dict['trades']
    pf_trans['dt'] = trades['trading_datetime']
    pf_trans['symbol'] = trades['order_book_id']
    pf_trans['price'] = trades['last_price']

    def _amount_side(se):
        if se['side'] == 'BUY':
            return abs(se['last_quantity'])
        elif se['side'] == 'SELL':
            return 0 - abs(se['last_quantity'])

    pf_trans['amount'] = trades.apply(_amount_side, axis=1)
    pf_trans.index = pd.to_datetime(pf_trans.index)
    # 时区问题
    if time_zone:
        pf_returns.index = pf_returns.index.tz_localize('UTC')
        pf_po.index = pf_po.index.tz_localize('UTC')
        pf_trans.index = pf_trans.index.tz_localize('UTC')
    return pf_returns, pf_po, pf_trans
