# -*- coding: utf-8 -*-
# @time: 2022/2/14 15:10
# @Author：lhf
# ----------------------
import sys
import numpy as np
import pandas as pd


def parse_rqalpha_trans_concat(trans_first, trans):
    """
    parse_rqalpha_first 函数得到的首日trans，需要与后面的trans避免重复
    :param trans_first:
    :param trans:
    :return:
    """
    day = list(set(trans_first.index.date))
    assert len(day) == 1
    after = trans[trans.index.date != day]
    new = pd.concat([trans_first, after]).sort_index(ascending=True)
    return new


def parse_rqalpha_first(sys_analyser_dict: dict, time_zone='UTC'):
    """
    截取的记录第一天就有持仓时，需要对交易记录进行完善；做到尽量完整的开平仓配对
    :param sys_analyser_dict:
    :param time_zone:
    :return: pyfolio transactions 结构
    """
    # fixme 好像不用调整后续也能计算pnl；或者参考更规范的写法 aifolio.pyfolio092.round_trips.add_closing_transactions
    trans_stock = []
    if 'stock_positions' in sys_analyser_dict.keys():
        temp = sys_analyser_dict['stock_positions']
        dt_min = min(temp.index)
        first = temp[temp.index == dt_min]
        for ind, se in first.iterrows():
            se = se.fillna(0)  # 填充nan否则相减为nan
            each = {
                'dt': ind,
                'symbol': se['order_book_id'],
                'price': se['last_price'],
                'amount': temp['quantity']
            }
            trans_stock.append(each)
    trans_stock = pd.DataFrame(trans_stock)
    if not trans_stock.empty:
        trans_stock.set_index('dt', inplace=True)

    trans_future = []
    if 'future_positions' in sys_analyser_dict.keys():
        temp = sys_analyser_dict['future_positions']
        dt_min = min(temp.index)
        first = temp[temp.index == dt_min]
        for ind, se in first.iterrows():
            se = se.fillna(0)  # 填充nan否则相减为nan
            each = {
                'dt': ind,
                'symbol': se['order_book_id'],
                'price': se['last_price'],
                'amount': se['contract_multiplier'] * (se['LONG_quantity'] - se['SHORT_quantity'])
            }
            trans_future.append(each)
    trans_future = pd.DataFrame(trans_future)
    if not trans_future.empty:
        trans_future.set_index('dt', inplace=True)

    trans = pd.concat([trans_stock, trans_future])
    if time_zone:
        trans.index = trans.index.tz_localize(time_zone)
    return trans


def parse_rqalpha(sys_analyser_dict: dict, time_zone='UTC'):
    """
    解析rqalpha框架的运行结果为pyfolio可使用的数据格式。(mod_sys_analyzer中存储的交易记录)

    关于期货交易的分析
    * 之前错误做法，将具体合约改underlying，会导致round_trips匹配有问题
    * 比较好的做法，传入sector_mappings参数，将具体合约映射到underlying，生成品种统计
    * 而对于underlying所属板块分析，需要手动统计计算一遍

    关于板块的汇总分析
    * 也是生成逐笔round_trips之后，再变更symbol为对应板块名。参考 aifolio/pyfolio092/tears.py:832

    :return:
    """
    this_name = sys._getframe().f_code.co_name

    # returns : pd.Series
    try:
        cumrets = sys_analyser_dict['portfolio']['unit_net_value']  # rq自带的分析是用单位净值；或者使用total_value
    except Exception as e:
        raise ValueError(f'{this_name}:total_value 数据缺失 {e}')
    # 要用交易日历；因为 empyrical.stats.annual_return 日收益率用252调整；
    # cumrets = cumrets.resample('1D').last().fillna(method='ffill')  # resample是自然日，使用上述函数num_years变大导致结果偏小

    # 要用普通收益率；因为 empyrical.stats.cum_returns_final 累计收益用的是连乘(r+1).prod()
    pf_returns = cumrets.pct_change()
    # pf_returns = np.log(cumrets / cumrets.shift(1)).fillna(0.0)  # 对数收益率
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
    # cash概念：应该是总账户-市值暴露(带正负)；期货账户时，不是指可用资金：总账户-保证金；从此处使用可看出 aifolio/pyfolio092/tears.py:817
    # fixme 股票账户时，rq结果的market_value是持仓市值，还是当日持仓收益
    # pf_po = pf_po.join(sys_analyser_dict['portfolio']['cash'].to_frame(name='cash'))
    temp = sys_analyser_dict['portfolio']['total_value'] - sys_analyser_dict['portfolio']['market_value']
    pf_po = pf_po.join(temp.to_frame(name='cash'))

    # transactions : pd.DataFrame, optional
    pf_trans = pd.DataFrame()
    trades = sys_analyser_dict['trades']
    pf_trans['dt'] = trades['trading_datetime']
    pf_trans['symbol'] = trades['order_book_id']
    pf_trans['price'] = trades['last_price']

    # 需要合约乘数调整；pyfolio只是用amount*price来计算pnl
    if 'future_positions' in sys_analyser_dict.keys():
        multiplier_dict = sys_analyser_dict['future_positions'][['order_book_id', 'contract_multiplier']
        ].drop_duplicates(subset=['order_book_id']).set_index('order_book_id')['contract_multiplier'].to_dict()
    else:
        multiplier_dict = {}

    def _amount_side(se):
        m = multiplier_dict.get(se['order_book_id'], 1)
        v = abs(se['last_quantity']) * m
        if se['side'] == 'BUY' and se['position_effect'] == 'OPEN':
            return v
        elif se['side'] == 'SELL' and se['position_effect'] == 'CLOSE':
            return 0 - v
        elif se['side'] == 'SELL' and se['position_effect'] == 'OPEN':
            return 0 - v
        elif se['side'] == 'BUY' and se['position_effect'] == 'CLOSE':
            return v
        else:
            raise RuntimeError(f"trade组合要求为[BUY SELL]和[OPEN CLOSE]! got {se['side'], se['position_effect']}")

    pf_trans['amount'] = trades.apply(_amount_side, axis=1)
    pf_trans['dt'] = pd.to_datetime(pf_trans['dt'])
    pf_trans.set_index('dt', inplace=True)

    # 时区问题
    if time_zone:
        pf_returns.index = pf_returns.index.tz_localize(time_zone)
        pf_po.index = pf_po.index.tz_localize(time_zone)
        pf_trans.index = pf_trans.index.tz_localize(time_zone)
    return pf_returns, pf_po, pf_trans
