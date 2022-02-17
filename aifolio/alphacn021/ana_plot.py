# -*- coding: utf-8 -*-
"""
@time: 2021/11/23 14:17
@file: ana_plot.py

"""
import itertools
import pandas as pd
from scipy.stats import morestats
from collections import Iterable
from cached_property import cached_property

from . import alphalens as al
from .ana_calc import AnalyzerCalc as FactorAnalyzer


class FactorAnalyzerResult(object):  # 参考jqfactor-1.32.9的analyze.py部分
    """因子分析结果

    用于访问因子分析的结果, 大部分为惰性属性, 在访问才会计算结果并返回

    所有属性列表:
        factor_data:返回因子值
            - 类型: pandas.Series
            - index: 为日期和股票代码的MultiIndex
        clean_factor_data: 去除 nan/inf, 整理后的因子值、forward_return 和分位数
            - 类型: pandas.DataFrame
            - index: 为日期和股票代码的MultiIndex
            - columns: 根据period选择后的forward_return
                    (如果调仓周期为1天, 那么 forward_return 为
                        [第二天的收盘价-今天的收盘价]/今天的收盘价),
                    因子值、行业分组、分位数数组、权重
        mean_return_by_quantile: 按分位数分组加权平均因子收益
            - 类型: pandas.DataFrame
            - index: 分位数分组
            - columns: 调仓周期
        mean_return_std_by_quantile: 按分位数分组加权因子收益标准差
            - 类型: pandas.DataFrame
            - index: 分位数分组
            - columns: 调仓周期
        mean_return_by_date: 按分位数及日期分组加权平均因子收益
            - 类型: pandas.DataFrame
            - index: 为日期和分位数的MultiIndex
            - columns: 调仓周期
        mean_return_std_by_date: 按分位数及日期分组加权因子收益标准差
            - 类型: pandas.DataFrame
            - index: 为日期和分位数的MultiIndex
            - columns: 调仓周期
        mean_return_by_group: 按分位数及行业分组加权平均因子收益
            - 类型: pandas.DataFrame
            - index: 为行业和分位数的MultiIndex
            - columns: 调仓周期
        mean_return_std_by_group: 按分位数及行业分组加权因子收益标准差
            - 类型: pandas.DataFrame
            - index: 为行业和分位数的MultiIndex
            - columns: 调仓周期
        mean_return_spread_by_quantile: 最高分位数因子收益减最低分位数因子收益每日均值
            - 类型: pandas.DataFrame
            - index: 日期
            - columns: 调仓周期
        mean_return_spread_std_by_quantile: 最高分位数因子收益减最低分位数因子收益每日标准差
            - 类型: pandas.DataFrame
            - index: 日期
            - columns: 调仓周期
        cumulative_return_by_quantile:各分位数每日累积收益
            - 类型: pandas.DataFrame
            - index: 日期
            - columns: 调仓周期和分位数
        cumulative_returns: 按因子值加权多空组合每日累积收益
            - 类型: pandas.DataFrame
            - index: 日期
            - columns: 调仓周期
        top_down_cumulative_returns: 做多最高分位做空最低分位多空组合每日累计收益
            - 类型: pandas.DataFrame
            - index: 日期
            - columns: 调仓周期
        ic: 信息比率
            - 类型: pandas.DataFrame
            - index: 日期
            - columns: 调仓周期
        ic_by_group: 分行业信息比率
            - 类型: pandas.DataFrame
            - index: 行业
            - columns: 调仓周期
        ic_monthly: 月度信息比率
            - 类型: pandas.DataFrame
            - index: 月度
            - columns: 调仓周期表
        quantile_turnover: 换手率
            - 类型: dict
            - 键: 调仓周期
                - index: 日期
                - columns: 分位数分组

    所有方法列表:
        calc_mean_return_by_quantile:
            计算按分位数分组加权因子收益和标准差
        calc_factor_returns:
            计算按因子值加权组合每日收益
        compute_mean_returns_spread:
            计算两个分位数相减的因子收益和标准差
        calc_factor_alpha_beta:
            计算因子的 alpha 和 beta
        calc_factor_information_coefficient:
            计算每日因子信息比率 (IC值)
        calc_mean_information_coefficient:
            计算因子信息比率均值 (IC值均值)
        calc_average_cumulative_return_by_quantile:
            按照当天的分位数算分位数未来和过去的收益均值和标准差
        calc_cumulative_return_by_quantile:
            计算各分位数每日累积收益
        calc_cumulative_returns:
            计算按因子值加权多空组合每日累积收益
        calc_top_down_cumulative_returns:
            计算做多最高分位做空最低分位多空组合每日累计收益
        calc_autocorrelation:
            根据调仓周期确定滞后期的每天计算因子自相关性
        calc_autocorrelation_n_days_lag:
            滞后 1 - n 天因子值自相关性
        calc_quantile_turnover_mean_n_days_lag:
            各分位数 1 - n 天平均换手率
        calc_ic_mean_n_days_lag:
            滞后 0 - n 天 forward return 信息比率

        plot_returns_table: 打印因子收益表
        plot_turnover_table: 打印换手率表
        plot_information_table: 打印信息比率(IC)相关表
        plot_quantile_statistics_table: 打印各分位数统计表
        plot_ic_ts: 画信息比率(IC)时间序列图
        plot_ic_hist: 画信息比率分布直方图
        plot_ic_qq: 画信息比率 qq 图
        plot_quantile_returns_bar: 画各分位数平均收益图
        plot_mean_quantile_returns_spread_time_series: 画最高分位减最低分位收益图
        plot_ic_by_group: 画按行业分组信息比率(IC)图
        plot_factor_auto_correlation: 画因子自相关图
        plot_top_bottom_quantile_turnover: 画最高最低分位换手率图
        plot_monthly_ic_heatmap: 画月度信息比率(IC)图
        plot_cumulative_returns: 画按因子值加权组合每日累积收益图
        plot_top_down_cumulative_returns: 画做多最大分位数做空最小分位数组合每日累积收益图
        plot_cumulative_returns_by_quantile: 画各分位数每日累积收益图
        plot_quantile_average_cumulative_return: 因子预测能力平均累计收益图
        plot_events_distribution: 画有效因子数量统计图

        create_summary_tear_sheet: 因子值特征分析
        create_returns_tear_sheet: 因子收益分析
        create_information_tear_sheet: 因子 IC 分析
        create_turnover_tear_sheet: 因子换手率分析
        create_event_returns_tear_sheet: 因子预测能力分析
        create_full_tear_sheet: 全部分析

        plot_disable_chinese_label: 关闭中文图例显示
    """

    __BANNED_ATTR = set([
        'gen_factor_data',
        'gen_clean_factor_and_forward_returns',
        'convert_result_to_dict',
    ])

    def __init__(self, factor_analyzer):
        if not isinstance(factor_analyzer, FactorAnalyzer):
            raise ValueError('needs a FactorAnalyzer instance')
        self._factor_analyzer = factor_analyzer

    def __getattr__(self, name):
        if not name.startswith('_') and name not in self.__BANNED_ATTR:
            try:
                return getattr(self._factor_analyzer, name)
            except AttributeError as e:
                if "has no attribute '{name}'".format(name=name) not in str(e):
                    raise

        raise AttributeError(
            "'%s' object has no attribute '%s'" % (self.__class__.__name__, name)
        )

    def __dir__(self):
        self_attr = [
            k for k in itertools.chain(dir(self.__class__), self.__dict__)
            if not k.endswith("__BANNED_ATTR") and not k.endswith('plot_quantile_returns_violin')
        ]
        analyzer_attr = [
            k for k in dir(self._factor_analyzer)
            if not k.startswith('_') and k not in self.__BANNED_ATTR
        ]
        return self_attr + analyzer_attr

    @cached_property
    def _alp(self):
        from . import alplot
        return alplot

    def plot_returns_table(self, demeaned=False, group_adjust=False):
        """打印因子收益表

        参数:
        demeaned:
        - True: 使用超额收益计算 (基准收益被认为是每日所有股票收益按照weight列中权重的加权的均值)
        - False: 不使用超额收益
        group_adjust:
        - True: 使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
        - False: 不使用行业中性收益
        """
        mean_return_by_quantile = self.calc_mean_return_by_quantile(
            by_date=False, by_group=False, demeaned=demeaned, group_adjust=group_adjust,
            _rename_forward_return_columns=False
        )[0].apply(al.rate_of_return, axis=0)

        mean_returns_spread, _ = self.compute_mean_returns_spread(
            upper_quant=self._factor_analyzer._quantiles,
            lower_quant=1,
            by_date=True,
            by_group=False,
            demeaned=demeaned,
            group_adjust=group_adjust,
            _rename_forward_return_columns=False
        )

        self._alp.plot_returns_table(
            self.calc_factor_alpha_beta(demeaned=demeaned, _rename_forward_return_columns=False),
            mean_return_by_quantile,
            mean_returns_spread
        )

    def plot_turnover_table(self):
        """打印换手率表"""
        # quantile_turnover = self.quantile_turnover
        # quantile_turnover = dict((int(k.rsplit('_', 1)[-1]), quantile_turnover[k])
        #                          for k in quantile_turnover)
        self._alp.plot_turnover_table(
            self.calc_autocorrelation(_rename_forward_return_columns=False),
            self.quantile_turnover
        )

    def plot_information_table(self, group_adjust=False, method=None):
        """打印信息比率 (IC)相关表

        参数:
        group_adjust:
        - True：使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
        - False：不使用行业中性收益
        method：
        - 'rank'：用秩相关系数计算IC值
        - 'normal':用相关系数计算IC值
        """
        ic = self.calc_factor_information_coefficient(group_adjust=group_adjust, by_group=False,
                                                      method=method,
                                                      _rename_forward_return_columns=False)
        self._alp.plot_information_table(ic)

    def plot_quantile_statistics_table(self):
        """打印各分位数统计表"""
        self._alp.plot_quantile_statistics_table(self._factor_analyzer._clean_factor_data)

    def plot_ic_ts(self, group_adjust=False, method=None):
        """画信息比率(IC)时间序列图

        参数:
        group_adjust:
        - True: 使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
        - False: 不使用行业中性收益
        method:
        - 'rank': 用秩相关系数计算IC值
        - 'normal':用相关系数计算IC值
        """
        ic = self.calc_factor_information_coefficient(group_adjust=group_adjust, by_group=False,
                                                      method=method,
                                                      _rename_forward_return_columns=False)
        self._alp.plot_ic_ts(ic)

    def plot_ic_hist(self, group_adjust=False, method=None):
        """画信息比率分布直方图

        参数:
        group_adjust:
        - True: 使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
        - False: 不使用行业中性收益
        method:
        - 'rank': 用秩相关系数计算IC值
        - 'normal': 用相关系数计算IC值
        """
        ic = self.calc_factor_information_coefficient(group_adjust=group_adjust, by_group=False,
                                                      method=method,
                                                      _rename_forward_return_columns=False)
        self._alp.plot_ic_hist(ic)

    def plot_ic_qq(self, group_adjust=False, method=None, theoretical_dist=None):
        """画信息比率 qq 图

        参数:
        group_adjust:
        - True: 使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
        - False: 不使用行业中性收益
        method:
        - 'rank': 用秩相关系数计算IC值
        - 'normal': 用相关系数计算IC值
        theoretical_dist:
        - 'norm': 正态分布
        - 't': t 分布
        """
        theoretical_dist = 'norm' if theoretical_dist is None else theoretical_dist
        theoretical_dist = morestats._parse_dist_kw(theoretical_dist)
        ic = self.calc_factor_information_coefficient(group_adjust=group_adjust, by_group=False,
                                                      method=method,
                                                      _rename_forward_return_columns=False)
        self._alp.plot_ic_qq(ic, theoretical_dist=theoretical_dist)

    def plot_quantile_returns_bar(self, by_group=False, demeaned=False, group_adjust=False):
        """画各分位数平均收益图

        参数:
        by_group:
        - True: 各行业的各分位数平均收益图
        - False: 各分位数平均收益图
        demeaned:
        - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        - True: 使用行业中性化后的收益计算累积收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        """
        mean_return_by_quantile = self.calc_mean_return_by_quantile(
            by_date=False, by_group=by_group, demeaned=demeaned, group_adjust=group_adjust,
            _rename_forward_return_columns=False
        )[0].apply(al.rate_of_return, axis=0)

        self._alp.plot_quantile_returns_bar(
            mean_return_by_quantile, by_group=by_group, ylim_percentiles=None
        )

    def plot_quantile_returns_violin(self, demeaned=False, group_adjust=False,
                                     ylim_percentiles=(1, 99)):
        """"""
        mean_return_by_date = self.calc_mean_return_by_quantile(
            by_date=True, by_group=False, demeaned=demeaned, group_adjust=group_adjust,
            _rename_forward_return_columns=False
        )[0].apply(al.rate_of_return, axis=0)

        self._alp.plot_quantile_returns_violin(mean_return_by_date,
                                               ylim_percentiles=ylim_percentiles)

    def plot_mean_quantile_returns_spread_time_series(self, demeaned=False, group_adjust=False,
                                                      bandwidth=1):
        """画最高分位减最低分位收益图

        参数:
        demeaned:
        - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        - True: 使用行业中性化后的收益计算累积收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        bandwidth: n, 加减 n 倍当日标准差
        """
        mean_returns_spread, mean_returns_spread_std = self.compute_mean_returns_spread(
            upper_quant=self._factor_analyzer._quantiles,
            lower_quant=1,
            by_date=True,
            by_group=False,
            demeaned=demeaned,
            group_adjust=group_adjust,
            _rename_forward_return_columns=False
        )

        self._alp.plot_mean_quantile_returns_spread_time_series(
            mean_returns_spread, std_err=mean_returns_spread_std,
            bandwidth=bandwidth, quantiles=self._factor_analyzer._quantiles
        )

    def plot_ic_by_group(self, group_adjust=False, method=None):
        """画按行业分组信息比率(IC)图

        参数:
        group_adjust:
        - True: 使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
        - False: 不使用行业中性收益
        method:
        - 'rank': 用秩相关系数计算IC值
        - 'normal': 用相关系数计算IC值
        """
        ic_by_group = self.calc_mean_information_coefficient(group_adjust=group_adjust,
                                                             by_group=True, method=method,
                                                             _rename_forward_return_columns=False)
        self._alp.plot_ic_by_group(ic_by_group)

    def plot_factor_auto_correlation(self, periods=None, rank=True):
        """画因子自相关图

        参数:
        periods: 滞后周期
        rank:
        - True: 用秩相关系数
        - False: 用相关系数
        """
        if periods is None:
            periods = self._factor_analyzer._periods
        if not isinstance(periods, Iterable):
            periods = (periods,)
        periods = tuple(periods)
        for p in periods:
            if p in self._factor_analyzer._periods:
                self._alp.plot_factor_rank_auto_correlation(
                    self.calc_autocorrelation(rank=rank,
                                              _rename_forward_return_columns=False)[p],
                    period=p
                )

    def plot_top_bottom_quantile_turnover(self, periods=None):
        """画最高最低分位换手率图

        参数:
        periods: 调仓周期
        """
        quantile_turnover = self.quantile_turnover
        quantile_turnover = dict((int(k.rsplit('_', 1)[-1]), quantile_turnover[k])
                                 for k in quantile_turnover)

        if periods is None:
            periods = self._factor_analyzer._periods
        if not isinstance(periods, Iterable):
            periods = (periods,)
        periods = tuple(periods)
        for p in periods:
            if p in self._factor_analyzer._periods:
                self._alp.plot_top_bottom_quantile_turnover(quantile_turnover[p], period=p)

    def plot_monthly_ic_heatmap(self, group_adjust=False):
        """画月度信息比率(IC)图

        参数:
        group_adjust:
        - True: 使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
        - False: 不使用行业中性收益
        """
        ic_monthly = self.calc_mean_information_coefficient(group_adjust=group_adjust,
                                                            by_group=False,
                                                            by_time="M",
                                                            _rename_forward_return_columns=False)
        self._alp.plot_monthly_ic_heatmap(ic_monthly)

    def plot_cumulative_returns(self, period=None, demeaned=False, group_adjust=False):
        """画按因子值加权组合每日累积收益图

        参数:
        periods: 调仓周期
        demeaned:
        详见 calc_factor_returns 中 demeaned 参数
        - True: 对因子值加权组合每日收益的权重去均值 (每日权重 = 每日权重 - 每日权重的均值),
                使组合转换为cash-neutral多空组合
        - False: 不对权重去均值
        group_adjust:
        详见 calc_factor_returns 中 group_adjust 参数
        - True: 对权重分行业去均值 (每日权重 = 每日权重 - 每日各行业权重的均值)，
                使组合转换为 industry-neutral 多空组合
        - False: 不对权重分行业去均值
        """
        if period is None:
            period = self._factor_analyzer._periods
        if not isinstance(period, Iterable):
            period = (period,)
        period = tuple(period)
        factor_returns = self.calc_factor_returns(demeaned=demeaned, group_adjust=group_adjust,
                                                  _rename_forward_return_columns=False)
        for p in period:
            if p in self._factor_analyzer._periods:
                self._alp.plot_cumulative_returns(factor_returns[p], period=p)

    def plot_top_down_cumulative_returns(self, period=None, demeaned=False, group_adjust=False):
        """画做多最大分位数做空最小分位数组合每日累积收益图

        period: 指定调仓周期
        demeaned:
        详见 calc_mean_return_by_quantile 中 demeaned 参数
        - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        详见 calc_mean_return_by_quantile 中 group_adjust 参数
        - True: 使用行业中性化后的收益计算累积收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        """
        if period is None:
            period = self._factor_analyzer._periods
        if not isinstance(period, Iterable):
            period = (period,)
        period = tuple(period)
        for p in period:
            if p in self._factor_analyzer._periods:
                factor_return = self.calc_top_down_cumulative_returns(
                    period=p, demeaned=demeaned, group_adjust=group_adjust,
                )
                self._alp.plot_top_down_cumulative_returns(
                    pd.DataFrame(factor_return, columns=[p]), period=p
                )

    def plot_cumulative_returns_by_quantile(self, period=None, demeaned=False, group_adjust=False):
        """画各分位数每日累积收益图

        参数:
        period: 调仓周期
        demeaned:
        详见 calc_mean_return_by_quantile 中 demeaned 参数
        - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        详见 calc_mean_return_by_quantile 中 group_adjust 参数
        - True: 使用行业中性化后的收益计算累积收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        """
        if period is None:
            period = self._factor_analyzer._periods
        if not isinstance(period, Iterable):
            period = (period,)
        period = tuple(period)
        mean_return_by_date, _ = self.calc_mean_return_by_quantile(
            by_date=True, by_group=False, demeaned=demeaned, group_adjust=group_adjust,
            _rename_forward_return_columns=False
        )
        for p in period:
            if p in self._factor_analyzer._periods:
                self._alp.plot_cumulative_returns_by_quantile(mean_return_by_date[p], period=p)

    def plot_quantile_average_cumulative_return(self, periods_before=5, periods_after=10,
                                                by_quantile=False, std_bar=False,
                                                demeaned=False, group_adjust=False):
        """因子预测能力平均累计收益图

        参数:
        periods_before: 计算过去的天数
        periods_after: 计算未来的天数
        by_quantile: 是否各分位数分别显示因子预测能力平均累计收益图
        std_bar:
        - True: 显示标准差
        - False: 不显示标准差
        demeaned:
        详见 calc_mean_return_by_quantile 中 demeaned 参数
        - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        详见 calc_mean_return_by_quantile 中 group_adjust 参数
        - True: 使用行业中性化后的收益计算累积收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        """
        average_cumulative_return_by_q = self.calc_average_cumulative_return_by_quantile(
            periods_before=periods_before, periods_after=periods_after,
            demeaned=demeaned, group_adjust=group_adjust
        )
        self._alp.plot_quantile_average_cumulative_return(average_cumulative_return_by_q,
                                                          by_quantile=by_quantile,
                                                          std_bar=std_bar,
                                                          periods_before=periods_before,
                                                          periods_after=periods_after)

    def plot_events_distribution(self, num_days=5):
        """画有效因子数量统计图

        参数:
        num_days: 统计间隔天数
        """
        self._alp.plot_events_distribution(
            events=self._factor_analyzer._clean_factor_data['factor'],
            num_days=num_days,
            full_dates=self._factor_analyzer._dates
        )

    def create_summary_tear_sheet(self, demeaned=False, group_adjust=False):
        """因子值特征分析

        参数:
        demeaned:
        详见 calc_mean_return_by_quantile 中 demeaned 参数
        - True: 使用超额收益计算 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        详见 calc_mean_return_by_quantile 中 group_adjust 参数
        - True: 使用行业中性化后的收益计算 (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        """
        self.plot_quantile_statistics_table()
        self.plot_returns_table(demeaned=demeaned, group_adjust=group_adjust)
        self.plot_quantile_returns_bar(by_group=False, demeaned=demeaned, group_adjust=group_adjust)
        self._alp.plt.show()
        self.plot_information_table(group_adjust=group_adjust)
        self.plot_turnover_table()

    def create_returns_tear_sheet(self, demeaned=False, group_adjust=False, by_group=False):
        """因子收益分析

        参数:
        demeaned:
        - True: 使用超额收益计算 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        - True: 使用行业中性化后的收益计算 (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        by_group:
        - True: 画各行业的各分位数平均收益图
        - False: 不画各行业的各分位数平均收益图
        """
        self.plot_returns_table(demeaned=demeaned, group_adjust=group_adjust)
        self.plot_quantile_returns_bar(by_group=False,
                                       demeaned=demeaned,
                                       group_adjust=group_adjust)
        self._alp.plt.show()
        self.plot_cumulative_returns(period=None, demeaned=demeaned, group_adjust=group_adjust)
        self._alp.plt.show()
        self.plot_top_down_cumulative_returns(
            period=None, demeaned=demeaned, group_adjust=group_adjust
        )
        self._alp.plt.show()
        self.plot_cumulative_returns_by_quantile(period=None,
                                                 demeaned=demeaned,
                                                 group_adjust=group_adjust)
        self._alp.plt.show()
        self.plot_mean_quantile_returns_spread_time_series(demeaned=demeaned,
                                                           group_adjust=group_adjust)
        self._alp.plt.show()
        if by_group:
            self.plot_quantile_returns_bar(by_group=True,
                                           demeaned=demeaned,
                                           group_adjust=group_adjust)
            self._alp.plt.show()

    def create_information_tear_sheet(self, group_adjust=False, by_group=False):
        """因子 IC 分析

        参数:
        group_adjust:
        - True: 使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
        - False: 不使用行业中性收益
        by_group:
        - True: 画按行业分组信息比率(IC)图
        - False: 画月度信息比率(IC)图
        """
        self.plot_ic_ts(group_adjust=group_adjust, method=None)
        self._alp.plt.show()
        self.plot_ic_qq(group_adjust=group_adjust)
        self._alp.plt.show()
        if by_group:
            self.plot_ic_by_group(group_adjust=group_adjust, method=None)
        else:
            self.plot_monthly_ic_heatmap(group_adjust=group_adjust)
        self._alp.plt.show()

    def create_turnover_tear_sheet(self, turnover_periods=None):
        """因子换手率分析

        参数:
        turnover_periods: 调仓周期
        """
        self.plot_turnover_table()
        self.plot_top_bottom_quantile_turnover(periods=turnover_periods)
        self._alp.plt.show()
        self.plot_factor_auto_correlation(periods=turnover_periods)
        self._alp.plt.show()

    def create_event_returns_tear_sheet(self, avgretplot=(5, 15),
                                        demeaned=False, group_adjust=False,
                                        std_bar=False):
        """因子预测能力分析

        参数:
        avgretplot: tuple 因子预测的天数
        -(计算过去的天数, 计算未来的天数)
        demeaned:
        详见 calc_mean_return_by_quantile 中 demeaned 参数
        - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        详见 calc_mean_return_by_quantile 中 group_adjust 参数
        - True: 使用行业中性化后的收益计算累积收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        std_bar:
        - True: 显示标准差
        - False: 不显示标准差
        """
        before, after = avgretplot
        self.plot_quantile_average_cumulative_return(
            periods_before=before, periods_after=after,
            by_quantile=False, std_bar=False,
            demeaned=demeaned, group_adjust=group_adjust
        )
        self._alp.plt.show()
        if std_bar:
            self.plot_quantile_average_cumulative_return(
                periods_before=before, periods_after=after,
                by_quantile=True, std_bar=True,
                demeaned=demeaned, group_adjust=group_adjust
            )
            self._alp.plt.show()

    def create_full_tear_sheet(self, demeaned=False, group_adjust=False, by_group=False,
                               turnover_periods=None, avgretplot=(5, 15), std_bar=False):
        """全部分析

        参数:
        demeaned:
        - True：使用超额收益计算 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False：不使用超额收益
        group_adjust:
        - True：使用行业中性化后的收益计算
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False：不使用行业中性化后的收益
        by_group:
        - True: 按行业展示
        - False: 不按行业展示
        turnover_periods: 调仓周期
        avgretplot: tuple 因子预测的天数
        -(计算过去的天数, 计算未来的天数)
        std_bar:
        - True: 显示标准差
        - False: 不显示标准差
        """
        self.plot_quantile_statistics_table()
        print("\n-------------------------\n")
        self.plot_returns_table(demeaned=demeaned, group_adjust=group_adjust)
        self.plot_quantile_returns_bar(by_group=False,
                                       demeaned=demeaned,
                                       group_adjust=group_adjust)
        self._alp.plt.show()
        self.plot_cumulative_returns(period=None, demeaned=demeaned, group_adjust=group_adjust)
        self._alp.plt.show()
        self.plot_top_down_cumulative_returns(
            period=None, demeaned=demeaned, group_adjust=group_adjust
        )
        self._alp.plt.show()
        self.plot_cumulative_returns_by_quantile(period=None,
                                                 demeaned=demeaned,
                                                 group_adjust=group_adjust)
        self._alp.plt.show()
        self.plot_mean_quantile_returns_spread_time_series(demeaned=demeaned,
                                                           group_adjust=group_adjust)
        self._alp.plt.show()
        if by_group:
            self.plot_quantile_returns_bar(by_group=True,
                                           demeaned=demeaned,
                                           group_adjust=group_adjust)
            self._alp.plt.show()
        print("\n-------------------------\n")
        self.plot_information_table(group_adjust=group_adjust)
        self.plot_ic_ts(group_adjust=group_adjust, method=None)
        self._alp.plt.show()
        self.plot_ic_qq(group_adjust=group_adjust)
        self._alp.plt.show()
        if by_group:
            self.plot_ic_by_group(group_adjust=group_adjust, method=None)
        else:
            self.plot_monthly_ic_heatmap(group_adjust=group_adjust)
        self._alp.plt.show()
        print("\n-------------------------\n")
        self.plot_turnover_table()
        self.plot_top_bottom_quantile_turnover(periods=turnover_periods)
        self._alp.plt.show()
        self.plot_factor_auto_correlation(periods=turnover_periods)
        self._alp.plt.show()
        print("\n-------------------------\n")
        before, after = avgretplot
        self.plot_quantile_average_cumulative_return(
            periods_before=before, periods_after=after,
            by_quantile=False, std_bar=False,
            demeaned=demeaned, group_adjust=group_adjust
        )
        self._alp.plt.show()
        if std_bar:
            self.plot_quantile_average_cumulative_return(
                periods_before=before, periods_after=after,
                by_quantile=True, std_bar=True,
                demeaned=demeaned, group_adjust=group_adjust
            )
            self._alp.plt.show()

    def plot_disable_chinese_label(self):
        """关闭中文图例显示

        画图时默认会从系统中查找中文字体显示以中文图例
        如果找不到中文字体则默认使用英文图例
        当找到中文字体但中文显示乱码时, 可调用此 API 关闭中文图例显示而使用英文
        """
        self._alp._use_chinese(False)
