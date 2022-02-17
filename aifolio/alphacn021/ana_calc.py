# -*- coding: utf-8 -*-
"""
@time: 2021/11/23 14:11
@file: analyze.py

"""
from functools import cached_property

import pandas as pd
from scipy.stats import spearmanr, pearsonr
from collections import Iterable
from fastcache import lru_cache

from . import alphalens as al
from .alphalens import MaxLossExceededError
from .utils import affirm


class AnalyzerCalc(object):
    """ 单因子分析，执行计算的部分 """

    # 参考jqfactor-1.32.9的analyze.py部分；去掉对数据源的依赖，改为用户自己传入所需数据
    # 根据 FactorAnalyzerResult 添加所需的对象方法；而非原始 FactorAnalyzer 全部的

    def __init__(self,
                 factor, prices,
                 groupby=None, groupby_labels=None, weights=None, binning_by_group=False,
                 quantiles=None, bins=None, periods=(1, 5, 10),
                 max_loss=0.5, zero_aware=False
                 ):
        """
        保存分析设定的参数；执行函数get_clean_factor_and_forward_returns，保存标准数据
        """
        self.factor = factor
        self.prices = prices
        self.groupby = groupby
        self.groupby_labels = groupby_labels
        self.weights = weights

        self._quantiles = quantiles
        self._bins = bins
        self._periods = tuple(periods) if isinstance(periods, Iterable) else (periods,)
        self._binning_by_group = binning_by_group
        self._max_loss = max_loss
        self._zero_aware = zero_aware

        self.__gen_clean_factor_and_forward_returns()

    def __gen_clean_factor_and_forward_returns(self):
        """格式化因子数据和定价数据"""
        try:
            self._clean_factor_data = al.get_clean_factor_and_forward_returns(
                self.factor,
                self.prices,
                groupby=self.groupby,
                binning_by_group=False,
                quantiles=self._quantiles,
                bins=None,
                periods=self._periods,
                filter_zscore=20,
                groupby_labels=self.groupby_labels,
                max_loss=self._max_loss
            )
            # 目前 Categorical Dtype 问题太多, 暂时放弃
            if 'group' in self._clean_factor_data:
                # al函数中传入groupby为None时，没有该列
                self._clean_factor_data['group'] = list(self._clean_factor_data['group'])
        except MaxLossExceededError as e:
            raise RuntimeError(e, "因子重复值或 nan 值太多, 无法完成分析")
        except ValueError as e:
            if 'Bin edges must be unique' in str(e):
                raise RuntimeError(e)

        # 添加weight列
        factor_copy = self._clean_factor_data.copy()
        weights = self.weights
        if weights is None:  # 为None时等权重
            weight_use = pd.Series(1.0, index=self._clean_factor_data.index)
            weight_use /= weight_use.sum()
        if weights is not None:  # fixme 其它格式weight计算是否准确
            if isinstance(weights, dict):
                diff = set(factor_copy.index.get_level_values(
                    'asset')) - set(weights.keys())
                if len(diff) > 0:
                    raise KeyError(
                        "Assets {} not in weights mapping".format(
                            list(diff)))
                ww = pd.Series(weights)
                weight_use = pd.Series(index=factor_copy.index,
                                       data=ww[factor_copy.index.get_level_values(
                                           'asset')].values)
            elif isinstance(weights, pd.DataFrame):
                weight_use = weights.stack()

        # 调整weight
        self._clean_factor_data['weight'] = weight_use.reindex(self._clean_factor_data.index)
        self._clean_factor_data['weight'] = self._clean_factor_data.set_index(
            'factor_quantile', append=True
        ).groupby(level=['date', 'factor_quantile']
                  )['weight'].apply(lambda s: s.divide(s.sum())).reset_index(
            'factor_quantile', drop=True
        )

    @property
    @al.rename_forward_return_columns
    def clean_factor_data(self):
        return self._clean_factor_data

    @al.rename_forward_return_columns
    @lru_cache(16)
    def calc_mean_return_by_quantile(self, by_date=False, by_group=False,
                                     demeaned=False, group_adjust=False):
        """计算按分位数分组因子收益和标准差

        因子收益为收益按照 weight 列中权重的加权平均值

        参数:
        by_date:
        - True: 按天计算收益
        - False: 不按天计算收益
        by_group:
        - True: 按行业计算收益
        - False: 不按行业计算收益
        demeaned:
        - True: 使用超额收益计算各分位数收益，超额收益=收益-基准收益
                (基准收益被认为是每日所有股票收益按照weight列中权重的加权的均值)
        - False: 不使用超额收益
        group_adjust:
        - True: 使用行业中性收益计算各分位数收益，行业中性收益=收益-行业收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
        - False: 不使用行业中性收益
        """
        return al.mean_return_by_quantile(self._clean_factor_data,
                                          by_date=by_date,
                                          by_group=by_group,
                                          demeaned=demeaned,
                                          group_adjust=group_adjust)

    @al.rename_forward_return_columns
    def compute_mean_returns_spread(self, upper_quant=None, lower_quant=None,
                                    by_date=True, by_group=False,
                                    demeaned=False, group_adjust=False):
        """计算两个分位数相减的因子收益和标准差

        参数:
        upper_quant: 用 upper_quant 选择的分位数减去 lower_quant 选择的分位数
        lower_quant: 用 upper_quant 选择的分位数减去 lower_quant 选择的分位数
        by_date:
        - True: 按天计算两个分位数相减的因子收益和标准差
        - False: 不按天计算两个分位数相减的因子收益和标准差
        by_group:
        - True: 分行业计算两个分位数相减的因子收益和标准差
        - False: 不分行业计算两个分位数相减的因子收益和标准差
        demeaned:
        - True: 使用超额收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        - True: 使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性收益
        """
        upper_quant = upper_quant if upper_quant is not None else self._quantiles
        lower_quant = lower_quant if lower_quant is not None else 1
        if not 1 <= upper_quant <= self._quantiles or not 1 <= lower_quant <= self._quantiles:
            raise ValueError("upper_quant 和 low_quant 的取值范围为 1 - %s 的整数" % self._quantiles)
        mean, std = self.calc_mean_return_by_quantile(by_date=by_date, by_group=by_group,
                                                      demeaned=demeaned, group_adjust=group_adjust,
                                                      _rename_forward_return_columns=False)
        mean = mean.apply(al.rate_of_return, axis=0)
        std = std.apply(al.rate_of_return, axis=0)
        return al.compute_mean_returns_spread(mean_returns=mean,
                                              upper_quant=upper_quant,
                                              lower_quant=lower_quant,
                                              std_err=std)

    @al.rename_forward_return_columns
    @lru_cache(4)
    def calc_factor_alpha_beta(self, demeaned=True, group_adjust=False):
        """计算因子的 alpha 和 beta

        因子值加权组合每日收益 = beta * 市场组合每日收益 + alpha

        因子值加权组合每日收益计算方法见 calc_factor_returns 函数
        市场组合每日收益是每日所有股票收益按照weight列中权重加权的均值
        结果中的 alpha 是年化 alpha

        参数:
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
        return al.factor_alpha_beta(self._clean_factor_data,
                                    demeaned=demeaned,
                                    group_adjust=group_adjust)

    @al.rename_forward_return_columns
    @lru_cache(2)
    def calc_autocorrelation(self, rank=True):
        """根据调仓周期确定滞后期的每天计算因子自相关性

        当日因子值和滞后period天的因子值的自相关性

        参数:
        rank:
        - True: 秩相关系数
        - False: 普通相关系数
        """
        return pd.concat(
            [
                al.factor_autocorrelation(self._clean_factor_data, period, rank=rank)
                for period in self._periods
            ],
            axis=1
        )

    @al.rename_forward_return_columns
    @lru_cache(8)
    def calc_factor_information_coefficient(self, group_adjust=False, by_group=False, method=None):
        """计算每日因子信息比率 (IC值)

        参数:
        group_adjust:
        - True: 使用行业中性收益计算 IC (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性收益
        by_group:
        - True: 分行业计算 IC
        - False: 不分行业计算 IC
        method:
        - 'rank': 用秩相关系数计算IC值
        - 'normal': 用普通相关系数计算IC值
        """
        if method is None:
            # method = self._ic_method
            method = 'rank'
        affirm(method in ('rank', 'normal'),
               "`method` should be chosen from ('rank' | 'normal')")

        if method == 'rank':
            method = spearmanr
        elif method == 'normal':
            method = pearsonr
        return al.factor_information_coefficient(self._clean_factor_data,
                                                 group_adjust=group_adjust,
                                                 by_group=by_group,
                                                 method=method)

    @al.rename_forward_return_columns
    @lru_cache(16)
    def calc_mean_information_coefficient(self, group_adjust=False, by_group=False,
                                          by_time=None, method=None):
        """计算因子信息比率均值 (IC值均值)

        参数:
        group_adjust:
        - True: 使用行业中性收益计算 IC (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性收益
        by_group:
        - True: 分行业计算 IC
        - False: 不分行业计算 IC
        by_time:
        - 'Y': 按年求均值
        - 'M': 按月求均值
        - None: 对所有日期求均值
        method:
        - 'rank': 用秩相关系数计算IC值
        - 'normal': 用普通相关系数计算IC值
        """
        if method is None:
            # method = self._ic_method
            method = 'rank'
        affirm(method in ('rank', 'normal'),
               "`method` should be chosen from ('rank' | 'normal')")

        if method == 'rank':
            method = spearmanr
        elif method == 'normal':
            method = pearsonr
        return al.mean_information_coefficient(
            self._clean_factor_data,
            group_adjust=group_adjust,
            by_group=by_group,
            by_time=by_time,
            method=method
        )

    @al.rename_forward_return_columns
    @lru_cache(4)
    def calc_factor_returns(self, demeaned=True, group_adjust=False):
        """计算按因子值加权组合每日收益

        权重 = 每日因子值 / 每日因子值的绝对值的和
        正的权重代表买入, 负的权重代表卖出

        参数:
        demeaned:
        - True: 对权重去均值 (每日权重 = 每日权重 - 每日权重的均值), 使组合转换为 cash-neutral 多空组合
        - False: 不对权重去均值
        group_adjust:
        - True: 对权重分行业去均值 (每日权重 = 每日权重 - 每日各行业权重的均值)，
                使组合转换为 industry-neutral 多空组合
        - False: 不对权重分行业去均值
        """
        return al.factor_returns(self._clean_factor_data,
                                 demeaned=demeaned,
                                 group_adjust=group_adjust)

    @lru_cache(20)
    def calc_top_down_cumulative_returns(self, period=None, demeaned=False, group_adjust=False):
        """计算做多最大分位，做空最小分位组合每日累积收益

        参数:
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
            period = self._periods[0]
        mean_returns, _ = self.calc_mean_return_by_quantile(
            by_date=True, by_group=False, demeaned=demeaned, group_adjust=group_adjust,
            _rename_forward_return_columns=False
        )

        upper_quant = mean_returns[period].xs(self._quantiles,
                                              level='factor_quantile')
        lower_quant = mean_returns[period].xs(1,
                                              level='factor_quantile')
        # * 0.5 保证总杠杆为 1
        # return al.cumulative_returns((upper_quant - lower_quant) * 0.5, period=period)
        return al.cumulative_returns(upper_quant - lower_quant, period=period)  # 去掉0.5；方便和分组收益对应上

    @lru_cache(16)
    def calc_average_cumulative_return_by_quantile(self, periods_before, periods_after,
                                                   demeaned=False, group_adjust=False):
        """按照当天的分位数算分位数未来和过去的收益均值和标准差

        参数:
        periods_before: 计算过去的天数
        periods_after: 计算未来的天数
        demeaned:
        - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        - True: 使用行业中性化后的收益计算累积收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        """
        return al.average_cumulative_return_by_quantile(
            self._clean_factor_data,
            prices=self.prices,
            periods_before=periods_before,
            periods_after=periods_after,
            demeaned=demeaned,
            group_adjust=group_adjust
        )

    @cached_property
    @al.rename_forward_return_columns
    def quantile_turnover(self):
        """换手率分析

        返回值一个 dict, key 是 period, value 是一个 DataFrame(index 是日期, column 是分位数)
        """

        quantile_factor = self._clean_factor_data['factor_quantile']

        quantile_turnover_rate = {
            p: pd.concat([al.quantile_turnover(quantile_factor, q, p)
                          for q in range(1, int(quantile_factor.max()) + 1)],
                         axis=1)
            for p in self._periods
        }

        return quantile_turnover_rate
