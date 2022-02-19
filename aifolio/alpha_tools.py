# -*- coding: utf-8 -*-
# @time: 2022/2/18 23:44
# @Author：lhf
# ----------------------
from aiutils.cache import PickleCache

from .settings import ALPHA_CHACHE_DIR, ALPHA_CHACHE_S
from .alphacn021.ana_calc import AnalyzerCalc


@PickleCache.cached_function_result_for_a_time(cache_dir=ALPHA_CHACHE_DIR, cache_second=ALPHA_CHACHE_S)
def cache_analyzer_calc(
        factor, prices,
        groupby=None, groupby_labels=None, weights=None, binning_by_group=False,
        quantiles=None, bins=None, periods=(1, 5, 10),
        max_loss=0.5, zero_aware=False
):
    factor = factor.sort_index()[list(sorted(factor.columns))]
    prices = prices.sort_index()[list(sorted(prices.columns))]
    return AnalyzerCalc(**locals())


def compute_forward_returns(prices, periods=(1, 5, 10), filter_zscore=None):
    from aifolio.alphacn021.alphalens import compute_forward_returns
    return compute_forward_returns(prices, periods=periods, filter_zscore=filter_zscore)
