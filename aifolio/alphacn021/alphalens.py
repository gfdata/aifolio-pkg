# -*- coding: utf-8 -*-
"""本地化 alphalens，基于 0.2.1"""

from __future__ import division, unicode_literals

from functools import wraps

import pandas as pd
import numpy as np
from scipy import stats

from .compatible import rolling_apply, OLS, add_constant


# utils.py

class NonMatchingTimezoneError(Exception):
    pass


class MaxLossExceededError(Exception):
    pass


def rethrow(exception, additional_message):
    """
    Re-raise the last exception that was active in the current scope
    without losing the stacktrace but adding an additional message.
    This is hacky because it has to be compatible with both python 2/3
    """
    e = exception
    m = additional_message
    if not e.args:
        e.args = (m,)
    else:
        e.args = (e.args[0] + m,) + e.args[1:]
    raise e


def non_unique_bin_edges_error(func):
    """
    Give user a more informative error in case it is not possible
    to properly calculate quantiles on the input dataframe (factor)
    """
    message = """
    An error occurred while computing bins/quantiles on the input provided.
    This usually happens when the input contains too many identical
    values and they span more than one quantile. The quantiles are choosen
    to have the same number of records each, but the same value cannot span
    multiple quantiles. Possible workarounds are:
    1 - Decrease the number of quantiles
    2 - Specify a custom quantiles range, e.g. [0, .50, .75, 1.] to get unequal
        number of records per quantile
    3 - Use 'bins' option instead of 'quantiles', 'bins' chooses the
        buckets to be evenly spaced according to the values themselves, while
        'quantiles' forces the buckets to have the same number of records.
    4 - for factors with discrete values use the 'bins' option with custom
        ranges and create a range for each discrete value
    Please see get_clean_factor_and_forward_returns documentation for
    full documentation of 'bins' and 'quantiles' options.
"""
    message = u"""
    根据输入的 quantiles 计算时发生错误.
    这通常发生在输入包含太多相同值, 使得它们跨越多个分位.
    每天的因子值是按照分位数平均分组的, 相同的值不能跨越多个分位数.
    可能的解决方法:
    1. 减少分位数
    2. 调整因子减少重复值
    3. 尝试不同的股票池
    """

    def dec(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            if 'Bin edges must be unique' in str(e):
                rethrow(e, message)
            raise

    return dec


@non_unique_bin_edges_error
def quantize_factor(factor_data, quantiles=5, bins=None, by_group=False, no_raise=False):
    """
    Computes period wise factor quantiles.
    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in get_clean_factor_and_forward_returns
    quantiles : int or sequence[float]
        Number of equal-sized quantile buckets to use in factor bucketing.
        Alternately sequence of quantiles, allowing non-equal-sized buckets
        e.g. [0, .10, .5, .90, 1.] or [.05, .5, .95]
        Only one of 'quantiles' or 'bins' can be not-None
    bins : int or sequence[float]
        Number of equal-width (valuewise) bins to use in factor bucketing.
        Alternately sequence of bin edges allowing for non-uniform bin width
        e.g. [-4, -2, -0.5, 0, 10]
        Only one of 'quantiles' or 'bins' can be not-None
    by_group : bool
        If True, compute quantile buckets separately for each group.
    no_raise: bool, optional
        If True, no exceptions are thrown and the values for which the
        exception would have been thrown are set to np.NaN
    Returns
    -------
    factor_quantile : pd.Series
        Factor quantiles indexed by date and asset.
    """
    if not ((quantiles is not None and bins is None) or (quantiles is None and bins is not None)):
        raise ValueError('Either quantiles or bins should be provided')

    def quantile_calc(x, _quantiles, _bins, _no_raise):
        try:
            if _quantiles is not None and _bins is None:
                return pd.qcut(x, _quantiles, labels=False) + 1
            elif _bins is not None and _quantiles is None:
                return pd.cut(x, _bins, labels=False) + 1
        except Exception as e:
            if _no_raise:
                return pd.Series(index=x.index)
            raise e

    grouper = [factor_data.index.get_level_values('date')]
    if by_group:
        grouper.append('group')

    factor_quantile = factor_data.groupby(grouper)['factor'] \
        .apply(quantile_calc, quantiles, bins, no_raise)
    factor_quantile.name = 'factor_quantile'

    return factor_quantile.dropna()


def compute_forward_returns(prices, periods=(1, 5, 10), filter_zscore=None):
    """
    Finds the N period forward returns (as percent change) for each asset
    provided.
    Parameters
    ----------
    prices : pd.DataFrame
        Pricing data to use in forward price calculation.
        Assets as columns, dates as index. Pricing data must
        span the factor analysis time period plus an additional buffer window
        that is greater than the maximum number of expected periods
        in the forward returns calculations.
    periods : sequence[int]
        periods to compute forward returns on.
    filter_zscore : int or float
        Sets forward returns greater than X standard deviations
        from the the mean to nan.
        Caution: this outlier filtering incorporates lookahead bias.
    Returns
    -------
    forward_returns : pd.DataFrame - MultiIndex
        Forward returns in indexed by date and asset.
        Separate column for each forward return window.
    """

    forward_returns = pd.DataFrame(
        index=pd.MultiIndex.from_product([prices.index, prices.columns], names=['date', 'asset'])
    )

    for period in periods:
        delta = prices.pct_change(period).shift(-period)

        if filter_zscore is not None:
            mask = abs(delta - delta.mean()) > (filter_zscore * delta.std())
            delta[mask] = np.nan

        forward_returns[period] = delta.stack()

    forward_returns.index = forward_returns.index.rename(['date', 'asset'])

    return forward_returns


def demean_forward_returns(factor_data, grouper=None):
    """
    Convert forward returns to returns relative to mean
    period wise all-universe or group returns.
    group-wise normalization incorporates the assumption of a
    group neutral portfolio constraint and thus allows allows the
    factor to be evaluated across groups.
    For example, if AAPL 5 period return is 0.1% and mean 5 period
    return for the Technology stocks in our universe was 0.5% in the
    same period, the group adjusted 5 period return for AAPL in this
    period is -0.4%.
    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        Forward returns in indexed by date and asset.
        Separate column for each forward return window.
    grouper : list
        If True, demean according to group.
    Returns
    -------
    adjusted_forward_returns : pd.DataFrame - MultiIndex
        DataFrame of the same format as the input, but with each
        security's returns normalized by group.
    """

    factor_data = factor_data.copy()

    if not grouper:
        grouper = factor_data.index.get_level_values('date')

    cols = get_forward_returns_columns(factor_data.columns)
    if 'weight' not in factor_data.columns:
        factor_data[cols] = factor_data.groupby(grouper, as_index=False)[cols] \
            .apply(lambda x: x - x.mean())
    else:
        factor_data[cols] = factor_data.groupby(grouper, as_index=False)[
            cols.append(pd.Index(['weight']))
        ].apply(lambda x: x[cols].subtract(np.average(x[cols], axis=0,
                                                      weights=x['weight'].fillna(0.0).values),
                                           axis=1))

    return factor_data


def print_table(table, name=None, fmt=None):
    """
    Pretty print a pandas DataFrame.
    Uses HTML output if running inside Jupyter Notebook, otherwise
    formatted text output.
    Parameters
    ----------
    table : pd.Series or pd.DataFrame
        Table to pretty-print.
    name : str, optional
        Table name to display in upper left corner.
    fmt : str, optional
        Formatter to use for displaying table elements.
        E.g. '{0:.2f}%' for displaying 100 as '100.00%'.
        Restores original setting after displaying.
    """
    from IPython.display import display

    if isinstance(table, pd.Series):
        table = pd.DataFrame(table)

    if isinstance(table, pd.DataFrame):
        table.columns.name = name

    prev_option = pd.get_option('display.float_format')
    if fmt is not None:
        pd.set_option('display.float_format', lambda x: fmt.format(x))

    display(table)

    if fmt is not None:
        pd.set_option('display.float_format', prev_option)


def get_clean_factor_and_forward_returns(
    factor,
    prices,
    groupby=None,
    binning_by_group=False,
    quantiles=5,
    bins=None,
    periods=(1, 5, 10),
    filter_zscore=20,
    groupby_labels=None,
    max_loss=0.25
):
    """
    Formats the factor data, pricing data, and group mappings
    into a DataFrame that contains aligned MultiIndex
    indices of timestamp and asset.
    Parameters
    ----------
    factor : pd.Series - MultiIndex
        A MultiIndex Series indexed by timestamp (level 0) and asset
        (level 1), containing the values for a single alpha factor.
        ::
            -----------------------------------
                date    |    asset   |
            -----------------------------------
                        |   AAPL     |   0.5
                        -----------------------
                        |   BA       |  -1.1
                        -----------------------
            2014-01-01  |   CMG      |   1.7
                        -----------------------
                        |   DAL      |  -0.1
                        -----------------------
                        |   LULU     |   2.7
                        -----------------------
    prices : pd.DataFrame
        A wide form Pandas DataFrame indexed by timestamp with assets
        in the columns. It is important to pass the
        correct pricing data in depending on what time of period your
        signal was generated so to avoid lookahead bias, or
        delayed calculations. Pricing data must span the factor
        analysis time period plus an additional buffer window
        that is greater than the maximum number of expected periods
        in the forward returns calculations.
        'Prices' must contain at least an entry for each timestamp/asset
        combination in 'factor'. This entry must be the asset price
        at the time the asset factor value is computed and it will be
        considered the buy price for that asset at that timestamp.
        'Prices' must also contain entries for timestamps following each
        timestamp/asset combination in 'factor', as many more timestamps
        as the maximum value in 'periods'. The asset price after 'period'
        timestamps will be considered the sell price for that asset when
        computing 'period' forward returns.
        ::
            ----------------------------------------------------
                        | AAPL |  BA  |  CMG  |  DAL  |  LULU  |
            ----------------------------------------------------
               Date     |      |      |       |       |        |
            ----------------------------------------------------
            2014-01-01  |605.12| 24.58|  11.72| 54.43 |  37.14 |
            ----------------------------------------------------
            2014-01-02  |604.35| 22.23|  12.21| 52.78 |  33.63 |
            ----------------------------------------------------
            2014-01-03  |607.94| 21.68|  14.36| 53.94 |  29.37 |
            ----------------------------------------------------
    groupby : pd.Series - MultiIndex or dict
        Either A MultiIndex Series indexed by date and asset,
        containing the period wise group codes for each asset, or
        a dict of asset to group mappings. If a dict is passed,
        it is assumed that group mappings are unchanged for the
        entire time period of the passed factor data.
    binning_by_group : bool
        If True, compute quantile buckets separately for each group.
        This is useful when the factor values range vary considerably
        across gorups so that it is wise to make the binning group relative.
        You should probably enable this if the factor is intended
        to be analyzed for a group neutral portfolio
    quantiles : int or sequence[float]
        Number of equal-sized quantile buckets to use in factor bucketing.
        Alternately sequence of quantiles, allowing non-equal-sized buckets
        e.g. [0, .10, .5, .90, 1.] or [.05, .5, .95]
        Only one of 'quantiles' or 'bins' can be not-None
    bins : int or sequence[float]
        Number of equal-width (valuewise) bins to use in factor bucketing.
        Alternately sequence of bin edges allowing for non-uniform bin width
        e.g. [-4, -2, -0.5, 0, 10]
        Chooses the buckets to be evenly spaced according to the values
        themselves. Useful when the factor contains discrete values.
        Only one of 'quantiles' or 'bins' can be not-None
    periods : sequence[int]
        periods to compute forward returns on.
    filter_zscore : int or float
        Sets forward returns greater than X standard deviations
        from the the mean to nan.
        Caution: this outlier filtering incorporates lookahead bias.
    groupby_labels : dict
        A dictionary keyed by group code with values corresponding
        to the display name for each group.
    max_loss : float, optional
        Maximum percentage (0.00 to 1.00) of factor data dropping allowed,
        computed comparing the number of items in the input factor index and
        the number of items in the output DataFrame index.
        Factor data can be partially dropped due to being flawed itself
        (e.g. NaNs), not having provided enough price data to compute
        forward returns for all factor values, or because it is not possible
        to perform binning.
        Set max_loss=0 to avoid Exceptions suppression.
    Returns
    -------
    merged_data : pd.DataFrame - MultiIndex
        A MultiIndex Series indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        ::
           -------------------------------------------------------------------
                      |       |  1  |  5  |  10  |factor|group|factor_quantile
           -------------------------------------------------------------------
               date   | asset |     |     |      |      |     |
           -------------------------------------------------------------------
                      | AAPL  | 0.09|-0.01|-0.079|  0.5 |  G1 |      3
                      --------------------------------------------------------
                      | BA    | 0.02| 0.06| 0.020| -1.1 |  G2 |      5
                      --------------------------------------------------------
           2014-01-01 | CMG   | 0.03| 0.09| 0.036|  1.7 |  G2 |      1
                      --------------------------------------------------------
                      | DAL   |-0.02|-0.06|-0.029| -0.1 |  G3 |      5
                      --------------------------------------------------------
                      | LULU  |-0.03| 0.05|-0.009|  2.7 |  G1 |      2
                      --------------------------------------------------------
    """

    if factor.index.levels[0].tz != prices.index.tz:
        raise NonMatchingTimezoneError(
            "The timezone of 'factor' is not the "
            "same as the timezone of 'prices'. See "
            "the pandas methods tz_localize and "
            "tz_convert."
        )

    initial_amount = float(len(factor.index))

    merged_data = compute_forward_returns(prices, periods, filter_zscore)

    factor = factor.copy()
    factor.index = factor.index.rename(['date', 'asset'])
    merged_data['factor'] = factor

    if groupby is not None:
        if isinstance(groupby, dict):
            diff = set(factor.index.get_level_values('asset')) - set(groupby.keys())
            if len(diff) > 0:
                raise KeyError("Assets {} not in group mapping".format(list(diff)))

            ss = pd.Series(groupby)
            groupby = pd.Series(
                index=factor.index, data=ss[factor.index.get_level_values('asset')].values
            )

        if groupby_labels is not None:
            diff = set(groupby.values) - set(groupby_labels.keys())
            if len(diff) > 0:
                raise KeyError("groups {} not in passed group names".format(list(diff)))

            sn = pd.Series(groupby_labels)
            groupby = pd.Series(index=groupby.index, data=sn[groupby.values].values)

        merged_data['group'] = groupby.astype('category')

    merged_data = merged_data.dropna()

    fwdret_amount = float(len(merged_data.index))

    no_raise = False if max_loss == 0 else True
    merged_data['factor_quantile'] = quantize_factor(
        merged_data, quantiles, bins, binning_by_group, no_raise
    )

    merged_data = merged_data.dropna()
    merged_data['factor_quantile'] = merged_data['factor_quantile'].astype(int)

    binning_amount = float(len(merged_data.index))

    tot_loss = (initial_amount - binning_amount) / initial_amount
    fwdret_loss = (initial_amount - fwdret_amount) / initial_amount
    bin_loss = tot_loss - fwdret_loss

    # print(
    #     "Dropped %.1f%% entries from factor data (%.1f%% after "
    #     "in forward returns computation and %.1f%% in binning phase). "
    #     "Set max_loss=0 to see potentially suppressed Exceptions." %
    #     (tot_loss * 100, fwdret_loss * 100, bin_loss * 100)
    # )

    if tot_loss > max_loss:
        message = (
            "max_loss (%.1f%%) exceeded %.1f%%, consider increasing it." %
            (max_loss * 100, tot_loss * 100)
        )
        raise MaxLossExceededError(message)

    return merged_data


def common_start_returns(
    factor, prices, before, after, cumulative=False, mean_by_date=False, demean_by=None
):
    """
    A date and equity pair is extracted from each index row in the factor
    dataframe and for each of these pairs a return series is built starting
    from 'before' the date and ending 'after' the date specified in the pair.
    All those returns series are then aligned to a common index (-before to
    after) and returned as a single DataFrame
    Parameters
    ----------
    factor : pd.DataFrame
        DataFrame with at least date and equity as index, the columns are
        irrelevant
    prices : pd.DataFrame
        A wide form Pandas DataFrame indexed by date with assets
        in the columns. Pricing data should span the factor
        analysis time period plus/minus an additional buffer window
        corresponding to after/before period parameters.
    before:
        How many returns to load before factor date
    after:
        How many returns to load after factor date
    cumulative: bool, optional
        Return cumulative returns
    mean_by_date: bool, optional
        If True, compute mean returns for each date and return that
        instead of a return series for each asset
    demean_by: pd.DataFrame, optional
        DataFrame with at least date and equity as index, the columns are
        irrelevant. For each date a list of equities is extracted from
        'demean_by' index and used as universe to compute demeaned mean
        returns (long short portfolio)
    Returns
    -------
    aligned_returns : pd.DataFrame
        Dataframe containing returns series for each factor aligned to the same
        index: -before to after
    """

    if cumulative:
        returns = prices
    else:
        returns = prices.pct_change(axis=0)

    all_returns = []

    for timestamp, df in factor.groupby(level='date'):

        equities = df.index.get_level_values('asset')

        try:
            day_zero_index = returns.index.get_loc(timestamp)
        except KeyError:
            continue

        starting_index = max(day_zero_index - before, 0)
        ending_index = min(day_zero_index + after + 1, len(returns.index))

        equities_slice = set(equities)
        if demean_by is not None:
            demean_equities = demean_by.loc[timestamp] \
                .index.get_level_values('asset')
            equities_slice |= set(demean_equities)

        series = returns.loc[returns.index[starting_index:ending_index], equities_slice]
        series.index = range(starting_index - day_zero_index, ending_index - day_zero_index)

        if cumulative:
            series = (series / series.loc[0, :]) - 1

        if demean_by is not None:
            mean = series.loc[:, demean_equities].mean(axis=1)
            series = series.loc[:, equities]
            series = series.sub(mean, axis=0)

        if mean_by_date:
            series = series.mean(axis=1)

        all_returns.append(series)

    return pd.concat(all_returns, axis=1)


def cumulative_returns(returns, period):
    """
    Builds cumulative returns from N-periods returns.
    When 'period' N is greater than 1 the cumulative returns plot is computed
    building and averaging the cumulative returns of N interleaved portfolios
    (started at subsequent periods 1,2,3,...,N) each one rebalancing every N
    periods.
    Parameters
    ----------
    returns: pd.Series
        pd.Series containing N-periods returns
    period: integer
        Period for which the returns are computed
    Returns
    -------
    pd.Series
        Cumulative returns series
    """

    returns = returns.fillna(0)

    if period == 1:
        return returns.add(1).cumprod()
    #
    # build N portfolios from the single returns Series
    #

    def split_portfolio(ret, period):
        return pd.DataFrame(np.diag(ret))

    sub_portfolios = returns.groupby(
        np.arange(len(returns.index)) // period, axis=0
    ).apply(split_portfolio, period)
    sub_portfolios.index = returns.index

    #
    # compute 1 period returns so that we can average the N portfolios
    #

    def rate_of_returns(ret, period):
        return ((np.nansum(ret) + 1)**(1. / period)) - 1

    sub_portfolios = rolling_apply(sub_portfolios, window=period, func=rate_of_returns,
                                   min_periods=1, args=(period,))
    sub_portfolios = sub_portfolios.add(1).cumprod()

    #
    # average the N portfolios
    #
    return sub_portfolios.mean(axis=1)


def rate_of_return(period_ret):
    """
    1-period Growth Rate: the average rate of 1-period returns
    """
    return period_ret.add(1).pow(1. / period_ret.name).sub(1)


def std_conversion(period_std):
    """
    1-period standard deviation (or standard error) approximation
    Parameters
    ----------
    period_std: pd.DataFrame
        DataFrame containing standard deviation or standard error values
        with column headings representing the return period.
    Returns
    -------
    pd.DataFrame
        DataFrame in same format as input but with one-period
        standard deviation/error values.
    """
    period_len = period_std.name
    return period_std / np.sqrt(period_len)


def get_forward_returns_columns(columns):
    return columns[columns.astype('str').str.isdigit()]


# performance.py


def factor_information_coefficient(factor_data, group_adjust=False, by_group=False,
                                   method=stats.spearmanr):
    """
    Computes the Spearman Rank Correlation based Information Coefficient (IC)
    between factor values and N period forward returns for each period in
    the factor index.
    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in get_clean_factor_and_forward_returns
    group_adjust : bool
        Demean forward returns by group before computing IC.
    by_group : bool
        If True, compute period wise IC separately for each group.
    Returns
    -------
    ic : pd.DataFrame
        Spearman Rank correlation between factor and
        provided forward returns.
    """

    def src_ic(group):
        f = group['factor']
        _ic = group[get_forward_returns_columns(factor_data.columns)] \
            .apply(lambda x: method(x, f)[0])
        return _ic

    factor_data = factor_data.copy()

    grouper = [factor_data.index.get_level_values('date')]

    if group_adjust:
        factor_data = demean_forward_returns(factor_data, grouper + ['group'])
    if by_group:
        grouper.append('group')

    with np.errstate(divide='ignore', invalid='ignore'):
        ic = factor_data.groupby(grouper).apply(src_ic)
    ic.columns = pd.Int64Index(ic.columns)

    return ic


def mean_information_coefficient(factor_data, group_adjust=False, by_group=False, by_time=None,
                                 method=stats.spearmanr):
    """
    Get the mean information coefficient of specified groups.
    Answers questions like:
    What is the mean IC for each month?
    What is the mean IC for each group for our whole timerange?
    What is the mean IC for for each group, each week?
    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in get_clean_factor_and_forward_returns
    group_adjust : bool
        Demean forward returns by group before computing IC.
    by_group : bool
        If True, take the mean IC for each group.
    by_time : str (pd time_rule), optional
        Time window to use when taking mean IC.
        See http://pandas.pydata.org/pandas-docs/stable/timeseries.html
        for available options.
    Returns
    -------
    ic : pd.DataFrame
        Mean Spearman Rank correlation between factor and provided
        forward price movement windows.
    """

    ic = factor_information_coefficient(factor_data, group_adjust, by_group, method=method)

    grouper = []
    if by_time is not None:
        grouper.append(pd.Grouper(freq=by_time))
    if by_group:
        grouper.append('group')

    if len(grouper) == 0:
        ic = ic.mean()

    else:
        ic = (ic.reset_index().set_index('date').groupby(grouper).mean())

    return ic


def factor_returns(factor_data, demeaned=True, group_adjust=False):
    """
    Computes period wise returns for portfolio weighted by factor
    values. Weights are computed by demeaning factors and dividing
    by the sum of their absolute value (achieving gross leverage of 1).
    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in get_clean_factor_and_forward_returns
    demeaned : bool
        Should this computation happen on a long short portfolio? if True,
        then factor values will be demeaned across factor universe when
        factor weighting the portfolio.
    group_adjust : bool
        Should this computation happen on a group neutral portfolio? If True,
        compute group neutral returns: each group will weight the same and
        returns demeaning will occur on the group level.
    Returns
    -------
    returns : pd.DataFrame
        Period wise returns of dollar neutral portfolio weighted by factor
        value.
    """

    def to_weights(group, is_long_short):
        if is_long_short:
            demeaned_vals = group - group.mean()
            return demeaned_vals / demeaned_vals.abs().sum()
        else:
            return group / group.abs().sum()

    grouper = [factor_data.index.get_level_values('date')]
    if group_adjust:
        grouper.append('group')

    weights = factor_data.groupby(grouper)['factor'] \
        .apply(to_weights, demeaned)

    if group_adjust:
        weights = weights.groupby(level='date').apply(to_weights, False)

    weighted_returns = \
        factor_data[get_forward_returns_columns(factor_data.columns)] \
        .multiply(weights, axis=0)

    returns = weighted_returns.groupby(level='date').sum()

    return returns


def factor_alpha_beta(factor_data, demeaned=True, group_adjust=False):
    """
    Compute the alpha (excess returns), alpha t-stat (alpha significance),
    and beta (market exposure) of a factor. A regression is run with
    the period wise factor universe mean return as the independent variable
    and mean period wise return from a portfolio weighted by factor values
    as the dependent variable.
    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in get_clean_factor_and_forward_returns
    demeaned : bool
        Control how to build factor returns used for alpha/beta computation
        -- see performance.factor_return for a full explanation
    group_adjust : bool
        Control how to build factor returns used for alpha/beta computation
        -- see performance.factor_return for a full explanation
    Returns
    -------
    alpha_beta : pd.Series
        A list containing the alpha, beta, a t-stat(alpha)
        for the given factor and forward returns.
    """

    returns = factor_returns(factor_data, demeaned, group_adjust)

    universe_ret, std_error_ret = weighted_mean_return(
        factor_data, grouper=[factor_data.index.get_level_values('date')]
    )

    if isinstance(returns, pd.Series):
        returns.name = universe_ret.columns.values[0]
        returns = pd.DataFrame(returns)

    alpha_beta = pd.DataFrame()
    for period in returns.columns.values:
        x = universe_ret[period].values
        y = returns[period].values
        x = add_constant(x)

        reg_fit = OLS(y, x).fit()
        alpha, beta = reg_fit.params

        alpha_beta.loc['Ann. alpha', period] = \
            (1 + alpha) ** (252.0 / period) - 1
        alpha_beta.loc['beta', period] = beta

    return alpha_beta


def weighted_mean_return(factor_data, grouper):
    """计算（年化）加权平均/标准差"""
    if 'weight' not in factor_data.columns:
        group_stats = factor_data.groupby(grouper)[
            get_forward_returns_columns(factor_data.columns)] \
            .agg(['mean', 'std', 'count'])
        mean_ret = group_stats.T.xs('mean', level=1).T

        std_error_ret = group_stats.T.xs('std', level=1).T \
            / np.sqrt(group_stats.T.xs('count', level=1).T)

    else:
        forward_returns_columns = get_forward_returns_columns(factor_data.columns)

        def agg(values, weights):
            count = len(values)
            average = np.average(values, weights=weights, axis=0)
            # Fast and numerically precise:
            variance = np.average((values - average)**2, weights=weights, axis=0
                                  ) * count / max((count - 1), 1)
            return pd.Series([average, np.sqrt(variance), count], index=['mean', 'std', 'count'])

        group_stats = factor_data.groupby(grouper)[
            forward_returns_columns.append(pd.Index(['weight']))] \
            .apply(lambda x: x[forward_returns_columns].apply(
                agg, weights=x['weight'].fillna(0.0).values
            ))

        mean_ret = group_stats.xs('mean', level=-1)

        std_error_ret = group_stats.xs('std', level=-1) \
            / np.sqrt(group_stats.xs('count', level=-1))

    return mean_ret, std_error_ret


def mean_return_by_quantile(
    factor_data, by_date=False, by_group=False, demeaned=True, group_adjust=False
):
    """
    Computes mean returns for factor quantiles across
    provided forward returns columns.
    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in get_clean_factor_and_forward_returns
    by_date : bool
        If True, compute quantile bucket returns separately for each date.
    by_group : bool
        If True, compute quantile bucket returns separately for each group.
    demeaned : bool
        Compute demeaned mean returns (long short portfolio)
    group_adjust : bool
        Returns demeaning will occur on the group level.
    Returns
    -------
    mean_ret : pd.DataFrame
        Mean period wise returns by specified factor quantile.
    std_error_ret : pd.DataFrame
        Standard error of returns by specified quantile.
    """

    if group_adjust:
        grouper = [factor_data.index.get_level_values('date')] + ['group']
        factor_data = demean_forward_returns(factor_data, grouper)
    elif demeaned:
        factor_data = demean_forward_returns(factor_data)
    else:
        factor_data = factor_data.copy()

    grouper = ['factor_quantile']
    if by_date:
        grouper.append(factor_data.index.get_level_values('date'))

    if by_group:
        grouper.append('group')

    mean_ret, std_error_ret = weighted_mean_return(factor_data, grouper=grouper)

    return mean_ret, std_error_ret


def compute_mean_returns_spread(mean_returns, upper_quant, lower_quant, std_err=None):
    """
    Computes the difference between the mean returns of
    two quantiles. Optionally, computes the standard error
    of this difference.
    Parameters
    ----------
    mean_returns : pd.DataFrame
        DataFrame of mean period wise returns by quantile.
        MultiIndex containing date and quantile.
        See mean_return_by_quantile.
    upper_quant : int
        Quantile of mean return from which we
        wish to subtract lower quantile mean return.
    lower_quant : int
        Quantile of mean return we wish to subtract
        from upper quantile mean return.
    std_err : pd.DataFrame
        Period wise standard error in mean return by quantile.
        Takes the same form as mean_returns.
    Returns
    -------
    mean_return_difference : pd.Series
        Period wise difference in quantile returns.
    joint_std_err : pd.Series
        Period wise standard error of the difference in quantile returns.
    """
    if isinstance(mean_returns.index, pd.MultiIndex):
        mean_return_difference = mean_returns.xs(upper_quant,
                                                 level='factor_quantile') \
            - mean_returns.xs(lower_quant, level='factor_quantile')
    else:
        mean_return_difference = mean_returns.loc[upper_quant] - mean_returns.loc[lower_quant]

    if isinstance(std_err.index, pd.MultiIndex):
        std1 = std_err.xs(upper_quant, level='factor_quantile')
        std2 = std_err.xs(lower_quant, level='factor_quantile')
    else:
        std1 = std_err.loc[upper_quant]
        std2 = std_err.loc[upper_quant]
    joint_std_err = np.sqrt(std1**2 + std2**2)

    return mean_return_difference, joint_std_err


def quantile_turnover(quantile_factor, quantile, period=1):
    """
    Computes the proportion of names in a factor quantile that were
    not in that quantile in the previous period.
    Parameters
    ----------
    quantile_factor : pd.Series
        DataFrame with date, asset and factor quantile.
    quantile : int
        Quantile on which to perform turnover analysis.
    period: int, optional
        Period over which to calculate the turnover
    Returns
    -------
    quant_turnover : pd.Series
        Period by period turnover for that quantile.
    """

    quant_names = quantile_factor[quantile_factor == quantile]
    quant_name_sets = quant_names.groupby(level=['date']
                                          ).apply(lambda x: set(x.index.get_level_values('asset')))
    new_names = (quant_name_sets - quant_name_sets.shift(period)).dropna()
    quant_turnover = new_names.apply(lambda x: len(x)) / quant_name_sets.apply(lambda x: len(x))
    quant_turnover.name = quantile
    return quant_turnover


def factor_autocorrelation(factor_data, period=1, rank=True):
    """
    Computes autocorrelation of mean factor ranks in specified time spans.
    We must compare period to period factor ranks rather than factor values
    to account for systematic shifts in the factor values of all names or names
    within a group. This metric is useful for measuring the turnover of a
    factor. If the value of a factor for each name changes randomly from period
    to period, we'd expect an autocorrelation of 0.
    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in get_clean_factor_and_forward_returns
    period: int, optional
        Period over which to calculate the autocorrelation
    Returns
    -------
    autocorr : pd.Series
        Rolling 1 period (defined by time_rule) autocorrelation of
        factor values.
    """

    grouper = [factor_data.index.get_level_values('date')]

    if rank:
        ranks = factor_data.groupby(grouper)[['factor']].rank()
    else:
        ranks = factor_data[['factor']]
    asset_factor_rank = ranks.reset_index().pivot(index='date', columns='asset', values='factor')

    autocorr = asset_factor_rank.corrwith(asset_factor_rank.shift(period), axis=1)
    autocorr.name = period
    return autocorr


def average_cumulative_return_by_quantile(
    factor_data,
    prices,
    periods_before=10,
    periods_after=15,
    demeaned=True,
    group_adjust=False,
    by_group=False
):
    """
    Plots average cumulative returns by factor quantiles in the period range
    defined by -periods_before to periods_after
    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in get_clean_factor_and_forward_returns
    prices : pd.DataFrame
        A wide form Pandas DataFrame indexed by date with assets
        in the columns. Pricing data should span the factor
        analysis time period plus/minus an additional buffer window
        corresponding to periods_after/periods_before parameters.
    periods_before : int, optional
        How many periods before factor to plot
    periods_after  : int, optional
        How many periods after factor to plot
    demeaned : bool, optional
        Compute demeaned mean returns (long short portfolio)
    group_adjust : bool
        Returns demeaning will occur on the group level (group
        neutral portfolio)
    by_group : bool
        If True, compute cumulative returns separately for each group
    Returns
    -------
    cumulative returns and std deviation : pd.DataFrame
        A MultiIndex DataFrame indexed by quantile (level 0) and mean/std
        (level 1) and the values on the columns in range from
        -periods_before to periods_after
        If by_group=True the index will have an additional 'group' level
        ::
            ---------------------------------------------------
                        |       | -2  | -1  |  0  |  1  | ...
            ---------------------------------------------------
              quantile  |       |     |     |     |     |
            ---------------------------------------------------
                        | mean  |  x  |  x  |  x  |  x  |
                 1      ---------------------------------------
                        | std   |  x  |  x  |  x  |  x  |
            ---------------------------------------------------
                        | mean  |  x  |  x  |  x  |  x  |
                 2      ---------------------------------------
                        | std   |  x  |  x  |  x  |  x  |
            ---------------------------------------------------
                ...     |                 ...
            ---------------------------------------------------
    """

    def cumulative_return(q_fact, demean_by):
        return common_start_returns(
            q_fact, prices, periods_before, periods_after, True, True, demean_by
        )

    def average_cumulative_return(q_fact, demean_by):
        q_returns = cumulative_return(q_fact, demean_by)
        return pd.DataFrame({'mean': q_returns.mean(axis=1), 'std': q_returns.std(axis=1)}).T

    if by_group:
        #
        # Compute quantile cumulative returns separately for each group
        # Deman those returns accordingly to 'group_adjust' and 'demeaned'
        #
        returns_bygroup = []

        for group, g_data in factor_data.groupby('group'):
            g_fq = g_data['factor_quantile']
            if group_adjust:
                demean_by = g_fq  # demeans at group level
            elif demeaned:
                demean_by = factor_data['factor_quantile']  # demean by all
            else:
                demean_by = None
            #
            # Align cumulative return from different dates to the same index
            # then compute mean and std
            #
            avgcumret = g_fq.groupby(g_fq).apply(average_cumulative_return, demean_by)
            avgcumret['group'] = group
            avgcumret.set_index('group', append=True, inplace=True)
            returns_bygroup.append(avgcumret)

        return pd.concat(returns_bygroup, axis=0)

    else:
        #
        # Compute quantile cumulative returns for the full factor_data
        # Align cumulative return from different dates to the same index
        # then compute mean and std
        # Deman those returns accordingly to 'group_adjust' and 'demeaned'
        #
        if group_adjust:
            all_returns = []
            for group, g_data in factor_data.groupby('group'):
                g_fq = g_data['factor_quantile']
                avgcumret = g_fq.groupby(g_fq).apply(cumulative_return, g_fq)
                all_returns.append(avgcumret)
            q_returns = pd.concat(all_returns, axis=1)
            q_returns = pd.DataFrame({'mean': q_returns.mean(axis=1), 'std': q_returns.std(axis=1)})
            return q_returns.unstack(level=1).stack(level=0)
        elif demeaned:
            fq = factor_data['factor_quantile']
            return fq.groupby(fq).apply(average_cumulative_return, fq)
        else:
            fq = factor_data['factor_quantile']
            return fq.groupby(fq).apply(average_cumulative_return, None)


# 自定义 function


def format_forward_return_column_name(origin_name):
    return 'period_' + str(origin_name)


def rename_forward_return_column(value):
    """将 forward_return 的 columns name 由 [1, 5] 改成 ['period_1', 'period_5']"""
    if isinstance(value, pd.DataFrame):
        origin = get_forward_returns_columns(value.columns)
        new = origin.map(format_forward_return_column_name)
        convert_dict = dict(zip(origin, new))
        if not convert_dict:
            return value
        return value.rename(columns=convert_dict)
    elif isinstance(value, pd.Series):
        origin = get_forward_returns_columns(value.index)
        new = origin.map(format_forward_return_column_name)
        convert_dict = dict(zip(origin, new))
        if not convert_dict:
            return value
        return value.rename(index=convert_dict)
    elif isinstance(value, tuple):  # 是个 tuple (返回多个值)
        return tuple(rename_forward_return_column(v) for v in value)
    elif isinstance(value, dict):  # 是个 dict
        origin = pd.Index(value.keys())
        new = origin.map(format_forward_return_column_name)
        convert_dict = dict(zip(origin, new))
        if not convert_dict:
            return value
        return dict((convert_dict[k], value[k]) for k in value)
    else:
        return value


def rename_forward_return_columns(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        need_rename = kwargs.pop('_rename_forward_return_columns', True)
        result = func(*args, **kwargs)

        if not need_rename:
            return result
        else:
            return rename_forward_return_column(result)

    return wrapper
