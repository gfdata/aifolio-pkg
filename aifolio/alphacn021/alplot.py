# encoding: utf-8

"""alphalens 画图函数"""

from __future__ import division, unicode_literals

from functools import wraps

import pandas as pd
import numpy as np
from scipy import stats
try:
    import matplotlib as mpl
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    import seaborn as sns
    PLOT_ENABLED = True
except Exception:
    PLOT_ENABLED = False

from .alphalens import print_table, cumulative_returns, rename_forward_return_column
from .compatible import rolling_mean, qqplot
from .utils import affirm, get_chinese_font, ignore_warning


DECIMAL_TO_BPS = 10000


def customize(func):
    """
    Decorator to set plotting context and axes style during function call.
    """

    @wraps(func)
    def call_w_context(*args, **kwargs):
        affirm(PLOT_ENABLED, 'matplotlib disabled')

        if not PlotConfig.FONT_SETTED:
            _use_chinese(True)

        set_context = kwargs.pop('set_context', True)
        if set_context:
            with plotting_context(), axes_style():
                sns.despine(left=True)
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return call_w_context


def plotting_context(context='notebook', font_scale=1.5, rc=None):
    """
    Create alphalens default plotting style context.
    Under the hood, calls and returns seaborn.plotting_context() with
    some custom settings. Usually you would use in a with-context.
    Parameters
    ----------
    context : str, optional
        Name of seaborn context.
    font_scale : float, optional
        Scale font by factor font_scale.
    rc : dict, optional
        Config flags.
        By default, {'lines.linewidth': 1.5}
        is being used and will be added to any
        rc passed in, unless explicitly overriden.
    Returns
    -------
    seaborn plotting context
    Example
    -------
    with alphalens.plotting.plotting_context(font_scale=2):
        alphalens.create_full_tear_sheet(..., set_context=False)
    See also
    --------
    For more information, see seaborn.plotting_context().
    """
    if rc is None:
        rc = {}

    rc_default = {'lines.linewidth': 1.5}

    # Add defaults if they do not exist
    for name, val in rc_default.items():
        rc.setdefault(name, val)

    return sns.plotting_context(context=context, font_scale=font_scale, rc=rc)


def axes_style(style='darkgrid', rc=None):
    """Create alphalens default axes style context.
    Under the hood, calls and returns seaborn.axes_style() with
    some custom settings. Usually you would use in a with-context.
    Parameters
    ----------
    style : str, optional
        Name of seaborn style.
    rc : dict, optional
        Config flags.
    Returns
    -------
    seaborn plotting context
    Example
    -------
    with alphalens.plotting.axes_style(style='whitegrid'):
        alphalens.create_full_tear_sheet(..., set_context=False)
    See also
    --------
    For more information, see seaborn.plotting_context().
    """
    if rc is None:
        rc = {}

    rc_default = {}

    # Add defaults if they do not exist
    for name, val in rc_default.items():
        rc.setdefault(name, val)

    return sns.axes_style(style=style, rc=rc)


def plot_returns_table(alpha_beta, mean_ret_quantile, mean_ret_spread_quantile):
    returns_table = pd.DataFrame()
    returns_table = returns_table.append(alpha_beta)
    returns_table.loc["Mean Period Wise Return Top Quantile (bps)"] = \
        mean_ret_quantile.iloc[-1] * DECIMAL_TO_BPS
    returns_table.loc["Mean Period Wise Return Bottom Quantile (bps)"] = \
        mean_ret_quantile.iloc[0] * DECIMAL_TO_BPS
    returns_table.loc["Mean Period Wise Spread (bps)"] = \
        mean_ret_spread_quantile.mean() * DECIMAL_TO_BPS

    print("收益分析")
    print_table(rename_forward_return_column(returns_table.apply(lambda x: x.round(3))))


def plot_turnover_table(autocorrelation_data, quantile_turnover):
    turnover_table = pd.DataFrame()
    for period in sorted(quantile_turnover.keys()):
        for quantile, p_data in quantile_turnover[period].iteritems():
            turnover_table.loc["Quantile {} Mean Turnover ".format(quantile), "{}".format(period)
                               ] = p_data.mean()
    auto_corr = pd.DataFrame()
    for period, p_data in autocorrelation_data.iteritems():
        auto_corr.loc["Mean Factor Rank Autocorrelation", "{}".format(period)] = p_data.mean()

    print("换手率分析")
    print_table(rename_forward_return_column(turnover_table.apply(lambda x: x.round(3))))
    print_table(rename_forward_return_column(auto_corr.apply(lambda x: x.round(3))))


def plot_information_table(ic_data):
    ic_summary_table = pd.DataFrame()
    ic_summary_table["IC Mean"] = ic_data.mean()
    ic_summary_table["IC Std."] = ic_data.std()
    ic_summary_table["IR"] = ic_data.mean() / ic_data.std()
    t_stat, p_value = stats.ttest_1samp(ic_data, 0)
    ic_summary_table["t-stat(IC)"] = t_stat
    ic_summary_table["p-value(IC)"] = p_value
    ic_summary_table["IC Skew"] = stats.skew(ic_data)
    ic_summary_table["IC Kurtosis"] = stats.kurtosis(ic_data)

    print("IC 分析")
    print_table(rename_forward_return_column(ic_summary_table.apply(lambda x: x.round(3)).T))


def plot_quantile_statistics_table(factor_data):
    quantile_stats = factor_data.groupby('factor_quantile') \
        .agg(['min', 'max', 'mean', 'std', 'count'])['factor']
    quantile_stats['count %'] = quantile_stats['count'] \
        / quantile_stats['count'].sum() * 100.

    print("分位数统计")
    print_table(quantile_stats)


@customize
def plot_ic_ts(ic, ax=None):
    """
    Plots Spearman Rank Information Coefficient and IC moving
    average for a given factor.
    Parameters
    ----------
    ic : pd.DataFrame
        DataFrame indexed by date, with IC for each forward return.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    ic = ic.copy()

    num_plots = len(ic.columns)
    if ax is None:
        f, ax = plt.subplots(num_plots, 1, figsize=(18, num_plots * 7))
        ax = np.asarray([ax]).flatten()

    ymin, ymax = (None, None)
    for a, (period_num, ic) in zip(ax, ic.iteritems()):
        ic.plot(alpha=0.7, ax=a, lw=0.7, color='steelblue')
        rolling_mean(ic, window=22).plot(ax=a, color='forestgreen', lw=2, alpha=0.8)

        a.axhline(0.0, linestyle='-', color='black', lw=1, alpha=0.8)
        a.set(ylabel='IC', xlabel="")
        if _use_chinese():
            a.set_title("{} 天 IC".format(period_num))
            a.legend(['IC', '1个月移动平均'], loc='upper right')
            a.text(
                .05,
                .95,
                "均值 {:.3f} \n方差 {:.3f}".format(ic.mean(), ic.std()),
                fontsize=16,
                bbox={
                    'facecolor': 'white',
                    'alpha': 1,
                    'pad': 5
                },
                transform=a.transAxes,
                verticalalignment='top'
            )
        else:
            a.set_title("{} Period Forward Return Information Coefficient (IC)".format(period_num))
            a.legend(['IC', '1 month moving avg'], loc='upper right')
            a.text(
                .05,
                .95,
                "Mean {:.3f} \nStd. {:.3f}".format(ic.mean(), ic.std()),
                fontsize=16,
                bbox={
                    'facecolor': 'white',
                    'alpha': 1,
                    'pad': 5
                },
                transform=a.transAxes,
                verticalalignment='top'
            )

        curr_ymin, curr_ymax = a.get_ylim()
        ymin = curr_ymin if ymin is None else min(ymin, curr_ymin)
        ymax = curr_ymax if ymax is None else max(ymax, curr_ymax)

    for a in ax:
        a.set_ylim([ymin, ymax])

    return ax


@ignore_warning(message='Using a non-tuple sequence for multidimensional indexing is deprecated',
                category=FutureWarning)
@customize
def plot_ic_hist(ic, ax=None):
    """
    Plots Spearman Rank Information Coefficient histogram for a given factor.
    Parameters
    ----------
    ic : pd.DataFrame
        DataFrame indexed by date, with IC for each forward return.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    ic = ic.copy()

    num_plots = len(ic.columns)

    v_spaces = ((num_plots - 1) // 3) + 1

    if ax is None:
        f, ax = plt.subplots(v_spaces, 3, figsize=(18, v_spaces * 6))
        ax = ax.flatten()

    for a, (period_num, ic) in zip(ax, ic.iteritems()):
        sns.distplot(ic.replace(np.nan, 0.), norm_hist=True, ax=a)
        a.set_xlim([-1, 1])
        if _use_chinese():
            a.set(title="%s 天 IC 分布直方图" % period_num, xlabel='IC')
            a.text(
                .05,
                .95,
                "均值 {:.3f} \n方差 {:.3f}".format(ic.mean(), ic.std()),
                fontsize=16,
                bbox={
                    'facecolor': 'white',
                    'alpha': 1,
                    'pad': 5
                },
                transform=a.transAxes,
                verticalalignment='top'
            )
        else:
            a.set(title="%s Period IC" % period_num, xlabel='IC')
            a.text(
                .05,
                .95,
                "Mean {:.3f} \nStd. {:.3f}".format(ic.mean(), ic.std()),
                fontsize=16,
                bbox={
                    'facecolor': 'white',
                    'alpha': 1,
                    'pad': 5
                },
                transform=a.transAxes,
                verticalalignment='top'
            )
        a.axvline(ic.mean(), color='w', linestyle='dashed', linewidth=2)

    if num_plots < len(ax):
        for a in ax[num_plots:]:
            a.set_visible(False)

    return ax


@customize
def plot_ic_qq(ic, theoretical_dist=stats.norm, ax=None):
    """
    Plots Spearman Rank Information Coefficient "Q-Q" plot relative to
    a theoretical distribution.
    Parameters
    ----------
    ic : pd.DataFrame
        DataFrame indexed by date, with IC for each forward return.
    theoretical_dist : scipy.stats._continuous_distns
        Continuous distribution generator. scipy.stats.norm and
        scipy.stats.t are popular options.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    ic = ic.copy()

    num_plots = len(ic.columns)

    v_spaces = ((num_plots - 1) // 3) + 1

    if ax is None:
        f, ax = plt.subplots(v_spaces, 3, figsize=(18, v_spaces * 6))
        ax = ax.flatten()

    if isinstance(theoretical_dist, stats.norm.__class__):
        dist_name = '正态' if _use_chinese() else 'Normal'
    elif isinstance(theoretical_dist, stats.t.__class__):
        dist_name = 'T'
    else:
        dist_name = '自定义' if _use_chinese() else 'Theoretical'

    for a, (period_num, ic) in zip(ax, ic.iteritems()):
        qqplot(ic.replace(np.nan, 0.).values, theoretical_dist, fit=True, line='45', ax=a)
        if _use_chinese():
            a.set(
                title="{} 天 IC {}分布 Q-Q 图".format(period_num, dist_name),
                ylabel='Observed Quantile',
                xlabel='{}分布分位数'.format(dist_name)
            )
        else:
            a.set(
                title="{} Period IC {} Dist. Q-Q".format(period_num, dist_name),
                ylabel='Observed Quantile',
                xlabel='{} Distribution Quantile'.format(dist_name)
            )

    if num_plots < len(ax):
        for a in ax[num_plots:]:
            a.set_visible(False)

    return ax


@customize
def plot_quantile_returns_bar(mean_ret_by_q, by_group=False, ylim_percentiles=None, ax=None):
    """
    Plots mean period wise returns for factor quantiles.
    Parameters
    ----------
    mean_ret_by_q : pd.DataFrame
        DataFrame with quantile, (group) and mean period wise return values.
    by_group : bool
        Disaggregated figures by group.
    ylim_percentiles : tuple of integers
        Percentiles of observed data to use as y limits for plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    mean_ret_by_q = mean_ret_by_q.copy()
    if _use_chinese():
        mean_ret_by_q.columns = mean_ret_by_q.columns.map(lambda x: str(x) + ' 天')

    if ylim_percentiles is not None:
        ymin = (np.nanpercentile(mean_ret_by_q.values, ylim_percentiles[0]) * DECIMAL_TO_BPS)
        ymax = (np.nanpercentile(mean_ret_by_q.values, ylim_percentiles[1]) * DECIMAL_TO_BPS)
    else:
        ymin = None
        ymax = None

    if by_group:
        num_group = len(mean_ret_by_q.index.get_level_values('group').unique())

        if ax is None:
            v_spaces = ((num_group - 1) // 2) + 1
            f, ax = plt.subplots(
                v_spaces,
                2,
                sharex=False,
                sharey=True,
                figsize=(
                    max(18,
                        mean_ret_by_q.index.get_level_values('factor_quantile').max()), 6 * v_spaces
                )
            )
            ax = ax.flatten()

        for a, (sc, cor) in zip(ax, mean_ret_by_q.groupby(level='group')):
            (cor.xs(sc, level='group').multiply(DECIMAL_TO_BPS).plot(kind='bar', title=sc, ax=a))

            a.set(xlabel='', ylabel='Mean Return (bps)', ylim=(ymin, ymax))

        if num_group < len(ax):
            for a in ax[num_group:]:
                a.set_visible(False)

        return ax

    else:
        if ax is None:
            f, ax = plt.subplots(
                1,
                1,
                figsize=(
                    max(18,
                        mean_ret_by_q.index.get_level_values('factor_quantile').max() // 2), 6
                )
            )
        if _use_chinese():
            (mean_ret_by_q.multiply(DECIMAL_TO_BPS).plot(kind='bar', title="各分位数平均收益", ax=ax))
        else:
            (
                mean_ret_by_q.multiply(DECIMAL_TO_BPS)
                .plot(kind='bar', title="Mean Period Wise Return By Factor Quantile", ax=ax)
            )
        ax.set(xlabel='', ylabel='Mean Return (bps)', ylim=(ymin, ymax))

        return ax


@customize
def plot_quantile_returns_violin(return_by_q, ylim_percentiles=None, ax=None):
    """
    Plots a violin box plot of period wise returns for factor quantiles.
    Parameters
    ----------
    return_by_q : pd.DataFrame - MultiIndex
        DataFrame with date and quantile as rows MultiIndex,
        forward return windows as columns, returns as values.
    ylim_percentiles : tuple of integers
        Percentiles of observed data to use as y limits for plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    return_by_q = return_by_q.copy()

    if ylim_percentiles is not None:
        ymin = (np.nanpercentile(return_by_q.values, ylim_percentiles[0]) * DECIMAL_TO_BPS)
        ymax = (np.nanpercentile(return_by_q.values, ylim_percentiles[1]) * DECIMAL_TO_BPS)
    else:
        ymin = None
        ymax = None

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    unstacked_dr = (return_by_q.multiply(DECIMAL_TO_BPS))
    unstacked_dr.columns = unstacked_dr.columns.set_names('forward_periods')
    unstacked_dr = unstacked_dr.stack()
    unstacked_dr.name = 'return'
    unstacked_dr = unstacked_dr.reset_index()

    sns.violinplot(
        data=unstacked_dr,
        x='factor_quantile',
        hue='forward_periods',
        y='return',
        orient='v',
        cut=0,
        inner='quartile',
        ax=ax
    )
    ax.set(
        xlabel='',
        ylabel='Return (bps)',
        title="Period Wise Return By Factor Quantile",
        ylim=(ymin, ymax)
    )

    ax.axhline(0.0, linestyle='-', color='black', lw=0.7, alpha=0.6)

    return ax


@customize
def plot_mean_quantile_returns_spread_time_series(
    mean_returns_spread, std_err=None, bandwidth=1, quantiles='最大', ax=None
):
    """
    Plots mean period wise returns for factor quantiles.
    Parameters
    ----------
    mean_returns_spread : pd.Series
        Series with difference between quantile mean returns by period.
    std_err : pd.Series
        Series with standard error of difference between quantile
        mean returns each period.
    bandwidth : float
        Width of displayed error bands in standard deviations.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if isinstance(mean_returns_spread, pd.DataFrame):
        if ax is None:
            ax = [None for a in mean_returns_spread.columns]

        ymin, ymax = (None, None)
        for (i, a), (name, fr_column) in zip(enumerate(ax), mean_returns_spread.iteritems()):
            stdn = None if std_err is None else std_err[name]
            a = plot_mean_quantile_returns_spread_time_series(
                fr_column, std_err=stdn, bandwidth=bandwidth, quantiles=quantiles, ax=a
            )
            ax[i] = a
            curr_ymin, curr_ymax = a.get_ylim()
            ymin = curr_ymin if ymin is None else min(ymin, curr_ymin)
            ymax = curr_ymax if ymax is None else max(ymax, curr_ymax)

        for a in ax:
            a.set_ylim([ymin, ymax])

        return ax

    periods = mean_returns_spread.name
    if _use_chinese():
        title = (
            '{} 分位收益减 1 分位收益 ({} 天)'.format(quantiles, (periods if periods is not None else ""))
        )
    else:
        title = (
            'Top Minus Bottom Quantile Mean Return ({} Period Forward Return)'
            .format(periods if periods is not None else "")
        )

    if ax is None:
        f, ax = plt.subplots(figsize=(18, 6))

    mean_returns_spread_bps = mean_returns_spread * DECIMAL_TO_BPS

    mean_returns_spread_bps.plot(alpha=0.4, ax=ax, lw=0.7, color='forestgreen')
    rolling_mean(mean_returns_spread_bps, window=22).plot(color='orangered', alpha=0.7, ax=ax)
    if _use_chinese():
        ax.legend(['当日收益 (加减 {:.2f} 倍当日标准差)'.format(bandwidth), '1 个月移动平均'], loc='upper right')
    else:
        ax.legend(['mean returns spread', '1 month moving avg'], loc='upper right')

    if std_err is not None:
        std_err_bps = std_err * DECIMAL_TO_BPS
        upper = mean_returns_spread_bps.values + (std_err_bps * bandwidth)
        lower = mean_returns_spread_bps.values - (std_err_bps * bandwidth)
        ax.fill_between(mean_returns_spread.index, lower, upper, alpha=0.3, color='steelblue')

    ylim = np.nanpercentile(abs(mean_returns_spread_bps.values), 95)
    ax.set(
        ylabel='Difference In Quantile Mean Return (bps)',
        xlabel='',
        title=title,
        ylim=(-ylim, ylim)
    )
    ax.axhline(0.0, linestyle='-', color='black', lw=1, alpha=0.8)

    return ax


@customize
def plot_ic_by_group(ic_group, ax=None):
    """
    Plots Spearman Rank Information Coefficient for a given factor over
    provided forward returns. Separates by group.
    Parameters
    ----------
    ic_group : pd.DataFrame
        group-wise mean period wise returns.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(max(18, len(ic_group)), 6))
    ic_group.plot(kind='bar', ax=ax)

    if _use_chinese():
        ax.set(title="分行业 IC", xlabel="")
    else:
        ax.set(title="Information Coefficient By Group", xlabel="")
    ax.set_xticklabels(ic_group.index, rotation=45)

    return ax


@customize
def plot_factor_rank_auto_correlation(factor_autocorrelation, period=1, ax=None):
    """
    Plots factor rank autocorrelation over time.
    See factor_rank_autocorrelation for more details.
    Parameters
    ----------
    factor_autocorrelation : pd.Series
        Rolling 1 period (defined by time_rule) autocorrelation
        of factor values.
    period: int, optional
        Period over which the autocorrelation is calculated
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    if _use_chinese():
        factor_autocorrelation.plot(title='因子自相关性 (滞后 {} 天)'.format(period), ax=ax)
        ax.set(ylabel='自相关性', xlabel='')
        ax.axhline(0.0, linestyle='-', color='black', lw=1)
        ax.text(
            .05,
            .95,
            "均值 %.3f" % factor_autocorrelation.mean(),
            fontsize=16,
            bbox={
                'facecolor': 'white',
                'alpha': 1,
                'pad': 5
            },
            transform=ax.transAxes,
            verticalalignment='top'
        )
    else:
        factor_autocorrelation.plot(title='{} Period Factor Autocorrelation'.format(period), ax=ax)
        ax.set(ylabel='Autocorrelation Coefficient', xlabel='')
        ax.axhline(0.0, linestyle='-', color='black', lw=1)
        ax.text(
            .05,
            .95,
            "Mean %.3f" % factor_autocorrelation.mean(),
            fontsize=16,
            bbox={
                'facecolor': 'white',
                'alpha': 1,
                'pad': 5
            },
            transform=ax.transAxes,
            verticalalignment='top'
        )

    return ax


@customize
def plot_top_bottom_quantile_turnover(quantile_turnover, period=1, ax=None):
    """
    Plots period wise top and bottom quantile factor turnover.
    Parameters
    ----------
    quantile_turnover: pd.Dataframe
        Quantile turnover (each DataFrame column a quantile).
    period: int, optional
        Period over which to calculate the turnover
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    max_quantile = quantile_turnover.columns.max()
    min_quantile = quantile_turnover.columns.min()
    turnover = pd.DataFrame()

    if _use_chinese():
        turnover['{:d} 分位换手率'.format(max_quantile)] = quantile_turnover[max_quantile]
        turnover['{:d} 分位换手率'.format(min_quantile)] = quantile_turnover[min_quantile]
        turnover.plot(title='{} 天换手率'.format(period), ax=ax, alpha=0.6, lw=0.8)
    else:
        turnover['top quantile turnover'] = quantile_turnover[max_quantile]
        turnover['bottom quantile turnover'] = quantile_turnover[min_quantile]
        turnover.plot(
            title='{} Period Top and Bottom Quantile Turnover'.format(period),
            ax=ax,
            alpha=0.6,
            lw=0.8
        )
    ax.set(ylabel='Proportion Of Names New To Quantile', xlabel="")

    return ax


@customize
def plot_monthly_ic_heatmap(mean_monthly_ic, ax=None):
    """
    Plots a heatmap of the information coefficient or returns by month.
    Parameters
    ----------
    mean_monthly_ic : pd.DataFrame
        The mean monthly IC for N periods forward.
    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    mean_monthly_ic = mean_monthly_ic.copy()

    num_plots = len(mean_monthly_ic.columns)

    v_spaces = ((num_plots - 1) // 3) + 1

    if ax is None:
        f, ax = plt.subplots(v_spaces, 3, figsize=(18, v_spaces * 6))
        ax = ax.flatten()

    new_index_year = []
    new_index_month = []
    for date in mean_monthly_ic.index:
        new_index_year.append(date.year)
        new_index_month.append(date.month)

    mean_monthly_ic.index = pd.MultiIndex.from_arrays(
        [new_index_year, new_index_month], names=["year", "month"]
    )

    for a, (periods_num, ic) in zip(ax, mean_monthly_ic.iteritems()):

        sns.heatmap(
            ic.unstack(),
            annot=True,
            alpha=1.0,
            center=0.0,
            annot_kws={"size": 7},
            linewidths=0.01,
            linecolor='white',
            cmap=cm.RdYlGn,
            cbar=False,
            ax=a
        )
        a.set(ylabel='', xlabel='')

        if _use_chinese():
            a.set_title("{} 天 IC 月度均值".format(periods_num))
        else:
            a.set_title("Monthly Mean {} Period IC".format(periods_num))

    if num_plots < len(ax):
        for a in ax[num_plots:]:
            a.set_visible(False)

    return ax


@customize
def plot_cumulative_returns(factor_returns, period=1, overlap=True, ax=None):
    """
    Plots the cumulative returns of the returns series passed in.
    Parameters
    ----------
    factor_returns : pd.Series
        Period wise returns of dollar neutral portfolio weighted by factor
        value.
    period: int, optional
        Period over which the daily returns are calculated
    overlap: boolean, optional
        Specify if subsequent returns overlaps.
        If 'overlap' is True and 'period' N is greater than 1, the cumulative
        returns plot is computed building and averaging the  cumulative returns
        of N interleaved portfolios (started at subsequent periods 1,2,3,...,N)
        each one rebalancing every N periods. This results in trading the
        factor at every value/signal computed by the factor and also the
        cumulative returns don't dependent on a specific starting date.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    overlapping_period = period if overlap else 1
    factor_returns = cumulative_returns(factor_returns, overlapping_period)

    factor_returns.plot(ax=ax, lw=3, color='forestgreen', alpha=0.6)
    if _use_chinese():
        ax.set(ylabel='累积收益', title='因子值加权多空组合累积收益 ({} 天平均)'.format(period), xlabel='')
    else:
        ax.set(
            ylabel='Cumulative Returns',
            title='''Factor Weighted Long/Short Portfolio Cumulative Return
                        ({} Fwd Period)'''.format(period),
            xlabel=''
        )
    ax.axhline(1.0, linestyle='-', color='black', lw=1)

    return ax


@customize
def plot_top_down_cumulative_returns(factor_returns, period=1, ax=None):

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    factor_returns.plot(ax=ax, lw=3, color='forestgreen', alpha=0.6)
    if _use_chinese():
        ax.set(
            ylabel='累积收益',
            title='做多最大分位做空最小分位组合累积收益 ({} 天平均)'.format(period),
            xlabel=''
        )
    else:
        ax.set(
            ylabel='Cumulative Returns',
            title='''Long Top/Short Bottom Factor Portfolio Cumulative Return
                        ({} Fwd Period)'''.format(period),
            xlabel=''
        )
    ax.axhline(1.0, linestyle='-', color='black', lw=1)

    return ax


@customize
def plot_cumulative_returns_by_quantile(quantile_returns, period=1, overlap=True, ax=None):
    """
    Plots the cumulative returns of various factor quantiles.
    Parameters
    ----------
    quantile_returns : pd.DataFrame
        Cumulative returns by factor quantile.
    period: int, optional
        Period over which the daily returns are calculated
    overlap: boolean, optional
        Specify if subsequent returns overlaps.
        If 'overlap' is True and 'period' N is greater than 1, the cumulative
        returns plot is computed building and averaging the  cumulative returns
        of N interleaved portfolios (started at subsequent periods 1,2,3,...,N)
        each one rebalancing every N periods. This results in trading the
        factor at every value/signal computed by the factor and also the
        cumulative returns don't dependent on a specific starting date.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    Returns
    -------
    ax : matplotlib.Axes
    """

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    ret_wide = quantile_returns.reset_index()\
        .pivot(index='date', columns='factor_quantile', values=period)

    overlapping_period = period if overlap else 1
    cum_ret = ret_wide.apply(cumulative_returns, args=(overlapping_period,))
    cum_ret = cum_ret.loc[:, ::-1]

    cum_ret.plot(lw=2, ax=ax, cmap=cm.RdYlGn_r)
    ax.legend()
    ymin, ymax = cum_ret.min().min(), cum_ret.max().max()
    if _use_chinese():
        ax.set(
            ylabel='累积收益(对数轴)',
            title='分位数 {} 天 Forward Return 累积收益 (对数轴)'.format(period),
            xlabel='',
            ylim=(ymin, ymax)
        )
    else:
        ax.set(
            ylabel='Log Cumulative Returns',
            title='''Cumulative Return by Quantile
                        ({} Period Forward Return)'''.format(period),
            xlabel='',
            ylim=(ymin, ymax)
        )
    ax.set_yscale('symlog', linthreshy=1)
    ax.set_yticks(np.linspace(ymin, ymax, 8))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.axhline(1.0, linestyle='-', color='black', lw=1)

    return ax


@customize
def plot_quantile_average_cumulative_return(
    avg_cumulative_returns,
    by_quantile=False,
    std_bar=False,
    title=None,
    ax=None,
    periods_before='',
    periods_after=''
):
    """
    Plots sector-wise mean daily returns for factor quantiles
    across provided forward price movement columns.
    Parameters
    ----------
    avg_cumulative_returns: pd.Dataframe
        The format is the one returned by
        performance.average_cumulative_return_by_quantile
    by_quantile : boolean, optional
        Disaggregated figures by quantile (useful to clearly see std dev bars)
    std_bar : boolean, optional
        Plot standard deviation plot
    title: string, optional
        Custom title
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    Returns
    -------
    ax : matplotlib.Axes
    """

    avg_cumulative_returns = avg_cumulative_returns.multiply(DECIMAL_TO_BPS)
    quantiles = len(avg_cumulative_returns.index.levels[0].unique())
    palette = [cm.RdYlGn_r(i) for i in np.linspace(0, 1, quantiles)]

    if by_quantile:

        if ax is None:
            v_spaces = ((quantiles - 1) // 2) + 1
            f, ax = plt.subplots(
                v_spaces, 2, sharex=False, sharey=False, figsize=(18, 6 * v_spaces)
            )
            ax = ax.flatten()

        for i, (quantile,
                q_ret) in enumerate(avg_cumulative_returns.groupby(level='factor_quantile')):

            mean = q_ret.loc[(quantile, 'mean')]
            mean.name = (str(quantile) + ' 分位') if _use_chinese() else ('Quantile ' + str(quantile))
            mean.plot(ax=ax[i], color=palette[i])
            ax[i].set_ylabel(('平均累积收益 (bps)' if _use_chinese() else 'Mean Return (bps)'))

            if std_bar:
                std = q_ret.loc[(quantile, 'std')]
                ax[i].errorbar(std.index, mean, yerr=std, fmt='none', ecolor=palette[i], label=None)

            ax[i].axvline(x=0, color='k', linestyle='--')
            ax[i].legend()
            i += 1

    else:

        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(18, 6))

        for i, (quantile,
                q_ret) in enumerate(avg_cumulative_returns.groupby(level='factor_quantile')):

            mean = q_ret.loc[(quantile, 'mean')]
            mean.name = (
                (str(quantile) + ' 分位') if _use_chinese() else ('Quantile ' + str(quantile))
            )
            mean.plot(ax=ax, color=palette[i])

            if std_bar:
                std = q_ret.loc[(quantile, 'std')]
                ax.errorbar(std.index, mean, yerr=std, fmt='none', ecolor=palette[i], label=None)
            i += 1

        ax.axvline(x=0, color='k', linestyle='--')
        ax.legend()
        if _use_chinese():
            ax.set(
                ylabel='平均累积收益 (bps)',
                title=(
                    "因子预测能力 (前 {} 天, 后 {} 天)".format(periods_before, periods_after)
                    if title is None else title
                ),
                xlabel='Periods'
            )
        else:
            ax.set(
                ylabel='Mean Return (bps)',
                title=("Average Cumulative Returns by Quantile" if title is None else title),
                xlabel='Periods'
            )

    return ax


@customize
def plot_events_distribution(events, num_days=5, full_dates=None, ax=None):
    """
    Plots the distribution of events in time.
    Parameters
    ----------
    events : pd.Series
        A pd.Series whose index contains at least 'date' level.
    num_bars : integer, optional
        Number of bars to plot
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    Returns
    -------
    ax : matplotlib.Axes
    """

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    if full_dates is None:
        full_dates = events.index.get_level_values('date').unique()

    group = pd.Series(range(len(full_dates)), index=full_dates) // num_days
    grouper_label = group.drop_duplicates()
    grouper = group.reindex(events.index.get_level_values('date'))

    count = events.groupby(grouper.values).count()
    count = count.reindex(grouper_label.values, fill_value=0)
    count.index = grouper_label.index.map(lambda x: x.strftime('%Y-%m-%d'))
    count.plot(kind="bar", grid=False, ax=ax)

    def annotateBars(x, dt, ax=ax):
        color = 'black'
        vertalign = 'top'
        ax.text(x, count.loc[dt], "{:d}".format(count.loc[dt]),
                rotation=45, color=color,
                horizontalalignment='center',
                verticalalignment=vertalign,
                fontsize=15, weight='heavy')
    [annotateBars(x, dt, ax=ax) for x, dt in enumerate(list(count.index))]

    if _use_chinese():
        ax.set(ylabel='因子数量', title='因子数量随时间分布', xlabel='Date')
    else:
        ax.set(ylabel='Number of events', title='Distribution of events in time', xlabel='Date')

    return ax


@customize
def plot_missing_events_distribution(events, num_days=5, full_dates=None, ax=None):
    """
    Plots the distribution of missing events ratio in time.
    Parameters
    ----------
    events : pd.Series
        A pd.Series whose index contains at least 'date' level.
    num_bars : integer, optional
        Number of bars to plot
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    Returns
    -------
    ax : matplotlib.Axes
    """

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    if full_dates is None:
        full_dates = events.index.get_level_values('date').unique()

    daily_count = events.groupby(level='date').count()
    most_common_count = np.argmax(np.bincount(daily_count))
    daily_missing = daily_count / most_common_count - 1
    daily_missing = daily_missing.reindex(full_dates, fill_value=-1.0)

    grouper = pd.Series(range(len(full_dates)), index=full_dates) // num_days
    grouper_label = grouper.drop_duplicates()

    missing = daily_missing.groupby(grouper.values).mean()
    missing = missing.reindex(grouper_label.values, fill_value=-1.0)
    missing.index = grouper_label.index.map(lambda x: x.strftime('%Y-%m-%d'))
    missing.plot(kind="bar", grid=False, ax=ax)

    def annotateBars(x, dt, ax=ax):
        color = 'black'
        vertalign = 'top'
        ax.text(x, missing.loc[dt], "{:+.1f}%".format(missing.loc[dt] * 100),
                rotation=45, color=color,
                horizontalalignment='center',
                verticalalignment=vertalign,
                fontsize=15, weight='heavy')
    [annotateBars(x, dt, ax=ax) for x, dt in enumerate(list(missing.index))]

    if _use_chinese():
        ax.set(ylabel='因子缺失率', title='因子缺失率随时间分布', xlabel='Date')
    else:
        ax.set(ylabel='Rate of missing events', title='Distribution of missing events in time',
               xlabel='Date')

    return ax


class PlotConfig(object):
    FONT_SETTED = False
    USE_CHINESE_LABEL = False
    MPL_FONT_FAMILY = mpl.rcParams['font.family'] if PLOT_ENABLED else None
    MPL_FONT = mpl.rcParams['font.sans-serif'] if PLOT_ENABLED else None


def _use_chinese(use=None):
    if not PLOT_ENABLED:
        return
    if use is None:
        return PlotConfig.USE_CHINESE_LABEL
    elif use:
        PlotConfig.USE_CHINESE_LABEL = use
        PlotConfig.FONT_SETTED = True
        _set_chinese_fonts()
    else:
        PlotConfig.USE_CHINESE_LABEL = use
        PlotConfig.FONT_SETTED = True
        _set_default_fonts()


def _set_chinese_fonts():
    chinese_font = get_chinese_font()
    if chinese_font:
        # 中文乱码
        mpl.rc('font', **{'sans-serif': [chinese_font], 'family': 'sans-serif'})
        # 负号乱码
        mpl.rcParams['axes.unicode_minus'] = False
    else:
        _use_chinese(False)


def _set_default_fonts():
    mpl.rc('font', **{'sans-serif': PlotConfig.MPL_FONT, 'family': PlotConfig.MPL_FONT_FAMILY})
    # 负号乱码
    mpl.rcParams['axes.unicode_minus'] = False
