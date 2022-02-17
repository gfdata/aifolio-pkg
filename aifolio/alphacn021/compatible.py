# encoding: utf-8

"""Python 版本与库版本兼容模块"""

import warnings

import pandas as pd

PD_VERSION = pd.__version__


# pandas
def rolling_apply(x, window, func, min_periods=None, freq=None, center=False,
                  args=tuple(), kwargs=dict()):

    if PD_VERSION >= '0.23.0':
        return x.rolling(
            window,
            min_periods=min_periods,
            center=center
        ).apply(
            func, False,
            args=args,
            kwargs=kwargs
        )
    elif PD_VERSION >= '0.18.0':
        return x.rolling(
            window,
            min_periods=min_periods,
            center=center
        ).apply(
            func,
            args=args,
            kwargs=kwargs
        )
    else:
        return pd.rolling_apply(
            x, window, func,
            min_periods=min_periods,
            freq=freq,
            center=center,
            args=args,
            kwargs=kwargs
        )


def rolling_mean(x, window, min_periods=None, freq=None, center=False, how=None):
    """freq, how 参数从 0.18.0 之后被 Deprecated 了，从 0.23.0 之后被删掉了"""
    if PD_VERSION >= '0.18.0':
        return x.rolling(window, min_periods=min_periods, center=center).mean()
    else:
        return pd.rolling_mean(
            x, window, min_periods=min_periods, freq=freq, center=center, how=how
        )


def rolling_std(x, window, min_periods=None, freq=None, center=False, how=None, ddof=1):
    """freq, how 参数从 0.18.0 之后被 Deprecated 了，从 0.23.0 之后被删掉了"""
    if PD_VERSION >= '0.18.0':
        return x.rolling(window, min_periods=min_periods, center=center).std(ddof=ddof)
    else:
        return pd.rolling_std(
            x, window,
            min_periods=min_periods,
            freq=freq,
            center=center,
            how=how,
            ddof=ddof
        )


def sortlevel(x, level=0, axis=0, ascending=True, inplace=False, sort_remaining=True):
    if PD_VERSION >= '0.20.0':
        return x.sort_index(
            level=level,
            axis=axis,
            ascending=ascending,
            inplace=inplace,
            sort_remaining=sort_remaining
        )
    else:
        return x.sortlevel(
            level=level,
            axis=axis,
            ascending=ascending,
            inplace=inplace,
            sort_remaining=sort_remaining
        )


# statsmodels
with warnings.catch_warnings():
    # 有的版本依赖的 pandas 库会有 deprecated warning
    warnings.simplefilter("ignore")
    import statsmodels
    from statsmodels.api import OLS, qqplot, ProbPlot
    from statsmodels.tools.tools import add_constant
