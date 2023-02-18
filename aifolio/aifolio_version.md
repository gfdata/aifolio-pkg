1.0.2

pyfolio_tools增加对rqalpha结果的转换

# 1.0.1

修改settings配置读入方式

增加alpha_tools

# 1.0.0

alphalens040 和 alphacn021 区别

* compute_forward_returns：alphacn021不用传factor参数，而alphalens040要传入

alphalens040

* 最新0.4.0版本，计算n period 的收益有问题(plot_cumulative_returns超过1D时)
* 计算累计收益时，依赖另一个包empyrical

alphacn021

* plot作图进行了部分汉化
* 基于alphalens021版本，也是jqfactor使用的版本
* 计算结果比较准确，处理函数比较简洁
