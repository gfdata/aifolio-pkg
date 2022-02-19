# -*- coding: utf-8 -*-
# @time: 2022/2/14 15:17
# @Author：lhf
# ----------------------
import os
import warnings
from pathlib import Path
from aiutils.config_load import ConfigLoading, config_dir

AIFOLIO_PROJ_DIR = Path(__file__).parent.parent  # 项目路径


def _this_config():
    name = Path(os.path.abspath(__file__)).parts[-2]  # 'ailocal'
    default_dir = os.path.abspath(os.path.expanduser(f'~/env'))
    # 默认配置
    _default = {
        # 文件存储的目录
        "cache_dir": os.path.join(default_dir, f'cache.{name}'),
        "alpha_cache_dir": os.path.join(default_dir, f'cache.{name}', 'alphaxx'),
        "alpha_cache_s": 3600 * 24 * 7,
    }
    # 其它位置的配置
    cf = ConfigLoading(_default, app_name=name)
    try:
        cf_file = config_dir(config_relative='env', proj_dir=AIFOLIO_PROJ_DIR)[-1].joinpath(f'{name}.json')
        cf.update_config_json(cf_file)
    except Exception as e:
        msg = f'\n 项目[{name}]加载配置失败，使用内置default配置 {e}'
        warnings.warn(msg, RuntimeWarning)
    return cf.attr_dict


AIFOLIO_CONFIG = _this_config()

# alphalens alphacn 分析包的缓存
ALPHA_CHACHE_DIR = AIFOLIO_CONFIG.alpha_cache_dir
ALPHA_CHACHE_S = AIFOLIO_CONFIG.alpha_cache_s
