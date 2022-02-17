# -*- coding: utf-8 -*-
# @time: 2022/2/14 14:59
# @Author：lhf
# ----------------------
from pathlib import Path


def get_version():
    """
    规则：读取同级目录下，{父目录}_version.md文件
    :return:
    """
    _dir = Path(__file__).parent
    file = _dir.joinpath(f'{_dir.name}_version.md')
    version = '0.0.0'
    with open(file, encoding='utf8') as f:
        while True:
            aline = f.readline()
            if aline == '':
                break

            version = aline.strip()
            if version:
                break

    return version


__version__ = get_version()
