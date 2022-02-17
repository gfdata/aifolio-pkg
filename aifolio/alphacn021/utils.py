# -*- coding: utf-8 -*-

import sys
import os
import re
import ast
import six
import math
import json
import types
import inspect
import hashlib
import warnings
import subprocess
from functools import wraps
from collections import OrderedDict, MutableMapping, Iterable
try:
    from thread import get_ident
except ImportError:
    from threading import get_ident


range = xrange if sys.version_info[0] < 3 else range


def setdefaultencoding(encoding='utf-8'):
    if sys.version_info[0] >= 3:
        return

    stdin, stdout, stderr = sys.stdin, sys.stdout, sys.stderr
    reload(sys)
    sys.setdefaultencoding(encoding)
    sys.stdin, sys.stdout, sys.stderr = stdin, stdout, stderr


def _make_assert_message(frame, regex):
    def extract_condition():
        code_context = inspect.getframeinfo(frame)[3]
        if not code_context:
            return ''
        match = re.search(regex, code_context[0])
        if not match:
            return ''
        return match.group(1).strip()

    class ReferenceFinder(ast.NodeVisitor):
        def __init__(self):
            self.names = []

        def find(self, tree, frame):
            self.visit(tree)
            nothing = object()
            deref = OrderedDict()
            for name in self.names:
                value = frame.f_locals.get(name, nothing) or frame.f_globals.get(name, nothing)
                if (value is not nothing and
                        not isinstance(value, (types.ModuleType, types.FunctionType))):
                    deref[name] = repr(value)
            return deref

        def visit_Name(self, node):
            self.names.append(node.id)

    condition = extract_condition()
    if not condition:
        return
    deref = ReferenceFinder().find(ast.parse(condition), frame)
    deref_str = ''
    if deref:
        deref_str = ' with ' + ', '.join('{}={}'.format(k, v) for k, v in deref.items())
    return 'assertion {} failed{}'.format(condition, deref_str)


def affirm(condition, message=None):
    if condition:
        return

    if message:
        raise AssertionError(str(message))

    frame = inspect.currentframe().f_back
    regex = r'affirm\s*\(\s*(.+)\s*\)'
    message = _make_assert_message(frame, regex)

    raise AssertionError(message)


def md5(data):
    if not isinstance(data, bytes):
        # bytes built-in is just an alias to the str type in Python 2.x
        data = data.encode('utf-8')
    return hashlib.md5(data).hexdigest()


def json_serial_fallback(obj):
    """JSON serializer for objects not serializable by default json code"""
    import numpy as np

    if isinstance(obj, np.int64):
        return int(obj)

    raise TypeError("{!r} (type: {!r}) not serializable".format(obj, type(obj)))


def json_dumps(obj):
    """Dump object to json"""
    return json.dumps(obj, default=json_serial_fallback)


class suppress(object):
    """Context manager to suppress specified exceptions

    After the exception is suppressed, execution proceeds with the next
    statement following the with statement.
         with suppress(FileNotFoundError):
             os.remove(somefile)
         # Execution still resumes here if the file was already removed

    copy from https://github.com/jazzband/contextlib2
    """

    def __init__(self, *exceptions):
        self._exceptions = exceptions

    def __enter__(self):
        pass

    def __exit__(self, exctype, excinst, exctb):
        # Unlike isinstance and issubclass, CPython exception handling
        # currently only looks at the concrete type hierarchy (ignoring
        # the instance and subclass checking hooks). While Guido considers
        # that a bug rather than a feature, it's a fairly hard one to fix
        # due to various internal implementation details. suppress provides
        # the simpler issubclass based semantics, rather than trying to
        # exactly reproduce the limitations of the CPython interpreter.
        #
        # See http://bugs.python.org/issue12029 for more details
        return exctype is not None and issubclass(exctype, self._exceptions)


def singleton(cls):
    """单例装饰器"""
    instances = {}

    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper


def filter_dict_values(dct, fields):
    """按给定字段的顺序过滤出字典的值到一个列表中

    如果给定的字段不在字典中，则该字段值为 None
    """
    return [dct.get(field) for field in fields]


class char_range(object):

    def __init__(self, start, stop):
        self.start, self.stop = ord(start), ord(stop) + 1

    def __iter__(self):
        for ci in range(self.start, self.stop):
            yield chr(ci)


def _recursive_repr(fillvalue='...'):
    """Decorator to make a repr function return fillvalue for a recursive call"""

    def decorating_function(user_function):
        repr_running = set()

        def wrapper(self):
            key = id(self), get_ident()
            if key in repr_running:
                return fillvalue
            repr_running.add(key)
            try:
                result = user_function(self)
            finally:
                repr_running.discard(key)
            return result

        # Can't use functools.wraps() here because of bootstrap issues
        wrapper.__module__ = getattr(user_function, '__module__')
        wrapper.__doc__ = getattr(user_function, '__doc__')
        wrapper.__name__ = getattr(user_function, '__name__')
        # wrapper.__qualname__ = getattr(user_function, '__qualname__')
        wrapper.__annotations__ = getattr(user_function, '__annotations__', {})
        return wrapper

    return decorating_function


class ChainMap(MutableMapping):
    """A ChainMap groups multiple dicts (or other mappings).

    A ChainMap groups multiple dicts (or other mappings)together
    to create a single, updateable view.

    The underlying mappings are stored in a list.  That list is public and can
    be accessed or updated using the *maps* attribute.  There is no other
    state.

    Lookups search the underlying mappings successively until a key is found.
    In contrast, writes, updates, and deletions only operate on the first
    mapping.
    """

    def __init__(self, *maps):
        """Initialize a ChainMap by setting *maps* to the given mappings.

        If no mappings are provided, a single empty dictionary is used.
        """
        self.maps = list(maps) or [{}]  # always at least one map

    def __missing__(self, key):
        raise KeyError(key)

    def __getitem__(self, key):
        for mapping in self.maps:
            try:
                return mapping[key]  # can't use 'key in mapping' with defaultdict
            except KeyError:
                pass
        return self.__missing__(key)  # support subclasses that define __missing__

    def get(self, key, default=None):
        return self[key] if key in self else default

    def __len__(self):
        return len(set().union(*self.maps))  # reuses stored hash values if possible

    def __iter__(self):
        d = {}
        for mapping in reversed(self.maps):
            d.update(mapping)  # reuses stored hash values if possible
        return iter(d)

    def __contains__(self, key):
        return any(key in m for m in self.maps)

    def __bool__(self):
        return any(self.maps)

    @_recursive_repr()
    def __repr__(self):
        return '{0.__class__.__name__}({1})'.format(self, ', '.join(map(repr, self.maps)))

    @classmethod
    def fromkeys(cls, iterable, *args):
        """Create a ChainMap with a single dict created from the iterable."""
        return cls(dict.fromkeys(iterable, *args))

    def copy(self):
        """New ChainMap or subclass with a new copy of maps[0] and refs to maps[1:]"""
        return self.__class__(self.maps[0].copy(), *self.maps[1:])

    __copy__ = copy

    def new_child(self, m=None):  # like Django's Context.push()
        """New ChainMap with a new map followed by all previous maps.

        If no map is provided, an empty dict is used.
        """
        if m is None:
            m = {}
        return self.__class__(m, *self.maps)

    @property
    def parents(self):  # like Django's Context.pop()
        """New ChainMap from maps[1:]."""
        return self.__class__(*self.maps[1:])

    def __setitem__(self, key, value):
        self.maps[0][key] = value

    def __delitem__(self, key):
        try:
            del self.maps[0][key]
        except KeyError:
            raise KeyError('Key not found in the first mapping: {!r}'.format(key))

    def popitem(self):
        """Remove and return an item pair from maps[0].

        Raise KeyError is maps[0] is empty.
        """
        try:
            return self.maps[0].popitem()
        except KeyError:
            raise KeyError('No keys found in the first mapping.')

    def pop(self, key, *args):
        """Remove *key* from maps[0] and return its value.

        Raise KeyError if *key* not in maps[0].
        """
        try:
            return self.maps[0].pop(key, *args)
        except KeyError:
            raise KeyError('Key not found in the first mapping: {!r}'.format(key))

    def clear(self):
        """Clear maps[0], leaving maps[1:] intact."""
        self.maps[0].clear()


def get_chinese_font():
    try:
        if sys.platform.startswith('win'):
            if os.path.exists('c:\\windows\\fonts\\simhei.ttf'):
                return 'simhei'  # 微软黑体
            else:
                return None
        elif sys.platform.startswith('linux'):
            cmd = 'fc-list :lang=zh -f "%{family}\n"'
            output = subprocess.check_output(cmd, shell=True)
            if isinstance(output, bytes):
                output = output.decode("utf-8")
            zh_fonts = [
                f.split(',', 1)[0] for f in output.split('\n') if f.split(',', 1)[0]
            ]
            return zh_fonts[0] if zh_fonts else None
    except Exception as e:
        warnings.warn(str(e), Warning)
        return None


def ignore_warning(message='', category=Warning, module='', lineno=0, append=False):
    # 忽略 warnings
    def decorator(func):
        @wraps(func)
        def func_wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message=message, category=category,
                                        module=module, lineno=lineno, append=append)
                return func(*args, **kwargs)
        return func_wrapper

    return decorator


def ensure_list(x):
    if isinstance(x, six.string_types) or not isinstance(x, Iterable):
        return [x]
    else:
        return list(x)


def text_shorten(text, width=60, placeholder="..."):
    return text[:width] + (text[width:] and placeholder)


def do_line_profile(func):
    from line_profiler import LineProfiler

    def profiled_func(*args, **kwargs):
        lp = LineProfiler()
        try:
            lp_wrap = lp(func)
            ret = lp_wrap(*args, **kwargs)
            return ret
        finally:
            lp.print_stats()

    return profiled_func


def get_method_instance(method):
    if sys.version_info.major < 3:
        return getattr(method, "im_self", None)
    else:
        return getattr(method, "__self__", None)


def assert_float_equal(left, right, error=1e-5):
    if math.isnan(left):
        assert math.isnan(left) and math.isnan(right)
    else:
        assert abs(left - right) < error


def assert_sequence_equal(left, right, error=1e-5, cmp_type=False):
    if cmp_type:
        assert type(left) == type(right)
    for x, y in zip(left, right):
        if isinstance(x, float):
            assert_float_equal(x, y, error)
        else:
            assert x == y


def assert_dict_equal(left, right, error=1e-5):
    for k, v in left.items():
        if isinstance(v, float):
            assert_float_equal(v, right[k], error)
        else:
            assert v == right[k]


def assert_series_equal(left, right, error=1e-5, cmp_length=True,
                        cmp_index=False):
    left_size = len(left)
    right_size = len(right)
    if cmp_length:
        assert left_size == right_size

    is_equal = True
    for i in range(left_size):
        left_index = left.index[i]
        right_index = right.index[i]
        if cmp_index and left_index != right_index:
            print("No. {}, index {} != {}".format(i, left_index, right_index))
            is_equal = False

        left_value = left.iloc[i]
        right_value = right.iloc[i]
        try:
            if isinstance(left_value, float):
                assert_float_equal(left_value, right_value, error)
            else:
                assert left_value == right_value
        except AssertionError:
            print("No. {}, value {}: {} != {}: {}".format(
                i, left_index, left_value, right_index, right_value
            ))
            is_equal = False

    assert is_equal
