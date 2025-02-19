import atexit
from functools import wraps

from line_profiler import LineProfiler
from time import perf_counter


# 定义装饰器，自动进行性能分析
def profile(func):
    profiler = LineProfiler()  # 只创建一个 LineProfiler 实例，用于累积所有调用的分析数据
    profiler.add_function(func)

    @wraps(func)  # 保持原函数的签名
    def wrapper(*args, **kwargs):
        profiler.enable_by_count()  # 启用分析
        result = func(*args, **kwargs)  # 执行原函数
        profiler.disable_by_count()  # 禁用分析
        return result

    atexit.register(profiler.print_stats)

    return wrapper


def time_ms():
    return perf_counter()*1000

