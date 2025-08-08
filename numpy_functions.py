#======================++++++++++++++++++++++++++++++========================
#
#                 all numpy functions i used rewritten for uC
#                            compatibility. less performant
#
#======================++++++++++++++++++++++++++++++========================
def ones(shape):
    return [[1.0 for _ in range(shape[1])] for _ in range(shape[0])]

def zeros(length):
    return [0.0 for _ in range(length)]

def zeros_like(array):
    return [0.0 for _ in array]

def sum_array(array):
    return sum(array)

def mean_array(array_2d):
    if not array_2d or not array_2d[0]:
        return []
    cols = len(array_2d[0])
    rows = len(array_2d)
    return [sum(row[i] for row in array_2d) / rows for i in range(cols)]

def clip_array(array, min_val, max_val):
    return [max(min(x, max_val), min_val) for x in array]

import math
def exp_array(array):
    return [math.exp(x) for x in array]

def any_negative(array):
    return any(x < 0 for x in array)
