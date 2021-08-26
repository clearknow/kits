import json
import numpy as np


class MyEncoder(json.JSONEncoder):
    """
    重写json模块JSONEncoder类中的default方法
    """
    def default(self, obj):
        # np整数转为内置int
        if isinstance(obj, np.integer):
            return int(obj)
        else:
            return super(json.JetEncoder, self).default(obj)
