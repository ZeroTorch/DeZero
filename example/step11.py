import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

import numpy as np
from tools.baseline import Variable
from tools.functions import Add, add

# xs = [Variable(np.array(2)), Variable(np.array(3))] # 리스트로 준비

x0 = Variable(np.array(2))
x1 = Variable(np.array(3))

# f = Add()
# ys = f(xs) 
# y = ys[0]
# print(y.data)

y = add(x0, x1)
print(y.data)