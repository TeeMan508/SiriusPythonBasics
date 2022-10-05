from casadi import *

x = SX.sym('x', 3)
a = sum2(x)
print(a)