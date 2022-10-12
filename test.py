from casadi import *

x = SX.sym("x", 1)
a = SX.sym("x", 2)
y = a**2
print(jacobian(y, a))