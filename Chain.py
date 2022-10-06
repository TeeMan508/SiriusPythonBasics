import numpy as np
import json
# from math import *
from casadi import *
import matplotlib.pyplot as plt


# Initial_Conditions = {"g": 10, "l": 1, "m": 1}
#
# with open("config.json", "w") as config_file:
#     json.dump(Initial_Conditions, config_file)


class Chain():
    def __init__(self, N):
        self.config = {}
        self.N = N
        self.I = 0

    def config_input(self, file_path):
        with open(file_path, "r") as config_file:
            self.config = json.load(config_file)
            print("Imported Config: " + str(self.config))
        self.I = self.config["m"] * self.config["l"] ** 2 / 12

    def L(self):
        x = SX.sym('x', self.N)
        y = SX.sym('y', self.N)
        t = SX.sym('t')
        tetha = SX.sym('tetha', self.N)
        sintetha = SX.sym("s", self.N)
        costetha = SX.sym("c", self.N)
        for i in range(0, self.N):
            sintetha[i] = sin(tetha[i])
            costetha[i] = cos(tetha[i])
        x[0] = self.config["l"] / 2 * sin(tetha[0])
        y[0] = self.config["l"] / 2 * cos(tetha[0])
        for i in range(1, self.N):
            x[i] = (sintetha[i] / 2 + sum1(sintetha[0:i])) * self.config["l"]
            y[i] = (cos(tetha[i]) / 2 + sum1(costetha[0:i])) * self.config["l"]

        J1tetha = jacobian(x, tetha)
        J2tetha = jacobian(y, tetha)
        tetha_d = SX.sym('d_tetha', self.N)
        tetha_dd = SX.sym('dd_tetha', self.N)

        x_d = J1tetha @ tetha_d
        y_d = J2tetha @ tetha_d


        L = (1 / 2) * self.config["m"] * (sumsqr(x_d) + sumsqr(y_d)) + (1 / 2) * self.I * sumsqr(tetha_d) - self.config[
            "m"] * self.config["g"] * sum1(y)


        left = jacobian(L, tetha).T
        right = jacobian(L, tetha_d).T
        right1 = jacobian(right, tetha)
        right2 = jacobian(right, tetha_d)
        equation_right = left - right1@tetha_d
        equation_left = right2
        print(solve(equation_left, equation_right).size())

        #print(left.size())#разобраться как сделать tetha функией от t



test = Chain(2)
test.config_input("config.json")
test.L()
