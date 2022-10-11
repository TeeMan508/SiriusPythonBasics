import numpy as np
import json
# from math import *
from casadi import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time


# Initial_Conditions = {"g": 10, "l": 1, "m": 1}
#
# with open("config.json", "w") as config_file:
#     json.dump(Initial_Conditions, config_file)


class Chain():
    def __init__(self, N):
        self.config = {}
        self.N = N
        self.I = 0
        self.count = 0

    def config_input(self, file_path):
        with open(file_path, "r") as config_file:
            self.config = json.load(config_file)
            print("Imported Config: " + str(self.config))
        self.I = self.config["m"] * self.config["l"] ** 2 / 12

    def calculate_symbol_equations(self):
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
        y[0] = -self.config["l"] / 2 * cos(tetha[0])
        for i in range(1, self.N):
            x[i] = (sintetha[i] / 2 + sum1(sintetha[0:i])) * self.config["l"]
            y[i] = -(cos(tetha[i]) / 2 + sum1(costetha[0:i])) * self.config["l"]

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
        equation_right = left - right1 @ tetha_d
        equation_left = right2
        dif_right = solve(equation_left, equation_right)
        dif_left = tetha_dd

        #return [dif_left, dif_right]
        f = Function('f', [tetha, tetha_d], [dif_right])
        return f


    def Runge_Kutta_4(self, f, t_min, t_max, dt):
        tetha0 = [pi/4 for i in range(0, self.N)]
        tetha_d0 = [0 for i in range(0, self.N)]
        tetha0 = np.array(tetha0)
        tetha_d0 = np.array(tetha_d0)
        res = [tetha0]
        self.count = int((t_max-t_min) // dt)

        def g(tetha_d):
            return tetha_d

        while (t_min < t_max):
            k1 = dt * f(tetha0, tetha_d0)
            q1 = dt * g(tetha_d0)

            k2 = dt * f(tetha0 + q1/2, tetha_d0+k1/2)

            q2 = dt * g(tetha_d0+k1/2)

            k3 = dt * f(tetha0 + q2 / 2, tetha_d0 + k2 / 2)
            q3 = dt * g(tetha_d0 + k2 / 2)

            k4 = dt * f(tetha0 + q3 / 2, tetha_d0 + k3 / 2)
            q4 = dt * g(tetha_d0 + k3 / 2)


            tetha1 = tetha0 + (k1+2*k2+2*k3+k4)/6
            tetha_d1 = tetha_d0 + (q1+2*q2+2*q3+q4)/6

            res.append(tetha1)

            tetha0 = tetha1
            tetha_d0 = tetha_d1
            t_min = t_min + dt

        return res


    def build_data(self, all_tetha):
        X = []
        Y = []

        for tetha in all_tetha:
            tetha = np.array(tetha)
            x = [sin(tetha[0])*self.config["l"]/2]
            y = [-cos(tetha[0])*self.config["l"]/2]
            for i in range(1, len(tetha)):
                x.append((sin(tetha[i])/2 + sum(np.sin(tetha[0:i])))*self.config["l"])
                y.append(-(cos(tetha[i])/2 + sum(np.cos(tetha[0:i]))) * self.config["l"])
            X.append(x)
            Y.append(y)

        return [X, Y]

    def build_animation(self, X, Y):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1,1,1)
        line, = ax.plot([], [])

        def init():
            line.set_data([], [])
            return line,

        def animate(i):
            line.set_data(X[i], Y[i])
            return line,

        anim = FuncAnimation(fig, animate, init_func=init, frames=self.count, interval=100)
        anim.save('test.gif', writer='imagemagick')
        plt.show()


    def buf(self, X, Y):
        for i in range(0, len(X)):
            plt.plot(X[i], Y[i])

        plt.show()







test = Chain(3)
test.config_input("config.json")
tetha = test.Runge_Kutta_4(test.calculate_symbol_equations(), 0, 3, 0.1)
data = test.build_data(tetha)
test.buf(data[0], data[1])
#test.build_animation(data[0], data[1])

