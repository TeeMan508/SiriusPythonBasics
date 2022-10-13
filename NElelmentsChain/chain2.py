import numpy as np
import json
from casadi import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time


class Initialization():
    def __init__(self, N):
        self.config = {}
        self.N = N
        self.I = 0

    def config_input(self, file_path):
        with open(file_path, "r") as config_file:
            self.config = json.load(config_file)
            print("Imported Config: " + str(self.config))
        self.I = self.config["m"] * self.config["l"] ** 2 / 12

    def calculate_symbol_equations(self):
        x = SX.sym('x', self.N)
        y = SX.sym('y', self.N)
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

        # return [dif_left, dif_right]
        f = Function('f', [tetha, tetha_d], [dif_right])
        return f


class Calculations():
    def __init__(self, t_min, t_max, tethastart, dtethastart, fps, main):
        self.fps = fps
        self.t_min = t_min
        self.t_max = t_max
        self.tethastart = tethastart
        self.dtethastart = dtethastart
        self.config = main.config
        self.N = main.N
        self.f = main.calculate_symbol_equations()

    def transform_data(self, all_tetha):
        X = []
        Y = []
        for tetha in all_tetha:
            tetha = np.array(tetha)
            x = [sin(tetha[0]) * self.config["l"] / 2]
            y = [-cos(tetha[0]) * self.config["l"] / 2]
            for i in range(1, len(tetha)):
                x.append((sin(tetha[i]) / 2 + sum(np.sin(tetha[0:i]))) * self.config["l"])
                y.append(-(cos(tetha[i]) / 2 + sum(np.cos(tetha[0:i]))) * self.config["l"])
            X.append(x)
            Y.append(y)
        resx = []
        resy = []
        for k in range(0, len(X)):
            bufx = [float(X[k][0] - sin(all_tetha[k][0]) * self.config["l"] / 2),
                    float(X[k][0] + sin(all_tetha[k][0]) * self.config["l"] / 2)]
            bufy = [float(Y[k][0] + cos(all_tetha[k][0]) * self.config["l"] / 2),
                    float(Y[k][0] - cos(all_tetha[k][0]) * self.config["l"] / 2)]
            for i in range(1, len(X[0])):
                bufx.append(float(X[k][i] + sin(all_tetha[k][i]) * self.config["l"] / 2))
                bufy.append(float(Y[k][i] - cos(all_tetha[k][i]) * self.config["l"] / 2))
            resx.append(bufx)
            resy.append(bufy)
        return [resx, resy]

    def Solve(self):
        tetha0 = [self.tethastart for i in range(0, self.N)]
        tetha_d0 = [self.dtethastart for i in range(0, self.N)]
        tetha0 = np.array(tetha0)
        tetha_d0 = np.array(tetha_d0)
        res = [tetha0]
        dt = 1 / self.fps

        def g(tetha_d):
            return tetha_d

        while (self.t_min <= self.t_max):
            k1 = dt * self.f(tetha0, tetha_d0)
            q1 = dt * g(tetha_d0)

            k2 = dt * self.f(tetha0 + q1 / 2, tetha_d0 + k1 / 2)
            q2 = dt * g(tetha_d0 + k1 / 2)

            k3 = dt * self.f(tetha0 + q2 / 2, tetha_d0 + k2 / 2)
            q3 = dt * g(tetha_d0 + k2 / 2)

            k4 = dt * self.f(tetha0 + q3 / 2, tetha_d0 + k3 / 2)
            q4 = dt * g(tetha_d0 + k3 / 2)

            tetha_d1 = tetha_d0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            tetha1 = tetha0 + (q1 + 2 * q2 + 2 * q3 + q4) / 6

            res.append(tetha1)

            tetha0 = tetha1
            tetha_d0 = tetha_d1
            self.t_min = self.t_min + dt

        return self.transform_data(res)


class Animation():
    def __init__(self, calc):
        self.data = calc.Solve()
        self.config = calc.config
        self.fps = calc.fps
        self.anim = None

    def build_animation(self):
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(1, 1, 1)
        plt.title("Number of elements: " + str(len(self.data[0][0]) - 1))
        ax.set_xlim([-self.config["l"] * (len(self.data[0][0]) + 1), self.config["l"] * (len(self.data[0][0]) + 1)])
        ax.set_ylim([-self.config["l"] * (len(self.data[0][0]) + 1), self.config["l"] * (len(self.data[0][0]) + 1)])
        ax.grid()
        line, = ax.plot(self.data[0][0], self.data[1][0], marker='o', markerfacecolor='r', linewidth=5, markersize=5)

        def animate(i):
            line.set_data(self.data[0][i], self.data[1][i])
            line.set_marker('o')
            line.set_markerfacecolor('r')
            line.set_linewidth(5)
            line.set_markersize(5)
            # line2.set_data(X[i], Y[i])
            # line2.set_marker('o')
            return line,

        cadr = [i for i in range(0, len(self.data[0]))]

        anim = FuncAnimation(fig, func=animate, frames=cadr, interval=1000 / self.fps, blit=True, repeat=False)
        plt.show()

# if __name__ == "__main__":
#     test_Init = Initialization(5)
#     test_Init.config_input("config.json")
#     test_Calc = Calculations(0, 3, pi / 2, 0, 60, test_Init)
#     test_Ani = Animation(test_Calc)
#     test_Ani.build_animation()
