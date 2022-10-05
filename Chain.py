import numpy as np
import json
#from math import *
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
        tetha = SX.sym('tetha', self.N)
        sintetha = SX.sym("s", self.N)
        costetha = SX.sym("c", self.N)
        for i in range(0, self.N):
            sintetha[i] = sin(tetha[i])
            costetha[i] = cos(tetha[i])
        x[0] = self.config["l"]/2 * sin(tetha[0])
        y[0] = self.config["l"]/2 * cos(tetha[0])
        for i in range(1, self.N):
            x[i] = (sintetha[i]/2 + sum1(sintetha[0:i])) * self.config["l"]
            y[i] = (cos(tetha[i])/2 + sum1(costetha[0:i])) * self.config["l"]
        


test = Chain(3)
test.config_input("config.json")
test.L()