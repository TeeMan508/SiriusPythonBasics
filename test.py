from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots()
x = np.arange(-2 * np.pi, 2 * np.pi, 0.1)
y = np.cos(x)

line, = ax.plot(x, y)


def update_cos(frame, line, x):
    # frame - параметр, который меняется от кадра к кадру
    # в данном случае - это начальная фаза (угол)
    # line - ссылка на объект Line2D
    line.set_ydata( np.cos(x+frame) )
    return [line]

phasa = np.arange(0, 4*np.pi, 0.1)

animation = FuncAnimation(
    fig,                # фигура, где отображается анимация
    func=update_cos,    # функция обновления текущего кадра
    frames=phasa,       # параметр, меняющийся от кадра к кадру
    fargs=(line, x),    # дополнительные параметры для функции update_cos
    interval=30,       # задержка между кадрами в мс
    blit=True,          # использовать ли двойную буферизацию
    repeat=False)

plt.show()
