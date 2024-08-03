import numpy as np
from matplotlib import pyplot as plt


def func(x, index):
    return index[0] * x[0] ** 2 + index[1] * x[1] ** 2 + index[2] * x[0] * x[1] + index[3] * x[0] + index[4] * x[1]


def plot_graph(x_1, x_2, w, x, y, constr1, constr2, text):
    plt.title(text)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.contour(x_1, x_2, w, 50)
    plt.plot([-1, 14], constr1, label='Ограничение 1')
    plt.plot([-1, 14], constr2, label='Ограничение 2')
    plt.plot(x, y, 'o-k', linewidth=2, label='Ход решения')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    index = np.array([-18, -18, 8, 132, 176])  # Значения всех аргументов
    x1scale = np.arange(-1, 16, 0.1)
    x2scale = np.arange(-1, 16, 0.1)
    x1scale, x2scale = np.meshgrid(x1scale, x2scale)
    w = func([x1scale, x2scale], index)

    x1 = [0, 3.65244, 4.035]
    x2 = [0, 4.86992, 4.647]

    y_constr1 = []
    y_constr2 = []
    y_constr1.append((84 - 7 * (-1)) / 12)
    y_constr1.append((84 - 7 * 14) / 12)
    y_constr2.append((80 - 10 * (-1)) / 8)
    y_constr2.append((80 - 10 * 14) / 8)

    plot_graph(x1scale, x2scale, w, x1, x2, y_constr1, y_constr2, "Метод проекции градиента")