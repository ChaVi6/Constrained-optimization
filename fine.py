import numpy as np
from scipy.optimize import minimize
from prettytable import PrettyTable
from matplotlib import pyplot as plt


def func(x, index):
    return -(index[0] * x[0] ** 2 + index[1] * x[1] ** 2 + index[2] * x[0] * x[1] + index[3] * x[0] + index[4] * x[1])


def constraint(x):
    return 4 * x[0] ** 2 + 9 * x[1] ** 2 - 144


def penalty_function(x, penalty, index):
    return func(x, index) + penalty * max(0, constraint(x)) ** 2


def print_table(x1, x2, f, penalty):
    table = PrettyTable()
    table.add_column("m", penalty)
    table.add_column("x1", x1)
    table.add_column("x2", x2)
    table.add_column("f", f)
    print(table)


def plot_graph(x_1, x_2, w, x, y, constr1, text):
    plt.title(text)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.contour(x_1, x_2, w, 50)
    plt.contour(x_1, x_2, constr1, 1, label='Ограничение')
    plt.plot(x, y, 'o-k', linewidth=2, label='Ход решения')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    index = np.array([-18, -18, 8, 132, 176])  # Значения всех аргументов
    x1scale = np.arange(-1, 16, 0.1)
    x2scale = np.arange(-1, 16, 0.1)
    x1scale, x2scale = np.meshgrid(x1scale, x2scale)
    w = func([x1scale, x2scale], index)
    constr = constraint([x1scale, x2scale])

    x0 = np.array([0, 0])
    x1s = [0]
    x2s = [0]
    fs = [func(x0, index)]
    iterations = [1]
    penalty = .00001
    penalties = [penalty]
    for i in range(9):
        res = minimize(lambda x: penalty_function(x, penalty, index), x0)
        penalty *= 10
        iterations.append(i)
        penalties.append(penalty)
        x1s.append(res.x[0])
        x2s.append(res.x[1])
        fs.append(-res.fun)

    print_table(x1s, x2s, fs, penalties)
    plot_graph(x1scale, x2scale, w, x1s, x2s, constr, "Метод штрафных функций")
