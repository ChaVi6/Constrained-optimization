import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from sympy import symbols, Eq, solve


def hesse(index):
    return np.array([[index[0] * 2, index[2]], [index[2], index[1] * 2]])


def objective(index, x):
    return index[0] * x[0] ** 2 + index[1] * x[1] ** 2 + index[2] * x[0] * x[1] + index[3] * x[0] + index[4] * x[1]


def derivative(index, x):
    dfX = np.array([2 * index[0], index[2], index[3], 2 * index[1], index[2], index[4]])
    return [dfX[0] * x[0] + dfX[1] * x[1] + dfX[2], dfX[3] * x[1] + dfX[4] * x[0] + dfX[5]]


def derivative_g(g, x):
    dg = np.array([2 * g[0], 2 * g[1]])
    return [dg[0] * x[0], dg[1] * x[1]]


def objective_g(g, x):
    return g[0] * x[0] ** 2 + g[1] * x[1] ** 2 + g[2]


def print_table(iterations, x1, x2, f):
    table = PrettyTable()
    table.add_column("step", iterations)
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


def zoit(index, g, x0):
    H = hesse(index)
    x = x0
    x1s = [x[0]]
    x2s = [x[1]]
    fs = [objective(index, x)]
    i = 0
    iterations = [i]
    x_pr = [2.0, 2.0]
    while np.all(abs(x_pr - x) > 0.000001):
        if objective_g(g, x) > 10 ** -6:
            K = -np.linalg.inv(H) @ derivative(index, x)
        else:
            df = derivative(index, x)
            dg = derivative_g(g, x)
            if (dg[1] == df[1]):
                k1 = 0
                if (dg[1] >= 0):
                    u = dg[1]
                    k2 = 1
                else:
                    u = -dg[1]
                    k2 = -1
            else:
                d = (df[0] * dg[1] - dg[0] * df[1]) / (dg[1] - df[1])
                if d >= 0:
                    u = d
                    k1 = 1
                else:
                    u = -d
                    k1 = -1
                k2 = (u - df[0] * k1) / df[1]
            K = np.array([k1, k2])
        t_ = -(derivative(index, x) @ K) / (K @ H @ K)
        t_pr_ = symbols("t_pr_")
        eq = Eq(objective_g(g, x + t_pr_ * K), 0)
        t_pr = max(solve(eq, t_pr_))
        t = min(t_, t_pr)
        x_pr = x
        x = x + t * K
        i += 1
        iterations.append(i)
        x1s.append(x[0])
        x2s.append(x[1])
        fs.append(objective(index, x))
    return iterations, x1s, x2s, fs


if __name__ == "__main__":
    index = np.array([-18, -18, 8, 132, 176])  # Значения всех аргументов
    g = np.array([-4, -9, 144])
    x1scale = np.arange(-1, 16, 0.1)
    x2scale = np.arange(-1, 16, 0.1)
    x1scale, x2scale = np.meshgrid(x1scale, x2scale)
    w = objective(index, [x1scale, x2scale])
    constr = objective_g(g, [x1scale, x2scale])

    x0 = np.array([1.0, 1.0])
    iterations, x1, x2, f = zoit(index, g, x0)
    print_table(iterations, x1, x2, f)
    plot_graph(x1scale, x2scale, w, x1, x2, constr, "Метод возможных направлений")
