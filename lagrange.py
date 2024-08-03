import numpy as np
from matplotlib import pyplot as plt
from sympy import symbols, Eq, solve


def func(x, index):
    return index[0] * x[0] ** 2 + index[1] * x[1] ** 2 + index[2] * x[0] * x[1] + index[3] * x[0] + index[4] * x[1]


def derivative(index):
    dfX = np.array([2 * index[0], index[2], index[3], 2 * index[1], index[2], index[4]])
    return dfX


def hesse(index):
    return np.array([[index[0] * 2, index[2]], [index[2], index[1] * 2]])


def plot_graph(x_1, x_2, w, x, y, text):
    plt.title(text)

    plt.plot(x, y, 'o-k', linewidth=2)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.contour(x_1, x_2, w, 50)
    plt.plot(x, y, color='black')
    plt.plot([x_2[0], x_2[len(x_2) - 1]], [y, y], ':')
    plt.plot([x, x], [x_2[0], x_2[len(x_2) - 1]], ':')
    plt.legend(['Точка максимума'])

    plt.show()


def solve_eq(index, constr):
    x1, x2, V = symbols('x1 x2 V')
    der = derivative(index)
    eq1 = Eq(der[0] * x1 + der[1] * x2 + der[2], 0)
    eq2 = Eq(der[3] * x2 + der[1] * x1 + der[4] + V, 0)
    eq3 = Eq(x2 - constr, 0)

    solution = solve((eq1, eq2, eq3), (x1, x2, V))
    print(solution)
    print(float(solution[V]))
    return float(solution[x1]), float(solution[x2]), func([float(solution[x1]), float(solution[x2])], index)


if __name__ == "__main__":
    index = np.array([-18, -18, 8, 132, 176])  # Значения всех аргументов
    x2 = 5

    x1scale = np.arange(0, 10, 0.1)
    x2scale = np.arange(0, 10, 0.1)
    x1scale, x2scale = np.meshgrid(x1scale, x2scale)
    w = func([x1scale, x2scale], index)
    x1_sol, x2_sol, f_sol = solve_eq(index, x2)
    f = func([x1_sol, x2_sol], index)
    print("Значение функции: f(x) = ", f)

    plot_graph(x1scale, x2scale, w, x1_sol, x2_sol, "Метод Лагранжа")
