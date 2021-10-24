import itertools
import math
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm

from utilities.Node import Node
from utilities.Perceptron import Perceptron


def generate_lineal_collection(class_qty):
    data = {"x": [], "y": [], "class_type": []}

    for i in range(class_qty):
        x_val = random.uniform(0.0, 0.8)
        y_val = random.uniform(0.0, 0.8)
        data["x"].append(x_val)
        data["y"].append(y_val)
        class_type = -1 if y_val >= x_val else 1
        data["class_type"].append(class_type)

    return pd.DataFrame(data)


def get_support_points(A, B, data):
    class_one_list = []
    class_two_list = []

    for i in range(len(data)):
        elem = data.iloc[i]
        distance = abs(A * elem["x"] + B - elem["y"])
        if elem["class_type"] == 1:
            class_one_list.append(Node(elem["x"], elem["y"], elem["class_type"], distance, i))
        else:
            class_two_list.append(Node(elem["x"], elem["y"], elem["class_type"], distance, i))

    class_one_list.sort(key=lambda n: n.distance)
    class_two_list.sort(key=lambda n: n.distance)
    return class_one_list, class_two_list


def plot_separation_function_and_data(a, b, test_df):
    # Graficamos el hiperplano de separacion
    x = np.linspace(0, 1, 10)
    y = a * x + b

    # Ordenamos los puntos y los graficamos en la recta
    x.sort()
    y.sort()
    fig, ax = plt.subplots()
    colors = itertools.cycle(["r", "b", "g"])
    class_one_df = test_df[test_df["class_type"] == 1]
    class_two_df = test_df[test_df["class_type"] == -1]
    ax.scatter(class_one_df["x"], class_one_df["y"], zorder=10, color=next(colors))
    ax.scatter(class_two_df["x"], class_two_df["y"], zorder=10, color=next(colors))
    ax.plot(x, y, 'k-', alpha=0.75, zorder=0)
    plt.show()


def plot_separation_and_compare(a, b, a1, b1, test_df):
    # Graficamos el hiperplano de separacion (ambos)
    x = np.linspace(0, 1, 10)
    y = a * x + b
    y1 = a1 * x + b1

    # Ordenamos los puntos y los graficamos en la recta
    x.sort()
    y.sort()
    colors = itertools.cycle(["r", "b", "y", "g", "m"])
    class_one_df = test_df[test_df["class_type"] == 1]
    class_two_df = test_df[test_df["class_type"] == -1]
    plt.scatter(class_one_df["x"], class_one_df["y"], zorder=10, color=next(colors))
    plt.scatter(class_two_df["x"], class_two_df["y"], zorder=10, color=next(colors))
    if current_type == 1:
        plt.scatter([class_one[0].x, class_one[1].x, class_two[0].x], [class_one[0].y, class_one[1].y, class_two[0].y],
                    zorder=10, color=next(colors))
    else:
        plt.scatter([class_two[0].x, class_two[1].x, class_one[0].x], [class_two[0].y, class_two[1].y, class_one[0].y],
                    zorder=10, color=next(colors))

    plt.plot(x, y, 'k-', alpha=0.75, zorder=0)
    plt.plot(x, y1, '--', alpha=0.75, zorder=0, color=next(colors))
    plt.legend(['Original', 'Optimal'])
    plt.show()


def calculate_best_separation(s1, s2, s3):
    # Trazo la recta entre los dos puntos de la misma clase
    line = [s1.x - s2.x, s1.y - s2.y]
    line = line / np.linalg.norm(line)

    # Calculo la pendiente
    m = line[1] / line[0]

    # Veo la distancia de los dos puntos
    d1 = np.linalg.norm([s3.x - s1.x, s3.y - s1.y])
    d2 = np.linalg.norm([s3.x - s2.x, s3.y - s2.y])

    # Me quedo con el de menor distancia
    if d2 < d1:
        selected = s2
    else:
        selected = s1

    # Calculo el punto medio entre el punto de la otra clase y el seleccionado
    middle_point = [(selected.x + s3.x) / 2, (selected.y + s3.y) / 2]

    # Retorno la pendiente y el nuevo b
    return m, middle_point[1] - m * middle_point[0]


def evaluate_optimal_line(a, b, data):
    total_distance = 0

    for i in range(len(data)):
        elem = data.iloc[i]
        total_distance += abs(a * elem["x"] + b - elem["y"])
        value = elem["class_type"] * (a * elem["x"] + b - elem["y"])
        if value < 0:
            return 9999

    return total_distance


def get_df_from_points(s1, s2, s3):
    x_list = [s1.x, s2.x, s3.x]
    y_list = [s1.y, s2.y, s3.y]
    class_list = [s1.class_t, s2.class_t, s3.class_t]

    data = {"x": x_list, "y": y_list, "class_type": class_list}
    return pd.DataFrame(data)


def generate_bad_collection(size):
    data = generate_lineal_collection(size)
    one_list, two_list = get_support_points(1, 0, data)

    for i in [one_list[0].idx, one_list[1].idx, two_list[0].idx, two_list[1].idx]:
        elem = data.iloc[i]
        data.loc[i, "class_type"] = -1 if elem["class_type"] == 1 else 1

    return data

def cortar():
    return False


def SVM(cota, df, b, w):
    i = 1
    kw = 0.01
    kb = 0.01
    # w = 0
    # b = 0
    c = 1
    wf = w
    bf = b
    while i < cota and not cortar():
        kw = 0.99*kw
        kb = 0.99*kb
        row = df.iloc[i % len(df["x"])]
        t = row["y"]*(w * row["x"] + b)
        if t < 1:
            sum1, sum2 = 0, 0
            for index, rowTemp in df.iterrows():
                sum1 += rowTemp["x"] * rowTemp["y"] * -1
                sum2 += rowTemp["y"] * -1
            wf = wf - kw*(w + c * sum1)
            bf = bf - kb*(c * sum2)
        else:
            wf = wf - w*kw
        i += 1
    return wf, bf


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

#   COMIENZO EJERCICIO 1

# Hacemos el conjunto de entrenamiento, creamos y entrenamos el perceptron
train_data = generate_lineal_collection(150)
p = Perceptron()
p.train(train_data, 0.1, 500)

# Creamos el conjunto de testeo y hacemos las predicciones
test_data = generate_lineal_collection(25)
p.predict(test_data)

# Ploteamos el resultado inicial de entrenar el perceptron
plot_separation_function_and_data(-p.weight[0] / p.weight[1], -p.weight[2] / p.weight[1], test_data)

# Obtenemos los puntos de soporte
norma = math.sqrt(math.pow(-p.weight[0] / p.weight[1], 2) + math.pow(-p.weight[2] / p.weight[1], 2))
class_one, class_two = get_support_points(-p.weight[0]/p.weight[1]/norma, -p.weight[2]/p.weight[1]/norma, test_data)

# Buscamos el optimo usando 3 puntos de soporte
current_a = 0
current_b = 0
current_distance = 99999999
current_type = 0

# Probamos con combinaciones de puntos distintas y nos quedamos con la mejor (parte 1 de 2)
new_a, new_b = calculate_best_separation(class_one[0], class_one[1], class_two[0])
distance_val = evaluate_optimal_line(new_a, new_b, get_df_from_points(class_one[0], class_one[1], class_two[0]))
if distance_val < current_distance:
    current_a = new_a
    current_b = new_b
    current_distance = distance_val
    current_type = 1

# Probamos con combinaciones de puntos distintas y nos quedamos con la mejor (parte 2 de 2)
new_a, new_b = calculate_best_separation(class_two[0], class_two[1], class_one[0])
distance_val = evaluate_optimal_line(new_a, new_b, get_df_from_points(class_one[0], class_one[1], class_two[0]))
if distance_val < current_distance:
    current_a = new_a
    current_b = new_b
    current_distance = distance_val
    current_type = 2

# Vemos la distancias promedio entre las dos rectas
d1 = evaluate_optimal_line(-p.weight[0] / p.weight[1], -p.weight[2] / p.weight[1], test_data)
d2 = evaluate_optimal_line(current_a, current_b, test_data)
print(d1, d2)

# Graficamos la del perceptron y la optima
plot_separation_and_compare(-p.weight[0] / p.weight[1], -p.weight[2] / p.weight[1], current_a, current_b, test_data)

#   COMIENZO EJERCICIO 2

train_data = generate_bad_collection(150)
p = Perceptron()
p.train(train_data, 0.1, 500)

test_data = generate_bad_collection(25)
plot_separation_function_and_data(-p.weight[0] / p.weight[1], -p.weight[2] / p.weight[1], test_data)

wf, bf = SVM(1000, test_data, current_b, current_a)
print("wf, bf = ", wf, bf)

#ejecricio B

train_data = generate_lineal_collection(50)
train_data = generate_bad_collection(50)
# classifier = svm.SVC(C=1, kernel='linear')
# train_data_x = train_data[['x', 'y']]
# train_data_y = train_data[['class_type']]
# clf = classifier.fit(train_data_x, train_data_y)
# pred = classifier.predict(train_data_x[['x', 'y']])
# ax = plt.gca()
#
# colors = itertools.cycle(["r", "b", "g", "y"])
# class_one_df = train_data[train_data["class_type"] == 1]
# class_two_df = train_data[train_data["class_type"] == -1]
# ax.scatter(class_one_df["x"], class_one_df["y"], zorder=10, color=next(colors))
# ax.scatter(class_two_df["x"], class_two_df["y"], zorder=10, color=next(colors))
#
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# xx = np.linspace(xlim[0], xlim[1], 30)
# yy = np.linspace(ylim[0], ylim[1], 30)
# YY, XX = np.meshgrid(yy, xx)
# xy = np.vstack([XX.ravel(), YY.ravel()]).T
# Z = classifier.decision_function(xy).reshape(XX.shape)
#
# # plot decision boundary and margins
# ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
#         linestyles=['--', '-', '--'])
# # plot support vectors
# ax.scatter(classifier.support_vectors_[:, 0], classifier.support_vectors_[:, 1], s=100,
#         linewidth=1, facecolors='none', edgecolors='k')
# plt.show()

svc = svm.SVC(kernel='linear', C=1E10)
train_data_x = train_data[['x', 'y']]
train_data_y = train_data[['class_type']]
svc.fit(train_data_x, train_data_y.values.ravel())
# x = np.linspace(0, 1, 10)
# w = svc.coef_[0]
# y_svm = x * (-w[0]/w[1]) - (svc.intercept_[0]/w[1])
# margin = 1 / np.sqrt(np.sum(svc.coef_ ** 2))
# yy_down = y_svm - np.sqrt(1 + x ** 2) * margin
# yy_up = y_svm + np.sqrt(1 + x ** 2) * margin
# plt.plot(x, y_svm, 'k-')
# # plt.plot(x, yy_down, 'k--')
# # plt.plot(x, yy_up, 'k--')
ax = plt.gca()
# ax.scatter(svc.support_vectors_[:, 0],
#                    svc.support_vectors_[:, 1],
#                    s=300, linewidth=1, facecolors='none');
colors = itertools.cycle(["r", "b", "g", "y"])
class_one_df = train_data[train_data["class_type"] == 1]
class_two_df = train_data[train_data["class_type"] == -1]
ax.scatter(class_one_df["x"], class_one_df["y"], zorder=10, color=next(colors))
ax.scatter(class_two_df["x"], class_two_df["y"], zorder=10, color=next(colors))
plot_svc_decision_function(svc)

# plt.plot(x, y_svm)

# plt.plot(x, x*best_line['m'] + best_line['b'], '--', color='orange')
# plt.plot(x, -x * p.weights[0]/p.weights[1] - p.bias/p.weights[1], '--', color='green')
plt.legend(['SVM'])

# best_margin = [min(distances_to_line(class1[:, :2],-w[0]/w[1], -svc.intercept_[0]/w[1])),
#               min(distances_to_line(class2[:, :2],-w[0]/w[1], -svc.intercept_[0]/w[1]))]
# print(best_margin)
plt.show()