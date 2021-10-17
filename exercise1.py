import itertools
import math
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utilities.Node import Node
from utilities.Perceptron import Perceptron


def generate_lineal_collection(class_qty):
    data = {"x": [], "y": [], "class_type": []}

    for i in range(class_qty):
        x_val = random.uniform(0.0, 1.0)
        y_val = random.uniform(0.0, 1.0)
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
            class_one_list.append(Node(elem["x"], elem["y"], elem["class_type"], distance))
        else:
            class_two_list.append(Node(elem["x"], elem["y"], elem["class_type"], distance))

    class_one_list.sort(key=lambda n: n.distance)
    class_two_list.sort(key=lambda n: n.distance)
    return class_one_list, class_two_list


def plot_separation_function_and_data(a, b, test_df):
    # Graficamos el hiperplano de separacion
    x = np.linspace(0, 1, 10)
    y = a * x + b
    norma = math.sqrt(math.pow(-p.weight[0] / p.weight[1], 2) + math.pow(-p.weight[2] / p.weight[1], 2))

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
    colors = itertools.cycle(["r", "b", "g"])
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


# Hacemos el conjunto de entrenamiento, creamos y entrenamos el perceptron
train_data = generate_lineal_collection(100)
p = Perceptron()
p.train(train_data, 0.1, 100)

# Creamos el conjunto de testeo y hacemos las predicciones
test_data = generate_lineal_collection(25)
p.predict(test_data)

# Ploteamos el resultado inicial de entrenar el perceptron
# plot_separation_function_and_data(-p.weight[0] / p.weight[1], -p.weight[2] / p.weight[1], test_data)

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