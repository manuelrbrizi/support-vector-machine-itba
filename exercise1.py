import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
import seaborn as sns
from matplotlib import pyplot as plt

from utilities.Perceptron import Perceptron


def generate_lineal_collection(class_qty):
    data = {"x": [], "y": [], "class_type": []}

    for i in range(class_qty):
        x = random.uniform(0.0, 0.5)
        y = random.uniform(0.5, 1.0)
        data["x"].append(x)
        data["y"].append(y)
        data["class_type"].append(1)

    for i in range(class_qty):
        x = random.uniform(0.5, 1.0)
        y = random.uniform(0.0, 0.5)
        data["x"].append(x)
        data["y"].append(y)
        data["class_type"].append(-1)

    return pd.DataFrame(data)


def scatterplot_df(other_df):
    your_palette = sns.color_palette('bright', 2)
    ax = sns.scatterplot(x="x", y="y", hue="class_type", data=other_df, palette=your_palette)
    return ax


train_data = generate_lineal_collection(20)
p = Perceptron()
p.train(train_data, 0.1, 1)
p.print_perceptron()

test_data = generate_lineal_collection(10)
p.predict(test_data)

x = np.linspace(0, 1, 100)
y = -p.weight[0]/p.weight[1]*x - p.weight[2]/p.weight[1]

# now sort it just to make it look like it's related
x.sort()
y.sort()

fig, ax = plt.subplots()
ax.scatter(test_data["x"], test_data["y"], zorder=10)
ax.plot(x, y, 'k-', alpha=0.75, zorder=0)
plt.show()

# # 100 linearly spaced numbers
# x = np.linspace(0,1,100)
#
# # the function, which is y = x^2 here
# y = x*p.weight[0] + p.weight[1]
#
# # setting the axes at the centre
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.spines['left'].set_position('center')
# ax.spines['bottom'].set_position('zero')
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
#
# # plot the function
# plt.plot(x,y, 'r')
#
# # show the plot
# plt.show()