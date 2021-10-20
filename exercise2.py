import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

def get_data(classes):
    dic = {"r": [], "g": [], "b": [], "class_type": []}
    arrayX = []
    arrayY = []
    for file, clase in classes:
        var = np.asarray(Image.open("resources/" + file))
        for i in range(len(var)):
            for j in range(len(var[i])):
                dic["r"].append(var[i][j][0] + 0)
                dic["g"].append(var[i][j][1] + 0)
                dic["b"].append(var[i][j][2] + 0)
                dic["class_type"].append(clase)
                arrayX.append(var[i][j])
                arrayY.append(clase)
    data = pd.DataFrame(dic)
    sample = data.sample(frac=0.10)
    return data, sample, arrayX, arrayY


def confusion_matrix(predicted, original):
    matrix = np.zeros(shape=(3, 3))
    for pred, truth in zip(predicted, original):
        matrix[truth][pred] += 1
    return matrix


def plot_matrix(matrix, c_val, kernel, data_set):
    fig, ax = plt.subplots()
    ax.matshow(matrix, cmap=plt.cm.Blues)

    for i, j in np.ndindex(matrix.shape):
        c = matrix[i][j]
        ax.text(i, j, str(c), va='center', ha='center')

    plt.xlabel("predicted")
    plt.ylabel("real value")
    # plt.savefig(f'{filename}', bbox_inches='tight')
    plt.title("confusion matrix c="+str(c_val)+" kernel "+kernel+" data set: " + data_set)
    plt.show()


def calculate_precision(confusion_matrix):
    sum = np.sum(confusion_matrix, axis=0)
    diagonal = np.diagonal(confusion_matrix)

    return np.mean(diagonal/sum)


def test_classifier(c=1.0, kernel='linear'):
    test_precision, train_precision = 0, 0
    classifier = svm.SVC(C=c, kernel=kernel)
    classifier.fit(X_train, Y_train)
    test_predicted = classifier.predict(X_test)
    train_predicted = classifier.predict(X_train)

    test_confusion = confusion_matrix(test_predicted, Y_test)
    train_confusion = confusion_matrix(train_predicted, Y_train)
    plot_matrix(test_confusion, c, kernel, "test")
    plot_matrix(train_confusion, c, kernel, "train")

    test_precision = calculate_precision(test_confusion)
    train_precision = calculate_precision(train_confusion)


    return test_precision, train_precision


def test_kernels():
    kernels = ['linear', 'poly', 'rbf']

    filename = 'kernel_results.csv'
    test_dic = {}
    train_dic = {}
    for kernel in kernels:
        test_precision, train_precision = test_classifier(c=1.0, kernel=kernel)
        print("kernel: ", kernel, "precision test:", test_precision, "precision train:", train_precision)
        test_dic[kernel].append(test_precision)
        train_dic[kernel].append(train_precision)

    return test_dic, train_dic


def test_c():

    for c in range(20):
        test_precision, train_precision = test_classifier(c=c*0.1+0.1)
        print("C: ", c*0.1+0.1, "precision test:", test_precision, "precision train:", train_precision)


def test_full_image(c_value=1.0, kernel='linear'):
    # multiclass support is handled by one-vs-one scheme
    classifier = svm.SVC(C=c_value, kernel=kernel)
    classifier.fit(X_train, Y_train)

    _, _, cowX, cowY = get_data([["cow.jpg", 4]])
    original = np.asarray(Image.open("resources/cow.jpg"))

    predicted = classifier.predict(cowX)
    result_array = []
    for value in predicted:
        if value == 0:
            result_array.append([0, 255, 0])
        if value == 1:
            result_array.append([255, 0, 0])
        if value == 2:
            result_array.append([0, 0, 255])

    result_image = np.array(result_array).reshape(original.shape)

    output = np.hstack([original, result_image])
    image = Image.fromarray(output.astype(np.uint8))
    image.save('side_by_side.png')
    image2 = Image.fromarray(result_image.astype(np.uint8))
    image2.save("predicted.png")


test_data, train_data, arrayX, arrayY = get_data([["cielo.jpg", 0], ["pasto.jpg", 1], ["vaca.jpg", 2]])

X_train, X_test, Y_train, Y_test = train_test_split(arrayX, arrayY, test_size=0.2, random_state=0)

test_c()
test_kernels()

test_full_image()

