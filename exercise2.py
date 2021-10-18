import pandas as pd
from PIL import Image
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

def get_data():
    dic = {"r": [], "g": [], "b": [], "class_type": []}
    for file, clase in [["cielo.jpg", 0], ["pasto.jpg", 1], ["vaca.jpg", 2]]:
        var = np.asarray(Image.open("resources/" + file))
        print(var[0][0])
        for i in range(len(var)):
            for j in range(len(var[i])):
                dic["r"].append(var[i][j][0] + 0)
                dic["g"].append(var[i][j][1] + 0)
                dic["b"].append(var[i][j][2] + 0)
                dic["class_type"].append(clase)
    data = pd.DataFrame(dic)
    sample = data.sample(frac=0.10)
    return data, sample


test_data, train_data = get_data()
