import math


class Perceptron:

    def __init__(self):
        self.weight = [0, 0, 0]
        self.min_error = 99999
        self.min_weight = []

    # todo error calculation
    def train(self, collection, learning_rate, epochs):
        error = 0

        for j in range(epochs):
            for i in range(len(collection)):
                excitation = self.weight[0] * collection.iloc[i].x + self.weight[1] * collection.iloc[i].y + self.weight[2]
                activation = 1 if excitation > 0 else -1
                self.weight[0] += learning_rate * (collection.iloc[i].class_type - activation) * collection.iloc[i].x
                self.weight[1] += learning_rate * (collection.iloc[i].class_type - activation) * collection.iloc[i].y
                self.weight[2] += learning_rate * (collection.iloc[i].class_type - activation)
                error += math.pow(collection.iloc[i].class_type - excitation, 2)

            error /= 2

            if error < self.min_error:
                self.min_error = error
                self.min_weight = self.weight

    def print_perceptron(self):
        print("W[0] = ", self.min_weight[0], ", W[1] = ", self.min_weight[1], ", W[3] = ", self.weight[2], ", Error: ", self.min_error)

    def predict(self, collection):
        for i in range(len(collection)):
            excitation = self.min_weight[0] * collection.iloc[i].x + self.min_weight[1] * collection.iloc[i].y
            activation = 1 if excitation > 0 else -1
            print("Expected = ", collection.iloc[i].class_type, ", Predicted = ", activation)
