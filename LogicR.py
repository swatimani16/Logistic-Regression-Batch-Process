# -*- coding: utf-8 -*-
"""

@author: Swati Mani
"""

import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:

    def __init__(self, alpha=0.001, epochs=100000):
        self.alpha = alpha
        self.epochs = epochs
        self.loss_threshold = 0.001
        self.iter_count = 0
        self.fpr = []
        self.tpr = []
        self.total_cost = []
        self.tot_iter = []
        self.norm_current_loss = []
        self.norm_grad = []

    def activ_sig(self, z):
        return 1 / (1 + np.exp(-z))

    def cross_entropy(self, predicted, actual):
        self.cost = (-actual * np.log(predicted) - (1 - actual) * np.log(1 - predicted)).mean()
        self.total_cost.append(self.cost)
        return self.cost

    def calc_S(self, actual_y, predicted_y):
        predicted_y = np.round(predicted_y)
        predicted_y = predicted_y.tolist()
        actual_y = actual_y.tolist()

        fp = 0
        for i in range(500):
            if predicted_y[i] != actual_y[i]:
                fp += 1

        tp = 0
        for i in range(500):
            if predicted_y[i + 500] == actual_y[i + 500]:
                tp += 1

        tpr = tp / 500
        fpr = fp / 500

        self.fpr.append(fpr)
        self.tpr.append(tpr)

    def getS(self):
        return self.fpr, self.tpr

    def learn(self, X, target):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

        self.weights = np.ones(X.shape[1])

        prev_loss = float('inf')
        for self.iter_count in range(self.epochs):
            net_val = np.dot(X, self.weights)
            pred = self.activ_sig(net_val)
            gradient = np.dot(X.T, (pred - target)) / target.size
            self.weights -= self.alpha * gradient
            current_loss = self.cross_entropy(pred, target)
            self.calc_S(target, pred)
            self.norm_grad.append(abs(gradient[0]) + abs(gradient[1]) + abs(gradient[2]))

            if (gradient == np.zeros(X.shape[1])).all():
                print("Cross - Entropy loss at epoch %s: %s" % (
                    self.iter_count + 1, self.cross_entropy(pred, target)))
                print("Gradient is zero!")
                print("total no. of iterations run: ", self.iter_count + 1)
                break

            if prev_loss - current_loss < self.loss_threshold:
                print("Cross - Entropy loss at epoch %s: %s" % (
                    self.iter_count + 1, self.cross_entropy(pred, target)))
                print("Loss optimized is less than threshold!")
                print("total no. of iterations run: ", self.iter_count + 1)
                break

            if self.iter_count % 50 == 0:
                print("Cross - Entropy loss at epoch %s: %s" % (
                    self.iter_count + 1, self.cross_entropy(pred, target)))

            if self.iter_count == self.epochs - 1:
                print("Cross - Entropy loss at epoch %s: %s" % (
                    self.iter_count + 1, self.cross_entropy(pred, target)))
                print("total no. of iterations run: ", self.iter_count + 1)

            self.norm_current_loss.append(abs(current_loss - prev_loss))
            prev_loss = current_loss
            self.store_iter = self.iter_count + 1
            self.tot_iter.append(self.store_iter)
        # print ("grad loss",self.norm_current_loss)
        return self.norm_current_loss, self.tot_iter, self.norm_grad

    def predict(self, X):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        return np.round(self.activ_sig(np.dot(X, self.weights)))


def generate_data(mean, variance, count):
    return np.random.multivariate_normal(mean, variance, count)


def calcAcc(predicted_y, test_y):
    predicted_y = predicted_y.tolist()
    test_y = test_y.tolist()

    count = 0
    for i in range(len(predicted_y)):
        if predicted_y[i] == test_y[i]:
            count += 1

    return (count / len(predicted_y)) * 100


def plot(fpr, tpr, alpha):
    plt.title("ROC curve for alpha = " + str(alpha))
    plt.plot(fpr, tpr)
    plt.show()

    auc = np.trapz(tpr, fpr)
    print('AUC:', auc)


max_epochs = 100000
x1 = generate_data([1, 0], [[1, 0.75], [0.75, 1]], 500)
x2 = generate_data([0, 1.5], [[1, 0.75], [0.75, 1]], 500)
X = np.vstack((x1, x2)).astype(np.float32)
y = np.hstack((np.zeros(500), np.ones(500)))

test_x1 = generate_data([1, 0], [[1, 0.75], [0.75, 1]], 500)
test_x2 = generate_data([0, 1.5], [[1, 0.75], [0.75, 1]], 500)
test_X = np.vstack((test_x1, test_x2)).astype(np.float32)
test_y = np.hstack((np.zeros(500), np.ones(500)))

print("\n\nLearning rate (Alpha): 1\nTotal Epochs: 100000")
LR = LogisticRegression(alpha=1, epochs=max_epochs)
LR.learn(X, y)
print("Final Weights: ", LR.weights)
predicted_y = LR.predict(test_X)
accuracy = calcAcc(predicted_y, test_y)
print("Accuracy: ", accuracy)
f, ax = plt.subplots(1, figsize=(5, 5))
ax.set_title("Norm Gradient")
ax.plot(range(0, len(LR.norm_grad)), LR.norm_grad, 'r', label=r'$\alpha = 1$')
ax.set_xlabel('Iterations')
ax.set_ylabel('Gradient')
ax.legend();
f, ax = plt.subplots(1, figsize=(5, 5))
plt.figure(figsize=(8, 8))
ax.set_title("Validation loss")
ax.plot(range(0, len(LR.total_cost)), LR.total_cost, 'r', label=r'$\alpha = 1$')
ax.set_xlabel('Iterations')
ax.set_ylabel('Training Error')
ax.legend();
fpr, tpr = LR.getS()
plt.figure(figsize=(5, 5))
plot(sorted(fpr), sorted(tpr), 1)

print("\n\nLearning rate (Alpha): 0.1\nTotal Epochs: 100000")
LR = LogisticRegression(alpha=0.1, epochs=max_epochs)
LR.learn(X, y)
print("Final Weights: ", LR.weights)
predicted_y = LR.predict(test_X)
accuracy = calcAcc(predicted_y, test_y)
print("Accuracy: ", accuracy)
f, ax = plt.subplots(1, figsize=(5, 5))
ax.set_title("Norm Gradient")
ax.plot(range(0, len(LR.norm_grad)), LR.norm_grad, 'r', label=r'$\alpha = 0.1$')
ax.set_xlabel('Iterations')
ax.set_ylabel('Gradient')
ax.legend();
f, ax = plt.subplots(1, figsize=(5, 5))
indices = np.float32(range(100000))
plt.figure(figsize=(8, 8))
ax.set_title("Validation loss")
ax.plot(range(0, len(LR.total_cost)), LR.total_cost, 'r', label=r'$\alpha = 0.1$')
ax.set_xlabel('Iterations')
ax.set_ylabel('Training Error')
ax.legend();
fpr, tpr = LR.getS()
plt.figure(figsize=(5, 5))
plot(sorted(fpr), sorted(tpr), 0.1)

print("\n\nLearning rate (Alpha): 0.01\nTotal Epochs: 100000")
LR = LogisticRegression(alpha=0.01, epochs=max_epochs)
LR.learn(X, y)
print("Final Weights: ", LR.weights)
predicted_y = LR.predict(test_X)
accuracy = calcAcc(predicted_y, test_y)
print("Accuracy: ", accuracy)
f, ax = plt.subplots(1, figsize=(5, 5))
ax.set_title("Norm Gradient")
ax.plot(range(0, len(LR.norm_grad)), LR.norm_grad, 'r', label=r'$\alpha = 0.01$')
ax.set_xlabel('Iterations')
ax.set_ylabel('Gradient')
ax.legend();
f, ax = plt.subplots(1, figsize=(5, 5))
indices = np.float32(range(100000))
plt.figure(figsize=(8, 8))
ax.set_title("Validation loss")
ax.plot(range(0, len(LR.total_cost)), LR.total_cost, 'r', label=r'$\alpha = 0.01$')
ax.set_xlabel('Iterations')
ax.set_ylabel('Training Error')
ax.legend();
fpr, tpr = LR.getS()
plt.figure(figsize=(5, 5))
plot(sorted(fpr), sorted(tpr), 0.01)

print("\n\nLearning rate (Alpha): 0.001\nTotal Epochs: 100000")
LR = LogisticRegression(alpha=0.001, epochs=max_epochs)
LR.learn(X, y)
print("Final Weights: ", LR.weights)
predicted_y = LR.predict(test_X)
accuracy = calcAcc(predicted_y, test_y)
print("Accuracy: ", accuracy)
f, ax = plt.subplots(1, figsize=(5, 5))
ax.set_title("Norm Gradient")
ax.plot(range(0, len(LR.norm_grad)), LR.norm_grad, 'r', label=r'$\alpha = 0.001$')
ax.set_xlabel('Iterations')
ax.set_ylabel('Gradient')
ax.legend();
f, ax = plt.subplots(1, figsize=(5, 5))
indices = np.float32(range(100000))
plt.figure(figsize=(8, 8))
ax.set_title("Validation loss")
ax.plot(range(0, len(LR.total_cost)), LR.total_cost, 'r', label=r'$\alpha = 0.001$')
ax.set_xlabel('Iterations')
ax.set_ylabel('Training Error')
ax.legend();
fpr, tpr = LR.getS()
plt.figure(figsize=(5, 5))
plot(sorted(fpr), sorted(tpr), 0.001)

plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=.4)
plt.show()

plt.figure(figsize=(5, 5))
x_d = np.linspace(-3, 4, 50)
y_d = (LR.weights[0] * x_d) / LR.weights[1]
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=.4)
plt.plot(x_d, y_d)
plt.show()