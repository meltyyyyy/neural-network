import sys, os
import numpy as np
sys.path.append(os.pardir)

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size,hidden_size)

        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size,output_size)

        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x,W1) + b1
        z1 = self.sigmoid(a1)
        a2 = np.dot(z1,W2) + b2
        y = self.softmax(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)

        return self.cross_entropy_error(y,t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        t = np.argmax(t,axis=1)

        accuracy = np.sum(y==t) / float(x.shape[0])
        return accuracy

    def gradient(self,x,t):
        loss_W = lambda W: self.loss(x,t)

        grads = {}
        grads['W1'] = self.numerical_gradient(loss_W,self.params['W1'])
        grads['b1'] = self.numerical_gradient(loss_W,self.params['b1'])
        grads['W2'] = self.numerical_gradient(loss_W,self.params['W2'])
        grads['b2'] = self.numerical_gradient(loss_W,self.params['b2'])

        return grads

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, a):
        c = np.max(a)
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a

        return y

    def numerical_gradient(self, f, x):
        h = 1e-4 # 0.0001
        grad = np.zeros_like(x)

        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = x[idx]
            x[idx] = tmp_val + h
            fxh1 = f(x) # f(x+h)

            x[idx] = tmp_val - h
            fxh2 = f(x) # f(x-h)
            grad[idx] = (fxh1 - fxh2) / (2*h)

            x[idx] = tmp_val # 値を元に戻す
            it.iternext()

        return grad

    def cross_entropy_error(self, y, t):
        if y.ndim == 1:
            t = t.reshape(1,t.size)
            y = y.reshape(1,y.size)

        batch_size = y.shape[0]
        return -np.sum(t * np.log(y + 1e-7)) / batch_size
