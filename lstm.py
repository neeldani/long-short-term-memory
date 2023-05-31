import numpy as np
from lstm import LSTMCell

class LSTM:
    
    def __init__(self, n_in, n_hidden, n_out, n_timestamps, weights=None):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.n_timestamps = n_timestamps
        
        if weights is None:
            self.weights = {}
            self.weights["Wi"] = np.random.randn(n_hidden, n_in + n_hidden)
            self.weights["bi"] = np.random.randn(n_hidden, 1)
            self.weights["Wf"] = np.random.randn(n_hidden, n_in + n_hidden)
            self.weights["bf"] = np.random.randn(n_hidden, 1)
            self.weights["Wo"] = np.random.randn(n_hidden, n_in + n_hidden)
            self.weights["bo"] = np.random.randn(n_hidden, 1)
            self.weights["Wg"] = np.random.randn(n_hidden, n_in + n_hidden)
            self.weights["bg"] = np.random.randn(n_hidden, 1)
            
            # dense layer
            self.weights["Wd"] = np.random.randn(n_out, n_hidden)
            self.weights["bd"] = np.random.randn(n_out, 1)
        
        self.model = [ LSTMCell() for _ in range(self.n_timestamps)]
        
    def forward(self, x, y):
        h = np.zeros((self.n_hidden, 1))
        c = np.zeros((self.n_hidden, 1))
        total_cost = 0

        for t in range(self.n_timestamps):
            lstm_cell = self.model[t]
            h, c, _ = lstm_cell.forward(x[:, [t]], h, c, self.weights)
            total_cost += lstm_cell.cost(y[:, [t]])

        return total_cost/self.n_timestamps

    def backward(self):
        T = self.n_timestamps
        dh_next = np.zeros((self.n_hidden, 1))
        dc_next = np.zeros((self.n_hidden, 1))
        gradients = {}

        for t in range(T-1, -1, -1):
            partial_gradients, dh_next, dc_next = self.model[t].backward(dh_next, dc_next, self.weights)

            for k, v in partial_gradients.items():
                if k not in gradients:
                    gradients[k] = 0
                gradients[k] += v/T

        return gradients

    def train(self, x, y, alpha):
        cost = self.forward(x, y)
        gradients = self.backward()

        # update weights
        for weight in self.weights.keys():
            self.weights[weight] -= alpha * gradients[weight]

        return cost

    def predict(self, x, h, c):
        y_preds = []

        for t in range(self.n_timestamps):
            lstm_cell = self.model[t]
            h, c, y_pred = lstm_cell.forward(x, h, c, self.weights)
            y_preds.append(y_pred)

        return y_preds
    def __init__(self, n_in, n_hidden, n_out, n_timestamps, weights=None):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.n_timestamps = n_timestamps
        
        if weights is None:
            self.weights = {}
            self.weights["Wi"] = np.random.randn(n_hidden, n_in + n_hidden)
            self.weights["bi"] = np.random.randn(n_hidden, 1)
            self.weights["Wf"] = np.random.randn(n_hidden, n_in + n_hidden)
            self.weights["bf"] = np.random.randn(n_hidden, 1)
            self.weights["Wo"] = np.random.randn(n_hidden, n_in + n_hidden)
            self.weights["bo"] = np.random.randn(n_hidden, 1)
            self.weights["Wg"] = np.random.randn(n_hidden, n_in + n_hidden)
            self.weights["bg"] = np.random.randn(n_hidden, 1)
            
            # dense layer
            self.weights["Wd"] = np.random.randn(n_out, n_hidden)
            self.weights["bd"] = np.random.randn(n_out, 1)
        
        self.model = [ LSTMCell() for _ in range(self.n_timestamps)]
        
    def forward(self, x, y):
        h = np.zeros((self.n_hidden, 1))
        c = np.zeros((self.n_hidden, 1))
        total_cost = 0

        for t in range(self.n_timestamps):
            lstm_cell = self.model[t]
            h, c, _ = lstm_cell.forward(x[:, [t]], h, c, self.weights)
            total_cost += lstm_cell.cost(y[:, [t]])

        return total_cost/self.n_timestamps

    def backward(self):
        T = self.n_timestamps
        dh_next = np.zeros((self.n_hidden, 1))
        dc_next = np.zeros((self.n_hidden, 1))
        gradients = {}

        for t in range(T-1, -1, -1):
            partial_gradients, dh_next, dc_next = self.model[t].backward(dh_next, dc_next, self.weights)

            for k, v in partial_gradients.items():
                if k not in gradients:
                    gradients[k] = 0
                gradients[k] += v/T

        return gradients

    def train(self, x, y, alpha):
        cost = self.forward(x, y)
        gradients = self.backward()

        # update weights
        for weight in self.weights.keys():
            self.weights[weight] -= alpha * gradients[weight]

        return cost

    def predict(self, x, h, c):
        y_preds = []

        for t in range(self.n_timestamps):
            lstm_cell = self.model[t]
            h, c, y_pred = lstm_cell.forward(x, h, c, self.weights)
            y_preds.append(y_pred)

        return y_preds