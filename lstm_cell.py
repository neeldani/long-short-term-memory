import numpy as np

class LSTMCell:
    def forward(self, x, h_prev, c_prev, weights):
        self.c_prev = c_prev
        self.x_shape = x.shape[0]
        self.x_h = np.vstack((x, h_prev))
        
        # compute gates
        self.i = self.softmax(np.dot(weights["Wi"], self.x_h) + weights["bi"])
        self.f = self.softmax(np.dot(weights["Wf"], self.x_h) + weights["bf"])
        self.o = self.softmax(np.dot(weights["Wo"], self.x_h) + weights["bo"])
        self.g = np.tanh(np.dot(weights["Wg"], self.x_h) + weights["bg"])
        
        # compute cell memory and hidden output
        self.c = np.multiply(self.i, self.g) + np.multiply(self.f, self.c_prev)
        self.h_ungated = np.tanh(self.c)
        self.h = np.multiply(self.o, self.h_ungated)
        
        # compute dense layer output
        self.y_pred = self.softmax(np.dot(weights["Wd"], self.h) + weights["bd"])
        
        return self.h, self.c, self.y_pred
    
    def cost(self, y_true):
        self.y_true = y_true
        loss = -np.sum(np.multiply(np.log(self.y_pred), self.y_true))
        return loss
    
    def backward(self, dh_next, dc_next, weights):
        grads = {}
        dh_prev = 0.0

        # output layer
        dL = self.y_pred - self.y_true 
        
        # dense layer (fix grads here)
        grads["bd"] = dL
        grads["Wd"] = np.dot(dL, self.h.T)
        
        dh = np.dot(weights["Wd"].T, dL) + dh_next
        
        # output gate
        dZo = np.multiply(np.multiply(dh, self.h_ungated), np.multiply(self.o, 1 - self.o))
        
        grads["bo"] = dZo
        grads["Wo"] =  np.dot(dZo, self.x_h.T)
        dh_prev += np.dot(weights["Wo"][:, self.x_shape:].T, dZo)
        
        dh_ungated = np.multiply(dh, self.o)
        dC = np.multiply(dh_ungated, 1 - np.power(self.h_ungated, 2)) + dc_next
        
        # gate gate
        dZg = np.multiply(np.multiply(dC, self.i), 1 - np.power(self.g, 2))
        grads["bg"] = dZg
        grads["Wg"] = np.dot(dZg, self.x_h.T)
        dh_prev += np.dot(weights["Wg"][:, self.x_shape:].T, dZg)
        
        # input gate
        dZi = np.multiply(np.multiply(dC, self.g), np.multiply(self.i, 1 - self.i))
        grads["bi"] = dZi
        grads["Wi"] = np.dot(dZi, self.x_h.T)
        dh_prev += np.dot(weights["Wi"][:, self.x_shape:].T, dZi)
        
        # forget gate
        dZf = np.multiply(np.multiply(dC, self.c_prev), np.multiply(self.f, 1 - self.f))
        grads["bf"] = dZf
        grads["Wf"] = np.dot(dZf, self.x_h.T)
        dh_prev += np.dot(weights["Wf"][:, self.x_shape].T, dZf)
        
        dC_prev = np.multiply(dC, self.f)
        
        return grads, dh_prev, dC_prev
        
    
    def softmax (self, z):
        if z.shape[0] == 1:
            return 1/ (1 + np.exp(-z))
        
        return np.exp(z)/np.sum(np.exp(z), axis = 0)
    def forward(self, x, h_prev, c_prev, weights):
        self.c_prev = c_prev
        self.x_shape = x.shape[0]
        self.x_h = np.vstack((x, h_prev))
        
        # compute gates
        self.i = self.softmax(np.dot(weights["Wi"], self.x_h) + weights["bi"])
        self.f = self.softmax(np.dot(weights["Wf"], self.x_h) + weights["bf"])
        self.o = self.softmax(np.dot(weights["Wo"], self.x_h) + weights["bo"])
        self.g = np.tanh(np.dot(weights["Wg"], self.x_h) + weights["bg"])
        
        # compute cell memory and hidden output
        self.c = np.multiply(self.i, self.g) + np.multiply(self.f, self.c_prev)
        self.h_ungated = np.tanh(self.c)
        self.h = np.multiply(self.o, self.h_ungated)
        
        # compute dense layer output
        self.y_pred = self.softmax(np.dot(weights["Wd"], self.h) + weights["bd"])
        
        return self.h, self.c, self.y_pred
    
    def cost(self, y_true):
        self.y_true = y_true
        loss = -np.sum(np.multiply(np.log(self.y_pred), self.y_true))
        return loss
    
    def backward(self, dh_next, dc_next, weights):
        grads = {}
        dh_prev = 0.0

        # output layer
        dL = self.y_pred - self.y_true 
        
        # dense layer (fix grads here)
        grads["bd"] = dL
        grads["Wd"] = np.dot(dL, self.h.T)
        
        dh = np.dot(weights["Wd"].T, dL) + dh_next
        
        # output gate
        dZo = np.multiply(np.multiply(dh, self.h_ungated), np.multiply(self.o, 1 - self.o))
        
        grads["bo"] = dZo
        grads["Wo"] =  np.dot(dZo, self.x_h.T)
        dh_prev += np.dot(weights["Wo"][:, self.x_shape:].T, dZo)
        
        dh_ungated = np.multiply(dh, self.o)
        dC = np.multiply(dh_ungated, 1 - np.power(self.h_ungated, 2)) + dc_next
        
        # gate gate
        dZg = np.multiply(np.multiply(dC, self.i), 1 - np.power(self.g, 2))
        grads["bg"] = dZg
        grads["Wg"] = np.dot(dZg, self.x_h.T)
        dh_prev += np.dot(weights["Wg"][:, self.x_shape:].T, dZg)
        
        # input gate
        dZi = np.multiply(np.multiply(dC, self.g), np.multiply(self.i, 1 - self.i))
        grads["bi"] = dZi
        grads["Wi"] = np.dot(dZi, self.x_h.T)
        dh_prev += np.dot(weights["Wi"][:, self.x_shape:].T, dZi)
        
        # forget gate
        dZf = np.multiply(np.multiply(dC, self.c_prev), np.multiply(self.f, 1 - self.f))
        grads["bf"] = dZf
        grads["Wf"] = np.dot(dZf, self.x_h.T)
        dh_prev += np.dot(weights["Wf"][:, self.x_shape].T, dZf)
        
        dC_prev = np.multiply(dC, self.f)
        
        return grads, dh_prev, dC_prev
        
    
    def softmax (self, z):
        if z.shape[0] == 1:
            return 1/ (1 + np.exp(-z))
        
        return np.exp(z)/np.sum(np.exp(z), axis = 0)