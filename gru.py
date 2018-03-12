# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

class RN(object):
    def __init__(self, fan_in, fan_out):
#        self.random = np.random.mtrand._rand.rand
#        self.random = lambda n: np.random.uniform(-1.0, 1.0, n)
        self.Wrx = self.random(fan_in, fan_out)
        self.Wrh = self.random(fan_out, fan_out)
        self.br = self.random()
        self.Whrh = self.random(fan_out, fan_out)
        self.Whx = self.random(fan_in, fan_out)
        self.bh = self.random()
        self.Wzx = self.random(fan_out, fan_out)
        self.Wzh = self.random(fan_out, fan_out)
        self.bz = self.random()
        self.ht_1 = np.zeros((fan_out,))
        
    def random(self, *args, mode=1):
        if mode==0:
            return np.random.mtrand._rand.rand(args[0], args[1])
        else:
            return np.random.uniform(-1.0, 1.0, args)
        
    def activate(self, x):
        x = np.array(x, dtype=np.float64)
        rt = self.sigmoid(np.dot(x, self.Wrx) + np.dot(self.ht_1, self.Wrh) + self.br)
        zt = self.sigmoid(np.dot(x, self.Wzx) + np.dot(self.ht_1, self.Wzh) + self.bz)
        h_t = np.tanh(np.dot(rt*self.ht_1, self.Whrh) + np.dot(x, self.Whx) + self.bh)
        ht = (1 - zt) * self.ht_1 + zt * h_t
        self.ht_1 = ht
        return ht
            
    # Source: https://gist.github.com/jovianlin/805189d4b19332f8b8a79bbd07e3f598
    def sigmoid(self, x, derivative=False):
        return x*(1-x) if derivative else 1/(1+np.exp(-x)+0.0001)
#        f = lambda z: 1/(1+np.exp(-z))
#        if derivative:
#            f =lambda z: z*(1-z)
#        return [f(z) for z in x]
        
    
def test_rn(fan_in, fan_out, plot=True):
    rn = RN(fan_in, fan_out)
    inut = []
#    np_range = np.arange(-6.0, 6.0, 0.1)
    np_range = np.linspace(0, 4*np.pi, 100)
    phase = 0.45
    for i in range(fan_in):
        inut.append(np.sin(np_range + phase))
        phase = np.random.random()
    assert len(inut) == fan_in
    inut = np.array(inut, dtype=np.float64)
    inut = np.reshape(inut, (inut.shape[1], inut.shape[0]))
    out = np.array([rn.activate(i) for i in inut])
    original_input = np.sin(np_range + phase)
    original_input = np.reshape(original_input, (original_input.shape[0],1))
    
    if plot:
        out = np.hstack((out, inut))
        plt.plot(np_range, out)
    return out
