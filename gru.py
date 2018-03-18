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
        self.Wzx = self.random(fan_in, fan_out)
        self.Wzh = self.random(fan_out, fan_out, mode=0)
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
        
        return x*(1-x) if derivative else 1/(1+np.exp(-x)+(np.random.normal(0,0.5)*0.01))
    
#        f = lambda z: 1/(1+np.exp(-z))
#        if derivative:
#            f =lambda z: z*(1-z)
#        return [f(z) for z in x]
        
    
def test_rn(fan_in, fan_out, plot=True):
    rn = RN(fan_in, fan_out)
    inut = []
#    np_range = np.arange(-6.0, 6.0, 0.1)
    np_range = np.linspace(-12.0, 4*np.pi, 1000)
    phase = 0.0
    for i in range(fan_in):
        f = np.random.choice([np.sin, np.cos])
        inut.append(f(np_range + phase))
#        nx = np.linspace(0, (r+i+1)*np.pi, 1000)
        phase += 0.002
    assert len(inut) == fan_in
    inut = np.array(inut, dtype=np.float64)
    inut = inut.T
    out = np.array([rn.activate(i) for i in inut])
    original_input = np.sin(np_range + phase)
    original_input = np.reshape(original_input, (original_input.shape[0],1))
    
    if plot:
#        out = np.hstack((out, inut))
        for t, color in enumerate(['r--', 'b--', 'y--','m--', 'g--','k--'], start=0):
            if t >= inut.shape[1]:
                break
            plt.plot(np_range, inut[:,t], color)
        for i, color in enumerate(['r-', 'b-', 'y-','m-', 'g-','k-'], start=0):
            if i >= out.shape[1]:
                break
            plt.plot(np_range, out[:,i], color)
    return out

def test_layered(fan_in, fan_out, hidden=[], plot=True, **kwargs):
    def get_layers(fan_in, fan_out, hidden=[]):
        rn = []
        
        for i in range(len(hidden)+1):
            if i == 0:
                rn.append((fan_in, hidden[0]))
            elif i == len(hidden):
                rn.append((hidden[i-1], fan_out))
            else:
                rn.append((hidden[i-1], hidden[i]))
        return rn
    
    def get_activations(rn_layers, x_in):
        rx_in = x_in
        
        activations_out = []
        final_out = None
        for rn in rn_layers:
            _out = np.array([rn.activate(i + np.random.uniform(0,0.1)) for i in rx_in])
            activations_out.append(_out)
            rx_in = _out
        else:
            final_out = _out
        return final_out, activations_out
    
    pi_multiplier = 4
    if 'pi_multiplier' in kwargs:
        pi_multiplier = kwargs['pi_multiplier']
        print("Setting pi_multiplier to: {}".format(pi_multiplier))
        
    n_points = 1000
    if 'n_points' in kwargs:
        n_points = kwargs['n_points']
        print("n_points: {}".format(n_points))
    
    f_choice = [np.sin, np.cos]
    if 'f_choice' in kwargs:
        assert type(kwargs['f_choice']) == list
        f_choice.extend(kwargs['f_choice'])
        _ff = set(f_choice)
        f_choice = list(_ff)
        print("f_choices: {}".format(f_choice))
        
    in_freq = 1.0
    if 'in_freq' in kwargs:
        in_freq = kwargs['in_freq']
        print("in_freq: {}".format(in_freq))

    if len(hidden) > 0:
        layers = get_layers(fan_in, fan_out, hidden)
        rn = []
        for layer in layers:
            rn.append(RN(layer[0],layer[1]))
            
        # Prepare input (this is the same used in test_rn)
        inut = []
        np_range = np.linspace(0, pi_multiplier*np.pi, n_points)
        phase = 0.0
        for i in range(fan_in):
            f = np.random.choice(f_choice)
            print(f)
            inut.append(f(in_freq*np_range - phase))
            phase += 0.5
        assert len(inut) == fan_in
        inut = np.array(inut, dtype=np.float64)
        inut = inut.T
        x_in = inut
        
        out, activations = get_activations(rn, x_in)

        if plot:
    
            for t, color in enumerate(['r--', 'b--', 'y--','m--', 'g--','k--'], start=0):
                if t == inut.shape[1]:
                    break
                plt.plot(np_range, inut[:,t], color)
            for i, color in enumerate(['r-', 'b-', 'y-','m-', 'g-','k-'], start=0):
                if i == out.shape[1]:
                    break
                plt.plot(np_range, out[:,i], color)
        
        return out, activations
        
    else:
        return test_rn(fan_in, fan_out)
        