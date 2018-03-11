# -*- coding: utf-8 -*-

class RN(object):
   ...:     def __init__(self, fan_in, fan_out):
   ...:         self.random = np.random.mtrand._rand.rand
   ...:         self.Wrx = self.random(fan_in, fan_out)
   ...:         self.Wrh = self.random(fan_out, fan_out)
   ...:         self.br = self.random(fan_out)
   ...:         self.Whrh = self.random(fan_out, fan_out)
   ...:         self.Whx = self.random(fan_in, fan_out)
   ...:         self.bh = self.random(fan_out)
   ...:         self.Wz = self.random(fan_out, fan_out)
   ...:         self.bz = self.random(fan_out)
   ...:         self.ht_1 = np.zeros((fan_out,))
   ...:     def activate(self, x):
   ...:         rt = self.sigmoid(np.dot(self.Wrx, x) + np.dot(self.Wrh, self.ht_1) + self.br)
   ...:         
   ...:     # Source: https://gist.github.com/jovianlin/805189d4b19332f8b8a79bbd07e3f598
   ...:     def sigmoid(x, derivative=False):
   ...:         return x*(1-x) if derivative else 1/(1+np.exp(-x))