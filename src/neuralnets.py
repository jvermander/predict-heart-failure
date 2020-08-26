# For creating neural networks

import numpy as np
import random as rand
import math
import scipy.optimize as opt
from types import MethodType

import warnings
np.set_printoptions(edgeitems=5)
np.core.arrayprint._line_width = 180
warnings.filterwarnings("ignore", category=RuntimeWarning) 

class Neural(object):

  # fix for 0-index
  # for classifications in (1 ... degree)
  # instead of in (0 ... degree-1)
  def binarize_ground_truth( y, degree ):
    assert(isinstance(y, (np.ndarray)))
    assert(y.ndim == 1)

    result = np.zeros((y.shape[0], degree))
    for i in range(0, degree):
      result[:, i] = (y == i)
    return result

  # For classification problems
  def binary_cross_entropy( self ):
    p = self.fp()

    J = (-1 / self.m) * \
        np.sum(self.y * np.log(p) +
        (1 - self.y) * np.log(1 - p))

    J += self.l / 2 / self.m * np.sum(self.weight ** 2)     
    return J  

  def binary_cross_entropy_deriv( self ):
    return self.p - self.y

  def sigmoid( self, x ):
    return 1 / (1 + np.exp(-x))

  # assumes a = sigmoid(z)
  def sigmoid_deriv( self, a ):
    return a * (1 - a)
    
  def init_random( self, rows, cols ):
    epsilon = math.sqrt(6) / math.sqrt(rows + cols)
    return [ rand.uniform(0, 1) * 2 * epsilon - epsilon for i in range(rows * cols) ]

  MAX_LAYERS = 10
  MIN_LAYERS = 2

  # all derivatives assume the input is in terms of the activation itself, not the weighted input
  BCE = { 'f' : binary_cross_entropy, 'deriv' : binary_cross_entropy_deriv }
  SIG = { 'f' : sigmoid, 'deriv' : sigmoid_deriv, 'init' : init_random }

  COST = { 'BCE' : BCE }
  ACT = { 'SIG' : SIG }
  INIT = { 'SIG' : init_random }

  def __init__( self, layers, X, y, l=0, cost='BCE', act='SIG' ):
    assert(isinstance(layers, (np.ndarray)))
    # assert(np.issubdtype(layers.dtype, int))
    assert(layers.ndim == 1)
    assert(layers.shape[0] >= Neural.MIN_LAYERS)
    assert(layers.shape[0] <= Neural.MAX_LAYERS)

    self.cost = MethodType(Neural.COST[cost]['f'], self)
    self.cost_deriv = MethodType(Neural.COST[cost]['deriv'], self)
    self.act = MethodType(Neural.ACT[act]['f'], self)
    self.act_deriv = MethodType(Neural.ACT[act]['deriv'], self)
    self.init = MethodType(Neural.INIT[act], self)

    self.layer_size = layers
    self.num_layers = layers.shape[0]

    self.y = y.T               # ground truth
    self.p = None              # prediction using the current weights
    self.m = self.y.shape[1]   # training set size
    self.l = l                 # regularization constant

    self.weight = []
    self.bias = []

    for l in range(1, self.num_layers):
      self.weight += self.init(self.layer_size[l], self.layer_size[l-1])
      self.bias += self.init(self.layer_size[l], 1)
    
    self.weight = np.array(self.weight)
    self.bias = np.array(self.bias)

    # all layer activations for each training example, including the input matrix
    self.a = np.zeros(self.m * (X.shape[1] + self.bias.shape[0]))
    print(X.shape[1])
    print(self.bias.shape[0])
    self.a[:X.shape[1] * self.m] = X.T.flatten()


  # For the current weights and biases, run the training set through the network
  # Returns a prediction matrix
  def fp( self ):
    weight_start = weight_end = \
    bias_start = bias_end = \
    act_start = act_end = 0
    
    for l in range(1, self.num_layers):
      weight_end += self.layer_size[l] * self.layer_size[l-1]
      bias_end += self.layer_size[l]
      act_end += self.layer_size[l-1] * self.m

      W = self.weight[weight_start : weight_end].reshape(self.layer_size[l], self.layer_size[l-1])
      b = self.bias[bias_start : bias_end].reshape(self.layer_size[l], 1)
      a = self.a[act_start : act_end].reshape(self.layer_size[l-1], self.m)

      self.a[act_end : act_end + self.layer_size[l] * self.m] = self.act(W.dot(a) + b).flatten()

      weight_start = weight_end
      bias_start = bias_end
      act_start = act_end

    degree = self.layer_size[self.num_layers-1]
    self.p = self.a[-(degree * self.m):].reshape(degree, self.m)
    return self.p

  def bp( self ):
    assert(self.p is not None)
    gradient = np.zeros(self.weight.shape)
    b_gradient = np.zeros(self.bias.shape)

    delta = self.cost_deriv() 
    grad_start = 0
    b_grad_start = 0
    a_start = self.layer_size[self.num_layers-1] * self.m
    w_start = 0

    for l in range(self.num_layers-1, 0, -1):
      output = self.layer_size[l]
      input = self.layer_size[l-1]

      grad_end = grad_start
      grad_start += output * input

      b_grad_end = b_grad_start
      b_grad_start += output

      if(l < self.num_layers-1):
        prev = self.layer_size[l+1]
        w_end = w_start
        w_start += prev * output
        w = self.weight[-w_start : None if w_end is 0 else -w_end].reshape(prev, output)
        delta = (w.T.dot(delta)) * self.act_deriv(a)

      a_end = a_start
      a_start += input * self.m
      a = self.a[ -a_start : -a_end ].reshape(input, self.m)
      
      regularize = self.l * self.weight[-grad_start : None if grad_end is 0 else -grad_end]

      gradient[ -grad_start : None if grad_end is 0 else -grad_end ] = delta.dot(a.T).flatten() + regularize
      b_gradient[ -b_grad_start : None if b_grad_end is 0 else -b_grad_end ] = np.sum(delta, axis=1).flatten()
  
    result = np.concatenate((gradient, b_gradient)) / self.m
    return result

  def debug_bp():
    layers = np.array([3, 5, 3])
    m = 5
    X = np.random.randn(m, 3)
    y = np.mod(np.arange(m), 3)
    y = Neural.binarize_ground_truth(y, 3)

    net = Neural(layers, X, y)
    net.fp()
    grad = net.bp()
    w = net.weight
    b = net.bias

    numgrad = np.zeros(grad.shape)
    e = 0.0001
    perturb = np.zeros(w.shape[0])
    i = 0
    for i in range(w.shape[0]):
      perturb[i] = e
      net.weight = w + perturb
      loss1 = net.binary_cross_entropy()
      net.weight = w - perturb
      loss2 = net.binary_cross_entropy()
      perturb[i] = 0

      numgrad[i] = (loss1 - loss2) / (2 * e)

    i += 1
    net.weight = w
    perturb = np.zeros(b.shape[0])
    for j in range(b.shape[0]):
      perturb[j] = e
      net.bias = b + perturb
      loss1 = net.binary_cross_entropy()
      net.bias = b - perturb
      loss2 = net.binary_cross_entropy()
      perturb[j] = 0 

      numgrad[i + j] = (loss1 - loss2) / (2 * e)

    for i in range(grad.shape[0]):
      print("%f / %f" % (numgrad[i], grad[i]))

    return np.linalg.norm(numgrad - grad) / np.linalg.norm(grad + numgrad)

  def parametrize( self, iter=100, alg='L-BFGS-B' ):
    theta = np.concatenate((self.weight, self.bias))
    result = opt.minimize(Neural.cost, theta, args=(self), jac=Neural.gradient,
      method=alg, options={'maxiter': iter})
    print(result)
    self.assign(result.x)
    return

  def assign( self, theta ):
    numweight = self.weight.shape[0]
    self.weight = theta[:numweight]
    self.bias = theta[numweight:]

  def cost( theta, network ):
    network.assign(theta)
    return network.cost()
    
  
  def gradient( theta, network ):
    network.assign(theta)
    return network.bp()

  # fix for 0-index
  def predict( self, X ):
    assert(X.shape[1] == self.layer_size[0])
    self.a[:X.shape[1] * self.m] = X.T.flatten()
    p = self.fp()
    print(p.shape)
    degree = p.shape[0]
    p = np.round(p)
    return p.reshape(-1)