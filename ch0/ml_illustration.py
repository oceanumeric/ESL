import numpy as np
import matplotlib.pyplot as plt
import math
import time
import torch


class MlExample:
  """
  A class to illustrate
    - what the learning rate is
    - the convergent and divergent behaviors of learning rate 
    - the tradeoff among:
        * the model
        * the learning rate
        * the traning time (iteration)
    - what the Tensor is ans why it is so useful
  """

  def __init__(self) -> None:
    self.x = np.linspace(-math.pi, math.pi, 2000)
    self.y = np.sin(self.x)
    self.pred = None  # prediction 
    self.L = None  # loss
    self.model = None
    self.model_label = None
    self.alpha = None
    self.iter = None

  
  def train_model1(self, learning_rate, iteration):
    self.alpha = learning_rate
    self.iter = iteration 
    # weights
    a = np.random.randn()
    b = np.random.randn()
    c = np.random.randn()
    d = np.random.randn()
    # loss 
    L = []
    start = time.time()
    for t in range(iteration):
      y_pred = a + b*self.x + c*self.x**2 + d*self.x**3

      # compute and print loss
      loss = np.square(self.y - y_pred).sum()
      if t % 100 == 99:
        L.append(loss)

      grad_y_pred = 2.0 * (y_pred - self.y)
      grad_a = grad_y_pred.sum()
      grad_b = (grad_y_pred*self.x).sum()
      grad_c = (grad_y_pred*self.x**2).sum()
      grad_d = (grad_y_pred*self.x**3).sum()

      # update the weight
      a -= learning_rate * grad_a
      b -= learning_rate * grad_b
      c -= learning_rate * grad_c
      d -= learning_rate * grad_d

    end = time.time()
    es = round(end-start, 3)
    print(f"The training time is: {es} seconds")
    self.model = f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3'
    print(self.model)
    self.model_label = r"$y = a+bx+cx^2+dx^3$"
    self.pred = y_pred
    self.L = L
    return y_pred, L 

  def train_model2(self, learning_rate, iteration):
    self.alpha = learning_rate
    self.iter = iteration 
    # weights
    a = np.random.randn()
    b = np.random.randn()
    c = np.random.randn()
    d = np.random.randn()
    e = np.random.randn()
    # loss 
    L = []
    start = time.time()
    for t in range(iteration):
      y_pred = a + b*self.x + c*self.x**2 + d*self.x**3 + e*self.x**4

      # compute and print loss
      loss = np.square(self.y - y_pred).sum()
      if t % 100 == 99:
        L.append(loss)

      grad_y_pred = 2.0 * (y_pred - self.y)
      grad_a = grad_y_pred.sum()
      grad_b = (grad_y_pred*self.x).sum()
      grad_c = (grad_y_pred*self.x**2).sum()
      grad_d = (grad_y_pred*self.x**3).sum()
      grad_e = (grad_y_pred*self.x**4).sum()

      # update the weight
      a -= learning_rate * grad_a
      b -= learning_rate * grad_b
      c -= learning_rate * grad_c
      d -= learning_rate * grad_d
      e -= learning_rate * grad_e

    end = time.time()
    es = round(end-start, 3)
    print(f"The training time is: {es} seconds")
    self.model = f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3 + {e} x^4'
    print(self.model)
    self.model_label = r"$y = a+bx+cx^2+dx^3+ex^4$"
    self.pred = y_pred
    self.L = L 
    return y_pred, L 

  def train_model3(self, learning_rate, iteration):
    self.alpha = learning_rate
    self.iter = iteration 
    # weights
    a = np.random.randn()
    b = np.random.randn()
    c = np.random.randn()
    d = np.random.randn()
    e = np.random.randn()
    f = np.random.randn()
    # loss 
    L = []
    start = time.time()
    for t in range(iteration):
        y_pred = a + b*self.x + c*self.x**2 + d*self.x**3 + e*self.x**4 + f*self.x**5

        # compute and print loss
        loss = np.square(self.y - y_pred).sum()
        if t % 100 == 99:
          L.append(loss)

        grad_y_pred = 2.0 * (y_pred - self.y)
        grad_a = grad_y_pred.sum()
        grad_b = (grad_y_pred*self.x).sum()
        grad_c = (grad_y_pred*self.x**2).sum()
        grad_d = (grad_y_pred*self.x**3).sum()
        grad_e = (grad_y_pred*self.x**4).sum()
        grad_f = (grad_y_pred*self.x**5).sum()

        # update the weight
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
        c -= learning_rate * grad_c
        d -= learning_rate * grad_d
        e -= learning_rate * grad_e
        f -= learning_rate * grad_f 

    end = time.time()
    es = round(end-start, 3)
    print(f"The training time is: {es} seconds")
    self.model = f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3 + {e} x^4 + {f} x^5'
    print(self.model)
    self.model_label = r"$y = a+bx+cx^2+dx^3+ex^4+fx^5$"
    self.pred = y_pred
    self.L = L
    return y_pred, L 

  def plot_results(self, gpu=False):
      fig, ax = plt.subplots(1, 2, figsize=(14, 5))
      ax[0].plot(self.L)
      ax[0].set_title(f"The loss function with alpha {self.alpha} and Iter {self.iter}")
      ax[0].annotate(round(self.L[-1], 3), (len(self.L)-5, self.L[-1]))
      if gpu:
        ax[1].plot(self.x.cpu().numpy(), self.y.cpu().numpy(), label=r"$\sin(x)$")
        ax[1].plot(self.x.cpu().numpy(), self.pred.cpu().numpy(), label=self.model_label)
      else:
        ax[1].plot(self.x, self.y, label=r"$\sin(x)$")
        ax[1].plot(self.x, self.pred, label=self.model_label)
      ax[1].axhline(y=0, color='k')
      ax[1].legend()
      fig.show()

  def train_with_gpu(self, learning_rate, iteration):
    self.alpha = learning_rate
    self.iter = iteration 

    # use gpu
    dtype = torch.float
    device = torch.device("cuda:0")
    # x and y 
    self.x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
    self.y = torch.sin(self.x)
    # weights
    a = torch.randn((), device=device, dtype=dtype)
    b = torch.randn((), device=device, dtype=dtype)
    c = torch.randn((), device=device, dtype=dtype)
    d = torch.randn((), device=device, dtype=dtype)
    e = torch.randn((), device=device, dtype=dtype)
    f = torch.randn((), device=device, dtype=dtype)
    # loss 
    L = []
    start = time.time()
    for t in range(iteration):
        y_pred = a + b*self.x + c*self.x**2 + d*self.x**3 + e*self.x**4 + f*self.x**5

        # Compute and print loss
        loss = (y_pred - self.y).pow(2).sum().item()
        if t % 100 == 99:
            L.append(loss)

        grad_y_pred = 2.0 * (y_pred - self.y)
        grad_a = grad_y_pred.sum()
        grad_b = (grad_y_pred*self.x).sum()
        grad_c = (grad_y_pred*self.x**2).sum()
        grad_d = (grad_y_pred*self.x**3).sum()
        grad_e = (grad_y_pred*self.x**4).sum()
        grad_f = (grad_y_pred*self.x**5).sum()

        # update the weight
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
        c -= learning_rate * grad_c
        d -= learning_rate * grad_d
        e -= learning_rate * grad_e
        f -= learning_rate * grad_f 

    end = time.time()
    es = round(end-start, 3)
    print(f"The training time is: {es} seconds")
    self.model = f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3 + {e} x^4 + {f} x^5'
    print(self.model)
    self.model_label = r"$y = a+bx+cx^2+dx^3+ex^4+fx^5$"
    self.pred = y_pred
    self.L = L
    return y_pred, L 

    