import numpy as np
import matplotlib.pyplot as plt
import math


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
    self.y = np.sin(x)
    self.pred = None  # prediction 
    self.L = None  # loss

  
  def train_model1(self, learning_rate, iteration):
    # weights
    a = np.random.randn()
    b = np.random.randn()
    c = np.random.randn()
    d = np.random.randn()
    # loss 
    L = []
    for t in range(iteration):
      y_pred = a + b*x + c*x**2 + d*x**3

      # compute and print loss
      loss = np.square(y - y_pred).sum()
      if t % 100 == 99:
        L.append(loss)

      grad_y_pred = 2.0 * (y_pred - y)
      grad_a = grad_y_pred.sum()
      grad_b = (grad_y_pred*x).sum()
      grad_c = (grad_y_pred*x**2).sum()
      grad_d = (grad_y_pred*x**3).sum()

      # update the weight
      a -= learning_rate * grad_a
      b -= learning_rate * grad_b
      c -= learning_rate * grad_c
      d -= learning_rate * grad_d

    print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')
    self.pred = y_pred
    self.L = L
    return y_pred, L 

  def train_model2(self, learning_rate, iteration):
    # weights
    a = np.random.randn()
    b = np.random.randn()
    c = np.random.randn()
    d = np.random.randn()
    e = np.random.randn()
    # loss 
    L = []
    for t in range(iteration):
      y_pred = a + b*x + c*x**2 + d*x**3 + e*x**4

      # compute and print loss
      loss = np.square(y - y_pred).sum()
      if t % 100 == 99:
        L.append(loss)

      grad_y_pred = 2.0 * (y_pred - y)
      grad_a = grad_y_pred.sum()
      grad_b = (grad_y_pred*x).sum()
      grad_c = (grad_y_pred*x**2).sum()
      grad_d = (grad_y_pred*x**3).sum()
      grad_e = (grad_y_pred*x**4).sum()

      # update the weight
      a -= learning_rate * grad_a
      b -= learning_rate * grad_b
      c -= learning_rate * grad_c
      d -= learning_rate * grad_d
      e -= learning_rate * grad_e

    print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3 + {e} x^4')
    self.pred = y_pred
    self.L = L 
    return y_pred, L 

  def train_model3(self, learning_rate, iteration):
    # weights
    a = np.random.randn()
    b = np.random.randn()
    c = np.random.randn()
    d = np.random.randn()
    e = np.random.randn()
    f = np.random.randn()
    # loss 
    L = []
    for t in range(iteration):
        y_pred = a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5

        # compute and print loss
        loss = np.square(y - y_pred).sum()
        if t % 100 == 99:
          L.append(loss)

        grad_y_pred = 2.0 * (y_pred - y)
        grad_a = grad_y_pred.sum()
        grad_b = (grad_y_pred*x).sum()
        grad_c = (grad_y_pred*x**2).sum()
        grad_d = (grad_y_pred*x**3).sum()
        grad_e = (grad_y_pred*x**4).sum()
        grad_f = (grad_y_pred*x**5).sum()

        # update the weight
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
        c -= learning_rate * grad_c
        d -= learning_rate * grad_d
        e -= learning_rate * grad_e
        f -= learning_rate * grad_f 

    print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3 + {e} x^4 + {f} x^5')
    self.pred = y_pred
    self.L = L
    return y_pred, L 

    def plot_results(self):
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        ax[0].plot(self.L)
        ax[0].set_title("The loss function")
        ax[0].annotate(round(self.L[-1], 3), (len(self.L)-10, self.L[-1]))
        ax[1].plot(self.x, self.y, label=r"$\sin(x)$")
        ax[1].plot(self.x, self.pred, label=r"$y = a+bx+cx^2+dx^3+ex^4$")
        ax[1].axhline(y=0, color='k')
        ax[1].legend()
        fig.show()

    