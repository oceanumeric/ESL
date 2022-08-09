"""
Chapter 2: overview of supervised learning
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import filterfalse, product
import random 


class OverviewSL:
    """
    Overview of supervised learning:
        - two approaches: linear regression vs. nearest neighbors 
    """

    # generate data as class variables
    size = 100
    cov = np.identity(2)
    mk_orange = np.random.multivariate_normal([0, 1], cov, 10)
    mk_orange = random.choice(mk_orange)
    mk_blue = np.random.multivariate_normal([1, 0], cov, 10)
    mk_blue = random.choice(mk_blue)
    orange = []
    blue = []
    for i in range(size):
        orange.append(
            np.random.multivariate_normal(mk_orange, cov/5)
        )
        blue.append(
            np.random.multivariate_normal(mk_blue, cov/5)
        )
    orange = np.array(orange)
    blue = np.array(blue)

    def __init__(self) -> None:
        self.X = np.r_[self.orange, self.blue]
        self.Y = np.r_[np.ones(self.size), np.zeros(self.size)]
        self.weights = None
        self.pred = None

    @classmethod
    def plot_data(cls):
        '''
        plot the simulated data
        '''
        fig, axes = plt.subplots(1, 1, figsize=(8, 8))
        axes.scatter(cls.orange[:, 0], cls.orange[:, 1],
                        color='#d68904', facecolor='none', s=70)
        axes.scatter(cls.blue[:, 0], cls.blue[:, 1],
                        color='#1f6f9c', facecolor='none', s=70)
        fig.show()

    def fit_with_linear_model(self):
        '''
        Fit with linear regression model with closed solution 
        '''
        X = np.c_[np.ones((self.X.shape[0], 1)), self.X]
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ self.Y
        self.pred = np.dot(X, self.weights)
        print("hello michael")
        print(f"The estimated coefficents are {self.weights}")

    def plot_linear_model(self):
        '''
        Plot the linear model 
        '''
        is_orange = lambda x: np.dot(np.r_[1, x], self.weights) > 0.5  # create a filter
        fig, axes = plt.subplots(1, 1, figsize=(8, 8))
        axes.scatter(self.orange[:, 0], self.orange[:, 1],
                        color='#d68904', facecolor='none', s=70)
        axes.scatter(self.blue[:, 0], self.blue[:, 1],
                        color='#1f6f9c', facecolor='none', s=70)
        xlim = axes.get_xlim()
        ylim = axes.get_ylim()
        # * unpact sequences or iterators 
        grid = np.array([*product(np.linspace(*xlim, 50),
                                  np.linspace(*ylim, 50))])
        orange_grid = np.array([*filter(is_orange, grid)])
        blue_grid = np.array([*filterfalse(is_orange, grid)])
        axes.plot(orange_grid[:, 0], orange_grid[:, 1], '.',
                  zorder = 0.001, color='orange', alpha = 0.3,
                  scalex = False, scaley = False)
        axes.plot(blue_grid[:, 0], blue_grid[:, 1], '.',
                  zorder = 0.001, color='blue', alpha = 0.3,
                  scalex = False, scaley = False)
        # a + x1*alpha + x2*beta = Y (1/0)
        find_y = lambda x: (0.5 - self.weights[0] - x* self.weights[1]) / self.weights[2]
        axes.plot(xlim, [*map(find_y, xlim)], color='k', 
                  scalex = False, scaley = False)






