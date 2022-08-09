"""
Chapter 2: overview of supervised learning
"""
import random 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import filterfalse, product


class OverviewSL:
    """
    Overview of supervised learning:
        - two approaches: linear regression vs. nearest neighbors 
    """

    # generate data as class variables
    size = 100
    cov = np.identity(2)
    mk_orange = np.random.multivariate_normal([0, 1], cov, 10)
    mk_blue = np.random.multivariate_normal([1, 0], cov, 10)
    orange = np.array()
    blue = np.random.multivariate_normal(random.choice(mk_blue), cov/5, size)

    def __init__(self) -> None:
        self.X = np.r_[self.orange, self.blue]
        self.Y = np.r_[np.zeros(self.size), np.ones(self.size)]
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
        self.pred = np.dot(self.weights, X)
        print(f"The estimated coefficents are {self.weights}")

    def plot_linear_model(self):
        '''
        Plot the linear model 
        '''
        is_orange = lambda x: np.dot(self.weights, x) > 0.5  # create a filter
        fig, axes = plt.subplots(1, 1, figsize=(8, 8))
        axes.scatter(self.orange[:, 0], self.orange[:, 1],
                        color='#d68904', facecolor='none', s=70)
        axes.scatter(self.blue[:, 0], self.blue[:, 1],
                        color='#1f6f9c', facecolor='none', s=70)
        xlim = axes.get_xlim()
        ylim = axes.get_ylim()
        # * unpact sequences or iterators 
        grid = np.array(*[product(np.linspace(*xlim, 50),
                                  np.linspace(*ylim, 50))])
        orange_grid = np.array([*filter(is_orange, grid)])
        blue_grid = np.array([*filterfalse(is_orange, grid)])
        axes.plot(orange_grid, blue_grid)




