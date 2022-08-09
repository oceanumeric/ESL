"""
Chapter 2: overview of supervised learning
"""
import random 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import filterfalse, product
from scipy.stats import multivariate_normal



class OverviewSL:
    """
    Overview of supervised learning:
        - two approaches: linear regression vs. nearest neighbors 
    """
    # set seed
    random.seed(666)
    np.random.seed(667)
    # generate data as class variables
    size = 100
    cov = np.identity(2)
    mk_orange_10 = np.random.multivariate_normal([0, 1], cov, 10)
    mk_orange = random.choice(mk_orange_10)
    mk_blue_10 = np.random.multivariate_normal([1, 0], cov, 10)
    mk_blue = random.choice(mk_blue_10)
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
                  zorder = 0.001, color='#d68904', alpha = 0.3,
                  scalex = False, scaley = False)
        axes.plot(blue_grid[:, 0], blue_grid[:, 1], '.',
                  zorder = 0.001, color='#1f6f9c', alpha = 0.3,
                  scalex = False, scaley = False)
        # a + x1*alpha + x2*beta = Y (1/0)
        find_y = lambda x: (0.5 - self.weights[0] - x* self.weights[1]) / self.weights[2]
        axes.plot(xlim, [*map(find_y, xlim)], color='k', 
                  scalex = False, scaley = False)
        axes.set_title("Linear Regression of 0/1 Response")
        
    def fit_with_nearest_neighbors(self, k, boundary_line=False):
        '''
        calculate the mean based on the distance 
        There is NO weights 
        '''
        self.k = k 
        def __predict(x):
            # calcualte the distance 
            distance = ((self.X - x)**2).sum(axis=1)
            # elements within the distance, sort and find top K
            elems = distance.argpartition(self.k)[:self.k]
            y_pred = self.Y[elems]
            return np.mean(y_pred)  
        
        # majority votes (mean > 0.5)
        is_orange = lambda x: __predict(x) > 0.5
        
        # plot the classification 
        fig, axes = plt.subplots(1, 1, figsize=(8, 8))
        axes.scatter(self.orange[:, 0], self.orange[:, 1],
                        color='#d68904', facecolor='none', s=70)
        axes.scatter(self.blue[:, 0], self.blue[:, 1],
                        color='#1f6f9c', facecolor='none', s=70)
        xlim = axes.get_xlim()
        ylim = axes.get_ylim()
        grid = np.array([*product(np.linspace(*xlim, 50),
                                  np.linspace(*ylim, 50))])
        orange_grid = np.array([*filter(is_orange, grid)])
        blue_grid = np.array([*filterfalse(is_orange, grid)])
        axes.plot(orange_grid[:, 0], orange_grid[:, 1], '.',
                  zorder = 0.001, color='#d68904', alpha = 0.3,
                  scalex = False, scaley = False)
        axes.plot(blue_grid[:, 0], blue_grid[:, 1], '.',
                  zorder = 0.001, color='#1f6f9c', alpha = 0.3,
                  scalex = False, scaley = False)
        # plot the boundary
        if boundary_line:
            blue_sort = blue_grid[np.argsort(-blue_grid[:, 1])]
            _, idx = np.unique(blue_sort[:, 0], return_index=True)
            boundary = blue_sort[idx]
            axes.plot(boundary[:, 0], boundary[:, 1], 'k-')
        axes.set_title(f"{k}-Nearest Neighbor Classifier")
        
        
    def bayes_classifier(self):
        # plot the classification 
        fig, axes = plt.subplots(1, 1, figsize=(8, 8))
        axes.scatter(self.orange[:, 0], self.orange[:, 1],
                        color='#d68904', facecolor='none', s=70)
        axes.scatter(self.blue[:, 0], self.blue[:, 1],
                        color='#1f6f9c', facecolor='none', s=70)
        xlim = axes.get_xlim()
        ylim = axes.get_ylim()
        # shape = 2500 x 2
        grid = np.array([*product(np.linspace(*xlim, 50),
                                  np.linspace(*ylim, 50))])
        # np.random.multivariate_normal([0, 1], cov, 10)
        # mean = a vector [1.11742176 2.0701749 ]
        # p(x|orange) calculated by simulated model
        # in practce we estimate it by using taining data
        orange_pdf = np.mean([
            multivariate_normal.pdf(grid, mean=m, cov=np.eye(2)/5)
            for m in self.mk_orange_10], axis=0)
        # shape = 2500 x 1 ; 2500 can be taken as the sample size
        blue_pdf = np.mean([
            multivariate_normal.pdf(grid, mean=m, cov=np.eye(2)/5)
            for m in self.mk_blue_10], axis=0)
        orange_grid = grid[orange_pdf >= blue_pdf]
        blue_grid = grid[orange_pdf < blue_pdf]
        axes.plot(orange_grid[:, 0], orange_grid[:, 1], '.',
                  zorder = 0.001, color='#d68904', alpha = 0.3,
                  scalex = False, scaley = False)
        axes.plot(blue_grid[:, 0], blue_grid[:, 1], '.',
                  zorder = 0.001, color='#1f6f9c', alpha = 0.3,
                  scalex = False, scaley = False)
        axes.set_title(r"Bayes Optimal Classifier $\mathbb{P}(orange) = \mathbb{P}(blue) = 1/2$")
        
        
        






