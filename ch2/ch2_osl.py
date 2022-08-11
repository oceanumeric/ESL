"""
Chapter 2: overview of supervised learning
"""
import random
from turtle import fillcolor 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import filterfalse, product, combinations
from scipy.stats import multivariate_normal
from matplotlib.patches import Rectangle, Circle, PathPatch
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.ticker import FormatStrFormatter



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
        fig, axes = plt.subplots(1, 1, figsize=(5, 5))
        axes.scatter(cls.orange[:, 0], cls.orange[:, 1],
                        color='#d68904', facecolor='none', s=55)
        axes.scatter(cls.blue[:, 0], cls.blue[:, 1],
                        color='#1f6f9c', facecolor='none', s=55)
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
        fig, axes = plt.subplots(1, 1, figsize=(5, 5))
        axes.scatter(self.orange[:, 0], self.orange[:, 1],
                        color='#d68904', facecolor='none', s=55)
        axes.scatter(self.blue[:, 0], self.blue[:, 1],
                        color='#1f6f9c', facecolor='none', s=55)
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
        fig, axes = plt.subplots(1, 1, figsize=(5, 5))
        axes.scatter(self.orange[:, 0], self.orange[:, 1],
                        color='#d68904', facecolor='none', s=55)
        axes.scatter(self.blue[:, 0], self.blue[:, 1],
                        color='#1f6f9c', facecolor='none', s=55)
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
        fig, axes = plt.subplots(1, 1, figsize=(5, 5))
        axes.scatter(self.orange[:, 0], self.orange[:, 1],
                        color='#d68904', facecolor='none', s=55)
        axes.scatter(self.blue[:, 0], self.blue[:, 1],
                        color='#1f6f9c', facecolor='none', s=55)
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
        
        
    def plot_sparsity(self):
        mu = 0 
        sigma = 3
        self.one_dim = np.random.normal(mu, sigma, 100)
        self.two_dim = np.random.multivariate_normal([0, 0],
                                                     np.identity(2)*sigma, 100)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].scatter(self.one_dim, [0]*self.one_dim.shape[0],
                        facecolor='none', edgecolor='r')
        axes[0].set_title("One dimension: not much sparsity")
        axes[1].scatter(self.two_dim[:, 0], self.two_dim[:, 1],
                        facecolor='none', edgecolor='g')
        axes[1].set_title("Two dimension: some sparsity")
        axes[1].set_aspect('equal', adjustable='box')

    def plot_denstity_example(self):
        x = np.linspace(0, 1, 5)
        xx = np.array([*product(x, x)])
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].plot(x, [0]*x.shape[0], 'or-')
        axes[0].set_title("Sampling of density 5 in one dimension")
        axes[0].set_aspect('equal', adjustable='box')
        axes[0].axis('off')
        axes[1].scatter(xx[:,0], xx[:, 1],
                        facecolor='g', edgecolor='g')
        axes[1].set_title("Sampling of density 5 in"\
                        "two dimension (more points are needed")
        axes[1].set_aspect('equal', adjustable='box')

    def plot_cube(self):
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(121, projection='3d')
        r = [0, 5]
        for s, e in combinations(np.array(list(product(r,r,r))), 2):
            if np.sum(np.abs(s-e)) == r[1]-r[0]:
                ax.plot3D(*zip(s,e), color="k")
        c = [0, 1]
        for s, e in combinations(np.array(list(product(c,c,c))), 2):
            if np.sum(np.abs(s-e)) == c[1]-c[0]:
                ax.plot3D(*zip(s,e), color="r")
        ax.grid(False)
        ax.view_init(17)
        ax.set_title('The share of unit cube is 0.8%')
        ax = fig.add_subplot(122)
        x = np.linspace(0, 1, 5)
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
        ax.set_aspect('equal', adjustable='box')
        ax.plot(x, [0]*x.shape[0], 'k')
        ax.plot(x, [1]*x.shape[0], 'k')
        ax.plot([1]*x.shape[0], x, 'k')
        x = np.linspace(0, 5, 10)
        xx = np.array([*product(x, x)])
        ax.scatter(xx[:,0], xx[:, 1],
                        facecolor='none', edgecolor='b')
        ax.set_title('The share of unit cube is 4% (it covers 4 points)')

    def plot_dimension_curse(self):
        def _edge_fraction(r, p):
            return np.power(r, 1/p)
        
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(121, projection='3d')
        r = [0, 5]
        for s, e in combinations(np.array(list(product(r,r,r))), 2):
            if np.sum(np.abs(s-e)) == r[1]-r[0]:
                ax.plot3D(*zip(s,e), color="grey")
        c = [0, 1]
        for s, e in combinations(np.array(list(product(c,c,c))), 2):
            if np.sum(np.abs(s-e)) == c[1]-c[0]:
                ax.plot3D(*zip(s,e), color="#d68904")
        ax.grid(False)
        ax.view_init(17)
        ax.set_title('The share of unit cube is 0.8%')
        ax = fig.add_subplot(122)
        ax.set_xlim(0, 0.8)
        ax.set_ylim(0, 1)
        fraction = np.linspace(0, 0.8, 100)
        for i in [1, 2, 3, 10]:
            ax.plot(fraction, _edge_fraction(fraction, i),
            label=f'p={i}')
        ax.legend(loc='lower right')
        ax.axvline(x=0.3, ls=':', color='k')
        ax.set_aspect(0.8, adjustable='box')
        ax.set_xlabel('fraction of data')
        ax.set_ylabel('distance of edge')
        ax.set_title('The curse of dimensionality')
        
    def plot_simulated_data_2_7_1(self):
        
        def _generate_training_data(p, n):
            """
            p - dimension
            n - sample size 
            """
            X = np.array(
                [np.random.uniform(-1, 1, p)
                 for _ in range(n)]
            )
            Y = np.apply_along_axis(np.linalg.norm, 1, X).reshape(-1, 1)
            Y = np.exp(-8*np.power(Y, 2))
            return X, Y 
        
        X, Y = _generate_training_data(1, 30) 
        # plot the function in one dimension
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        x = np.linspace(-1, 1, 1000)
        axes[0].plot(x, np.exp(-8*np.power(x, 2)), color='g')
        axes[0].scatter(X, Y, color='#1FBFC3')
        axes[0].axvline(x=0, color='k')
        axes[0].axvline(x=0.11, color='k', ls=":", ymax=0.87)
        axes[0].annotate("the nearest neighbor",
                         xy=(0.13, 0.93),)
        axes[0].annotate("is very close to 0", xy=(0.16, 0.85))
        ticks = [-1, -0.5, 0, 0.5, 1]
        axes[0].set_xticks(ticks)
        for xx in X:
            axes[0].axvline(xx, ymax=0.03, color='grey')
        axes[0].set_title("1-NN in One Dimension")
        axes[0].annotate(r"$f(x) = e^{-8||x||^2}$",
                         xy=(-1, 0.85), fontsize=14)
        axes[0].set_aspect(2)
        axes[0].set_xlabel("x")
        # plot 2 dimension
        x2, y = _generate_training_data(2, 30)
        axes[1].scatter(x2[:, 0], x2[:, 1], color='#1FBFC3')
        axes[1].set_aspect(0.9)
        axes[1].set_xticks(ticks)
        axes[1].set_yticks(ticks)
        axes[1].axvspan(-0.05, 0.05, alpha=0.3, color='#F8A331')
        axes[1].scatter(0, 0, color='k', s=15)
        circle = plt.Circle((0, 0), 0.12, color='b', alpha=0.2)
        axes[1].add_patch(circle)
        axes[1].set_title("1-NN in One vs. Two Dimensions")
        axes[1].set_xlabel("x1")
        axes[1].set_ylabel("x2")
    
    @classmethod    
    def __simulation(cls, p :int, n :int, nsim :int, model :str) -> dict:
        
        def _generate_training_data(p :int, n :int) -> np.ndarray:
            """
            p - dimension \\
            n - sample size 
            """
            X = np.array(
                [np.random.uniform(-1, 1, p)
                 for _ in range(n)]
            )
            Y = np.apply_along_axis(np.linalg.norm, 1, X).reshape(-1, 1)
            Y = np.exp(-8*np.power(Y, 2))
            return X, Y 
        
        def _generate_training_data2(p :int, n :int) -> np.ndarray:
            """
            p - dimension \\
            n - sample size 
            """
            X = np.array(
                [np.random.uniform(-1, 1, p)
                 for _ in range(n)]
            )
            # Y constant in all but one dimension
            Y = np.ones(X.shape)
            Y[:, 0] = 1/2 * np.power((X[:, 0]+1.0), 3)
            return X, Y
        
        res = {'average_distance': 0}
        estimated_y_nsim = []
        distance_nsim = []
        for _ in range(nsim):
            if model == "1":
                x, y = _generate_training_data(p, n)
            else:
                x, y = _generate_training_data2(p, n)
            # find the nearest point at [0] or [0, 0, ..,0]
            if p == 1:
                x_norm = np.abs(x)
            else:
                # 1000 times p, calculate norm arlong axis = 1
                x_norm = np.linalg.norm(x, axis=1)
            nearest_idx = x_norm.argmin()
            nearest_x, nearest_distance = x[nearest_idx], x_norm[nearest_idx]
            # estimated y conditional x
            if model == "1":
                nearest_y = np.exp(-8*np.power(np.linalg.norm(nearest_x), 2))
            else:
                # constant in all but one dimension 
                nearest_y = np.ones(nearest_x.shape)
                nearest_y[0] = 1/2 * np.power(nearest_x[0]+1.0, 3)
            estimated_y_nsim.append(nearest_y)
            distance_nsim.append(nearest_distance)
        res['average_distance'] = np.mean(distance_nsim)
        estimated_y_nsim = np.array(estimated_y_nsim)
        if model == "1":
            res['variance'] = estimated_y_nsim.var()
        else:
            res['variance'] = estimated_y_nsim[:, 0].var()
        if model == "1":
            # calculate bias f(0) = 1; bias = 1 - nearest_y
            res['squared_bias'] = np.mean(np.power((1-estimated_y_nsim),2))
        else:
            if p == 1:
                # calculate the bias f(0) = 1/2(0+1)^3 = 0.5
                res['squared_bias'] = np.mean(np.power((0.5-estimated_y_nsim), 2))
            else:
                # 1 times p dimension
                y_0 = np.ones((100, p))
                y_0[:, 0] = 0.5
                res['squared_bias'] = np.mean(np.power((y_0[0]-estimated_y_nsim[0]), 2))
                                        
                
                
        
        return res
        
    def plot_simulated_data_2_7_2(self):
        """
        we are estimation f(0), therefore the nearest
        point should have the smallest L2-norm
        """
        
        # plot the graph
        nsim = 100
        data = {p: self.__simulation(p, 1000, nsim, model="1") for p in range(1, 11)}
        dimension = list(data.keys())
        average_distance = [d['average_distance'] for p, d in data.items()]
        variance = np.array([d['variance'] for p, d in data.items()])
        squared_bias = np.array([d['squared_bias'] for p, d in data.items()])
        mse = variance + squared_bias
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].set_title('Distance to 1-NN vs. Dimension')
        axes[0].plot(dimension, average_distance, 'ro--')
        axes[0].set_xlabel('Dimension')
        axes[0].set_ylabel('Average Distance to Nearest Neighbor')

        axes[1].set_title('MSE vs. Dimension')
        axes[1].plot(dimension, mse, 'o-', label='MSE')
        axes[1].plot(dimension, variance, 'o-', label='Variance')
        axes[1].plot(dimension, squared_bias, 'o-', label='Squared Bias')
        axes[1].set_xlabel('Dimension')
        axes[1].set_ylabel('MSE')
        axes[1].legend()
        
    def plot_simulated_data_2_8(self):
        
        def _generate_training_data(p :int, n :int) -> np.ndarray:
            """
            p - dimension \\
            n - sample size 
            """
            X = np.array(
                [np.random.uniform(-1, 1, p)
                 for _ in range(n)]
            )
            Y = 1/2 * np.power((X[:, 0]+1.0), 3)
            return X, Y 
        
        X, Y = _generate_training_data(1, 30) 
        # plot the function in one dimension
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        x = np.linspace(-1, 1, 1000)
        axes[0].plot(x, 1/2 * np.power((x+1.0), 3), color='b')
        axes[0].scatter(X, Y, color='#1FBFC3')
        axes[0].axvline(x=0, color='k')
        axes[0].axvline(x=0.05, color='k', ls=":", ymax=0.2)
        axes[0].annotate("the nearest neighbor",
                         xy=(0.13, 0.83),)
        axes[0].annotate("is very close to 0", xy=(0.16, 0.65))
        ticks = [-1, -0.5, 0, 0.5, 1]
        axes[0].set_xticks(ticks)
        for xx in X:
            axes[0].axvline(xx, ymax=0.03, color='orange')
        axes[0].set_title("1-NN in One Dimension")
        axes[0].annotate(r"$f(X) = \frac{1}{2}(X_1 + 1)^3$",
                         xy=(-1, 3.0), fontsize=14)
        axes[0].set_xlabel("x")
        
        # plot the mean squared error
        nsim = 100
        data = {p: self.__simulation(p, 1000, nsim, model="2") for p in range(1, 11)}
        dimension = list(data.keys())
        average_distance = [d['average_distance'] for p, d in data.items()]
        variance = np.array([d['variance'] for p, d in data.items()])
        squared_bias = np.array([d['squared_bias'] for p, d in data.items()])
        mse = variance + squared_bias
        
        axes[1].set_title('MSE vs. Dimension')
        axes[1].plot(dimension, mse, 'o-', label='MSE')
        axes[1].plot(dimension, variance, 'o-', label='Variance')
        axes[1].plot(dimension, squared_bias, 'o-', label='Squared Bias')
        axes[1].set_xlabel('Dimension')
        axes[1].set_ylabel('MSE')
        axes[1].legend()
        ticks = [2, 4, 6, 8, 10]
        for xticks in ticks:
            axes[1].axvline(xticks, ymax=0.02, color='grey')
        axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.subplots_adjust(wspace=0.3)
        

            
            
            
 

        



        
        
        
        






