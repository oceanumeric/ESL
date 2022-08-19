"""
Linear Methods for Regression
"""
import math
import time
import itertools
import collections
from turtle import color
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context='notebook', style='ticks')


class LinearRegression:
    
    def __init__(self) -> None:
        # load the dataset
        url = "https://hastie.su.domains/ElemStatLearn/datasets/prostate.data"
        self.data = pd.read_csv(url, delimiter="\t").iloc[:, 1:]
        self.feature_idx = self.data.columns[:-2]  # feature names index 
        self.y = self.data['lpsa']
        self.selection_performance = {}
        
        
    def normalize_dataset(self) -> None:
        # normalize the dataset
        features = self.data[self.feature_idx]
        self.normalize_features = sp.stats.zscore(features)
        
    def split_dataset(self) -> None:
        # split the normalized dataset: train and test
        self.x_train = self.normalize_features[self.data['train'] == 'T']
        self.x_test = self.normalize_features[self.data['train'] == 'F']
        self.y_train = self.y[self.data['train'] == 'T'].values.reshape(-1, 1)
        self.y_test = self.y[self.data['train'] == 'F'].values.reshape(-1, 1)
        for i in [self.x_train, self.x_test, self.y_train, self.y_test]:
            print("The shape of splitted dataset is:", i.shape)
        
    @classmethod
    def fit_with_ols(cls, X :np.ndarray, Y :np.ndarray) -> np.ndarray:
        """
        Input:
            - X 
            - Y 
        
        Return: 
            - beta
            - fitted_y
            - sum_rss
        """
        # add constant values
        cnst = np.ones((X.shape[0], 1))
        X = np.hstack((cnst, X))
        # beta.shape = 9 x 1 
        beta = np.linalg.inv(X.T @ X) @ X.T @ Y
        fitted_y = X @ beta
        sum_rss = np.power(Y - fitted_y, 2).sum()
        return beta, fitted_y, sum_rss
    
    def select_subsets(self, k : int) -> None:
        """
        Select all subsets from the set of features \\
        It uses all the sample \\
        Return: \\
            - a list of features
            - e.g. [(1, 2, 6), [1, 3, 8], ..] when k = 3
        """
        subset_idx = itertools.combinations(self.feature_idx, k)
        return list(subset_idx)
    
    def fit_with_subsets(self) -> list:
        """
        fit OLS model for all subsets of features (variables) \\
        Using the training dataset 
        Return:
            - a dict {'k': , 'features': , 'beta':, 'res': }
        """
        res = []
        # fit with zero subsets
        x = np.ones((self.x_train.shape[0], 1))
        beta = np.linalg.inv(x.T @ x) @ x.T @ self.y_train
        rss = np.power(self.y_train - x @ beta, 2).sum()
        res.append(
            {
                'k': 0,
                'features': 'constant',
                'beta': beta,
                'rss': rss
            }
        ) 
        for i in range(1, self.x_train.shape[1]+1):
            # i = number of combinations n=8 choose k 
            features = self.select_subsets(k=i)
            for ss in features:
                x_subset = self.x_train[list(ss)]
                beta, _, rss = self.fit_with_ols(x_subset, self.y_train)
                res.append(
                    {
                        'k': i,
                        'features': ss,
                        'beta': beta,
                        'rss': rss
                    }
                )
        
        self.subsets_ols = res
        
    def plot_figure_3_5(self):
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        res = [(res['k'], res['rss']) for res in self.subsets_ols]
        res = np.array(res)
        ax.scatter(res[:, 0], res[:, 1], color='grey', s=6)
        ax.set_ylim(0, 100)
        ax.set_xticks(range(9))
        ax.set_xlabel('Subset Size k')
        ax.set_ylabel('Residual Sum-of-Squares')
        # plot the linked line
        res = pd.DataFrame(res)
        res.columns = ['k', 'value']
        ll = res.groupby('k')['value'].min()
        ax.plot(ll, 'o--', color='#FC0D1B', markersize=4)
        
    def find_best_subset(self):
        """
        Use brutal force to find the best subset and print out the results
        """
        tic = time.time()
        self.fit_with_subsets()
        toc = time.time()
        all_results = pd.DataFrame(self.subsets_ols)
        print("The brutal force takes", round(toc-tic, 3), "seconds.")
        return all_results.loc[all_results['rss'].argmin()]
    
        
    def forward_stepwise(self, using_test_data=False) -> None:
        """
        Find the best subset based on finding the minimal Rss \\
        Use the same training dataset \\
        Algorithm:
            - it starts from the subset k = 1 (it assumes rss is bigger when \\
                k = 0)
        """
        # initialize the list, start from the intercept
        model_info = pd.DataFrame(columns=['features', 'rss'])
        tic = time.time()
        subset_list = []
        for i in range(1, self.x_train.shape[1]+1):
            remaining_var = [p for p in self.x_train.columns 
                             if p not in subset_list]
            results = []
            for p in remaining_var:
                # add new features in the forward manner  
                forward_features = subset_list + [p]
                x = self.x_train[forward_features]
                beta, _, rss = self.fit_with_ols(x, self.y_train)
                if using_test_data:
                    x = self.x_test[forward_features]
                    cnst = np.ones((x.shape[0], 1))
                    x = np.hstack((cnst, x))
                    fitted_y = x @ beta
                    rss = np.power(fitted_y - self.y_test, 2).sum()
                    res = {'features': p, 'beta': beta, 'rss': rss}
                else:
                    res = {'features': p, 'beta': beta, 'rss': rss}
                results.append(res)
            models = pd.DataFrame(results)
            # find the best one within this selection
            best_model = models.loc[models['rss'].argmin()]
            # now append it to model_info
            model_info.loc[i] = best_model
            # update the subset_list
            subset_list.append(model_info.loc[i]['features'])
        toc = time.time()
        print("Forward Selection takes:", round(toc-tic, 3), "seconds.")
        fig, ax = plt.subplots(1,1, figsize=(7, 5))
        ax.plot(model_info['features'], model_info['rss'], 'o--',
                color='#E49E25')
        ax.set_xlabel("variables")
        ax.set_ylabel("residual sum of squares (RSS)")
        for a, b in zip(model_info['features'], model_info['rss']):
            ax.annotate(f"{b:.3f}", (a, b),
                        textcoords='offset points',
                        # text position, lift annotation 
                        xytext=(0, 7))
        if using_test_data:
            ax.set_title("Forward-stepwise selection with the test dataset")
            ax.axvline('gleason', ymax=0.15, color='grey', linestyle=':')
            ax.axvline('lweight', ymax=0.3, color='grey', linestyle=':')
            ax.annotate('best subset area', ('gleason', 12.2),
                        textcoords='offset points',
                        # text position, move left
                        xytext=(-2, 0))
        else:
            ax.set_title("Forward-stepwise selection with the training dataset")
            
    @staticmethod
    def index_tenfold(n:int) ->np.ndarray:
        """Produce index array for tenfold CV with dataframe length n."""
        original_indices = np.arange(n)
        tenfold_indices = np.zeros(n)

        div, mod = divmod(n, 10)
        unit_sizes = [div for _ in range(10)]
        for i in range(mod):
            unit_sizes[i] += 1

        for k, unit_size in enumerate(unit_sizes):
            tenfold = np.random.choice(original_indices, unit_size,
                                    replace=False)
            tenfold_indices[tenfold] = k
            original_indices = np.delete(
                original_indices,
                [np.argwhere(original_indices == val) for val in tenfold],
            )
            # print(tenfold, original_indices)
        return tenfold_indices 
    
    @classmethod
    def __edf(cls, singular_sigma :np.ndarray, interval :int) -> np.ndarray:
            """
            Calculate 
            """
            p = singular_sigma.shape[0]
            edfs = np.linspace(0.5, p-0.5, (p-1)*interval+1)
            threshold = 1e-3
            lambdas = []
            for edf in edfs:
                # Newton-Raphson
                lambda0 = (p-edf)/edf
                lambda1 = 1e6
                diff = lambda1 - lambda0
                while diff > threshold:
                    num = (singular_sigma/(singular_sigma+lambda0)).sum()-edf
                    denom = (singular_sigma/((singular_sigma+lambda0)**2)).sum()
                    lambda1 = lambda0 + num/denom
                    diff = lambda1 - lambda0
                    lambda0 = lambda1
                lambdas.append(lambda1)
            lambdas.append(0)
            
            edfs = np.concatenate(([0], edfs, [p]))
            return edfs, np.array(lambdas)
    
    def plot_figure_3_8(self):
        """
        Plot figure 3.8 
        """
        intercept = self.y_train.mean()  # intercept is just mean now 
        # centering y
        y_train_centered = (self.y_train-intercept)
        
        # get sample of train and test dataset
        var_size = self.x_train.shape[1]
        
        u, s, v = np.linalg.svd(self.x_train, full_matrices=False)
        s_squared = s**2
        edfs, lambdas = self.__edf(s_squared, 4)
        # initialize the beta_ridge 
        beta_ridge = [np.zeros(var_size)]
        for lam in lambdas:
            Sigma = np.diag(s/(s_squared+lam))
            # X = U Sigma V^T 
            beta_estimation = v.T @ Sigma @ u.T @ y_train_centered
            beta_ridge.append(beta_estimation.flatten())
        beta_ridge = np.array(beta_ridge)
        
        fig, ax = plt.subplots(1, 1, figsize=(7, 8))
        ax.plot(edfs, beta_ridge, 'o-', markersize=2, color='#0B24FB', alpha=0.7)
        ax.set_xlabel(r'$df(\lambda)$')
        ax.set_ylabel("Coefficients")
        ax.set_title("Profiles of ridge coefficients for the prostate cancer example")
        ax.axvline(x=5, linestyle='--', color='#FC0D1B', alpha=0.5)
        ax.axhline(y=0, linestyle='--', color='k', alpha=0.8)
        # add annotation
        for idx, v in enumerate(self.x_train.columns):
            ax.text(8.1, beta_ridge[-1, idx], v, size='small',
                    horizontalalignment='left')
        ax.set_xlim(-0.5, 9)
        
    def cross_validation_ridge(self):
        """
        10-fold cross validation: use the full dataset 
        """
        # centering y 
        intercept = self.y_train.mean()  # intercept is just mean now 
        # centering y
        y_train_centered = (self.y_train-intercept)
        cv10_indices = LinearRegression.index_tenfold(self.x_train.shape[0])
        cv_beta = collections.defaultdict(list)
        cv_rss = collections.defaultdict(list)
        for cv_idx in range(10):
            # create a mask 
            cv_mask = cv10_indices != cv_idx  
            one_fold_size = (cv_mask == True).size 
            cv_x = self.x_train[cv_mask]
            cv_y = y_train_centered[cv_mask]
            # calculate the intercept
            intercept = cv_y.mean()
            rss0 = ((cv_y - intercept)**2).sum()/one_fold_size
            cv_rss[0].append(rss0)
            # singular decomposition
            u, s, vt = np.linalg.svd(cv_x, full_matrices=False)
            edfs, lambdas = self.__edf(s**2, 2)
            
            for edf, lamb in zip(edfs[1:], lambdas):
                mat_diag = np.diag(s/(s**2+lamb))
                beta_ridge = vt.T @ mat_diag @ u.T @ cv_y
                cv_beta[edf].append(beta_ridge)
                cv_y_fitted = cv_x @ beta_ridge
                cv_rss[edf].append(((cv_y-cv_y_fitted)**2).sum()/one_fold_size)
                
        cv_rss_mean = [np.array(rss).mean() for _, rss in cv_rss.items()]
        cv_rss_std = [np.array(rss).std() for _, rss in cv_rss.items()]
    
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot(edfs, cv_rss_mean, 'o-', color='C1')
        for idx, (ave, std) in enumerate(zip(cv_rss_mean, cv_rss_std)):
            ax.plot([idx/2, idx/2], [ave-std, ave+std], color='#5BB5E7')
            ax.plot([idx/2-0.1, idx/2+0.1], [ave-std, ave-std], color='#5BB5E7',
                    linewidth=1)
            ax.plot([idx/2-0.1, idx/2+0.1], [ave+std, ave+std], color='#5BB5E7',
                    linewidth=1)
        ax.set_xlabel("Degrees of Freedom")
        ax.set_ylabel("CV Error")
        ax.axvline(x=5, linestyle='--', alpha=0.5, color='#9F32EC')
        ax.axhline(cv_rss_mean[10], linestyle='--', alpha=0.5, color='#9F32EC')
    
        lambda_cv = lambdas[np.where(edfs==5)[0]-1]
        u, s, vt = np.linalg.svd(cv_x, full_matrices=False)
        
        mat_diag = np.diag(s/(s**2+lambda_cv))
        beta_ridge = vt.T @ mat_diag @ u.T @ cv_y
        
        y_test_fitted = self.x_test @ beta_ridge
        y_test_mena = self.y_test.mean()
        test_error = ((self.y_test - y_test_mena - y_test_fitted)**2).sum()/self.y_test.shape[0]
        
        col1 = []
        col1.append(round(self.y_train.mean(), 3))
        col1 += [round(x[0],3) for x in beta_ridge]
        col1 += [round(list(test_error)[0], 3)]
        col0 = ['Intercept'] + list(self.feature_idx)+['Test Error']
        dt = {'variable': col0, 'Value': col1}
        return pd.DataFrame(dt)
        
            
        
            
            
            
           
    
            
            
            
            
class SVD:
    """
    A class to illustrate svd decomposition
    """
    
    def __init__(self) -> None:
        self.svd = None
    
    @staticmethod
    def plotVectors(ax, vecs :np.ndarray, cols :np.ndarray, alpha: float=1.0):
        """
        Plot set of vectors.

        Parameters
        ----------
        ax: axes 
        vecs : array-like
            Coordinates of the vectors to plot. Each vectors is in an array. For
            instance: [[1, 3], [2, 2]] can be used to plot 2 vectors.
        cols : array-like
            Colors of the vectors. For instance: ['red', 'blue'] will display the
            first vector in red and the second in blue.
        alpha : float
            Opacity of vectors

        Returns:

        fig : instance of matplotlib.figure.Figure
            The figure of the vectors
        """
        ax.axvline(x=0, color='#A9A9A9', zorder=0)
        ax.axhline(y=0, color='#A9A9A9', zorder=0)

        for i in range(len(vecs)):
            x = np.concatenate([[0,0],vecs[i]])
            ax.quiver([x[0]],
                    [x[1]],
                    [x[2]],
                    [x[3]],
                    angles='xy', scale_units='xy', scale=1, color=cols[i],
                    alpha=alpha)
            
    @staticmethod
    def matrixToPlot(ax, matrix, vectorsCol=['#FF9A13', '#1190FF']):
        """
        Modify the unit circle and basis vector by applying a matrix.
        Visualize the effect of the matrix in 2D.

        Parameters
        ----------
        matrix : array-like
            2D matrix to apply to the unit circle.
        vectorsCol : HEX color code
            Color of the basis vectors

        Returns:

        fig : instance of matplotlib.figure.Figure
            The figure containing modified unit circle and basis vectors.
        """
        # Unit circle
        x = np.linspace(-1, 1, 100000)
        y = np.sqrt(1-(x**2))

        # Modified unit circle (separate negative and positive parts)
        x1 = matrix[0,0]*x + matrix[0,1]*y
        y1 = matrix[1,0]*x + matrix[1,1]*y
        x1_neg = matrix[0,0]*x - matrix[0,1]*y
        y1_neg = matrix[1,0]*x - matrix[1,1]*y

        # Vectors
        u1 = [matrix[0,0],matrix[1,0]]
        v1 = [matrix[0,1],matrix[1,1]]

        SVD.plotVectors(ax, [u1, v1], cols=[vectorsCol[0], vectorsCol[1]])

        ax.plot(x1, y1, '#18A75A', alpha=0.5)
        ax.plot(x1_neg, y1_neg, '#18A75A', alpha=0.5)
            
            
            
        
            
            
    
        