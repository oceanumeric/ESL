"""
Linear Methods for Regression
"""
import math
import time
import itertools
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
            
            
            
        
            
            
    
        