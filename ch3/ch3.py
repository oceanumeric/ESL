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
    
        
    def forward_stepwise(self) -> None:
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
                forward_features = subset_list + [p]
                x = self.x_train[forward_features]
                beta, _, rss = self.fit_with_ols(x, self.y_train)
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
        print(subset_list)
                
            
            
            
        
            
            
    
        