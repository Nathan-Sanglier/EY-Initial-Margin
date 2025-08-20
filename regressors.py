import  numpy                                       as      np
import  statsmodels.api                             as      sm
from    utils                                       import  get_basis
from    sklearn.linear_model                        import  LinearRegression, Ridge, HuberRegressor
from    statsmodels.nonparametric.kernel_regression import  KernelReg
from    sklearn.neighbors                           import  KNeighborsRegressor

class RawMomentsRegressor:
    '''
    Models for estimating conditional raw moments E[Y^j | X] by regression.

    Attributes:
    - setting: Configuration settings for the regression model.
        -> 'method':
            'LR'    = linear regression
            'GLM'   = generalized linear model (only for even moments, otherwise LR)
            'HR'    = Huber regression
            'KR'    = kernel regression
        -> 'ridge': Ridge regularization strength for LR and GLM
        -> 'epsilon': HR loss parameter
        -> 'basis_type': Type of basis functions used for parametric regression (LR, GLM, HR)
            'laguerre':     Laguerre polynomials
            'canonical':    monomials 1, X, X^2, ...
        -> 'order': Truncature order of the basis functions for parametric regression
        -> 'bandwidth': Bandwidth for KR
            'silverman': Silverman's rule of thumb
            'cv_ls': Cross-validated least squares
        -> 'regress mean': if False, E[Y | X] is set to zero
    - order_moms: The maximal order of the moment to be estimated.
    - models: List of fitted models for each moment.

    Methods:
    - fit: Fit the models to the training data.
    - predict: Predict the moments for the test data.
    '''
    def __init__(self, setting, order_moms):
        self.setting    = setting
        self.order_moms = order_moms

    def fit(self, x_train, y_train):
        method          = self.setting['method']
        ind_start       = 1 if self.setting['regress_mean']==True else 2
        self.models     = []

        if (method == 'LR') or (method == 'HR') or (method == 'GLM'):
            X_train = get_basis(x_train, self.setting['basis_type'], self.setting['order'])
            for j in range(ind_start, self.order_moms+1):
                if (j%2 != 0) and (method == 'GLM'):
                    model = LinearRegression(fit_intercept=False)
                    model.fit(X_train, y_train[:, j-1])
                elif (method == 'LR') and (self.setting['ridge']>0):
                    model = Ridge(alpha=self.setting['ridge'], fit_intercept=False)
                elif method == 'LR':
                    model = LinearRegression(fit_intercept=False)
                elif method == 'HR':
                    model = HuberRegressor(epsilon=self.setting['epsilon'], alpha=self.setting['ridge'] ,fit_intercept=False)
                elif (method == 'GLM') and (self.setting['ridge']>0):
                    model = sm.GLM(y_train[:, j-1], X_train, family=sm.families.Gaussian(link=sm.families.links.Log())).fit_regularized(method='elastic_net', alpha=self.setting['ridge'], L1_wt=0)
                elif method == 'GLM':
                    model = sm.GLM(y_train[:, j-1], X_train, family=sm.families.Gaussian(link=sm.families.links.Log())).fit()
                
                if method != 'GLM':
                    model.fit(X_train, y_train[:, j-1])

                self.models.append(model)
        
        elif method =='KR':
            X_train = x_train.reshape(-1, 1)
            if self.setting['bandwidth'] == 'silverman':
                iqr = np.quantile(x_train, 0.75) - np.quantile(x_train, 0.25)
                bdw = [0.9*min(np.std(x_train), iqr/1.34) * len(x_train)**(-1/5)]
            elif self.setting['bandwidth'] == 'cv_ls':
                bdw = 'cv_ls'
            else:
                bdw = [self.setting['bandwidth']]
            for j in range(ind_start, self.order_moms+1):
                model = KernelReg(endog=y_train[:, j-1], exog=X_train, var_type='c', bw=bdw)
                self.models.append(model)

    def predict(self, x_test):
        method      = self.setting['method']
        ind_start   = 1 if self.setting['regress_mean']==True else 2
        yhat_test   = np.zeros((len(x_test), self.order_moms))
        ind_model   = 0
        for j in range(ind_start, self.order_moms+1):
            model = self.models[ind_model]
            if (method == 'LR') or (method == 'HR') or (method == 'GLM'):
                X_test = get_basis(x_test, self.setting['basis_type'], self.setting['order'])
                yhat_test[:, j-1] = model.predict(X_test)
            elif method =='KR':
                X_test              = x_test.reshape(-1, 1)
                yhat_test[:, j-1]   = model.fit(X_test)[0]
            ind_model += 1
        
        return yhat_test


class JohnsonSupportRegressor:
    '''
    Model for estimating the value-at-risk function based on the support values and the associated Johnson theoretical quantiles.

    Attributes:
    - setting: Configuration settings for the model.
        -> 'method':
            'LR'    = linear regression
            'kNN'   = k-nearest neighbors
            'KR'    = kernel regression
        -> 'basis_type': Type of basis functions used for LR
            'laguerre':     Laguerre polynomials
            'canonical':    monomials 1, X, X^2, ...
        -> 'n_neighbors': Number of neighbors for kNN
        -> 'bandwidth': Bandwidth for KR
            'silverman': Silverman's rule of thumb
            'cv_ls': Cross-validated least squares
    - model: The fitted model.
    - warning_empty: Flag indicating if there were not enough valid support values to fit the model.
    '''
    def __init__(self, setting):
        self.setting = setting

    def fit(self, x_train, y_train):
        # It may be possible that there are not enough valid support values to fit the model
        if (len(x_train) == 0) or ((self.setting['method']=='kNN') and (len(x_train)<self.setting['n_neighbors'])):
            self.warning_empty = True
            return
        self.warning_empty = False
        method = self.setting['method']
        if (method == 'LR'):
            X_train     = get_basis(x_train, self.setting['basis_type'], self.setting['order'])
            self.model  = LinearRegression(fit_intercept=False)
        elif (method == 'kNN') or (method == 'KR'):
            X_train = x_train.reshape(-1, 1)
            if method == 'kNN':
                self.model = KNeighborsRegressor(n_neighbors=self.setting['n_neighbors'], weights='distance', algorithm='auto', leaf_size=30, p=2)
            elif method == 'KR':
                if self.setting['bandwidth'] == 'silverman':
                    iqr = np.quantile(x_train, 0.75) - np.quantile(x_train, 0.25)
                    bdw = [0.9*min(np.std(x_train), iqr/1.34) * len(x_train)**(-1/5)]
                elif self.setting['bandwidth'] == 'cv_ls':
                    bdw = 'cv_ls'
                else:
                    bdw = [self.setting['bandwidth']]
                self.model = KernelReg(endog=y_train, exog=X_train, var_type='c', bw=bdw)

        if (method == 'LR') or (method == 'kNN'):
            self.model.fit(X_train, y_train)

    def predict(self, x_test):
        if self.warning_empty:
            return np.zeros_like(x_test)
        method      = self.setting['method']
        if (method == 'LR'):
            X_test = get_basis(x_test, self.setting['basis_type'], self.setting['order'])
            yhat_test = self.model.predict(X_test)
        elif (method == 'kNN') or (method == 'KR'):
            X_test = x_test.reshape(-1, 1)
            if method == 'kNN':
                yhat_test = self.model.predict(X_test)
            elif method == 'KR':
                yhat_test = self.model.fit(X_test)[0]
        return yhat_test