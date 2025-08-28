import  torch
import  torch.nn                as      nn
import  numpy                   as      np
from    abc                     import  ABC, abstractmethod
from    utils                   import  get_mtmdiff, get_centred_moms, get_quantile_normal, get_skewkurt, moment_matching_johnson, percentile_matching_johnson, get_quantile_johnson, get_mtmdiff_nmc
from    regressors              import  RawMomentsRegressor, JohnsonSupportRegressor
from    torch.utils.data        import  DataLoader, TensorDataset
from    sklearn.preprocessing   import  StandardScaler


class ForwardInitialMarginModel(ABC):
    '''
    Needs to be inherited.
    Model for Forward IM that takes the form of a list of functions at each timestep.

    Attributes:
    - level: The confidence level for the value-at-risk.
    - mpor: The margin period of risk.
    - time_grid: The time grid on which we estimate initial margin.

    Methods:
    - generate_paths: Generate paths for the initial margin based on the portfolio price paths and potentially the risk factors paths.
    - generate_im_profile: Generate the initial margin profile based on the paths, ie. empirical mean and quantiles at 5% and 95% of estimated IM at each timestep of the MC simulation.
    - compute_mse: Compute the mean squared error between the estimated initial margin and the groundtruth initial margin (either given by analytical formula or nested MC)
    '''
    def __init__(self, level, mpor, time_grid):
        self.level      = level
        self.mpor       = mpor
        self.time_grid  = time_grid

    def generate_im_profile(self, initial_margin_hat_paths):
        # We use np.nanquantile (which enables to discard NaN values) as our estimation of IM may return NaN values in some failing cases (eg. GLSMC when stdev < 0)
        return np.nanquantile(initial_margin_hat_paths, 0.05, axis=0), np.nanmean(initial_margin_hat_paths, axis=0), np.nanquantile(initial_margin_hat_paths, 0.95, axis=0)
    
    @abstractmethod
    def plot_im_profile(self, im_profile_paths, ax, *args):
        pass
    
    def compute_mse(self, initial_margin_hat_paths, initial_margin_paths):
        return np.nanmean((initial_margin_hat_paths - initial_margin_paths)**2)
    
    @abstractmethod
    def generate_paths(self, *args):
        pass


class GaussianLeastSquaresMonteCarlo(ForwardInitialMarginModel):
    '''
    Gaussian least-squares Monte-Carlo (GLSMC).

    Attributes:
    - moments_regression_setting: Configuration settings for the estimation of raw moments by regression.
    - moments_regressors: List of conditional raw moments functions for each time step.
    '''
    def __init__(self, level, mpor, time_grid, moments_regression_setting):
        super().__init__(level, mpor, time_grid)
        self.moments_regression_setting = moments_regression_setting

    def fit(self, mtm_train_paths):
        mtmdiff_train_paths     = get_mtmdiff(mtm_train_paths, self.mpor, self.time_grid)
        self.moments_regressors = []
        for i in range(len(self.time_grid.grid)):
            x_train             = mtm_train_paths[:, i]
            y_train             = mtmdiff_train_paths[:, i].reshape(-1, 1)**np.arange(1, 3)
            # If we are at maturity, we know that IM is zero as we look for VaR(V_T - V_T | F_T) = 0, so no need to train a model
            if (i<len(self.time_grid.grid)-1) or (np.any(mtmdiff_train_paths[:, -1])):
                moments_regressor   = RawMomentsRegressor(self.moments_regression_setting, 2)
                moments_regressor.fit(x_train, y_train)
                self.moments_regressors.append(moments_regressor)

    def generate_paths(self, mtm_test_paths):
        paths = np.zeros_like(mtm_test_paths)
        for i in range(len(self.time_grid.grid)):
            if (i<len(self.time_grid.grid)-1) or (len(self.moments_regressors)==len(self.time_grid.grid)):
                murawhat                = self.moments_regressors[i].predict(mtm_test_paths[:, i])
                muhat, maskhat          = get_centred_moms(murawhat)
                varhat                  = np.nan * np.ones_like(mtm_test_paths[:, i])
                varhat[maskhat[:, 0]]   = get_quantile_normal(murawhat[maskhat[:, 0], 0], muhat[maskhat[:, 0], 0], self.level).reshape(-1)
                paths[:, i]             = np.maximum(varhat, 0)

        return paths
    
    def plot_im_profile(self, im_profile_paths, ax):
        ax.plot(self.time_grid.grid, im_profile_paths[2], linestyle='--', color='red', label='95% CI - GLSMC')
        ax.plot(self.time_grid.grid, im_profile_paths[1], color='red', label='DIM - GLSMC')
        ax.plot(self.time_grid.grid, im_profile_paths[0], linestyle='--', color='red')


class JohnsonMonteCarlo(ForwardInitialMarginModel):
    '''
    Needs to be inherited.
    Regroups methods making the assumption of Johnson distribution for change in portfolio price over the MPOR.

    Attributes:
    - matching_setting: Configuration settings for the matching of moments, ie. the estimation of conditional raw moments up to order four by regression.
    - support_values_setting: Configuration settings for the support values.
        -> 'initial nb': Number of support values before adding finer resolution in the tails of empirical distribution.
        -> 'perc add tails': Total percentage of support values to add in the tails, ie. half in each tail of empirical distribution.
        -> 'add ends': Whether to add the min and max values of empirical distribution to the support values.
    - quantile_function_setting: Configuration settings for the JohnsonSupportRegressor.

    Methods:
    - get_percent_values: Returns the percentiles of empirical distribution associated to the support values.
    '''
    def __init__(self, level, mpor, time_grid, matching_setting, support_values_setting, quantile_function_setting):
        super().__init__(level, mpor, time_grid)
        self.matching_setting           = matching_setting
        self.support_values_setting     = support_values_setting
        self.quantile_function_setting  = quantile_function_setting

    def get_percent_values(self):
        Nq      = self.support_values_setting['initial nb']
        Nqtail  = int(self.support_values_setting['perc add tails']*Nq)//2
        q_main  = np.arange(1, Nq) / Nq
        q_ltail = np.linspace(0, 0.01, Nqtail+2)[1:-1]
        q_utail = np.linspace(0.99, 1, Nqtail+2)[1:-1]
        q_all   = np.sort(np.unique(np.concatenate([q_ltail, q_main, q_utail])))
        if self.support_values_setting['add ends']:
            q_all = np.concatenate([[0], q_all, [1]])
        q_all = np.unique(q_all)
        return q_all
    
    def generate_paths(self, mtm_test_paths):
        paths = np.zeros_like(mtm_test_paths)
        for i in range(len(self.time_grid.grid)):
            if (i<len(self.time_grid.grid)-1) or (len(self.quantile_functions)==len(self.time_grid.grid)):
                varhat      = self.quantile_functions[i].predict(mtm_test_paths[:, i])
                paths[:, i] = np.maximum(varhat, 0)
        return paths


class JohnsonLeastSquaresMonteCarlo(JohnsonMonteCarlo):
    '''
    Johnson least-squares Monte-Carlo (JLSMC).

    Attributes:
    - quantile_functions: List of quantile functions, ie. value-at-risk functions determined using support values, for each time step.
    '''
    def __init__(self, level, mpor, time_grid, moments_matching_setting, support_values_setting, quantile_function_setting):
        super().__init__(level, mpor, time_grid, moments_matching_setting, support_values_setting, quantile_function_setting)

    def fit(self, mtm_train_paths):
        self.quantile_functions = []
        mtmdiff_train_paths     = get_mtmdiff(mtm_train_paths, self.mpor, self.time_grid)
        q_all                   = self.get_percent_values()
        for i in range(len(self.time_grid.grid)):
            if (i<len(self.time_grid.grid)-1) or (np.any(mtmdiff_train_paths[:, -1])):
                x_train             = mtm_train_paths[:, i]
                y_train             = mtmdiff_train_paths[:, i].reshape(-1, 1)**np.arange(1, 5)
                moments_regressor   = RawMomentsRegressor(self.matching_setting, 4)
                moments_regressor.fit(x_train, y_train)
                x_supp                      = np.quantile(x_train, q_all, method='inverted_cdf')
                murawhat_supp               = moments_regressor.predict(x_supp)
                muhat_supp, maskhat_supp    = get_centred_moms(murawhat_supp)
                skewhat_supp, kurthat_supp, _, _, mask_tothat_supp  = get_skewkurt(muhat_supp, maskhat_supp)
                jparamshat_supp, jtypehat_supp, mask_hat_supp       = moment_matching_johnson(murawhat_supp[:, 0], muhat_supp[:, 0], skewhat_supp, kurthat_supp, mask_tothat_supp, False)
                jparamshat_supp, jtypehat_supp, x_supp              = jparamshat_supp[mask_hat_supp], jtypehat_supp[mask_hat_supp], x_supp[mask_hat_supp]
                quanthat_supp, mask_hat_supp_johnson                = get_quantile_johnson(jparamshat_supp, jtypehat_supp, np.array([self.level]))
                x_supp, quanthat_supp                               = x_supp[mask_hat_supp_johnson.reshape(-1)], quanthat_supp[mask_hat_supp_johnson]
                quantile_func = JohnsonSupportRegressor(self.quantile_function_setting)
                quantile_func.fit(x_supp, quanthat_supp)
                self.quantile_functions.append(quantile_func)

    def plot_im_profile(self, im_profile_paths, ax):
        ax.plot(self.time_grid.grid, im_profile_paths[2], linestyle='--', color='blue', label='95% CI - JLSMC')
        ax.plot(self.time_grid.grid, im_profile_paths[1], color='blue', label='DIM - JLSMC')
        ax.plot(self.time_grid.grid, im_profile_paths[0], linestyle='--', color='blue')
    

class JohnsonPercentileMatchingMonteCarlo(JohnsonMonteCarlo):
    '''
    Johnson percentile matching Monte-Carlo (JPMMC).

    Attributes:
    - quantile_functions: List of quantile functions, ie. value-at-risk functions determined using support values, for each time step.
    '''
    def __init__(self, level, mpor, time_grid, moments_percentile_setting, support_values_setting, quantile_function_setting):
        super().__init__(level, mpor, time_grid, moments_percentile_setting, support_values_setting, quantile_function_setting)

    def fit(self, risk_factors_train_paths, mtm_train_paths, pricing_engine):
        self.quantile_functions = []
        q_all                   = self.get_percent_values()
        for i in range(len(self.time_grid.grid)):
            if (i<len(self.time_grid.grid)-1) or (len(self.time_grid.grid) < mtm_train_paths.shape[1]):
                xx_train    = risk_factors_train_paths[:, i]
                x_train     = mtm_train_paths[:, i]
                x_supp      = np.quantile(x_train, q_all, method='inverted_cdf')
                jparamshat_supp, jtypehat_supp, mask_hat_supp   = percentile_matching_johnson(self.time_grid.grid[i], x_supp, xx_train, x_train, pricing_engine, self.mpor, self.matching_setting['Nnmc'], self.matching_setting['z'])                
                jparamshat_supp, jtypehat_supp, x_supp          = jparamshat_supp[mask_hat_supp], jtypehat_supp[mask_hat_supp], x_supp[mask_hat_supp]
                quanthat_supp, mask_hat_supp_johnson            = get_quantile_johnson(jparamshat_supp, jtypehat_supp, np.array([self.level]))
                x_supp, quanthat_supp                           = x_supp[mask_hat_supp_johnson.reshape(-1)], quanthat_supp[mask_hat_supp_johnson]
                quantile_func = JohnsonSupportRegressor(self.quantile_function_setting)
                quantile_func.fit(x_supp, quanthat_supp)
                self.quantile_functions.append(quantile_func)

    def plot_im_profile(self, im_profile_paths, ax):
        ax.plot(self.time_grid.grid, im_profile_paths[2], linestyle='--', color='green', label='95% CI - JPMMC')
        ax.plot(self.time_grid.grid, im_profile_paths[1], color='green', label='DIM - JPMMC')
        ax.plot(self.time_grid.grid, im_profile_paths[0], linestyle='--', color='green')


class NonLinearNN(nn.Module):
    '''
    Non-linear neural network with softplus activation functions.

    Attributes:
    - net: PyTorch neural network.

    Methods:
    - forward: Forward pass through the network.
    '''
    def __init__(self, input_dim, n_neurons, n_hidden_layers):
        '''
        Arguments:
        - input_dim: Dimensionality of the input features.
        - n_neurons: Number of neurons in each hidden layer.
        - n_hidden_layers: Number of hidden layers.
        '''
        super().__init__()
        layers = []
        for i in range(n_hidden_layers):
            layers.append(nn.Linear(input_dim if i == 0 else n_neurons, n_neurons, bias=False))
            layers.append(nn.Softplus(beta=1, threshold=20))
        layers.append(nn.Linear(n_neurons, 1, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NeuralQuantileRegression(ForwardInitialMarginModel):
    '''
    Model that uses quantile regression over neural networks.

    Attributes:
    - quantile_function_setting: Configuration settings for the neural network.
        -> 'batch_size'
        -> 'n_neurons': Number of neurons in each hidden layer.
        -> 'n_hidden_layers': Number of hidden layers.
        -> 'optimizer': optimization procedure for learning.
            'sgd': Stochastic Gradient Descent
            'adam': Adam
        -> 'learning_rate': Learning rate for the optimizer.
        -> 'n_epochs': Number of training epochs.
    - quantile_functions: List of quantile functions, ie. fitted neural networks, for each time step.
    '''
    def __init__(self, level, mpor, time_grid, quantile_function_setting):
        super().__init__(level, mpor, time_grid)
        self.quantile_function_setting = quantile_function_setting

    def fit(self, risk_factors_train_paths, mtm_train_paths):
        self.quantile_functions = []
        self.scalers            = []
        self.scalers_pred       = []
        mtmdiff_train_paths = get_mtmdiff(mtm_train_paths, self.mpor, self.time_grid)
        pinball_loss = lambda yhat, y: torch.maximum(y - yhat, torch.tensor(0.0))/(1-self.level) + yhat
        for i in range(len(self.time_grid.grid)-1, -1, -1): # from last to first timestep to leverage transfer learning
            if (i<len(self.time_grid.grid)-1) or (len(self.time_grid.grid) < mtm_train_paths.shape[1]):
                X           = torch.tensor(risk_factors_train_paths[:, i].reshape(-1, 1), dtype=torch.float32)
                scaler      = StandardScaler()
                X           = scaler.fit_transform(X, with_std=(X.std() > 0))
                self.scalers.insert(0, scaler)
                ones        = torch.ones(X.size(0), 1, dtype=torch.float32)
                X           = torch.cat((ones, X), dim=1)
                y           = torch.tensor(mtmdiff_train_paths[:, i].reshape(-1, 1), dtype=torch.float32)
                scaler_pred = StandardScaler()
                y           = scaler_pred.fit_transform(y, with_std=(y.std() > 0))
                self.scalers_pred.insert(0, scaler_pred)
                dataset     = TensorDataset(X, y)
                dataloader  = DataLoader(dataset, batch_size=self.quantile_function_setting['batch_size'], shuffle=True)
                model       = NonLinearNN(input_dim=2, n_neurons=self.quantile_function_setting['n_neurons'], n_hidden_layers=self.quantile_function_setting['n_hidden_layers'])
                if len(self.quantile_functions) > 0: # transfer learning: initialize weights at timestep i with weights from timestep i+1. If we are at last timestep, weights init according to default procedure.
                    model.load_state_dict(self.quantile_functions[0].state_dict())
                crit        = lambda y_hat, y: torch.mean(pinball_loss(y_hat, y))
                if self.quantile_function_setting['optimizer'] == 'sgd':
                    optim = torch.optim.SGD(model.parameters(), lr=self.quantile_function_setting['learning_rate'], momentum=0.9, weight_decay=self.quantile_function_setting['ridge'])
                elif self.quantile_function_setting['optimizer'] == 'adam':
                    optim = torch.optim.Adam(model.parameters(), lr=self.quantile_function_setting['learning_rate'], weight_decay=self.quantile_function_setting['ridge'])
                best_loss = float("inf")
                patience_counter = 0
                for epoch in range(self.quantile_function_setting['n_epochs']):
                    model.train()
                    epoch_loss = 0
                    for X_batch, y_batch in dataloader:
                        optim.zero_grad()
                        y_hat   = model(X_batch)
                        loss    = crit(y_hat, y_batch)
                        loss.backward()
                        optim.step()
                        epoch_loss += loss.item() * X_batch.size(0)
                    epoch_loss /= len(dataloader.dataset)
                    if epoch_loss < best_loss - 10**(-4):
                        best_loss = epoch_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    if patience_counter > self.quantile_function_setting['patience']:
                        break
                self.quantile_functions.insert(0, model)

    def generate_paths(self, risk_factors_test_paths):
        paths = np.zeros((risk_factors_test_paths.shape[0], self.time_grid.num_steps+1))
        for i in range(len(self.time_grid.grid)):
            if (i<len(self.time_grid.grid)-1) or (len(self.quantile_functions)==len(self.time_grid.grid)):
                X       = torch.tensor(risk_factors_test_paths[:, i].reshape(-1, 1), dtype=torch.float32)
                X       = self.scalers[i].transform(X)
                ones    = torch.ones(X.size(0), 1, dtype=torch.float32)
                X       = torch.cat((ones, X), dim=1)
                varhat  = self.scalers_pred[i].inverse_transform(self.quantile_functions[i](X).detach().numpy()).reshape(-1)
                paths[:, i] = np.maximum(varhat, 0)
        return paths
    
    def plot_im_profile(self, im_profile_paths, ax):
        ax.plot(self.time_grid.grid, im_profile_paths[2], linestyle='--', color='orange', label='95% CI - ML')
        ax.plot(self.time_grid.grid, im_profile_paths[1], color='orange', label='DIM - ML')
        ax.plot(self.time_grid.grid, im_profile_paths[0], linestyle='--', color='orange')


class AnalyticalForwardInitialMargin(ForwardInitialMarginModel):
    '''
    Model associated to an analytical formula for forward IM.
    Of course, it depends on the portfolio and risk factors used.

    Attributes:
    - pricing_engine: a pricing engine for which forward IM has an analytical formula.
    '''
    def __init__(self, level, mpor, time_grid, pricing_engine):
        super().__init__(level, mpor, time_grid)
        self.pricing_engine = pricing_engine

    def generate_paths(self, risk_factors_test_paths, mtm_test_paths):
        paths = np.zeros_like(mtm_test_paths)
        for i in range(len(self.time_grid.grid)):
            if (i<len(self.time_grid.grid)-1) or (len(self.time_grid.grid) < mtm_test_paths.shape[1]):
                delta   = min(self.time_grid.grid[-1] - self.time_grid.grid[i], self.mpor)
                varhat = self.pricing_engine.get_conditional_quantile(self.level, delta, self.time_grid.grid[i], risk_factors_test_paths[:, i])
                paths[:, i] = np.maximum(varhat - mtm_test_paths[:, i], 0)
        return paths
    
    def plot_im_profile(self, im_profile_paths, ax):
        ax.plot(self.time_grid.grid, im_profile_paths[2], linestyle='--', color='black', label='95% CI - theo')
        ax.plot(self.time_grid.grid, im_profile_paths[1], color='black', label='DIM - theo')
        ax.plot(self.time_grid.grid, im_profile_paths[0], linestyle='--', color='black')
    

class NestedMonteCarloForwardInitialMargin(ForwardInitialMarginModel):
    '''
    Model that computes forward IM using nested Monte-Carlo.

    Attributes:
    - pricing_engine: the portfolio and risk factors, needed to perform the nested MC.
    - num_inner_paths: number of inner paths for the nested MC simulations.
    '''
    def __init__(self, level, mpor, time_grid, pricing_engine, num_inner_paths):
        super().__init__(level, mpor, time_grid)
        self.pricing_engine = pricing_engine
        self.num_inner_paths = num_inner_paths

    def generate_paths(self, risk_factors_test_paths, mtm_test_paths):
        paths               = np.zeros((mtm_test_paths.shape[0], self.time_grid.num_steps+1))
        original_timegrid   = self.pricing_engine.pricing_model.time_grid
        for i in range(len(self.time_grid.grid)):
            ind = original_timegrid.find_index(self.time_grid.grid[i])
            if self.time_grid.grid[i] == 0:
                mtmdiff_nmc = get_mtmdiff_nmc(self.num_inner_paths, self.pricing_engine, risk_factors_test_paths[0:1, ind], mtm_test_paths[0:1, ind], self.mpor, self.time_grid.grid[i])
            else:
                mtmdiff_nmc = get_mtmdiff_nmc(self.num_inner_paths, self.pricing_engine, risk_factors_test_paths[:, ind], mtm_test_paths[:, ind], self.mpor, self.time_grid.grid[i])
            varhat = np.quantile(mtmdiff_nmc, self.level, axis=1)
            paths[:, i] = np.maximum(varhat, 0)
        return paths

    def plot_im_profile(self, im_profile_paths, ax):
        ax.scatter(self.time_grid.grid, im_profile_paths[2], s=30, facecolors='none', color='black', label='95% CI - NMC')
        ax.scatter(self.time_grid.grid, im_profile_paths[1], s=30, color='black', label='DIM - NMC')
        ax.scatter(self.time_grid.grid, im_profile_paths[0], s=30, facecolors='none', color='black')