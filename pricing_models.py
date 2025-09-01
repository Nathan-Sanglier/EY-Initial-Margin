import  numpy           as      np
from    math            import  sqrt, exp
from    scipy.signal    import  lfilter
from    abc             import  ABC, abstractmethod

class PricingModel(ABC):
    '''
    Needs to be inherited.
    Stochastic process used to describe the evolution of risk factors.
    Slightly different from a "vanilla" stochastic process, as it calibrates the process parameters to market data when possible.

    Attributes:
    - spot: initial value of the risk factors
    - time_grid: time grid for the simulation of risk factors

    Methods:
    - set_time_grid: sets the time grid for the simulation, to call before generating paths
    - set_spot: sets the initial value of the risk factors
    '''
    def __init__(self, spot):
        self.spot = spot

    @abstractmethod
    def generate_paths(self, num_paths):
        pass

    def set_time_grid(self, time_grid):
        self.time_grid = time_grid

    def set_spot(self, spot):
        self.spot = spot


class BlackScholes(PricingModel):
    '''
    Unidimensional Black-Scholes process.

    Attributes:
    - volatility: volatility of the risk factor
    - riskfree_rate: risk-free interest rate
    '''
    def __init__(self, volatility, riskfree_rate, spot):
        super().__init__(spot)
        self.volatility     = volatility
        self.riskfree_rate  = riskfree_rate

    def generate_paths(self, num_paths):
        paths           = np.zeros((num_paths, self.time_grid.num_steps + 1))
        paths[:, 0]     = self.spot
        Z               = np.random.normal(0, 1, size=(num_paths, self.time_grid.num_steps))
        increments      = np.exp((self.riskfree_rate - 0.5 * self.volatility ** 2) * self.time_grid.timestep + self.volatility * sqrt(self.time_grid.timestep) * Z)
        paths[:, 1:]    = self.spot * np.cumprod(increments, axis=1)
        return paths
    

class YieldCurve:
    '''
    Specific ZC yield curve defined as y(T) = C1 + C2 * exp(C3 * T). Needed for interest rates models.

    Attributes:
    - C1, C2, C3: coefficients of the yield curve

    Methods:
    - zero_coupon_yield: ZC yield y(T)
    - d1_zero_coupon_yield: first derivative of ZC yield y'(T)
    - d2_zero_coupon_yield: second derivative of ZC yield y''(T)
    - instantaneous_forward_rate: observed instantaneous forward rate f(0, T) = y'(T) * T + y(T)
    - d1_instantaneous_forward_rate: first derivative of observed instantaneous forward rate wrt T f'(0, T)
    - d2_instantaneous_forward_rate: second derivative of observed instantaneous forward rate wrt T f''(0, T)
    - zero_coupon_price: observed ZC price P(0, T) = exp(-y(T) * T)
    '''
    def __init__(self, C1, C2, C3):
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
    
    def zero_coupon_yield(self, maturity):
        return self.C1 + self.C2 * np.exp(self.C3 * maturity)
    
    def d1_zero_coupon_yield(self, maturity):
        return self.C2 * self.C3 * np.exp(self.C3*maturity)
    
    def d2_zero_coupon_yield(self, maturity):
        return self.C3 * self.d1_zero_coupon_yield(maturity)
    
    def instantaneous_forward_rate(self, maturity):
        return  self.d1_zero_coupon_yield(maturity)*maturity + self.zero_coupon_yield(maturity)
    
    def d1_instantaneous_forward_rate(self, maturity):
        return self.d2_zero_coupon_yield(maturity)*maturity + 2*self.d1_zero_coupon_yield(maturity)
    
    def zero_coupon_price(self, maturity):
        return np.exp(-self.zero_coupon_yield(maturity) * maturity)


class OneFactorHullWhite(PricingModel):
    '''
    Unidimensional one-factor Hull-White model calibrated to the initial ZC curve.

    Attributes:
    - volatility: volatility of the short rate
    - mean_reversion_speed: speed of mean reversion
    - yield_curve: yield curve object
    - longterm_mean: long-term mean function given by calibration to the initial ZC curve

    Methods:
    - beta: function appearing in the mean of the short rate at time t given the filtration at time s<t
    '''
    def __init__(self, volatility, mean_reversion_speed, yield_curve):
        super().__init__(yield_curve.zero_coupon_yield(0))
        self.volatility             = volatility
        self.mean_reversion_speed   = mean_reversion_speed
        self.yield_curve            = yield_curve
        self.longterm_mean          = lambda t: yield_curve.d1_instantaneous_forward_rate(t) + mean_reversion_speed*yield_curve.instantaneous_forward_rate(t) + volatility**2/(2*mean_reversion_speed) * (1 - np.exp(-2*mean_reversion_speed*t))

    def beta(self, t):
        return self.yield_curve.zero_coupon_yield(t) + self.volatility**2/(2*self.mean_reversion_speed**2) * (1 - np.exp(-self.mean_reversion_speed*t))**2
    
    def generate_paths(self, num_paths):
        # We use $X_{t_i} - e^{-kh} X_{t_{i-1}} = \beta(t_i) - \beta(t_{i-1}) e^{-kh} + \sqrt{\frac{\sigma^2}{2k} (1 - e^{-2kh})} Z_i$ where $Z_i$ standard Gaussian.
        beta_vals   = self.beta(self.time_grid.grid)
        m_vals      = beta_vals[1:] - beta_vals[:-1]*exp(-self.mean_reversion_speed*self.time_grid.timestep)
        Z           = np.random.normal(0, 1, size=(num_paths, self.time_grid.num_steps))
        X           = m_vals[None, :] + np.sqrt(self.volatility**2/(2*self.mean_reversion_speed)*(1-np.exp(-2*self.mean_reversion_speed*self.time_grid.timestep))) * Z
        y, _        = lfilter(b=[1.0], a=[1.0, -exp(-self.mean_reversion_speed*self.time_grid.timestep)], x=X, axis=1, zi=np.full((num_paths, 1), self.spot))
        paths       = np.concatenate([self.spot * np.ones((num_paths, 1)), y], axis=1)
        return paths