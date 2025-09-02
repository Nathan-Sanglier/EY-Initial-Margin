import  numpy           as      np
from    math            import  exp, sqrt
from    abc             import  ABC, abstractmethod
from    scipy.stats     import  norm
from    backend.utils   import  TimeGrid
from    scipy.optimize  import  root as root_finder


class PricingEngine(ABC):
    '''
    Needs to be inherited.
    Combines the risk factors model (pricing model) with the financial asset considered.
    Enables to price a given derivative with a given model.

    Attributes:
    - maturity: The maturity date of the financial derivative.
    - pricing_model: The risk factors model.

    Methods:
    - generate_paths: Generates paths of derivative price based on paths of risk factors.
    '''
    def __init__(self, maturity, pricing_model):
        self.maturity       = maturity
        self.pricing_model  = pricing_model

    @abstractmethod
    def generate_paths(self, risk_factors_paths):
        pass


class AnalyticalPricingEngine(PricingEngine):
    '''
    Needs to be inherited.
    Special case of pricing engine where we have an analytical formula available for forward IM .

    Methods:
    - get_conditional_quantile: returns the conditional quantile of derivative price at time t+delta.
    '''
    def __init__(self, maturity, pricing_model):
        super().__init__(maturity, pricing_model)

    @abstractmethod
    def get_conditional_quantile(self, level, delta, tref, risk_factors_vals):
        '''
        Arguments:
        - level: The confidence level for value-at-risk.
        - delta: The time increment for the MPOR.
        - tref: The reference time for the MPOR.
        - risk_factors_vals: The values of the risk factors at the reference time.
        '''
        pass


class PutBlackScholes(AnalyticalPricingEngine):
    '''
    European put option where stock price follows a BlackScholes model.

    Attributes:
    - strike: The strike price of the put option.
    '''
    def __init__(self, strike, maturity, black_scholes):
        super().__init__(maturity, black_scholes)
        self.strike = strike
    
    def generate_paths(self, stock_price_paths):
        def d1(stock_price, t):
            return (np.log(stock_price/self.strike) + (self.pricing_model.riskfree_rate + 0.5*self.pricing_model.volatility**2)*(self.maturity-t)) / (self.pricing_model.volatility*np.sqrt(self.maturity-t))
        def d2(stock_price, t):
            return d1(stock_price, t) - self.pricing_model.volatility*np.sqrt(self.maturity-t)
        def put_price(stock_price, t):
            d1_val = d1(stock_price, t)
            d2_val = d2(stock_price, t)
            return self.strike*np.exp(-self.pricing_model.riskfree_rate*(self.maturity-t))*norm.cdf(-d2_val) - stock_price*norm.cdf(-d1_val)

        num_paths       = stock_price_paths.shape[0]
        time_grid       = self.pricing_model.time_grid
        paths           = np.zeros((num_paths, time_grid.num_steps+1))
        paths[:, :-1]   = put_price(stock_price_paths[:, :-1], time_grid.grid[:-1])
        if time_grid.grid[-1] == self.maturity: # need to to do separately the case where we are at maturity (price = payoff)
            paths[:, -1] = np.maximum(self.strike-stock_price_paths[:, -1], 0)
        else:
            paths[:, -1] = put_price(stock_price_paths[:, -1], time_grid.grid[-1])
        return paths
    

    def get_conditional_quantile(self, level, delta, tref, risk_factors_vals):
        condquant_risk_factors  = risk_factors_vals * exp((self.pricing_model.riskfree_rate-0.5*self.pricing_model.volatility**2)*delta + self.pricing_model.volatility*sqrt(delta)*norm.ppf(1-level))
        init_time_grid          = self.pricing_model.time_grid
        # we need to change time grid of pricing model accordingly as we want to generate value(s) of derivative at a given time t+delta
        self.pricing_model.set_time_grid(TimeGrid(init_time_grid.timestep, 0, tref+delta))
        cond_quant = self.generate_paths(condquant_risk_factors.reshape(-1, 1))
        self.pricing_model.set_time_grid(init_time_grid)
        return cond_quant.reshape(-1)


class ZeroCouponOneFactorHullWhite(PricingEngine):
    '''
    ZC bond where the riskfree rate follows a Hull-White model.
    '''
    def __init__(self, maturity, hull_white):
        super().__init__(maturity, hull_white)

    def generate_paths(self, riskfree_rate_paths):
        # Affine model: B(t, T) = e^{m(t, T) - n(t, T) r_t}
        k = self.pricing_model.mean_reversion_speed
        def n(t):
            return (1-np.exp(-k*(self.maturity-t)))/k
        def m(t):
            obs_zc_price    = lambda t: self.pricing_model.yield_curve.zero_coupon_price(t)
            obs_yield_curve = lambda t: self.pricing_model.yield_curve.zero_coupon_yield(t)
            return np.log(obs_zc_price(self.maturity)/obs_zc_price(t)) + n(t)*obs_yield_curve(t) - self.pricing_model.volatility**2/(4*k)*n(t)**2*(1 - np.exp(-2*k*t))
    
        return np.exp(m(self.pricing_model.time_grid.grid) - n(self.pricing_model.time_grid.grid) * riskfree_rate_paths)
    

class PutZeroCouponHullWhite(PricingEngine):
    '''
    European put option on a ZC bond (with maturity higher than put maturity) where the riskfree rate follows a one factor Hull-White model.

    Attributes:
    - strike: The strike price of the put option.
    - zc_maturity: The maturity of the ZC bond.
    '''
    def __init__(self, strike, maturity, zc_maturity, hull_white):
        super().__init__(maturity, hull_white)
        self.strike         = strike
        self.zc_maturity    = zc_maturity

    def generate_paths(self, riskfree_rate_paths, zc1_paths, zc2_paths):
        '''
        Arguments:
        - riskfree_rate_paths: Simulated paths of the risk-free rate.
        - zc1_paths: Simulated paths of the ZC bond with maturity = T = put maturity.
        - zc2_paths: Simulated paths of the ZC bond with maturity = S>T = ZC bond maturity.
        '''
        def var_integral(t): # time-weighted variance in the put BS formula on a ZC bond
            k = self.pricing_model.mean_reversion_speed
            return self.pricing_model.volatility/k * (1 -exp(-k*(self.zc_maturity-self.maturity))) * np.sqrt((1 - np.exp(-2*k*(self.maturity-t)))/(2*k))
        def d1(zc1_price, zc2_price, var):
            return (np.log(zc2_price/(self.strike*zc1_price)) + 0.5*var) / np.sqrt(var)
        def d2(zc1_price, zc2_price, var):
            return d1(zc1_price, zc2_price, var) - np.sqrt(var)
        def put_price(zc1_price, zc2_price, t):
            var     = var_integral(t)
            d1_val  = d1(zc1_price, zc2_price, var)
            d2_val  = d2(zc1_price, zc2_price, var)
            return self.strike*zc1_price*norm.cdf(-d2_val) - zc2_price*norm.cdf(-d1_val)
        
        num_paths       = riskfree_rate_paths.shape[0]
        time_grid       = self.pricing_model.time_grid
        paths           = np.zeros((num_paths, time_grid.num_steps+1))
        paths[:, :-1]   = put_price(zc1_paths[:, :-1], zc2_paths[:, :-1], time_grid.grid[:-1])
        if time_grid.grid[-1] == self.maturity:
            paths[:, -1] = np.maximum(self.strike-zc2_paths[:, -1], 0)
        else:
            paths[:, -1] = put_price(zc1_paths[:, -1],  zc2_paths[:, -1], time_grid.grid[-1])
        return paths


class SwaptionOneFactorHullWhite(AnalyticalPricingEngine):
    '''
    Swaption on a swap starting at maturity T where the riskfree rate follows a one factor Hull-White model.

    Attributes:
    - swap_strike: The strike price of the underlying swap.
    - swap_time_grid: The time grid for the swap cash flow payments, where time origin = first payment date.
    - swap_notional: The notional amount of the swap.
    '''
    def __init__(self, maturity, swap_strike, swap_time_grid, swap_notional, hull_white):
        super().__init__(maturity, hull_white)
        self.swap_strike            = swap_strike
        self.swap_time_grid         = swap_time_grid
        self.swap_notional          = swap_notional

    def generate_paths(self, riskfree_rate_paths):
        k               = self.pricing_model.mean_reversion_speed
        fixing_dates    = self.swap_time_grid.grid + self.maturity + self.swap_time_grid.timestep
        # $B(T_0, T_k) = e^{\mu(T_0, T_k) - \psi(T_0, T_k) Y_{T_0}}$ where $Y_{T_0}$ is standard Gaussian under the $T_0$-forward measure.
        def psi(t, Tfix):
            return self.pricing_model.volatility/k * (exp(-k*t) - np.exp(-k*Tfix))
        def mu(t, Tfix):
            obs_zc_price    = lambda t: self.pricing_model.yield_curve.zero_coupon_price(t)
            int1            = self.pricing_model.volatility**2/k**3 * (1 + np.exp(-k*Tfix) - exp(-k*t) - np.exp(-k*(Tfix-t)) + 1/4*(np.exp(-2*k*(Tfix-t)) - 1))
            int2            = self.pricing_model.volatility**2/k**3 * (np.exp(-k*(Tfix-t)) - np.exp(-k*Tfix) - 1 + exp(-k*t) - np.exp(-k*(Tfix+self.maturity-2*t)) + np.exp(-k*(Tfix+self.maturity)) + exp(-k*(self.maturity-t)) - exp(-k*(t+self.maturity)))
            return np.log(obs_zc_price(Tfix)/ obs_zc_price(t)) - int1 + int2

        c           = np.concatenate([self.swap_time_grid.timestep*self.swap_strike*np.ones(self.swap_time_grid.num_steps), [1+self.swap_time_grid.timestep*self.swap_strike]], axis=0)
        K           = 1
        mu_vals     = mu(self.maturity, fixing_dates)
        psi_vals    = psi(self.maturity, fixing_dates)
        # Jamshidian trick
        f           = lambda y: np.sum(c*np.exp(mu_vals - psi_vals*y)) - K
        ybar        = root_finder(f, 0, method='hybr').x
        Kjam        = np.exp(mu_vals - psi_vals*ybar)
        zc1         = ZeroCouponOneFactorHullWhite(self.maturity, self.pricing_model)
        zc1_paths   = zc1.generate_paths(riskfree_rate_paths)
        paths       = np.zeros_like(riskfree_rate_paths)
        # swaption price = weighted sum of prices of put options on ZC bonds
        for j in range(len(fixing_dates)):
            zc2         = ZeroCouponOneFactorHullWhite(fixing_dates[j], self.pricing_model)
            zc2_paths   = zc2.generate_paths(riskfree_rate_paths)
            put         = PutZeroCouponHullWhite(Kjam[j], self.maturity, fixing_dates[j], self.pricing_model)
            put_paths   = put.generate_paths(riskfree_rate_paths, zc1_paths, zc2_paths)
            paths       += c[j] * put_paths
        paths *= self.swap_notional
        return paths
    
    def get_conditional_quantile(self, level, delta, tref, risk_factors_vals):
        k                       = self.pricing_model.mean_reversion_speed
        mean                    = risk_factors_vals*exp(-k*delta) + self.pricing_model.beta(tref+delta) - self.pricing_model.beta(tref) * exp(-k*delta)
        variance                = self.pricing_model.volatility**2/(2*k) * (1 - np.exp(-2*k*delta))
        condquant_risk_factors  = mean + np.sqrt(variance)*norm.ppf(level)
        init_time_grid          = self.pricing_model.time_grid
        self.pricing_model.set_time_grid(TimeGrid(init_time_grid.timestep, 0, tref+delta))
        cond_quant = self.generate_paths(condquant_risk_factors.reshape(-1, 1))
        self.pricing_model.set_time_grid(init_time_grid)
        return cond_quant.reshape(-1)