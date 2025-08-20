import  numpy           as      np
from    scipy.special   import  eval_genlaguerre, comb
from    as99            import  f_johnson_M
from    math            import  sqrt, exp
from   scipy.stats      import  norm, johnsonsu, johnsonsb, lognorm


class TimeGrid:
    '''
    Time grid object for simulation of risk factors/portfolio price/forward IM paths.

    Attributes:
    - start_time: The start time of the simulation (included).
    - timestep: The time step for the simulation.
    - time_horizon: The total time horizon for the simulation.
    - grid: np.linspace time grid.
    - num_steps: The number of time steps in the simulation.

    Methods:
    - find_index: Find the index of a given time in the time grid (provided it is within the grid).
    '''
    def __init__(self, timestep, time_horizon, start_time=0):
        self.start_time     = start_time
        self.timestep       = timestep
        self.time_horizon   = time_horizon
        self.grid = np.linspace(start_time, start_time+time_horizon, int(time_horizon/timestep)+1)
        self.num_steps      = len(self.grid) - 1
    
    def find_index(self, t):
        return int((t-self.start_time) / self.timestep)
    

def get_mtmdiff(mtm_paths, mpor, time_grid):
    '''
    Computes $V_{(t+\delta) \wedge T} - V_t$ for each path and at each timestep.
    '''
    inds_offset     = np.clip(np.arange(time_grid.num_steps+1) + time_grid.find_index(mpor), 0, time_grid.num_steps)
    mtmdiff_paths   = mtm_paths[:, inds_offset] - mtm_paths
    return mtmdiff_paths

def get_basis(x, basis_type, order):
    '''
    Returns the basis functions evaluated at x.
    '''
    if basis_type == 'laguerre':
        return np.column_stack([eval_genlaguerre(s, 0, x) for s in range(order+1)])
    elif basis_type == 'canonical':
        return np.column_stack([x**s for s in range(order+1)])
    
def get_centred_moms(yhat):
    '''
    Computes the centred moments (starting from oder two) based on the given raw moments (starting from order one), as a binomial sum.
    '''
    M, K        = yhat.shape
    zhat        = np.column_stack([np.ones((M, 1)), yhat])
    pows        = np.arange(K+1)
    m1hat_pows  = zhat[:, 1:2]**pows
    temp        = np.ones(M, dtype=bool)

    muhat       = np.zeros((M, K-1))
    maskhat     = np.ones_like(muhat, dtype=bool)
    for k in range(2, K+1):
        # compute centred momentof order k
        coeffs          = (comb(k, pows[:k+1]) * (-1)**pows[k::-1]).reshape(1, -1)
        terms           = coeffs * zhat[:, :k+1] * m1hat_pows[:, k::-1]
        muhat[:, k-2]   = np.sum(terms, axis=1)
        if k%2 == 0:
            temp            &= (yhat[:, k-1]>0) # Only keep moment where the k-th raw moment is positive, with k even
            maskhat[:, k-2] = temp&(muhat[:, k-2]>0) # In addition, only keep moment if the k-th centred moment is positive, with k even
        else:
            maskhat[:, k-2] = temp

    return muhat, maskhat


def get_skewkurt(muhat, maskhat=None):
    '''
    Computes skewness and kurtosis based on the centred moments of order two to four.
    '''
    get_skew = lambda mu2, mu3: mu3 / mu2**(3/2)
    get_kurt = lambda mu2, mu4: mu4 / mu2**2

    if maskhat is None:
        maskhat = np.ones_like(muhat, dtype=bool)
    
    mask_skewhat            = (maskhat[:, 0])&(maskhat[:, 1]) # Only keep skewness if both centred moments of order two and three are defined correctly
    skewhat                 = np.nan * np.ones(muhat.shape[0])
    skewhat[mask_skewhat]   = get_skew(muhat[mask_skewhat, 0], muhat[mask_skewhat, 1])
    mask_kurthat            = (maskhat[:, 0])&(maskhat[:, 2]) # Only keep kurtosis if both centred moments of order two and four are defined correctly
    kurthat                 = np.nan * np.ones(muhat.shape[0])
    kurthat[mask_kurthat]   = get_kurt(muhat[mask_kurthat, 0], muhat[mask_kurthat, 2])
    
    mask_tothat = (mask_skewhat)&(mask_kurthat)
    
    return skewhat, kurthat, mask_skewhat, mask_kurthat, mask_tothat


def moment_matching_johnson(mu1rawhat, mu2hat, skewhat, kurthat, mask_tothat, PRINT_MESS=True):
    '''
    Moment-matching procedure of Johnson distribution based on AS99 algorithm.
    '''
    jtypes_map  = {'SL': 1, 'SU': 2, 'SB': 3, 'SN': 4, 'ST': 5}
    jparamshat  = np.zeros((len(mu1rawhat), 4))
    jtypehat    = np.zeros(len(mu1rawhat))
    mask_hat    = np.zeros_like(mu1rawhat, dtype=bool)
    for i in range(len(mu1rawhat)):
        if mask_tothat[i]:
            jcoeffs, jf, mess = f_johnson_M(mu1rawhat[i], sqrt(mu2hat[i]), skewhat[i], kurthat[i])
            if mess == '(b2 < b1+one)': 
                if PRINT_MESS:
                    print(f'Warning: impossible region for support point n°{i} ({jf}): beta2 = {kurthat[i]:.2f} < beta1+1 = {skewhat[i]**2+1:.2f}')
                jparamshat[i]  = np.repeat(np.nan, 4)
                jtypehat[i]    = None
            else:
                if np.any(np.isnan(jcoeffs)):
                    if PRINT_MESS:
                        print(f'Warning: NaN coefficients for support point n°{i}')
                        print(f'    (mu1, mu2, skew, kurt)      = ({mu1rawhat[i]:.2f}, {mu2hat[i]:.2f}, {skewhat[i]:.2f}, {kurthat[i]:.2f})')
                        print(f'    (gamma, delta, xi, lambda)  = ({jcoeffs[0]:.2f}, {jcoeffs[1]:.2f}, {jcoeffs[2]:.2f}, {jcoeffs[3]:.2f})')
                    jparamshat[i]   = np.repeat(np.nan, 4)
                    jtypehat[i]     = None
                else:
                    mask_hat[i]     = True
                    jparamshat[i]   = jcoeffs
                    jtypehat[i]     = jtypes_map[jf]
    return jparamshat, jtypehat, mask_hat


def percentile_matching_johnson(tref, mtm_vals, risk_factors_train_vals, mtm_train_vals, pricing_engine, mpor, num_inner_paths, z):
    'Percentile-matching procedure of Johnson distribution based on Shapiro method.'
    jtypes_map  = {'SL': 1, 'SU': 2, 'SB': 3, 'SN': 4, 'ST': 5}
    percents    = norm.cdf([3*z, z, -z, -3*z])
    inds        = (mtm_train_vals[:, None] == mtm_vals).argmax(axis=0)
    mtmdiff_nmc = get_mtmdiff_nmc(num_inner_paths, pricing_engine, risk_factors_train_vals[inds], mtm_vals, mpor, tref)
    quant_nmc   = np.zeros((len(percents), mtm_vals.shape[0]))
    for i, perc in enumerate(percents):
        quant_nmc[i, :] = np.quantile(mtmdiff_nmc, perc, axis=1)
        
    m = quant_nmc[0, :] - quant_nmc[1, :]
    n = quant_nmc[2, :] - quant_nmc[3, :]
    p = quant_nmc[1, :] - quant_nmc[2, :]

    jtypehat    =  np.zeros_like(p, dtype=object)
    mask_hat    = np.zeros_like(jtypehat, dtype=bool)
    jparamshat  = np.zeros((len(jtypehat), 4))
    for i in range(len(p)):
        if (p[i]==0):
            jparamshat[i]  = np.repeat(np.nan, 4)
            jtypehat[i]    = None
            continue
        else:
            d = (m[i]*n[i])/p[i]**2
            if (d<0.999) or (d<1 and m[i]<=p[i]):
                jtype = jtypes_map['SB']
            elif (d>1.001) or ((d>1) and (m[i]<=p[i])):
                jtype = jtypes_map['SU']
            else:
                jtype = jtypes_map['SL']
        if jtype == jtypes_map['SU']:
            jparamshat[i, 1]    = 2*z/np.arccosh(1/2*(m[i]/p[i] + n[i]/p[i]))
            jparamshat[i, 0]    = jparamshat[i, 1] * np.arcsinh((n[i]/p[i] - m[i]/p[i]) / (2*np.sqrt(m[i]/p[i] * n[i]/p[i] - 1)))
            jparamshat[i, 2]    = (quant_nmc[1, i] + quant_nmc[2, i])/2 + p[i]*(n[i]/p[i] - m[i]/p[i])/(2*(m[i]/p[i] + n[i]/p[i] - 2))
            jparamshat[i, 3]    = 2*p[i]*np.sqrt(m[i]/p[i] * n[i]/p[i] - 1) / ((m[i]/p[i] + n[i]/p[i] - 2) * np.sqrt(m[i]/p[i] + n[i]/p[i] + 2))
            mask_hat[i]         = True
            jtypehat[i]         = jtype
        elif jtype == jtypes_map['SB']:
            if (m[i]==0) or (n[i]==0):
                # print(f'Warning: impossible region for support point n°{i} - d = {d[i]} (SB): m = {m[i]} ; n = {n[i]}')
                jparamshat[i]  = np.repeat(np.nan, 4)
                jtypehat[i]    = None
            else:
                jparamshat[i, 1]    = z/np.arccosh(1/2*np.sqrt((1+p[i]/m[i]) * (1+p[i]/n[i])))
                jparamshat[i, 0]    = jparamshat[i, 1] * np.arcsinh((p[i]/n[i] - p[i]/m[i]) * np.sqrt((1+p[i]/m[i]) * (1+p[i]/n[i]) - 4) / (2*(p[i]/m[i] * p[i]/n[i] - 1)))
                jparamshat[i, 3]    = p[i]*np.sqrt(((1+p[i]/m[i]) * (1+p[i]/n[i]) - 2)**2 - 4) / (p[i]/m[i]*p[i]/n[i] - 1)
                jparamshat[i, 2]    = (quant_nmc[1, i] + quant_nmc[2, i])/2 - jparamshat[i, 3]/2 + p[i]*(p[i]/n[i] - p[i]/m[i])/(2*(p[i]/m[i] * p[i]/n[i] - 1))
                mask_hat[i]         = True
                jtypehat[i]         = jtype
        elif jtype == jtypes_map['SL']:
            if (m[i]<=p[i]):
                # print(f'Warning: impossible region for support point n°{i} - d = {d[i]} (SL): m = {m[i]} ; p = {p[i]}')
                jparamshat[i]  = np.repeat(np.nan, 4)
                jtypehat[i]    = None
            else:
                jparamshat[i, 1]    = 2*z/np.log(m[i]/p[i])
                jparamshat[i, 0]    = jparamshat[i, 1] * np.log((m[i]/p[i] - 1) / (p[i] * np.sqrt(m[i]/p[i])))
                jparamshat[i, 2]    = (quant_nmc[1, i] + quant_nmc[2, i])/2 - p[i]/2 * (m[i]/p[i] + 1) / (m[i]/p[i] - 1)
                jparamshat[i, 3]    = 1
                mask_hat[i]         = True
                jtypehat[i]         = jtype

    return jparamshat, jtypehat, mask_hat


def get_quantile_normal(mu1rawhat, mu2hat, alpha):
    '''
    Quantile at level alpha of normal distribution with mean mu1rawhat and variance mu2hat.
    '''
    quanthat = norm.ppf(alpha, loc=mu1rawhat.reshape(-1, 1), scale=np.sqrt(mu2hat.reshape(-1, 1)))
    return quanthat


def get_quantile_johnson(jparamshat, jtypehat, alpha):
    '''
    Quantile at level alpha of Johnson distribution.
    '''
    jtypes_map  = {'SL': 1, 'SU': 2, 'SB': 3, 'SN': 4, 'ST': 5}
    quanthat    = np.zeros((jparamshat.shape[0], len(alpha)))
    maskhat     = np.ones_like(quanthat, dtype=bool)
    for i in range(jparamshat.shape[0]):
        gamma_, delta_, xi_, lambda_    = jparamshat[i:i+1, 0], jparamshat[i:i+1, 1], jparamshat[i:i+1, 2], jparamshat[i:i+1, 3]
        absdelta_, abslambda_           = np.abs(delta_), np.abs(lambda_)
        if jtypehat[i] == jtypes_map['SL']:
            quanthat[i] = lognorm.ppf(alpha, s=1/absdelta_, loc=xi_, scale=lambda_*exp(-gamma_/delta_))
        elif jtypehat[i] == jtypes_map['SU']:
            quanthat[i] = johnsonsu.ppf(alpha, a=gamma_, b=delta_, loc=xi_, scale=lambda_)
        elif (jtypehat[i]==jtypes_map['SB']) or (jtypehat[i]==jtypes_map['ST']):
            quanthat[i] = johnsonsb.ppf(alpha, a=gamma_, b=delta_, loc=xi_, scale=lambda_)
        elif jtypehat[i] == jtypes_map['SN']:
            quanthat[i] = norm.ppf(alpha, loc=(xi_-gamma_*lambda_)/delta_, scale=abslambda_/absdelta_)
        maskhat[i, :] = np.isfinite(quanthat[i])
    return quanthat, maskhat


def get_mtmdiff_nmc(num_inner_paths, pricing_engine, risk_factors_vals, mtm_vals, mpor, tref):
    '''
    Makes nested MC simulations at a given time t for several risk factors values, diffusing risk factors and portfolio price from t to $(t+mpor) \wedge T$.
    '''
    num_paths       = len(risk_factors_vals)
    mtmdiff         = np.zeros((num_paths, num_inner_paths))
    init_time_grid  = pricing_engine.pricing_model.time_grid
    init_spot       = pricing_engine.pricing_model.spot
    delta           = min(init_time_grid.grid[-1] - tref, mpor)
    new_time_grid   = TimeGrid(init_time_grid.timestep, delta, tref)
    for m in range(num_paths):
        # Don't forget to change timegrid for simulation as well as spot value
        pricing_engine.pricing_model.set_time_grid(new_time_grid)
        pricing_engine.pricing_model.set_spot(risk_factors_vals[m])
        risk_factors_nested_paths = pricing_engine.pricing_model.generate_paths(num_inner_paths)
        pricing_engine.pricing_model.set_time_grid(TimeGrid(init_time_grid.timestep, 0, tref+delta))
        mtm_nested_vals = pricing_engine.generate_paths(risk_factors_nested_paths[:, -1:])
        mtmdiff[m, :]   = mtm_nested_vals[:, 0] - mtm_vals[m]
    
    # Reste time grid and spot to their initial values
    pricing_engine.pricing_model.set_time_grid(init_time_grid)
    pricing_engine.pricing_model.set_spot(init_spot)

    return mtmdiff