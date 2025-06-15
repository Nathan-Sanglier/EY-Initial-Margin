import  numpy                                       as      np
import  statsmodels.api                             as      sm
from    statsmodels.nonparametric.kernel_regression import  KernelReg
from    scipy.stats                                 import  norm, johnsonsu, johnsonsb, lognorm
from    scipy.special                               import  eval_genlaguerre, comb
from    time                                        import  time
from    math                                        import  sqrt, exp
from    sklearn.linear_model                        import  LinearRegression, Ridge, HuberRegressor
from    sklearn.neighbors                           import  KNeighborsRegressor
from    as99                                        import  f_johnson_M

def beta(t, r):
    return np.exp(-r*t)

def gen_riskfactors(S0, r, sigma, dt, M, N):
    paths           = np.zeros((M, N+1))
    paths[:, 0]     = S0
    Z               = np.random.normal(0, 1, size=(M, N))
    increments      = np.exp((r-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
    paths[:, 1:]    = S0 * np.cumprod(increments, axis=1)
    return paths

def gen_mtm(S_paths, r, sigma, K, T, time_grid):
    M   = S_paths.shape[0]
    N   = S_paths.shape[1]-1
    mtm = np.zeros((M, N+1))
    for i in range(N+1):
        if time_grid[i] == T:
            mtm[:, i]   = np.maximum(K-S_paths[:, i], 0)
        else:
            d1          = (np.log(S_paths[:, i]/K) + (r + 0.5*sigma**2)*(T-time_grid[i])) / (sigma*np.sqrt(T-time_grid[i]))
            d2          = d1 - sigma*np.sqrt(T-time_grid[i])
            mtm[:, i]   = K*beta(T-time_grid[i], r)*norm.cdf(-d2) - S_paths[:, i]*norm.cdf(-d1)
        
    return mtm


def get_mtmdiff_nmc(M_in, S, mtm, r, sigma, dt, K, T, ind_tref, ind_delta, time_grid):
    gen_riskfactors_in = lambda S0, M, N: gen_riskfactors(S0, r, sigma, dt, M, N)
    gen_mtm_in         = lambda S_paths, time_grid: gen_mtm(S_paths, r, sigma, K, T, time_grid)
    M               = S.shape[0]
    mtmdiff         = np.zeros((M, M_in))
    ind_tdelta      = ind_tref + ind_delta
    if ind_tdelta >= len(time_grid):
        ind_tdelta = len(time_grid) - 1
    ind_offset      = ind_tdelta - ind_tref
    for m in range(M):
        S_paths_nested  = gen_riskfactors_in(S[m], M_in, ind_offset)
        mtm_vals_nested = gen_mtm_in(S_paths_nested[:, -1:], time_grid[ind_tdelta:ind_tdelta+1])
        mtmdiff[m, :]   = mtm_vals_nested[:, 0] - mtm[m]

    return mtmdiff

def get_mtmdiff(mtm_paths, ind_delta):
    N               = mtm_paths.shape[1] - 1
    inds_offset     = np.clip(np.arange(N+1) + ind_delta, 0, N)
    mtmdiff_paths   = mtm_paths[:, inds_offset] - mtm_paths
    mtmdiff_paths   = mtmdiff_paths[:, :-1]
    return mtmdiff_paths


def get_basis(mtm, basis_type, order):
    if basis_type == 'laguerre':
        return np.column_stack([eval_genlaguerre(s, 0, mtm) for s in range(order+1)])
    elif basis_type == 'canonical':
        return np.column_stack([mtm**s for s in range(order+1)])


def regress_moms(mtm_train, y_train, mtm_pred_list, setting, order_moms):
    start = time()
    method          = setting['method']
    ind_start       = 1 if setting['regress_mean']==True else 2
    yhat_pred_list  = []
    X_pred_list     = []

    for mtm in mtm_pred_list:
        yhat_pred   = np.zeros((len(mtm), order_moms))
        yhat_pred_list.append(yhat_pred)
    if (method == 'LR') or (method == 'HR') or (method == 'GLM'):
        X_train = get_basis(mtm_train, setting['basis_type'], setting['order'])
        for mtm in mtm_pred_list:
            X_pred = get_basis(mtm, setting['basis_type'], setting['order'])
            X_pred_list.append(X_pred)
        for j in range(ind_start, order_moms+1):
            if j%2 != 0:
                if (method == 'LR') and (setting['ridge']>0):
                    model = Ridge(setting['ridge'], fit_intercept=False)
                elif method == 'HR':
                    model = HuberRegressor(epsilon=setting['epsilon'], alpha=setting['ridge'], fit_intercept=False)
                elif (method == 'LR') or (method == 'GLM'):
                    model = LinearRegression(fit_intercept=False)
                
                model.fit(X_train, y_train[:, j-1])
            else:
                if (method == 'LR') and (setting['ridge']>0):
                    model = Ridge(alpha=setting['ridge'], fit_intercept=False)
                elif method == 'LR':
                    model = LinearRegression(fit_intercept=False)
                elif method == 'HR':
                    model = HuberRegressor(epsilon=setting['epsilon'], alpha=setting['ridge'] ,fit_intercept=False)
                elif (method == 'GLM') and (setting['ridge']>0):
                    model = sm.GLM(y_train[:, j-1], X_train, family=sm.families.Gaussian(link=sm.families.links.Log())).fit_regularized(method='elastic_net', alpha=setting['ridge'], L1_wt=0)
                elif method == 'GLM':
                    model = sm.GLM(y_train[:, j-1], X_train, family=sm.families.Gaussian(link=sm.families.links.Log())).fit()
                
                if method != 'GLM':
                    model.fit(X_train, y_train[:, j-1])

            for i, mtm in enumerate(mtm_pred_list):
                yhat_pred_list[i][:, j-1] = model.predict(X_pred_list[i])
    
    elif method =='KR':

        X_train = mtm_train.reshape(-1, 1)
        if setting['bandwidth'] == 'silverman':
            iqr     = np.quantile(mtm_train, 0.75) - np.quantile(mtm_train, 0.25)
            bdw     =[0.9*min(np.std(mtm_train), iqr/1.34) * len(mtm_train)**(-1/5)]
        elif setting['bandwidth'] == 'cv_ls':
            bdw     = 'cv_ls'
        else:
            bdw     = [setting['bandwidth']]
        for j in range(ind_start, order_moms+1):
            kr  = KernelReg(endog=y_train[:, j-1], exog=X_train, var_type='c', bw=bdw)
            for i, mtm in enumerate(mtm_pred_list):
                X_pred = mtm.reshape(-1, 1)
                yhat_pred_list[i][:, j-1] = kr.fit(X_pred)[0]
    end = time()
    
    return yhat_pred_list, end-start


def get_centred_moms(yhat):
    M, K        = yhat.shape
    zhat        = np.column_stack([np.ones((M, 1)), yhat])
    pows        = np.arange(K+1)
    m1hat_pows  = zhat[:, 1:2]**pows
    temp        = np.ones(M, dtype=bool)

    muhat       = np.zeros((M, K-1))
    maskhat     = np.ones_like(muhat, dtype=bool)
    for k in range(2, K+1):
        coeffs          = (comb(k, pows[:k+1]) * (-1)**pows[k::-1]).reshape(1, -1)
        terms           = coeffs * zhat[:, :k+1] * m1hat_pows[:, k::-1]
        muhat[:, k-2]   = np.sum(terms, axis=1)
        if k%2 == 0:
            temp            &= (yhat[:, k-1]>0)
            maskhat[:, k-2] = temp&(muhat[:, k-2]>0)
        else:
            maskhat[:, k-2] = temp

    return muhat, maskhat

def get_skewkurt(muhat, maskhat=None):

    get_skew = lambda mu2, mu3: mu3 / mu2**(3/2)
    get_kurt = lambda mu2, mu4: mu4 / mu2**2

    if maskhat is None:
        maskhat = np.ones_like(muhat, dtype=bool)
    
    mask_skewhat            = (maskhat[:, 0])&(maskhat[:, 1])
    skewhat                 = np.nan * np.ones(muhat.shape[0])
    skewhat[mask_skewhat]   = get_skew(muhat[mask_skewhat, 0], muhat[mask_skewhat, 1])
    mask_kurthat            = (maskhat[:, 0])&(maskhat[:, 2])
    kurthat                 = np.nan * np.ones(muhat.shape[0])
    kurthat[mask_kurthat]   = get_kurt(muhat[mask_kurthat, 0], muhat[mask_kurthat, 2])
    
    mask_tothat = (mask_skewhat)&(mask_kurthat)
    
    return skewhat, kurthat, mask_skewhat, mask_kurthat, mask_tothat

def moment_matching_johnson(mu1rawhat, mu2hat, skewhat, kurthat, mask_tothat, PRINT_MESS=True):
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


def percentile_matching_johnson(mtm, S_train, mtm_train, r, sigma, dt, K, T, ind_tref, ind_delta, time_grid, Nnmc, z):
    jtypes_map  = {'SL': 1, 'SU': 2, 'SB': 3, 'SN': 4, 'ST': 5}
    percents    = norm.cdf([3*z, z, -z, -3*z])
    inds        = (mtm_train[:, None] == mtm).argmax(axis=0)
    mtmdiff_nmc = get_mtmdiff_nmc(Nnmc, S_train[inds], mtm, r, sigma, dt, K, T, ind_tref, ind_delta, time_grid)
    quant_nmc   = np.zeros((len(percents), mtm.shape[0]))
    for i, perc in enumerate(percents):
        quant_nmc[i, :] = np.quantile(mtmdiff_nmc, perc, method='inverted_cdf', axis=1)
        
    m = quant_nmc[0, :] - quant_nmc[1, :]
    n = quant_nmc[2, :] - quant_nmc[3, :]
    p = quant_nmc[1, :] - quant_nmc[2, :]
    d = (m*n)/p**2

    conds       = [(d<0.999)|((d<1)&(m<=p)), (d>1.001)|((d>1)&(m<=p)), (d>=0.999)&(d<=1.001)&(m>p)]
    conds_jtype = [jtypes_map['SB'], jtypes_map['SU'], jtypes_map['SL']]
    jtypehat    = np.select(conds, conds_jtype).astype(object)
    mask_hat    = np.zeros_like(jtypehat, dtype=bool)
    jparamshat  = np.zeros((len(jtypehat), 4))
    for i, jtype in enumerate(jtypehat):
        if jtype == jtypes_map['SU']:
            if p[i] == 0:
                # print(f'Warning: impossible region for support point n°{i} - d = {d[i]:.4f} (SU): p = {p[i]:.4f}')
                jparamshat[i]  = np.repeat(np.nan, 4)
                jtypehat[i]    = None
            else:
                jparamshat[i, 1]    = 2*z/np.arccosh(1/2*(m[i]/p[i] + n[i]/p[i]))
                jparamshat[i, 0]    = jparamshat[i, 1] * np.arcsinh((n[i]/p[i] - m[i]/p[i]) / (2*np.sqrt(m[i]/p[i] * n[i]/p[i] - 1)))
                jparamshat[i, 2]    = (quant_nmc[1, i] + quant_nmc[2, i])/2 + p[i]*(n[i]/p[i] - m[i]/p[i])/(2*(m[i]/p[i] + n[i]/p[i] - 2))
                jparamshat[i, 3]    = 2*p[i]*np.sqrt(m[i]/p[i] * n[i]/p[i] - 1) / ((m[i]/p[i] + n[i]/p[i] - 2) * np.sqrt(m[i]/p[i] + n[i]/p[i] + 2))
                mask_hat[i]         = True
        elif jtype == jtypes_map['SB']:
            if (m[i] == 0) or (n[i] == 0):
                # print(f'Warning: impossible region for support point n°{i} - d = {d[i]} (SB): m = {m[i]} ; n = {n[i]}')
                jparamshat[i]  = np.repeat(np.nan, 4)
                jtypehat[i]    = None
            else:
                jparamshat[i, 1]    = z/np.arccosh(1/2*np.sqrt((1+p[i]/m[i]) * (1+p[i]/n[i])))
                jparamshat[i, 0]    = jparamshat[i, 1] * np.arcsinh((p[i]/n[i] - p[i]/m[i]) * np.sqrt((1+p[i]/m[i]) * (1+p[i]/n[i]) - 4) / (2*(p[i]/m[i] * p[i]/n[i] - 1)))
                jparamshat[i, 3]    = p[i]*np.sqrt(((1+p[i]/m[i]) * (1+p[i]/n[i]) - 2)**2 - 4) / (p[i]/m[i]*p[i]/n[i] - 1)
                jparamshat[i, 2]    = (quant_nmc[1, i] + quant_nmc[2, i])/2 - jparamshat[i, 3]/2 + p[i]*(p[i]/n[i] - p[i]/m[i])/(2*(p[i]/m[i] * p[i]/n[i] - 1))
                mask_hat[i]         = True
        elif jtype == jtypes_map['SL']:
            if (m[i] < p[i]) or (p[i] == 0):
                # print(f'Warning: impossible region for support point n°{i} - d = {d[i]} (SL): m = {m[i]} ; p = {p[i]}')
                jparamshat[i]  = np.repeat(np.nan, 4)
                jtypehat[i]    = None
            else:
                jparamshat[i, 1]    = 2*z/np.log(m[i]/p[i])
                jparamshat[i, 0]    = jparamshat[i, 1] * np.log((m[i]/p[i] - 1) / (p[i] * np.sqrt(m[i]/p[i])))
                jparamshat[i, 2]    = (quant_nmc[1, i] + quant_nmc[2, i])/2 - p[i]/2 * (m[i]/p[i] + 1) / (m[i]/p[i] - 1)
                jparamshat[i, 3]    = 1
                mask_hat[i]         = True

    return jparamshat, jtypehat, mask_hat


def get_quantile_johnson(jparamshat, jtypehat, alpha):
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


def get_pdf_johnson(jparamshat, jtypehat, mtmdiff_pdf):
    jtypes_map  = {'SL': 1, 'SU': 2, 'SB': 3, 'SN': 4, 'ST': 5}
    pdfhat      = np.zeros((jparamshat.shape[0], len(mtmdiff_pdf)))
    for i in range(jparamshat.shape[0]):
        gamma_, delta_, xi_, lambda_    = jparamshat[i, 0], jparamshat[i, 1], jparamshat[i, 2], jparamshat[i, 3]
        absdelta_, abslambda_           = np.abs(delta_), np.abs(lambda_)
        if jtypehat[i] == jtypes_map['SL']:
            pdfhat[i, :] = lognorm.pdf(mtmdiff_pdf, s=1/absdelta_, loc=xi_, scale=lambda_*exp(-gamma_/delta_))
        elif jtypehat[i] == jtypes_map['SU']:
            pdfhat[i, :] = johnsonsu.pdf(mtmdiff_pdf, a=gamma_, b=delta_, loc=xi_, scale=lambda_)
        elif (jtypehat[i]==jtypes_map['SB']) or (jtypehat[i]==jtypes_map['ST']):
            pdfhat[i, :] = johnsonsb.pdf(mtmdiff_pdf, a=gamma_, b=delta_, loc=xi_, scale=lambda_)
        elif jtypehat[i] == jtypes_map['SN']:
            pdfhat[i, :] = norm.pdf(mtmdiff_pdf, loc=(xi_-gamma_*lambda_)/delta_, scale=abslambda_/absdelta_)
    return pdfhat


def get_quantile_normal(mu1rawhat, mu2hat, alpha):
    quanthat = norm.ppf(alpha, loc=mu1rawhat.reshape(-1, 1), scale=np.sqrt(mu2hat.reshape(-1, 1)))
    return quanthat


def get_pdf_normal(mu1rawhat, mu2hat, mtmdiff_pdf):
    pdfhat = norm.pdf(mtmdiff_pdf, loc=mu1rawhat.reshape(-1, 1), scale=np.sqrt(mu2hat.reshape(-1, 1)))
    return pdfhat


def get_var_jlsmc(mtm_supp, quanthat_supp, mtm_pred_list, setting):
    method          = setting['method']
    yhat_pred_list  = []
    X_pred_list     = []

    for mtm in mtm_pred_list:
        yhat_pred   = np.zeros(len(mtm))
        yhat_pred_list.append(yhat_pred)
    if (method == 'LR'):
        X_train = get_basis(mtm_supp, setting['basis_type'], setting['order'])
        for mtm in mtm_pred_list:
            X_pred = get_basis(mtm, setting['basis_type'], setting['order'])
            X_pred_list.append(X_pred)
        model = LinearRegression(fit_intercept=False)
    elif (method == 'kNN') or (method == 'KR'):
        X_train = mtm_supp.reshape(-1, 1)
        for mtm in mtm_pred_list:
            X_pred = mtm.reshape(-1, 1)
            X_pred_list.append(X_pred)
        if method == 'kNN':
            model = KNeighborsRegressor(n_neighbors=setting['n_neighbors'], weights='distance', algorithm='auto', leaf_size=30, p=2)
        elif method == 'KR':
            if setting['bandwidth'] == 'silverman':
                iqr     = np.quantile(mtm_supp, 0.75) - np.quantile(mtm_supp, 0.25)
                bdw     =[0.9*min(np.std(mtm_supp), iqr/1.34) * len(mtm_supp)**(-1/5)]
            elif setting['bandwidth'] == 'cv_ls':
                bdw     = 'cv_ls'
            else:
                bdw     = [setting['bandwidth']]
            model = KernelReg(endog=quanthat_supp, exog=X_train, var_type='c', bw=bdw)

    if (method == 'LR') or (method == 'kNN'):
        model.fit(X_train, quanthat_supp)
        for i, mtm in enumerate(mtm_pred_list):
            yhat_pred_list[i] = model.predict(X_pred_list[i])
    elif method == 'KR':
        for i, mtm in enumerate(mtm_pred_list):
            yhat_pred_list[i] =model.fit(X_pred_list[i])[0]
        
    return yhat_pred_list


'''
def get_var_put(S, mtm, r, sigma, K, T, alpha, delta, ind_tref, time_grid):
    ind_delta = int(delta/(time_grid[1]-time_grid[0]))
    ind_tdelta  = ind_tref + ind_delta
    var_S       = S * exp((r-0.5*sigma**2)*delta + sigma*sqrt(delta)*norm.ppf(1-alpha))
    temp        = gen_mtm(var_S.reshape(-1, 1), r, sigma, K, T, time_grid[ind_tdelta:(ind_tdelta+1)]).reshape(-1)
    varhat      = temp - mtm
    return varhat
'''


def get_var_put(S, mtm, r, sigma, K, T, alpha, delta, ind_tref, time_grid):
    ind_delta   = int(delta/(time_grid[1]-time_grid[0]))
    ind_tdelta  = ind_tref + ind_delta 
    if ind_tdelta >= len(time_grid):
        ind_tdelta = len(time_grid) - 1
    time_gap    = time_grid[ind_tdelta] - time_grid[ind_tref]
    var_S       = S * exp((r-0.5*sigma**2)*time_gap + sigma*sqrt(time_gap)*norm.ppf(1-alpha))
    temp        = gen_mtm(var_S.reshape(-1, 1), r, sigma, K, T, time_grid[ind_tdelta:(ind_tdelta+1)]).reshape(-1)
    varhat      = temp - mtm
    return varhat