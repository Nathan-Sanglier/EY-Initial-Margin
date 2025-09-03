# Modeling Forward Initial Margin and Counterparty Risk in Uncleared Derivatives

## 📘 Overview 
This project relates to my internship in the Quantitative Advisory Services (QAS) team of EY France, on the modeling of forward initial margin (IM) in uncleared derivatives that arises in counterparty credit risk topics. More precisely, it contains all the numerical results (and other stuff) of the two case studies that are presented in my [internship report](report.pdf). A short summary of my work, used for my internship defense, is available as [slides](report.pdf). The code has been done in `python`, using `jupyter` notebooks for the analysis and `.py` files for the forward IM engines. Run `pip install -r requirements.txt` to install required dependencies.

## 🚀 Main Features
- 🟢 [`tutorial.ipynb`](tutorial.ipynb) : Notebook that shows how to train and evaluate a given forward IM model for the two case studies, relying on object-oriented programming.
- 🟢 [`analyses`](analyses/) : 
    - [`analysis_moments_regression.ipynb`](analyses/analysis_moments_regression.ipynb) : Analysis of the estimation of raw moments of $\left(\Delta V_{t_i+\delta} \mid v\right)$ obtained by regression, for the two case studies. We also analyze the resulting centred moments, skewness and kurtosis estimates based on these raw moments estimates. The regression methods investigated are linear regression, generalized linear models, Hubert regression, Ridge regression, and kernel regression.
    - [`analysis_jlsmc.ipynb`](analyses/analysis_jlsmc.ipynb) : Analysis of the JLSMC method at a specific timestep chosen by the user, for the two case studies. In particular, we investigate the impact of the methods chosen for estimating raw moments of $\left(\Delta V_{t_i+\delta} \mid v\right)$, defining the support values, and estimating the value-at-risk function $g_i(\cdot)$ based on the support values and the associated quantiles of the fitted Johnson distributions through moment-matching. Moreover, we analyze the Gaussian and Johnson assumptions of $\left(\Delta V_{t_i+\delta} \mid v\right)$ for the same timestep, and for a given value $v$ of portfolio price.
    - [`analysis_jpmmc.ipynb`](analyses/analysis_jpmmc.ipynb) : We perform the same kind of work as for [`analysis_jlsmc.ipynb`](analysis_jlsmc.ipynb), except that we focus on the JPMMC method. It means that we use the Johnson percentile-matching (instead of moment-matching) procedure to fit a Johnson distribution.
    - [`analysis_forward_im_profile.ipynb`](analyses/analysis_forward_im_profile.ipynb) : This notebook enables to automatically compute the forward IM profiles (and related performance metrics) obtained by the GLSMC, JLSMC, JPMMC, and neural network quantile regression, for a wide range of settings. These profiles have been already pre-computed and stored in [`results_forward_im_profile`](analyses/results_forward_im_profile/) folder.
    - [`analysis_im_formula_swaption_hw1f.ipynb`](analyses/analysis_im_formula_swaption_hw1f.ipynb) :Analysis of the validity of our approximation leading to an analytical formula for the case of the swaption and one factor Hull-White model.

## 📑 Other Features

- 🟡 [`others`](others/) : 
    - 🟡 [`stylized_graphs.ipynb`](others/stylized_graphs.ipynb) : Code for stylized graphs that are presented in the report, not of much interest for the reader of the report.

- 🟡 [`ressources`](ressources/) : Folder containing several research papers on forward IM, counterparty credit risk, quantile regression, etc.

## ⚙️ Back-end Architecture
The back-end architecture is stored locally as a `python` package named `backend`, automatically setup when running `pip install -r requirements.txt` thanks to [`pyproject.toml`](pyproject.toml).
- 🔵 [`as99.py`](as99.py) : AS99 algorithm for fitting a Johnson distribution through moment-matching, adapted from Matlab by [`maxdevblock`](https://github.com/maxdevblock) and for which I've corrected some mistakes.
- 🔵 [`utils.py`](utils.py) : Contains all the functions that can be used in the different notebooks or classes of this project.
- 🔵 [`pricing_models.py`](pricing_models.py) : Contains the stochastic processes used to describe the evolution of risk factors (`BlackScholes`, `OneFactorHullWhite`) and specific parametrization of the zero-coupon yield curve (`YieldCurve`).
- 🔵 [`pricing_engines.py`](pricing_engines.py) : Contains the risk factors model combined with the financial asset considered; it enables to price a given derivative with a given model for the two case studies (`PutBlackScholes`, `SwaptionOneFactorHullWhite`).
- 🔵 [`forward_im_models.py`](forward_im_models.py) : Contains the forward IM models investigated during my internship:
    - Monte-Carlo with Gaussian distribution (`GaussianLeastSquaresMonteCarlo`).
    - Monte-Carlo with Johnson distribution (`JohnsonLeastSquaresMonteCarlo`, `JohnsonPercentileMatchingMonteCarlo`).
    - Neural networks quantile regression (`NeuralQuantileRegression`).
    - Nested Monte-Carlo (`NestedMonteCarloForwardInitialMargin`).
    - Cases for which the forward IM has an (potentially approximate) analytical formula (`AnalyticalForwardInitialMargin`).

## 👤 Authors
- 🟣 [`SANGLIER Nathan`](https://github.com/Nathan-Sanglier)