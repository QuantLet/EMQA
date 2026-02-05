#!/usr/bin/env python3
"""Generate Lecture 3 (GARCH) and Lecture 4 (Risk/Trading) quantlet notebooks."""
import json, os

BASE = os.path.dirname(os.path.abspath(__file__))

COLAB_BASE = "https://colab.research.google.com/github/QuantLet/EMQA/blob/main/EN/quantlets"

STYLE_CODE = '''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.facecolor': 'none',
    'axes.facecolor': 'none',
    'savefig.facecolor': 'none',
    'savefig.transparent': True,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 11,
    'figure.figsize': (12, 6),
})

COLORS = {
    'blue': '#1A3A6E', 'red': '#CD0000', 'green': '#2E7D32',
    'orange': '#E67E22', 'purple': '#8E44AD', 'gray': '#808080',
    'cyan': '#00BCD4', 'amber': '#B5853F'
}

def save_fig(fig, name):
    fig.savefig(name, bbox_inches='tight', transparent=True, dpi=300)
    print(f"Saved: {name}")
'''

YFINANCE_CODE = '''import yfinance as yf

def fetch(ticker, start='2020-01-01', end='2025-12-31'):
    d = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        return d['Close'].squeeze().dropna()
    return d['Close'].dropna()
'''

ELEC_CODE = '''import os
def load_elec():
    paths = [
        '../../charts/electricity_cache.csv',
        '/Users/danielpele/Documents/Energy MBA/charts/electricity_cache.csv',
    ]
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p, parse_dates=[0], index_col=0)
            return df
    raise FileNotFoundError("No electricity data found")

elec = load_elec()
if 'price' in elec.columns:
    price_col = 'price'
elif 'Price' in elec.columns:
    price_col = 'Price'
else:
    price_col = elec.columns[0]
elec_price = elec[price_col].dropna()
'''

def nb(cells):
    nb_cells = []
    for ctype, src in cells:
        if ctype == 'md':
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": [src]})
        else:
            nb_cells.append({"cell_type": "code", "metadata": {}, "source": [src], "outputs": [], "execution_count": None})
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"}
        },
        "cells": nb_cells
    }

def write_nb(name, cells):
    folder = os.path.join(BASE, name)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.ipynb")
    with open(path, 'w') as f:
        json.dump(nb(cells), f, indent=1)
    print(f"Created: {path}")


def colab_url(name):
    return f"{COLAB_BASE}/{name}/{name}.ipynb"


def badge(name):
    return f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url(name)})"


# ---------------------------------------------------------------------------
# Lecture 3 - GARCH (7 quantlets)
# ---------------------------------------------------------------------------

def gen_vol_clustering():
    name = "EMQA_vol_clustering"
    cells = [
        ("md", f"""{badge(name)}

# {name}
Volatility clustering in energy returns -- visual evidence from Brent crude oil.
**Output:** `vol_clustering.pdf`"""),
        ("code", STYLE_CODE),
        ("code", YFINANCE_CODE),
        ("code", '''# Fetch Brent crude
brent = fetch('BZ=F', start='2018-01-01')
log_ret = np.log(brent / brent.shift(1)).dropna()
print(f"Brent returns: {len(log_ret)} observations, {log_ret.index[0].date()} to {log_ret.index[-1].date()}")'''),
        ("code", '''# 3-panel: Returns, Squared returns, Rolling volatility
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# Panel 1 - Returns
axes[0].plot(log_ret.index, log_ret.values, color=COLORS['blue'], linewidth=0.5)
axes[0].set_ylabel('Log Returns')
axes[0].set_title('Brent Crude Oil - Volatility Clustering')

# Panel 2 - Squared returns
axes[1].plot(log_ret.index, log_ret.values**2, color=COLORS['red'], linewidth=0.5)
axes[1].set_ylabel('Squared Returns')

# Panel 3 - Rolling volatility (30-day, annualized)
roll_vol = log_ret.rolling(30).std() * np.sqrt(252)
axes[2].plot(roll_vol.index, roll_vol.values, color=COLORS['green'], linewidth=1)
axes[2].set_ylabel('Annualized Volatility')
axes[2].set_xlabel('Date')

# Highlight COVID (Mar-Jun 2020) and Ukraine (Feb-Jun 2022)
for ax in axes:
    ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-06-30'),
               alpha=0.15, color=COLORS['orange'], label='COVID-19')
    ax.axvspan(pd.Timestamp('2022-02-01'), pd.Timestamp('2022-06-30'),
               alpha=0.15, color=COLORS['purple'], label='Ukraine conflict')

# Legend on bottom axis only
handles, labels = axes[2].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
axes[2].legend(by_label.values(), by_label.keys(),
               loc='upper center', bbox_to_anchor=(0.5, -0.10), frameon=False, ncol=2)

plt.tight_layout()
save_fig(fig, 'vol_clustering.pdf')
plt.show()'''),
    ]
    write_nb(name, cells)


def gen_news_impact():
    name = "EMQA_news_impact"
    cells = [
        ("md", f"""{badge(name)}

# {name}
News impact curve: GARCH vs GJR-GARCH -- showing asymmetric volatility response.
**Output:** `news_impact_curve.pdf`"""),
        ("code", STYLE_CODE),
        ("code", '''# News impact curves (analytical)
# GARCH(1,1): h_t = omega + alpha * eps^2 + beta * h_{t-1}
# GJR-GARCH:  h_t = omega + (alpha + gamma * I(eps<0)) * eps^2 + beta * h_{t-1}

omega = 0.00001
alpha_garch = 0.10
beta_ = 0.85
alpha_gjr = 0.05
gamma_gjr = 0.10

# Assume h_{t-1} = unconditional variance
h_unc_garch = omega / (1 - alpha_garch - beta_)
h_unc_gjr = omega / (1 - alpha_gjr - 0.5 * gamma_gjr - beta_)

eps = np.linspace(-4, 4, 500)

# GARCH news impact
nic_garch = omega + alpha_garch * eps**2 + beta_ * h_unc_garch

# GJR-GARCH news impact
indicator = (eps < 0).astype(float)
nic_gjr = omega + (alpha_gjr + gamma_gjr * indicator) * eps**2 + beta_ * h_unc_gjr

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(eps, nic_garch, color=COLORS['blue'], linewidth=2, label='GARCH(1,1)')
ax.plot(eps, nic_gjr, color=COLORS['red'], linewidth=2, label='GJR-GARCH(1,1)')
ax.axvline(0, color=COLORS['gray'], linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Standardized Shock ($\\\\epsilon_{t-1}$)')
ax.set_ylabel('Conditional Variance ($h_t$)')
ax.set_title('News Impact Curve: Symmetric vs Asymmetric GARCH')

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), frameon=False, ncol=2)

plt.tight_layout()
save_fig(fig, 'news_impact_curve.pdf')
plt.show()'''),
    ]
    write_nb(name, cells)


def gen_garch_oil():
    name = "EMQA_garch_oil"
    cells = [
        ("md", f"""{badge(name)}

# {name}
GARCH(1,1) estimation on Brent crude oil returns.
**Output:** `garch_oil.pdf`"""),
        ("code", STYLE_CODE),
        ("code", YFINANCE_CODE),
        ("code", '''# Fetch Brent returns
brent = fetch('BZ=F', start='2018-01-01')
log_ret = (np.log(brent / brent.shift(1)).dropna()) * 100  # percentage returns
print(f"Returns: {len(log_ret)} obs")'''),
        ("code", '''# Fit GARCH(1,1)
try:
    from arch import arch_model
    am = arch_model(log_ret, vol='Garch', p=1, q=1, dist='normal', mean='Constant')
    res = am.fit(disp='off')
    print(res.summary())
    cond_vol = res.conditional_volatility
    use_arch = True
except ImportError:
    print("arch package not available -- simulating GARCH(1,1)")
    use_arch = False
    omega, alpha, beta_ = 0.01, 0.08, 0.90
    n = len(log_ret)
    cond_var = np.zeros(n)
    cond_var[0] = log_ret.var()
    ret_vals = log_ret.values
    for t in range(1, n):
        cond_var[t] = omega + alpha * ret_vals[t-1]**2 + beta_ * cond_var[t-1]
    cond_vol = pd.Series(np.sqrt(cond_var), index=log_ret.index)'''),
        ("code", '''# 2-panel plot
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Panel 1 - Returns with +/- 2 sigma bands
axes[0].plot(log_ret.index, log_ret.values, color=COLORS['gray'], linewidth=0.4, alpha=0.7, label='Returns')
axes[0].plot(cond_vol.index, 2 * cond_vol.values, color=COLORS['red'], linewidth=1, label='+2$\\\\sigma$')
axes[0].plot(cond_vol.index, -2 * cond_vol.values, color=COLORS['red'], linewidth=1, label='-2$\\\\sigma$')
axes[0].set_ylabel('Returns (%)')
axes[0].set_title('Brent Crude - GARCH(1,1) Conditional Volatility Bands')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), frameon=False, ncol=3)

# Panel 2 - Conditional volatility
axes[1].plot(cond_vol.index, cond_vol.values, color=COLORS['blue'], linewidth=1)
axes[1].set_ylabel('Conditional Volatility (%)')
axes[1].set_xlabel('Date')
axes[1].set_title('GARCH(1,1) Conditional Volatility')

plt.tight_layout()
save_fig(fig, 'garch_oil.pdf')
plt.show()'''),
    ]
    write_nb(name, cells)


def gen_garch_estimation():
    name = "EMQA_garch_estimation"
    cells = [
        ("md", f"""{badge(name)}

# {name}
GARCH parameter estimation and interpretation -- visualizing omega, alpha, beta and persistence.
**Output:** `garch_estimation.pdf`"""),
        ("code", STYLE_CODE),
        ("code", YFINANCE_CODE),
        ("code", '''# Fetch Brent and fit GARCH
brent = fetch('BZ=F', start='2018-01-01')
log_ret = (np.log(brent / brent.shift(1)).dropna()) * 100

try:
    from arch import arch_model
    am = arch_model(log_ret, vol='Garch', p=1, q=1, dist='normal', mean='Constant')
    res = am.fit(disp='off')
    omega = res.params['omega']
    alpha = res.params['alpha[1]']
    beta_ = res.params['beta[1]']
    use_arch = True
except ImportError:
    omega, alpha, beta_ = 0.01, 0.08, 0.90
    use_arch = False

persistence = alpha + beta_
print(f"omega  = {omega:.6f}")
print(f"alpha  = {alpha:.4f}")
print(f"beta   = {beta_:.4f}")
print(f"alpha+beta (persistence) = {persistence:.4f}")'''),
        ("code", '''# 3-panel: Parameter bar chart, Persistence bar, News impact curve
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1 - Parameter bar chart
params = ['$\\\\omega$', '$\\\\alpha$', '$\\\\beta$']
values = [omega, alpha, beta_]
bar_colors = [COLORS['blue'], COLORS['red'], COLORS['green']]
bars = axes[0].bar(params, values, color=bar_colors, width=0.5, edgecolor='white')
for bar, val in zip(bars, values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=10)
axes[0].set_ylabel('Parameter Value')
axes[0].set_title('GARCH(1,1) Parameters')

# Panel 2 - Persistence gauge (horizontal bar)
axes[1].barh(['Persistence\\n($\\\\alpha+\\\\beta$)'], [persistence], color=COLORS['orange'],
             height=0.4, edgecolor='white')
axes[1].barh(['Persistence\\n($\\\\alpha+\\\\beta$)'], [1 - persistence], left=[persistence],
             color=COLORS['gray'], alpha=0.2, height=0.4, edgecolor='white')
axes[1].set_xlim(0, 1.05)
axes[1].set_xlabel('Value')
axes[1].set_title(f'Volatility Persistence = {persistence:.4f}')
axes[1].axvline(1.0, color=COLORS['red'], linestyle='--', linewidth=1, alpha=0.5)

# Panel 3 - News impact curve from fitted parameters
h_unc = omega / (1 - alpha - beta_) if persistence < 1 else omega / 0.01
eps = np.linspace(-4, 4, 500)
nic = omega + alpha * eps**2 + beta_ * h_unc
axes[2].plot(eps, nic, color=COLORS['purple'], linewidth=2)
axes[2].axvline(0, color=COLORS['gray'], linestyle='--', linewidth=0.8, alpha=0.5)
axes[2].set_xlabel('Standardized Shock ($\\\\epsilon_{t-1}$)')
axes[2].set_ylabel('Conditional Variance ($h_t$)')
axes[2].set_title('Fitted News Impact Curve')

plt.tight_layout()
save_fig(fig, 'garch_estimation.pdf')
plt.show()'''),
    ]
    write_nb(name, cells)


def gen_vol_dynamics():
    name = "EMQA_vol_dynamics"
    cells = [
        ("md", f"""{badge(name)}

# {name}
Conditional volatility dynamics -- GARCH conditional vol vs realized (rolling) volatility.
**Output:** `vol_dynamics.pdf`"""),
        ("code", STYLE_CODE),
        ("code", YFINANCE_CODE),
        ("code", '''# Fetch and compute returns
brent = fetch('BZ=F', start='2018-01-01')
log_ret = (np.log(brent / brent.shift(1)).dropna()) * 100

# Realized vol (30-day rolling, annualized)
realized_vol = log_ret.rolling(30).std() * np.sqrt(252)

# GARCH conditional vol
try:
    from arch import arch_model
    am = arch_model(log_ret, vol='Garch', p=1, q=1, dist='normal', mean='Constant')
    res = am.fit(disp='off')
    cond_vol = res.conditional_volatility * np.sqrt(252)  # annualize
except ImportError:
    omega, alpha, beta_ = 0.01, 0.08, 0.90
    n = len(log_ret)
    cond_var = np.zeros(n)
    cond_var[0] = log_ret.var()
    ret_vals = log_ret.values
    for t in range(1, n):
        cond_var[t] = omega + alpha * ret_vals[t-1]**2 + beta_ * cond_var[t-1]
    cond_vol = pd.Series(np.sqrt(cond_var) * np.sqrt(252), index=log_ret.index)

print(f"Mean GARCH vol:     {cond_vol.mean():.2f}%")
print(f"Mean Realized vol:  {realized_vol.mean():.2f}%")'''),
        ("code", '''# 2-panel: Overlaid volatilities, Ratio
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Panel 1 - Both volatilities overlaid
axes[0].plot(realized_vol.index, realized_vol.values, color=COLORS['blue'],
             linewidth=1, label='Realized Vol (30-day rolling)')
axes[0].plot(cond_vol.index, cond_vol.values, color=COLORS['red'],
             linewidth=1, label='GARCH Conditional Vol')
axes[0].set_ylabel('Annualized Volatility (%)')
axes[0].set_title('Conditional vs Realized Volatility - Brent Crude')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), frameon=False, ncol=2)

# Panel 2 - Ratio
common_idx = realized_vol.dropna().index.intersection(cond_vol.dropna().index)
ratio = cond_vol.loc[common_idx] / realized_vol.loc[common_idx]
axes[1].plot(common_idx, ratio.values, color=COLORS['purple'], linewidth=0.8)
axes[1].axhline(1.0, color=COLORS['gray'], linestyle='--', linewidth=1)
axes[1].set_ylabel('GARCH / Realized Ratio')
axes[1].set_xlabel('Date')
axes[1].set_title('Volatility Ratio (GARCH / Realized)')

plt.tight_layout()
save_fig(fig, 'vol_dynamics.pdf')
plt.show()'''),
    ]
    write_nb(name, cells)


def gen_var_calculation():
    name = "EMQA_var_calculation"
    cells = [
        ("md", f"""{badge(name)}

# {name}
Value at Risk (VaR) with GARCH conditional volatility at 95% and 99% confidence.
**Output:** `var_calculation.pdf`"""),
        ("code", STYLE_CODE),
        ("code", YFINANCE_CODE),
        ("code", '''# Fetch Brent returns
brent = fetch('BZ=F', start='2018-01-01')
log_ret = (np.log(brent / brent.shift(1)).dropna()) * 100

# GARCH conditional vol
try:
    from arch import arch_model
    am = arch_model(log_ret, vol='Garch', p=1, q=1, dist='normal', mean='Constant')
    res = am.fit(disp='off')
    cond_vol = res.conditional_volatility
    mu = res.params.get('mu', res.params.iloc[0])
except ImportError:
    omega, alpha, beta_ = 0.01, 0.08, 0.90
    mu = log_ret.mean()
    n = len(log_ret)
    cond_var = np.zeros(n)
    cond_var[0] = log_ret.var()
    ret_vals = log_ret.values
    for t in range(1, n):
        cond_var[t] = omega + alpha * ret_vals[t-1]**2 + beta_ * cond_var[t-1]
    cond_vol = pd.Series(np.sqrt(cond_var), index=log_ret.index)

from scipy.stats import norm
z_95 = norm.ppf(0.05)
z_99 = norm.ppf(0.01)

var_95 = mu + z_95 * cond_vol
var_99 = mu + z_99 * cond_vol

# Violations
viol_95 = log_ret < var_95
viol_99 = log_ret < var_99

print(f"VaR 95% violations: {viol_95.sum()} / {len(log_ret)} = {viol_95.mean()*100:.2f}% (expected 5%)")
print(f"VaR 99% violations: {viol_99.sum()} / {len(log_ret)} = {viol_99.mean()*100:.2f}% (expected 1%)")'''),
        ("code", '''# Plot returns with VaR thresholds and violations
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(log_ret.index, log_ret.values, color=COLORS['gray'], linewidth=0.4, alpha=0.6, label='Returns')
ax.plot(var_95.index, var_95.values, color=COLORS['orange'], linewidth=1, label='VaR 95%')
ax.plot(var_99.index, var_99.values, color=COLORS['red'], linewidth=1, label='VaR 99%')

# Mark violations
viol_99_dates = log_ret.index[viol_99]
viol_99_vals = log_ret[viol_99]
ax.scatter(viol_99_dates, viol_99_vals, color=COLORS['red'], s=15, zorder=5, label='99% Exceedance')

viol_95_only = viol_95 & ~viol_99
viol_95_dates = log_ret.index[viol_95_only]
viol_95_vals = log_ret[viol_95_only]
ax.scatter(viol_95_dates, viol_95_vals, color=COLORS['orange'], s=10, zorder=4, label='95% Exceedance')

ax.set_xlabel('Date')
ax.set_ylabel('Returns (%)')
ax.set_title('Value at Risk - GARCH-based VaR on Brent Crude')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), frameon=False, ncol=5)

plt.tight_layout()
save_fig(fig, 'var_calculation.pdf')
plt.show()'''),
    ]
    write_nb(name, cells)


def gen_garch_electricity():
    name = "EMQA_garch_electricity"
    cells = [
        ("md", f"""{badge(name)}

# {name}
GARCH(1,1) on electricity returns -- higher and spikier volatility than oil.
**Output:** `garch_electricity.pdf`"""),
        ("code", STYLE_CODE),
        ("code", ELEC_CODE),
        ("code", '''# Compute daily returns from electricity prices
elec_ret = elec_price.pct_change().dropna() * 100
# Winsorize extreme outliers for numerical stability
clip_val = elec_ret.abs().quantile(0.995)
elec_ret = elec_ret.clip(-clip_val, clip_val)
print(f"Electricity returns: {len(elec_ret)} obs, std={elec_ret.std():.2f}%")'''),
        ("code", '''# Fit GARCH(1,1)
try:
    from arch import arch_model
    am = arch_model(elec_ret, vol='Garch', p=1, q=1, dist='normal', mean='Constant')
    res = am.fit(disp='off')
    print(res.summary())
    cond_vol = res.conditional_volatility
except ImportError:
    print("arch not available -- simulating GARCH")
    omega, alpha, beta_ = 0.5, 0.15, 0.80
    n = len(elec_ret)
    cond_var = np.zeros(n)
    cond_var[0] = elec_ret.var()
    ret_vals = elec_ret.values
    for t in range(1, n):
        cond_var[t] = omega + alpha * ret_vals[t-1]**2 + beta_ * cond_var[t-1]
    cond_vol = pd.Series(np.sqrt(np.maximum(cond_var, 0)), index=elec_ret.index)'''),
        ("code", '''# 2-panel plot
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Panel 1 - Returns with bands
axes[0].plot(elec_ret.index, elec_ret.values, color=COLORS['gray'], linewidth=0.4, alpha=0.7, label='Returns')
axes[0].plot(cond_vol.index, 2 * cond_vol.values, color=COLORS['red'], linewidth=1, label='+2$\\\\sigma$')
axes[0].plot(cond_vol.index, -2 * cond_vol.values, color=COLORS['red'], linewidth=1, label='-2$\\\\sigma$')
axes[0].set_ylabel('Returns (%)')
axes[0].set_title('Electricity Returns - GARCH(1,1) Conditional Bands')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), frameon=False, ncol=3)

# Panel 2 - Conditional volatility
axes[1].plot(cond_vol.index, cond_vol.values, color=COLORS['orange'], linewidth=1)
axes[1].set_ylabel('Conditional Volatility (%)')
axes[1].set_xlabel('Date')
axes[1].set_title('GARCH(1,1) Conditional Volatility - Electricity')

plt.tight_layout()
save_fig(fig, 'garch_electricity.pdf')
plt.show()'''),
    ]
    write_nb(name, cells)


# ---------------------------------------------------------------------------
# Lecture 4 - Risk/Trading (10 quantlets)
# ---------------------------------------------------------------------------

def gen_hedge_ratio():
    name = "EMQA_hedge_ratio"
    cells = [
        ("md", f"""{badge(name)}

# {name}
Optimal hedge ratio via OLS regression of Brent on WTI returns.
**Output:** `hedge_ratio.pdf`"""),
        ("code", STYLE_CODE),
        ("code", YFINANCE_CODE),
        ("code", '''# Fetch Brent and WTI
brent = fetch('BZ=F')
wti = fetch('CL=F')

# Align and compute returns
common = brent.index.intersection(wti.index)
brent = brent.loc[common]
wti = wti.loc[common]
ret_b = np.log(brent / brent.shift(1)).dropna()
ret_w = np.log(wti / wti.shift(1)).dropna()
common2 = ret_b.index.intersection(ret_w.index)
ret_b = ret_b.loc[common2]
ret_w = ret_w.loc[common2]

# OLS hedge ratio
from numpy.polynomial.polynomial import polyfit
coeffs = np.polyfit(ret_w.values, ret_b.values, 1)
hedge_ratio = coeffs[0]
intercept = coeffs[1]
print(f"Optimal hedge ratio (beta): {hedge_ratio:.4f}")
print(f"Intercept: {intercept:.6f}")'''),
        ("code", '''# 2-panel: Scatter + regression, Portfolio variance vs hedge ratio
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1 - Scatter + regression line
axes[0].scatter(ret_w.values, ret_b.values, alpha=0.3, s=8, color=COLORS['blue'])
x_line = np.linspace(ret_w.min(), ret_w.max(), 100)
axes[0].plot(x_line, hedge_ratio * x_line + intercept, color=COLORS['red'], linewidth=2,
             label=f'h* = {hedge_ratio:.3f}')
axes[0].set_xlabel('WTI Returns')
axes[0].set_ylabel('Brent Returns')
axes[0].set_title('OLS Regression: Brent on WTI')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), frameon=False)

# Panel 2 - Portfolio variance vs hedge ratio
h_range = np.linspace(0, 2, 200)
var_b = ret_b.var()
var_w = ret_w.var()
cov_bw = np.cov(ret_b.values, ret_w.values)[0, 1]
port_var = var_b - 2 * h_range * cov_bw + h_range**2 * var_w
min_h = cov_bw / var_w
axes[1].plot(h_range, port_var, color=COLORS['blue'], linewidth=2)
axes[1].axvline(min_h, color=COLORS['red'], linestyle='--', linewidth=1.5, label=f'h* = {min_h:.3f}')
axes[1].scatter([min_h], [port_var[np.argmin(np.abs(h_range - min_h))]], color=COLORS['red'], s=80, zorder=5)
axes[1].set_xlabel('Hedge Ratio (h)')
axes[1].set_ylabel('Portfolio Variance')
axes[1].set_title('Hedged Portfolio Variance vs Hedge Ratio')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), frameon=False)

plt.tight_layout()
save_fig(fig, 'hedge_ratio.pdf')
plt.show()'''),
    ]
    write_nb(name, cells)


def gen_hedge_compare():
    name = "EMQA_hedge_compare"
    cells = [
        ("md", f"""{badge(name)}

# {name}
Hedging strategy comparison: No Hedge, Full Hedge, and Optimal Hedge.
**Output:** `hedge_compare.pdf`"""),
        ("code", STYLE_CODE),
        ("code", '''# Simulate spot price path and futures hedge
np.random.seed(42)
n_days = 252
S0 = 80  # initial spot price
mu = 0.0002
sigma = 0.02

# Spot price path
z = np.random.randn(n_days)
log_ret = mu + sigma * z
spot = S0 * np.exp(np.cumsum(log_ret))
spot = np.insert(spot, 0, S0)

# Futures price tracks spot with small basis
basis = np.random.randn(n_days + 1) * 0.3
futures = spot + basis

# Effective cost under different strategies
h_opt = 0.85  # optimal hedge ratio (estimated)

cost_no_hedge = spot
cost_full_hedge = np.full_like(spot, spot[0]) + np.cumsum(np.insert(np.diff(spot) - np.diff(futures), 0, 0))
cost_opt_hedge = spot - h_opt * (futures - futures[0])

days = np.arange(n_days + 1)'''),
        ("code", '''fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1 - Effective cost over time
axes[0].plot(days, cost_no_hedge, color=COLORS['red'], linewidth=1.5, label='No Hedge')
axes[0].plot(days, cost_full_hedge, color=COLORS['blue'], linewidth=1.5, label='Full Hedge (h=1)')
axes[0].plot(days, cost_opt_hedge, color=COLORS['green'], linewidth=1.5, label=f'Optimal Hedge (h={h_opt})')
axes[0].set_xlabel('Day')
axes[0].set_ylabel('Effective Cost ($)')
axes[0].set_title('Effective Cost Under Different Hedging Strategies')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), frameon=False, ncol=3)

# Panel 2 - Bar chart of cost volatility
vol_no = np.std(np.diff(cost_no_hedge))
vol_full = np.std(np.diff(cost_full_hedge))
vol_opt = np.std(np.diff(cost_opt_hedge))

labels = ['No Hedge', 'Full Hedge', 'Optimal Hedge']
vols = [vol_no, vol_full, vol_opt]
bar_colors = [COLORS['red'], COLORS['blue'], COLORS['green']]
bars = axes[1].bar(labels, vols, color=bar_colors, width=0.5, edgecolor='white')
for bar, val in zip(bars, vols):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10)
axes[1].set_ylabel('Cost Volatility ($ / day)')
axes[1].set_title('Daily Cost Volatility by Strategy')

plt.tight_layout()
save_fig(fig, 'hedge_compare.pdf')
plt.show()'''),
    ]
    write_nb(name, cells)


def gen_portfolio():
    name = "EMQA_portfolio"
    cells = [
        ("md", f"""{badge(name)}

# {name}
Portfolio optimization and efficient frontier for energy commodities.
**Output:** `portfolio_frontier.pdf`"""),
        ("code", STYLE_CODE),
        ("code", YFINANCE_CODE),
        ("code", '''# Fetch 4 energy tickers
tickers = {'Brent': 'BZ=F', 'WTI': 'CL=F', 'NatGas': 'NG=F', 'HeatingOil': 'HO=F'}
prices = pd.DataFrame()
for label, tk in tickers.items():
    try:
        s = fetch(tk)
        prices[label] = s
    except Exception as e:
        print(f"Warning: {tk} failed ({e})")

prices = prices.dropna()
returns = np.log(prices / prices.shift(1)).dropna()
print(f"Assets: {list(returns.columns)}, Observations: {len(returns)}")

mu_annual = returns.mean() * 252
cov_annual = returns.cov() * 252
print("\\nAnnualized returns:")
print(mu_annual.round(4))'''),
        ("code", '''# Monte Carlo portfolios
np.random.seed(42)
n_portfolios = 10000
n_assets = len(returns.columns)

results = np.zeros((n_portfolios, 3))
weights_record = np.zeros((n_portfolios, n_assets))

for i in range(n_portfolios):
    w = np.random.dirichlet(np.ones(n_assets))
    weights_record[i] = w
    port_ret = np.dot(w, mu_annual.values)
    port_vol = np.sqrt(np.dot(w.T, np.dot(cov_annual.values, w)))
    sharpe = port_ret / port_vol if port_vol > 0 else 0
    results[i] = [port_vol, port_ret, sharpe]

# Min variance and max Sharpe
idx_min_var = results[:, 0].argmin()
idx_max_sharpe = results[:, 2].argmax()'''),
        ("code", '''fig, ax = plt.subplots(figsize=(10, 7))

# Scatter all portfolios
sc = ax.scatter(results[:, 0] * 100, results[:, 1] * 100, c=results[:, 2],
                cmap='RdYlGn', s=5, alpha=0.5)
plt.colorbar(sc, ax=ax, label='Sharpe Ratio', shrink=0.8)

# Mark min-variance
ax.scatter(results[idx_min_var, 0] * 100, results[idx_min_var, 1] * 100,
           color=COLORS['blue'], marker='*', s=300, zorder=5, edgecolors='black',
           linewidths=0.8, label='Min Variance')

# Mark max-Sharpe
ax.scatter(results[idx_max_sharpe, 0] * 100, results[idx_max_sharpe, 1] * 100,
           color=COLORS['red'], marker='*', s=300, zorder=5, edgecolors='black',
           linewidths=0.8, label='Max Sharpe')

# Mark individual assets
asset_colors = [COLORS['blue'], COLORS['green'], COLORS['orange'], COLORS['purple']]
for j, col in enumerate(returns.columns):
    w_single = np.zeros(n_assets)
    w_single[j] = 1.0
    ret_j = mu_annual.values[j] * 100
    vol_j = np.sqrt(cov_annual.values[j, j]) * 100
    ax.scatter(vol_j, ret_j, color=asset_colors[j % len(asset_colors)],
               marker='D', s=100, zorder=6, edgecolors='black', linewidths=0.8, label=col)

ax.set_xlabel('Annualized Volatility (%)')
ax.set_ylabel('Annualized Return (%)')
ax.set_title('Energy Portfolio - Efficient Frontier (Monte Carlo)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), frameon=False, ncol=3)

plt.tight_layout()
save_fig(fig, 'portfolio_frontier.pdf')
plt.show()'''),
    ]
    write_nb(name, cells)


def gen_crack_spread():
    name = "EMQA_crack_spread"
    cells = [
        ("md", f"""{badge(name)}

# {name}
Crack spread: Oil vs Gasoline -- a key refining margin indicator.
**Output:** `crack_spread.pdf`"""),
        ("code", STYLE_CODE),
        ("code", YFINANCE_CODE),
        ("code", '''# Fetch Brent and RBOB Gasoline
brent = fetch('BZ=F')
gasoline = fetch('RB=F')

common = brent.index.intersection(gasoline.index)
brent = brent.loc[common]
gasoline = gasoline.loc[common]

# Crack spread (simplified 1:1): Gasoline (per gallon * 42 gal/bbl) - Oil (per bbl)
crack = gasoline * 42 - brent
print(f"Crack spread: mean={crack.mean():.2f}, std={crack.std():.2f}")'''),
        ("code", '''fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Panel 1 - Normalized prices
brent_norm = brent / brent.iloc[0] * 100
gas_norm = gasoline / gasoline.iloc[0] * 100
axes[0].plot(brent_norm.index, brent_norm.values, color=COLORS['blue'], linewidth=1.2, label='Brent Crude')
axes[0].plot(gas_norm.index, gas_norm.values, color=COLORS['orange'], linewidth=1.2, label='RBOB Gasoline')
axes[0].set_ylabel('Normalized Price (base=100)')
axes[0].set_title('Brent Crude vs RBOB Gasoline (Normalized)')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), frameon=False, ncol=2)

# Panel 2 - Spread with mean +/- std bands
crack_mean = crack.mean()
crack_std = crack.std()
axes[1].plot(crack.index, crack.values, color=COLORS['green'], linewidth=1, label='Crack Spread')
axes[1].axhline(crack_mean, color=COLORS['blue'], linestyle='-', linewidth=1.5, label=f'Mean ({crack_mean:.1f})')
axes[1].axhline(crack_mean + crack_std, color=COLORS['red'], linestyle='--', linewidth=1, alpha=0.7, label=f'+1 Std ({crack_mean + crack_std:.1f})')
axes[1].axhline(crack_mean - crack_std, color=COLORS['red'], linestyle='--', linewidth=1, alpha=0.7, label=f'-1 Std ({crack_mean - crack_std:.1f})')
axes[1].fill_between(crack.index, crack_mean - crack_std, crack_mean + crack_std,
                      alpha=0.1, color=COLORS['red'])
axes[1].set_ylabel('Crack Spread ($/bbl)')
axes[1].set_xlabel('Date')
axes[1].set_title('Crack Spread (Gasoline x 42 - Brent)')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), frameon=False, ncol=4)

plt.tight_layout()
save_fig(fig, 'crack_spread.pdf')
plt.show()'''),
    ]
    write_nb(name, cells)


def gen_spark_spread():
    name = "EMQA_spark_spread"
    cells = [
        ("md", f"""{badge(name)}

# {name}
Spark spread: Gas vs Electricity -- profitability of gas-fired power generation.
**Output:** `spark_spread.pdf`"""),
        ("code", STYLE_CODE),
        ("code", YFINANCE_CODE),
        ("code", ELEC_CODE),
        ("code", '''# Fetch Natural Gas
gas = fetch('NG=F')

# Align electricity and gas on common dates
elec_daily = elec_price.copy()
elec_daily.index = pd.to_datetime(elec_daily.index)
gas.index = pd.to_datetime(gas.index)
common = elec_daily.index.intersection(gas.index)

if len(common) < 30:
    # If not enough overlap, use approximate alignment
    elec_daily = elec_daily.resample('B').last().dropna()
    gas_aligned = gas.reindex(elec_daily.index, method='ffill').dropna()
    common = elec_daily.index.intersection(gas_aligned.index)
    elec_aligned = elec_daily.loc[common]
    gas_aligned = gas_aligned.loc[common]
else:
    elec_aligned = elec_daily.loc[common]
    gas_aligned = gas.loc[common]

# Spark spread = Electricity price - Gas / heat_rate
# Heat rate ~ 7 MMBtu/MWh for a typical gas plant
heat_rate = 7.0
# Gas is in $/MMBtu, Electricity in EUR/MWh (approximate conversion)
spark = elec_aligned.values - gas_aligned.values * heat_rate

spark_series = pd.Series(spark, index=common)
print(f"Spark spread: mean={spark_series.mean():.2f}, std={spark_series.std():.2f}")'''),
        ("code", '''fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Panel 1 - Both series
ax1 = axes[0]
ln1 = ax1.plot(elec_aligned.index, elec_aligned.values, color=COLORS['orange'], linewidth=1, label='Electricity')
ax1.set_ylabel('Electricity Price', color=COLORS['orange'])
ax2 = ax1.twinx()
ln2 = ax2.plot(gas_aligned.index, gas_aligned.values, color=COLORS['blue'], linewidth=1, label='Natural Gas')
ax2.set_ylabel('Gas Price ($/MMBtu)', color=COLORS['blue'])
ax2.spines['top'].set_visible(False)
lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.05), frameon=False, ncol=2)
ax1.set_title('Electricity and Natural Gas Prices')

# Panel 2 - Spark spread with bands
sp_mean = spark_series.mean()
sp_std = spark_series.std()
axes[1].plot(spark_series.index, spark_series.values, color=COLORS['green'], linewidth=1, label='Spark Spread')
axes[1].axhline(sp_mean, color=COLORS['blue'], linewidth=1.5, label=f'Mean ({sp_mean:.1f})')
axes[1].axhline(sp_mean + sp_std, color=COLORS['red'], linestyle='--', linewidth=1, alpha=0.7)
axes[1].axhline(sp_mean - sp_std, color=COLORS['red'], linestyle='--', linewidth=1, alpha=0.7)
axes[1].fill_between(spark_series.index, sp_mean - sp_std, sp_mean + sp_std,
                      alpha=0.1, color=COLORS['red'])
axes[1].axhline(0, color=COLORS['gray'], linestyle=':', linewidth=0.8)
axes[1].set_ylabel('Spark Spread')
axes[1].set_xlabel('Date')
axes[1].set_title(f'Spark Spread (Elec - Gas x {heat_rate} heat rate)')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), frameon=False, ncol=3)

plt.tight_layout()
save_fig(fig, 'spark_spread.pdf')
plt.show()'''),
    ]
    write_nb(name, cells)


def gen_pairs_trading():
    name = "EMQA_pairs_trading"
    cells = [
        ("md", f"""{badge(name)}

# {name}
Brent-WTI pairs trading strategy using z-score signals.
**Output:** `pairs_trading.pdf`"""),
        ("code", STYLE_CODE),
        ("code", YFINANCE_CODE),
        ("code", '''# Fetch Brent and WTI
brent = fetch('BZ=F')
wti = fetch('CL=F')

common = brent.index.intersection(wti.index)
brent = brent.loc[common]
wti = wti.loc[common]

# Spread and z-score
spread = brent - wti
roll_mean = spread.rolling(30).mean()
roll_std = spread.rolling(30).std()
zscore = ((spread - roll_mean) / roll_std).dropna()

print(f"Spread: mean={spread.mean():.2f}, std={spread.std():.2f}")
print(f"Z-score range: [{zscore.min():.2f}, {zscore.max():.2f}]")'''),
        ("code", '''# Generate trading signals
entry_threshold = 2.0
exit_threshold = 0.0

position = pd.Series(0.0, index=zscore.index)
for i in range(1, len(zscore)):
    if zscore.iloc[i] > entry_threshold:
        position.iloc[i] = -1  # short spread
    elif zscore.iloc[i] < -entry_threshold:
        position.iloc[i] = 1   # long spread
    elif abs(zscore.iloc[i]) < exit_threshold + 0.5:
        position.iloc[i] = 0   # exit
    else:
        position.iloc[i] = position.iloc[i-1]

# P&L: position * change in spread
spread_aligned = spread.loc[zscore.index]
daily_pnl = position.shift(1) * spread_aligned.diff()
daily_pnl = daily_pnl.fillna(0)
cum_pnl = daily_pnl.cumsum()

print(f"Total P&L: ${cum_pnl.iloc[-1]:.2f}")
print(f"Sharpe: {daily_pnl.mean() / daily_pnl.std() * np.sqrt(252):.2f}")'''),
        ("code", '''fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# Panel 1 - Z-score with thresholds
axes[0].plot(zscore.index, zscore.values, color=COLORS['blue'], linewidth=0.8, label='Z-score')
axes[0].axhline(entry_threshold, color=COLORS['red'], linestyle='--', linewidth=1, label=f'Entry (+/-{entry_threshold})')
axes[0].axhline(-entry_threshold, color=COLORS['red'], linestyle='--', linewidth=1)
axes[0].axhline(0, color=COLORS['gray'], linestyle=':', linewidth=0.8)
axes[0].fill_between(zscore.index, -entry_threshold, entry_threshold, alpha=0.05, color=COLORS['green'])
axes[0].set_ylabel('Z-score')
axes[0].set_title('Brent-WTI Pairs Trading: Z-score Signal')

# Mark long/short signals
long_sig = (position == 1) & (position.shift(1) != 1)
short_sig = (position == -1) & (position.shift(1) != -1)
axes[0].scatter(zscore.index[long_sig], zscore[long_sig], color=COLORS['green'],
                marker='^', s=40, zorder=5, label='Long signal')
axes[0].scatter(zscore.index[short_sig], zscore[short_sig], color=COLORS['red'],
                marker='v', s=40, zorder=5, label='Short signal')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), frameon=False, ncol=5)

# Panel 2 - Position
axes[1].fill_between(position.index, position.values, 0, where=position > 0,
                      color=COLORS['green'], alpha=0.4, label='Long')
axes[1].fill_between(position.index, position.values, 0, where=position < 0,
                      color=COLORS['red'], alpha=0.4, label='Short')
axes[1].set_ylabel('Position')
axes[1].set_title('Trading Position')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), frameon=False, ncol=2)

# Panel 3 - Cumulative P&L
axes[2].plot(cum_pnl.index, cum_pnl.values, color=COLORS['blue'], linewidth=1.5)
axes[2].fill_between(cum_pnl.index, 0, cum_pnl.values,
                      where=cum_pnl >= 0, alpha=0.2, color=COLORS['green'])
axes[2].fill_between(cum_pnl.index, 0, cum_pnl.values,
                      where=cum_pnl < 0, alpha=0.2, color=COLORS['red'])
axes[2].axhline(0, color=COLORS['gray'], linestyle=':', linewidth=0.8)
axes[2].set_ylabel('Cumulative P&L ($)')
axes[2].set_xlabel('Date')
axes[2].set_title('Cumulative Profit & Loss')

plt.tight_layout()
save_fig(fig, 'pairs_trading.pdf')
plt.show()'''),
    ]
    write_nb(name, cells)


def gen_ml_rf():
    name = "EMQA_ml_rf"
    cells = [
        ("md", f"""{badge(name)}

# {name}
Random Forest for energy price prediction -- feature importance analysis.
**Output:** `ml_rf_importance.pdf`"""),
        ("code", STYLE_CODE),
        ("code", YFINANCE_CODE),
        ("code", '''# Fetch Brent and create features
brent = fetch('BZ=F', start='2018-01-01')
df = pd.DataFrame({'price': brent})
df['return'] = np.log(df['price'] / df['price'].shift(1))

# Lag features
for lag in [1, 2, 3, 7, 14]:
    df[f'ret_lag_{lag}'] = df['return'].shift(lag)

# Rolling statistics
df['roll_mean_5'] = df['return'].rolling(5).mean()
df['roll_std_5'] = df['return'].rolling(5).std()
df['roll_mean_20'] = df['return'].rolling(20).mean()
df['roll_std_20'] = df['return'].rolling(20).std()
df['roll_skew_20'] = df['return'].rolling(20).skew()

# Target: next-day return
df['target'] = df['return'].shift(-1)
df = df.dropna()

feature_cols = [c for c in df.columns if c not in ['price', 'return', 'target']]
X = df[feature_cols].values
y = df['target'].values

# Train/test split (80/20)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Features: {feature_cols}")
print(f"Train: {len(X_train)}, Test: {len(X_test)}")'''),
        ("code", '''from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.6f}")
print(f"R-squared: {r2:.4f}")

# Feature importance
importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=True)'''),
        ("code", '''fig, ax = plt.subplots(figsize=(10, 7))

colors = [COLORS['blue'] if v < feat_imp.quantile(0.75) else COLORS['red'] for v in feat_imp.values]
ax.barh(feat_imp.index, feat_imp.values, color=colors, edgecolor='white', height=0.6)
ax.set_xlabel('Feature Importance')
ax.set_title(f'Random Forest Feature Importance (MAE={mae:.5f}, $R^2$={r2:.3f})')

for i, (val, name_f) in enumerate(zip(feat_imp.values, feat_imp.index)):
    ax.text(val + 0.002, i, f'{val:.3f}', va='center', fontsize=9)

plt.tight_layout()
save_fig(fig, 'ml_rf_importance.pdf')
plt.show()'''),
    ]
    write_nb(name, cells)


def gen_ml_compare():
    name = "EMQA_ml_compare"
    cells = [
        ("md", f"""{badge(name)}

# {name}
ML model comparison: Linear Regression, Random Forest, Gradient Boosting.
**Output:** `ml_compare.pdf`"""),
        ("code", STYLE_CODE),
        ("code", YFINANCE_CODE),
        ("code", '''# Fetch Brent and create features (same as ml_rf)
brent = fetch('BZ=F', start='2018-01-01')
df = pd.DataFrame({'price': brent})
df['return'] = np.log(df['price'] / df['price'].shift(1))

for lag in [1, 2, 3, 7, 14]:
    df[f'ret_lag_{lag}'] = df['return'].shift(lag)

df['roll_mean_5'] = df['return'].rolling(5).mean()
df['roll_std_5'] = df['return'].rolling(5).std()
df['roll_mean_20'] = df['return'].rolling(20).mean()
df['roll_std_20'] = df['return'].rolling(20).std()
df['roll_skew_20'] = df['return'].rolling(20).skew()

df['target'] = df['return'].shift(-1)
df = df.dropna()

feature_cols = [c for c in df.columns if c not in ['price', 'return', 'target']]
X = df[feature_cols].values
y = df['target'].values

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]'''),
        ("code", '''from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42),
}

results = {}
for name_m, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Simple directional Sharpe proxy
    pred_sign = np.sign(y_pred)
    strat_ret = pred_sign * y_test
    sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(252) if strat_ret.std() > 0 else 0

    results[name_m] = {'MAE': mae, 'R2': r2, 'Sharpe': sharpe}
    print(f"{name_m}: MAE={mae:.6f}, R2={r2:.4f}, Sharpe={sharpe:.2f}")

res_df = pd.DataFrame(results).T'''),
        ("code", '''fig, axes = plt.subplots(1, 2, figsize=(14, 6))

bar_colors = [COLORS['blue'], COLORS['green'], COLORS['orange']]

# Panel 1 - MAE
bars1 = axes[0].bar(res_df.index, res_df['MAE'], color=bar_colors, width=0.5, edgecolor='white')
for bar, val in zip(bars1, res_df['MAE']):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00005,
                 f'{val:.5f}', ha='center', va='bottom', fontsize=10)
axes[0].set_ylabel('MAE')
axes[0].set_title('Mean Absolute Error')
axes[0].tick_params(axis='x', rotation=15)

# Panel 2 - R-squared
bars2 = axes[1].bar(res_df.index, res_df['R2'], color=bar_colors, width=0.5, edgecolor='white')
for bar, val in zip(bars2, res_df['R2']):
    offset = 0.002 if val >= 0 else -0.002
    va = 'bottom' if val >= 0 else 'top'
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                 f'{val:.4f}', ha='center', va=va, fontsize=10)
axes[1].set_ylabel('$R^2$')
axes[1].set_title('R-squared')
axes[1].tick_params(axis='x', rotation=15)
axes[1].axhline(0, color=COLORS['gray'], linestyle=':', linewidth=0.8)

plt.tight_layout()
save_fig(fig, 'ml_compare.pdf')
plt.show()'''),
    ]
    write_nb(name, cells)


def gen_case_momentum():
    name = "EMQA_case_momentum"
    cells = [
        ("md", f"""{badge(name)}

# {name}
Momentum trading strategy on Brent crude oil.
**Output:** `case_momentum.pdf`"""),
        ("code", STYLE_CODE),
        ("code", YFINANCE_CODE),
        ("code", '''# Fetch Brent
brent = fetch('BZ=F', start='2018-01-01')
log_ret = np.log(brent / brent.shift(1)).dropna()

# Momentum signal: buy when 20-day return > 0, else cash (0 return)
mom_20 = log_ret.rolling(20).sum()  # 20-day cumulative return
signal = (mom_20 > 0).astype(float).shift(1)  # signal known at end of day, trade next day
signal = signal.loc[log_ret.index].fillna(0)

# Strategy returns
strat_ret = signal * log_ret
buyhold_ret = log_ret

# Cumulative returns
cum_strat = strat_ret.cumsum()
cum_buyhold = buyhold_ret.cumsum()

sharpe_strat = strat_ret.mean() / strat_ret.std() * np.sqrt(252)
sharpe_bh = buyhold_ret.mean() / buyhold_ret.std() * np.sqrt(252)
print(f"Momentum Sharpe:   {sharpe_strat:.3f}")
print(f"Buy-Hold Sharpe:   {sharpe_bh:.3f}")
print(f"Momentum total:    {(np.exp(cum_strat.iloc[-1]) - 1)*100:.1f}%")
print(f"Buy-Hold total:    {(np.exp(cum_buyhold.iloc[-1]) - 1)*100:.1f}%")'''),
        ("code", '''fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(cum_strat.index, (np.exp(cum_strat) - 1) * 100, color=COLORS['blue'],
        linewidth=1.5, label=f'Momentum (Sharpe={sharpe_strat:.2f})')
ax.plot(cum_buyhold.index, (np.exp(cum_buyhold) - 1) * 100, color=COLORS['gray'],
        linewidth=1.5, label=f'Buy & Hold (Sharpe={sharpe_bh:.2f})')

# Shade momentum-in-market periods
ax.fill_between(signal.index, ax.get_ylim()[0], ax.get_ylim()[1],
                where=signal > 0, alpha=0.05, color=COLORS['green'], label='In Market')

ax.axhline(0, color=COLORS['gray'], linestyle=':', linewidth=0.8)
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Return (%)')
ax.set_title('Momentum Strategy (20-day) vs Buy & Hold - Brent Crude')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), frameon=False, ncol=3)

plt.tight_layout()
save_fig(fig, 'case_momentum.pdf')
plt.show()'''),
    ]
    write_nb(name, cells)


def gen_case_perf():
    name = "EMQA_case_perf"
    cells = [
        ("md", f"""{badge(name)}

# {name}
Strategy performance analysis: cumulative returns, drawdown, rolling Sharpe.
**Output:** `case_perf.pdf`"""),
        ("code", STYLE_CODE),
        ("code", YFINANCE_CODE),
        ("code", '''# Fetch Brent and compute momentum strategy (same as case_momentum)
brent = fetch('BZ=F', start='2018-01-01')
log_ret = np.log(brent / brent.shift(1)).dropna()

mom_20 = log_ret.rolling(20).sum()
signal = (mom_20 > 0).astype(float).shift(1).fillna(0)
strat_ret = signal * log_ret
buyhold_ret = log_ret

# Cumulative wealth
cum_strat = np.exp(strat_ret.cumsum())
cum_buyhold = np.exp(buyhold_ret.cumsum())

# Drawdown
peak_strat = cum_strat.cummax()
dd_strat = (cum_strat - peak_strat) / peak_strat * 100

peak_bh = cum_buyhold.cummax()
dd_bh = (cum_buyhold - peak_bh) / peak_bh * 100

# Rolling 60-day Sharpe
roll_sharpe_strat = strat_ret.rolling(60).mean() / strat_ret.rolling(60).std() * np.sqrt(252)
roll_sharpe_bh = buyhold_ret.rolling(60).mean() / buyhold_ret.rolling(60).std() * np.sqrt(252)

# Summary stats
total_ret_strat = (cum_strat.iloc[-1] - 1) * 100
total_ret_bh = (cum_buyhold.iloc[-1] - 1) * 100
max_dd_strat = dd_strat.min()
max_dd_bh = dd_bh.min()
sharpe_strat = strat_ret.mean() / strat_ret.std() * np.sqrt(252)
sharpe_bh = buyhold_ret.mean() / buyhold_ret.std() * np.sqrt(252)

print(f"{'Metric':<25} {'Momentum':>12} {'Buy & Hold':>12}")
print("-" * 50)
print(f"{'Total Return (%)':<25} {total_ret_strat:>12.1f} {total_ret_bh:>12.1f}")
print(f"{'Max Drawdown (%)':<25} {max_dd_strat:>12.1f} {max_dd_bh:>12.1f}")
print(f"{'Sharpe Ratio':<25} {sharpe_strat:>12.3f} {sharpe_bh:>12.3f}")'''),
        ("code", '''# 3-panel vertical
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# Panel 1 - Cumulative returns
axes[0].plot(cum_strat.index, (cum_strat - 1) * 100, color=COLORS['blue'], linewidth=1.5,
             label=f'Momentum ({total_ret_strat:.0f}%)')
axes[0].plot(cum_buyhold.index, (cum_buyhold - 1) * 100, color=COLORS['gray'], linewidth=1.5,
             label=f'Buy & Hold ({total_ret_bh:.0f}%)')
axes[0].axhline(0, color=COLORS['gray'], linestyle=':', linewidth=0.8)
axes[0].set_ylabel('Cumulative Return (%)')
axes[0].set_title('Strategy Performance Analysis - Brent Crude Momentum')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), frameon=False, ncol=2)

# Panel 2 - Drawdown
axes[1].fill_between(dd_strat.index, dd_strat.values, 0, color=COLORS['blue'], alpha=0.4,
                      label=f'Momentum (max={max_dd_strat:.1f}%)')
axes[1].fill_between(dd_bh.index, dd_bh.values, 0, color=COLORS['gray'], alpha=0.3,
                      label=f'Buy & Hold (max={max_dd_bh:.1f}%)')
axes[1].set_ylabel('Drawdown (%)')
axes[1].set_title('Drawdown')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), frameon=False, ncol=2)

# Panel 3 - Rolling 60-day Sharpe
axes[2].plot(roll_sharpe_strat.index, roll_sharpe_strat.values, color=COLORS['blue'],
             linewidth=1, label='Momentum')
axes[2].plot(roll_sharpe_bh.index, roll_sharpe_bh.values, color=COLORS['gray'],
             linewidth=1, label='Buy & Hold')
axes[2].axhline(0, color=COLORS['gray'], linestyle=':', linewidth=0.8)
axes[2].set_ylabel('Sharpe Ratio')
axes[2].set_xlabel('Date')
axes[2].set_title('Rolling 60-day Sharpe Ratio')
axes[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), frameon=False, ncol=2)

plt.tight_layout()
save_fig(fig, 'case_perf.pdf')
plt.show()'''),
    ]
    write_nb(name, cells)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    generators = [
        # Lecture 3 - GARCH
        gen_vol_clustering,
        gen_news_impact,
        gen_garch_oil,
        gen_garch_estimation,
        gen_vol_dynamics,
        gen_var_calculation,
        gen_garch_electricity,
        # Lecture 4 - Risk/Trading
        gen_hedge_ratio,
        gen_hedge_compare,
        gen_portfolio,
        gen_crack_spread,
        gen_spark_spread,
        gen_pairs_trading,
        gen_ml_rf,
        gen_ml_compare,
        gen_case_momentum,
        gen_case_perf,
    ]

    print(f"Generating {len(generators)} notebooks...\n")
    for gen_func in generators:
        gen_func()

    print(f"\nDone! Created {len(generators)} notebooks.")
