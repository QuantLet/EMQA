#!/usr/bin/env python3
"""Generate Lecture 2 (ARIMA) quantlet notebooks."""
import json, os

BASE = os.path.dirname(os.path.abspath(__file__))

COLAB_BASE = "https://colab.research.google.com/github/QuantLet/EMQA/blob/main/EN/quantlets"

STYLE_CODE = """import numpy as np
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
"""

YFINANCE_CODE = """import yfinance as yf

def fetch(ticker, start='2020-01-01', end='2025-12-31'):
    d = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        return d['Close'].squeeze().dropna()
    return d['Close'].dropna()
"""

ELEC_CODE = """import os
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
"""


def nb(cells):
    nb_cells = []
    for ctype, src in cells:
        if ctype == "md":
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": [src]})
        else:
            nb_cells.append({
                "cell_type": "code", "metadata": {}, "source": [src],
                "outputs": [], "execution_count": None
            })
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
    with open(path, "w") as f:
        json.dump(nb(cells), f, indent=1)
    print(f"Created: {path}")


# ---------------------------------------------------------------------------
# Helper to build the Colab badge + title markdown
# ---------------------------------------------------------------------------
def badge(name, title, description):
    return (
        f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        f"({COLAB_BASE}/{name}/{name}.ipynb)\n\n"
        f"# {name}\n\n"
        f"**{title}**\n\n"
        f"{description}"
    )


# ===================================================================
# 1. EMQA_AR1_simulation
# ===================================================================
def gen_ar1_simulation():
    NAME = "EMQA_AR1_simulation"
    write_nb(NAME, [
        ("md", badge(NAME, "AR(1) Process Simulation",
                      "Simulate AR(1) processes with different autoregressive coefficients "
                      "and observe how the parameter phi controls persistence and mean-reversion.")),
        ("code", STYLE_CODE),
        ("code", """from statsmodels.tsa.arima_process import ArmaProcess

np.random.seed(42)
n = 200

phis = [0.95, 0.5, -0.7, 0.99]
labels = [r'$\\phi = 0.95$', r'$\\phi = 0.5$', r'$\\phi = -0.7$', r'$\\phi = 0.99$']
subtitles = [
    'High persistence, slow mean-reversion',
    'Moderate persistence',
    'Negative autocorrelation (oscillating)',
    'Near unit root (almost random walk)'
]

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.ravel()

for i, (phi, lab, sub) in enumerate(zip(phis, labels, subtitles)):
    ar_params = np.array([1, -phi])
    ma_params = np.array([1])
    process = ArmaProcess(ar_params, ma_params)
    y = process.generate_sample(nsample=n)

    axes[i].plot(y, color=COLORS['blue'], linewidth=0.9)
    axes[i].axhline(0, color=COLORS['gray'], linewidth=0.5, linestyle='--')
    axes[i].set_title(f'{lab}  —  {sub}', fontsize=11)
    axes[i].set_xlabel('Time')
    axes[i].set_ylabel('Value')

fig.suptitle('AR(1) Process Simulations', fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
save_fig(fig, 'ar1_phi_comparison.pdf')
plt.show()
"""),
    ])


# ===================================================================
# 2. EMQA_ARMA_comparison
# ===================================================================
def gen_arma_comparison():
    NAME = "EMQA_ARMA_comparison"
    write_nb(NAME, [
        ("md", badge(NAME, "ARMA Process Comparison",
                      "Compare White Noise, AR(1), MA(1), and ARMA(1,1) processes side by side.")),
        ("code", STYLE_CODE),
        ("code", """from statsmodels.tsa.arima_process import ArmaProcess

np.random.seed(42)
n = 300

configs = [
    {'ar': [1], 'ma': [1], 'title': 'White Noise',
     'desc': 'No memory: each observation\\nis independent'},
    {'ar': [1, -0.9], 'ma': [1], 'title': r'AR(1)  $\\phi=0.9$',
     'desc': 'Strong persistence:\\ncurrent value depends on past'},
    {'ar': [1], 'ma': [1, 0.8], 'title': r'MA(1)  $\\theta=0.8$',
     'desc': 'Short memory: only one\\nlagged shock matters'},
    {'ar': [1, -0.7], 'ma': [1, 0.5], 'title': r'ARMA(1,1)  $\\phi=0.7,\\;\\theta=0.5$',
     'desc': 'Mixed: combines AR\\npersistence with MA shock'},
]

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.ravel()
colors_list = [COLORS['blue'], COLORS['red'], COLORS['green'], COLORS['purple']]

for i, cfg in enumerate(configs):
    proc = ArmaProcess(np.array(cfg['ar']), np.array(cfg['ma']))
    y = proc.generate_sample(nsample=n)
    axes[i].plot(y, color=colors_list[i], linewidth=0.8, alpha=0.9)
    axes[i].axhline(0, color=COLORS['gray'], linewidth=0.5, linestyle='--')
    axes[i].set_title(cfg['title'], fontsize=12, fontweight='bold')

    props = dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.7)
    axes[i].text(0.02, 0.97, cfg['desc'], transform=axes[i].transAxes,
                 fontsize=9, verticalalignment='top', bbox=props)
    axes[i].set_xlabel('Time')
    axes[i].set_ylabel('Value')

fig.suptitle('ARMA Process Comparison', fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
save_fig(fig, 'arma_processes_comparison.pdf')
plt.show()
"""),
    ])


# ===================================================================
# 3. EMQA_ACF_PACF_identification
# ===================================================================
def gen_acf_pacf_identification():
    NAME = "EMQA_ACF_PACF_identification"
    write_nb(NAME, [
        ("md", badge(NAME, "ACF / PACF Pattern Identification Guide",
                      "A 4x2 grid showing the theoretical ACF and PACF signatures "
                      "of White Noise, AR(1), MA(1), and ARMA(1,1).")),
        ("code", STYLE_CODE),
        ("code", """from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import acf, pacf

np.random.seed(42)
n = 500
nlags = 25

configs = [
    {'ar': [1], 'ma': [1], 'label': 'White Noise',
     'acf_note': 'All lags near zero', 'pacf_note': 'All lags near zero'},
    {'ar': [1, -0.9], 'ma': [1], 'label': 'AR(1)',
     'acf_note': 'Exponential decay', 'pacf_note': 'Cuts off after lag 1'},
    {'ar': [1], 'ma': [1, 0.8], 'label': 'MA(1)',
     'acf_note': 'Cuts off after lag 1', 'pacf_note': 'Exponential decay'},
    {'ar': [1, -0.7], 'ma': [1, 0.5], 'label': 'ARMA(1,1)',
     'acf_note': 'Tails off (mixed decay)', 'pacf_note': 'Tails off (mixed decay)'},
]

fig, axes = plt.subplots(4, 2, figsize=(14, 14))
conf_bound = 1.96 / np.sqrt(n)

for row, cfg in enumerate(configs):
    proc = ArmaProcess(np.array(cfg['ar']), np.array(cfg['ma']))
    y = proc.generate_sample(nsample=n)

    acf_vals = acf(y, nlags=nlags)
    pacf_vals = pacf(y, nlags=nlags)

    # ACF
    ax_acf = axes[row, 0]
    ax_acf.bar(range(nlags + 1), acf_vals, width=0.3, color=COLORS['blue'], alpha=0.85)
    ax_acf.axhline(conf_bound, color=COLORS['red'], linestyle='--', linewidth=0.8)
    ax_acf.axhline(-conf_bound, color=COLORS['red'], linestyle='--', linewidth=0.8)
    ax_acf.axhline(0, color='black', linewidth=0.5)
    ax_acf.set_title(f'{cfg["label"]} — ACF', fontsize=11, fontweight='bold')
    ax_acf.set_xlabel('Lag')
    ax_acf.set_ylabel('ACF')
    props = dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8)
    ax_acf.text(0.98, 0.95, cfg['acf_note'], transform=ax_acf.transAxes,
                fontsize=9, ha='right', va='top', bbox=props)

    # PACF
    ax_pacf = axes[row, 1]
    ax_pacf.bar(range(nlags + 1), pacf_vals, width=0.3, color=COLORS['green'], alpha=0.85)
    ax_pacf.axhline(conf_bound, color=COLORS['red'], linestyle='--', linewidth=0.8)
    ax_pacf.axhline(-conf_bound, color=COLORS['red'], linestyle='--', linewidth=0.8)
    ax_pacf.axhline(0, color='black', linewidth=0.5)
    ax_pacf.set_title(f'{cfg["label"]} — PACF', fontsize=11, fontweight='bold')
    ax_pacf.set_xlabel('Lag')
    ax_pacf.set_ylabel('PACF')
    ax_pacf.text(0.98, 0.95, cfg['pacf_note'], transform=ax_pacf.transAxes,
                 fontsize=9, ha='right', va='top', bbox=props)

fig.suptitle('ACF / PACF Identification Guide', fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()
save_fig(fig, 'arma_acf_pacf_identification.pdf')
plt.show()
"""),
    ])


# ===================================================================
# 4. EMQA_arima_intro
# ===================================================================
def gen_arima_intro():
    NAME = "EMQA_arima_intro"
    write_nb(NAME, [
        ("md", badge(NAME, "ARIMA Notation and Concepts",
                      "Visual comparison of a non-stationary Random Walk and a stationary "
                      "AR(1) process, with ARIMA(p,d,q) component annotations.")),
        ("code", STYLE_CODE),
        ("code", """np.random.seed(42)
n = 300

# Random walk (non-stationary)
rw = np.cumsum(np.random.randn(n))

# AR(1) stationary process
from statsmodels.tsa.arima_process import ArmaProcess
ar1 = ArmaProcess(np.array([1, -0.8]), np.array([1])).generate_sample(nsample=n)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Random Walk
axes[0].plot(rw, color=COLORS['red'], linewidth=1.0)
axes[0].axhline(0, color=COLORS['gray'], linestyle='--', linewidth=0.5)
axes[0].set_title('Random Walk (Non-Stationary)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Value')
props = dict(boxstyle='round,pad=0.5', facecolor='#FDEBD0', alpha=0.85)
rw_text = ("ARIMA(0,1,0)\\n"
           "d = 1: needs differencing\\n"
           "No constant mean\\n"
           "Variance grows with time")
axes[0].text(0.03, 0.97, rw_text, transform=axes[0].transAxes,
             fontsize=9, va='top', bbox=props)

# Panel 2: AR(1)
axes[1].plot(ar1, color=COLORS['blue'], linewidth=1.0)
axes[1].axhline(0, color=COLORS['gray'], linestyle='--', linewidth=0.5)
axes[1].set_title('AR(1) Process (Stationary)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Value')
ar_text = ("ARIMA(1,0,0)\\n"
           "d = 0: already stationary\\n"
           "Constant mean (reverts to 0)\\n"
           "Bounded variance")
axes[1].text(0.03, 0.97, ar_text, transform=axes[1].transAxes,
             fontsize=9, va='top', bbox=props)

# Central annotation
fig.text(0.5, -0.06,
         "ARIMA(p, d, q)\\n"
         "p = AR order (persistence)   |   d = differencing order (stationarity)   |   q = MA order (shocks)",
         ha='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.6', facecolor='#D5F5E3', alpha=0.9))

fig.suptitle('ARIMA: Notation and Core Concepts', fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
save_fig(fig, 'arima_intro.pdf')
plt.show()
"""),
    ])


# ===================================================================
# 5. EMQA_arima_oil
# ===================================================================
def gen_arima_oil():
    NAME = "EMQA_arima_oil"
    write_nb(NAME, [
        ("md", badge(NAME, "Oil Price ARIMA Analysis",
                      "Four-panel analysis of Brent crude oil prices: levels, log returns, "
                      "ACF, and return distribution.")),
        ("code", STYLE_CODE),
        ("code", YFINANCE_CODE),
        ("code", """from statsmodels.tsa.stattools import acf

brent = fetch('BZ=F')
log_ret = np.log(brent / brent.shift(1)).dropna()

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Panel 1: Price levels
axes[0, 0].plot(brent.index, brent.values, color=COLORS['blue'], linewidth=0.8)
axes[0, 0].set_title('Brent Crude Oil — Price Levels', fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Price (USD)')

# Panel 2: Log returns
axes[0, 1].plot(log_ret.index, log_ret.values, color=COLORS['red'], linewidth=0.5, alpha=0.8)
axes[0, 1].axhline(0, color=COLORS['gray'], linestyle='--', linewidth=0.5)
axes[0, 1].set_title('Log Returns', fontweight='bold')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Log Return')

# Panel 3: ACF of returns
nlags = 40
acf_vals = acf(log_ret, nlags=nlags)
axes[1, 0].bar(range(nlags + 1), acf_vals, width=0.4, color=COLORS['green'], alpha=0.85)
conf = 1.96 / np.sqrt(len(log_ret))
axes[1, 0].axhline(conf, color=COLORS['red'], linestyle='--', linewidth=0.8)
axes[1, 0].axhline(-conf, color=COLORS['red'], linestyle='--', linewidth=0.8)
axes[1, 0].axhline(0, color='black', linewidth=0.5)
axes[1, 0].set_title('ACF of Log Returns', fontweight='bold')
axes[1, 0].set_xlabel('Lag')
axes[1, 0].set_ylabel('ACF')

# Panel 4: Histogram with normal overlay
axes[1, 1].hist(log_ret, bins=60, density=True, color=COLORS['blue'], alpha=0.6, edgecolor='white')
x_range = np.linspace(log_ret.min(), log_ret.max(), 200)
mu, sigma = log_ret.mean(), log_ret.std()
from scipy.stats import norm
axes[1, 1].plot(x_range, norm.pdf(x_range, mu, sigma), color=COLORS['red'], linewidth=2,
                label='Normal fit')
axes[1, 1].set_title('Return Distribution', fontweight='bold')
axes[1, 1].set_xlabel('Log Return')
axes[1, 1].set_ylabel('Density')
axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), frameon=False)

fig.suptitle('Brent Crude Oil — ARIMA Analysis', fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
save_fig(fig, 'arima_oil_analysis.pdf')
plt.show()
"""),
    ])


# ===================================================================
# 6. EMQA_oil_acf_pacf
# ===================================================================
def gen_oil_acf_pacf():
    NAME = "EMQA_oil_acf_pacf"
    write_nb(NAME, [
        ("md", badge(NAME, "Oil Returns ACF/PACF for Order Selection",
                      "ACF and PACF plots of Brent oil log returns with 40 lags and 95% "
                      "confidence intervals to guide ARIMA order selection.")),
        ("code", STYLE_CODE),
        ("code", YFINANCE_CODE),
        ("code", """from statsmodels.tsa.stattools import acf, pacf

brent = fetch('BZ=F')
log_ret = np.log(brent / brent.shift(1)).dropna()

nlags = 40
acf_vals = acf(log_ret, nlags=nlags)
pacf_vals = pacf(log_ret, nlags=nlags)
conf = 1.96 / np.sqrt(len(log_ret))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ACF
axes[0].bar(range(nlags + 1), acf_vals, width=0.4, color=COLORS['blue'], alpha=0.85)
axes[0].axhline(conf, color=COLORS['red'], linestyle='--', linewidth=0.8, label='95% CI')
axes[0].axhline(-conf, color=COLORS['red'], linestyle='--', linewidth=0.8)
axes[0].axhline(0, color='black', linewidth=0.5)
axes[0].set_title('ACF of Brent Log Returns', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Lag')
axes[0].set_ylabel('ACF')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), frameon=False)

# Count significant ACF lags (excluding lag 0)
sig_acf = [k for k in range(1, nlags + 1) if abs(acf_vals[k]) > conf]
acf_note = f"Significant lags: {sig_acf[:5]}..." if len(sig_acf) > 5 else f"Significant lags: {sig_acf}"
props = dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8)
axes[0].text(0.98, 0.95, f"Suggested q (MA order)\\n{acf_note}",
             transform=axes[0].transAxes, fontsize=9, ha='right', va='top', bbox=props)

# PACF
axes[1].bar(range(nlags + 1), pacf_vals, width=0.4, color=COLORS['green'], alpha=0.85)
axes[1].axhline(conf, color=COLORS['red'], linestyle='--', linewidth=0.8, label='95% CI')
axes[1].axhline(-conf, color=COLORS['red'], linestyle='--', linewidth=0.8)
axes[1].axhline(0, color='black', linewidth=0.5)
axes[1].set_title('PACF of Brent Log Returns', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Lag')
axes[1].set_ylabel('PACF')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), frameon=False)

sig_pacf = [k for k in range(1, nlags + 1) if abs(pacf_vals[k]) > conf]
pacf_note = f"Significant lags: {sig_pacf[:5]}..." if len(sig_pacf) > 5 else f"Significant lags: {sig_pacf}"
axes[1].text(0.98, 0.95, f"Suggested p (AR order)\\n{pacf_note}",
             transform=axes[1].transAxes, fontsize=9, ha='right', va='top', bbox=props)

fig.suptitle('Brent Oil — ACF / PACF Order Selection', fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
save_fig(fig, 'oil_acf_pacf_selection.pdf')
plt.show()
"""),
    ])


# ===================================================================
# 7. EMQA_arima_diagnostics
# ===================================================================
def gen_arima_diagnostics():
    NAME = "EMQA_arima_diagnostics"
    write_nb(NAME, [
        ("md", badge(NAME, "ARIMA Residual Diagnostics",
                      "Fit ARIMA(1,1,1) to Brent crude oil prices and run a full residual "
                      "diagnostic suite: time series, histogram, ACF, and QQ-plot.")),
        ("code", STYLE_CODE),
        ("code", YFINANCE_CODE),
        ("code", """from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

brent = fetch('BZ=F')

# Fit ARIMA(1,1,1)
model = ARIMA(brent, order=(1, 1, 1))
result = model.fit()
print(result.summary())
"""),
        ("code", """resid = result.resid
std_resid = (resid - resid.mean()) / resid.std()

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Panel 1: Standardized residuals
axes[0, 0].plot(std_resid.index, std_resid.values, color=COLORS['blue'], linewidth=0.5, alpha=0.8)
axes[0, 0].axhline(0, color=COLORS['gray'], linestyle='--', linewidth=0.5)
axes[0, 0].axhline(2, color=COLORS['red'], linestyle=':', linewidth=0.8, alpha=0.5)
axes[0, 0].axhline(-2, color=COLORS['red'], linestyle=':', linewidth=0.8, alpha=0.5)
axes[0, 0].set_title('Standardized Residuals', fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Std. Residual')

# Panel 2: Histogram + Normal
axes[0, 1].hist(std_resid, bins=50, density=True, color=COLORS['blue'], alpha=0.6, edgecolor='white')
x_range = np.linspace(-4, 4, 200)
axes[0, 1].plot(x_range, stats.norm.pdf(x_range), color=COLORS['red'], linewidth=2, label='Normal')
axes[0, 1].set_title('Residual Distribution', fontweight='bold')
axes[0, 1].set_xlabel('Standardized Residual')
axes[0, 1].set_ylabel('Density')
axes[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), frameon=False)

# Panel 3: ACF of residuals
nlags = 30
acf_vals = acf(resid.dropna(), nlags=nlags)
conf = 1.96 / np.sqrt(len(resid.dropna()))
axes[1, 0].bar(range(nlags + 1), acf_vals, width=0.4, color=COLORS['green'], alpha=0.85)
axes[1, 0].axhline(conf, color=COLORS['red'], linestyle='--', linewidth=0.8)
axes[1, 0].axhline(-conf, color=COLORS['red'], linestyle='--', linewidth=0.8)
axes[1, 0].axhline(0, color='black', linewidth=0.5)
axes[1, 0].set_title('ACF of Residuals', fontweight='bold')
axes[1, 0].set_xlabel('Lag')
axes[1, 0].set_ylabel('ACF')

# Panel 4: QQ-plot
stats.probplot(std_resid.dropna(), dist='norm', plot=axes[1, 1])
axes[1, 1].get_lines()[0].set(color=COLORS['blue'], markersize=3, alpha=0.6)
axes[1, 1].get_lines()[1].set(color=COLORS['red'], linewidth=1.5)
axes[1, 1].set_title('Normal QQ-Plot', fontweight='bold')

fig.suptitle('ARIMA(1,1,1) Residual Diagnostics — Brent Oil', fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
save_fig(fig, 'arima_diagnostics.pdf')
plt.show()
"""),
        ("code", """# Ljung-Box test
lb_test = acorr_ljungbox(resid.dropna(), lags=[10, 20, 30], return_df=True)
print("Ljung-Box Test Results:")
print(lb_test)
print()
if (lb_test['lb_pvalue'] > 0.05).all():
    print("=> No significant autocorrelation in residuals (good model fit)")
else:
    print("=> Some residual autocorrelation detected (consider adjusting the model)")
"""),
    ])


# ===================================================================
# 8. EMQA_arima_forecast
# ===================================================================
def gen_arima_forecast():
    NAME = "EMQA_arima_forecast"
    write_nb(NAME, [
        ("md", badge(NAME, "ARIMA Forecast with Confidence Intervals",
                      "Fit an ARIMA model to Brent crude oil and produce a 30-step-ahead "
                      "forecast with 95% confidence intervals.")),
        ("code", STYLE_CODE),
        ("code", YFINANCE_CODE),
        ("code", """from statsmodels.tsa.arima.model import ARIMA

brent = fetch('BZ=F')

# Fit ARIMA(1,1,1)
model = ARIMA(brent, order=(1, 1, 1))
result = model.fit()
print(result.summary())
"""),
        ("code", """# Forecast 30 steps ahead
forecast_steps = 30
forecast = result.get_forecast(steps=forecast_steps)
fc_mean = forecast.predicted_mean
fc_ci = forecast.conf_int(alpha=0.05)

fig, ax = plt.subplots(figsize=(14, 6))

# Plot last 200 observations for context
hist_data = brent.iloc[-200:]
ax.plot(hist_data.index, hist_data.values, color=COLORS['blue'], linewidth=1.0,
        label='Historical')

# Forecast
ax.plot(fc_mean.index, fc_mean.values, color=COLORS['red'], linewidth=2.0,
        label='Forecast')

# Confidence interval
ax.fill_between(fc_ci.index, fc_ci.iloc[:, 0], fc_ci.iloc[:, 1],
                color=COLORS['red'], alpha=0.15, label='95% CI')

# Vertical line at forecast origin
ax.axvline(brent.index[-1], color=COLORS['gray'], linestyle='--', linewidth=0.8, alpha=0.7)

ax.set_title('ARIMA(1,1,1) Forecast — Brent Crude Oil', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), frameon=False, ncol=3)

fig.tight_layout()
save_fig(fig, 'arima_forecast.pdf')
plt.show()
"""),
    ])


# ===================================================================
# 9. EMQA_arima_electricity
# ===================================================================
def gen_arima_electricity():
    NAME = "EMQA_arima_electricity"
    write_nb(NAME, [
        ("md", badge(NAME, "Electricity Price SARIMA Forecast",
                      "Load electricity price data, resample to daily frequency, and fit "
                      "a seasonal ARIMA (SARIMAX) model with weekly seasonality.")),
        ("code", STYLE_CODE),
        ("code", ELEC_CODE),
        ("code", """from statsmodels.tsa.statespace.sarimax import SARIMAX

# Resample to daily mean
daily = elec_price.resample('D').mean().dropna()
print(f"Daily electricity prices: {len(daily)} observations")
print(f"Date range: {daily.index[0]} to {daily.index[-1]}")
print(daily.describe())
"""),
        ("code", """# Fit SARIMAX(1,1,1)(1,1,1,7) — weekly seasonality
try:
    model = SARIMAX(daily, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7),
                    enforce_stationarity=False, enforce_invertibility=False)
    result = model.fit(disp=False, maxiter=200)
    print(result.summary())
except Exception as e:
    print(f"SARIMAX(1,1,1)(1,1,1,7) failed: {e}")
    print("Falling back to simpler SARIMAX(1,1,1)(0,1,1,7)")
    model = SARIMAX(daily, order=(1, 1, 1), seasonal_order=(0, 1, 1, 7),
                    enforce_stationarity=False, enforce_invertibility=False)
    result = model.fit(disp=False, maxiter=200)
    print(result.summary())
"""),
        ("code", """# Forecast
forecast_steps = 30
forecast = result.get_forecast(steps=forecast_steps)
fc_mean = forecast.predicted_mean
fc_ci = forecast.conf_int(alpha=0.05)

fig, ax = plt.subplots(figsize=(14, 6))

# Last 120 days of history
hist = daily.iloc[-120:]
ax.plot(hist.index, hist.values, color=COLORS['blue'], linewidth=1.0, label='Historical')
ax.plot(fc_mean.index, fc_mean.values, color=COLORS['red'], linewidth=2.0, label='Forecast')
ax.fill_between(fc_ci.index, fc_ci.iloc[:, 0], fc_ci.iloc[:, 1],
                color=COLORS['red'], alpha=0.15, label='95% CI')
ax.axvline(daily.index[-1], color=COLORS['gray'], linestyle='--', linewidth=0.8, alpha=0.7)

ax.set_title('SARIMA Forecast — Daily Electricity Prices', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), frameon=False, ncol=3)

fig.tight_layout()
save_fig(fig, 'arima_electricity.pdf')
plt.show()
"""),
    ])


# ===================================================================
# 10. EMQA_model_selection
# ===================================================================
def gen_model_selection():
    NAME = "EMQA_model_selection"
    write_nb(NAME, [
        ("md", badge(NAME, "AIC / BIC Model Selection Grid",
                      "Fit multiple ARIMA(p,d,q) specifications to Brent oil returns "
                      "and display AIC values in a heatmap to identify the best model.")),
        ("code", STYLE_CODE),
        ("code", YFINANCE_CODE),
        ("code", """from statsmodels.tsa.arima.model import ARIMA
import itertools

brent = fetch('BZ=F')
log_ret = np.log(brent / brent.shift(1)).dropna()

p_range = range(0, 4)
d_range = range(0, 2)
q_range = range(0, 4)

results = []
for p, d, q in itertools.product(p_range, d_range, q_range):
    try:
        model = ARIMA(log_ret, order=(p, d, q))
        fit = model.fit()
        results.append({'p': p, 'd': d, 'q': q,
                        'AIC': fit.aic, 'BIC': fit.bic})
    except Exception:
        results.append({'p': p, 'd': d, 'q': q,
                        'AIC': np.nan, 'BIC': np.nan})

df_res = pd.DataFrame(results)
print(f"Tested {len(df_res)} model specifications")
print()

# Best models
best_aic = df_res.loc[df_res['AIC'].idxmin()]
best_bic = df_res.loc[df_res['BIC'].idxmin()]
print(f"Best AIC: ARIMA({int(best_aic.p)},{int(best_aic.d)},{int(best_aic.q)}) = {best_aic.AIC:.2f}")
print(f"Best BIC: ARIMA({int(best_bic.p)},{int(best_bic.d)},{int(best_bic.q)}) = {best_bic.BIC:.2f}")
"""),
        ("code", """# Heatmaps for d=0 and d=1
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, d_val in enumerate([0, 1]):
    subset = df_res[df_res['d'] == d_val].copy()
    pivot = subset.pivot_table(index='p', columns='q', values='AIC')

    im = axes[idx].imshow(pivot.values, cmap='RdYlGn_r', aspect='auto')
    axes[idx].set_xticks(range(len(pivot.columns)))
    axes[idx].set_xticklabels(pivot.columns.astype(int))
    axes[idx].set_yticks(range(len(pivot.index)))
    axes[idx].set_yticklabels(pivot.index.astype(int))
    axes[idx].set_xlabel('q (MA order)')
    axes[idx].set_ylabel('p (AR order)')
    axes[idx].set_title(f'AIC Heatmap — d = {d_val}', fontweight='bold')

    # Annotate cells
    for r in range(len(pivot.index)):
        for c in range(len(pivot.columns)):
            val = pivot.values[r, c]
            if not np.isnan(val):
                axes[idx].text(c, r, f'{val:.0f}', ha='center', va='center', fontsize=8,
                               color='white' if val > pivot.values[~np.isnan(pivot.values)].mean() else 'black')

    # Highlight best in this panel
    best_in_panel = subset.loc[subset['AIC'].idxmin()]
    bp = int(best_in_panel.p)
    bq = int(best_in_panel.q)
    r_idx = list(pivot.index).index(bp)
    c_idx = list(pivot.columns).index(bq)
    rect = plt.Rectangle((c_idx - 0.5, r_idx - 0.5), 1, 1,
                          linewidth=3, edgecolor=COLORS['red'], facecolor='none')
    axes[idx].add_patch(rect)

    plt.colorbar(im, ax=axes[idx], shrink=0.8)

fig.suptitle('ARIMA Model Selection — AIC Grid (Brent Oil Returns)',
             fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
save_fig(fig, 'model_selection_aic_bic.pdf')
plt.show()
"""),
        ("code", """# Summary table of top 10 models by AIC
top10 = df_res.nsmallest(10, 'AIC')[['p', 'd', 'q', 'AIC', 'BIC']].reset_index(drop=True)
top10.index += 1
top10.index.name = 'Rank'
print("Top 10 Models by AIC:")
print(top10.to_string())
"""),
    ])


# ===================================================================
# 11. EMQA_electricity_analysis
# ===================================================================
def gen_electricity_analysis():
    NAME = "EMQA_electricity_analysis"
    write_nb(NAME, [
        ("md", badge(NAME, "Electricity Price Statistics",
                      "Exploratory analysis of electricity prices: time series, distribution, "
                      "hourly boxplots, and monthly averages.")),
        ("code", STYLE_CODE),
        ("code", ELEC_CODE),
        ("code", """print("Electricity Price — Descriptive Statistics")
print("=" * 50)
print(elec_price.describe())
print(f"\\nSkewness:  {elec_price.skew():.4f}")
print(f"Kurtosis:  {elec_price.kurtosis():.4f}")
print(f"Date range: {elec_price.index[0]} to {elec_price.index[-1]}")
print(f"Observations: {len(elec_price)}")
"""),
        ("code", """fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Panel 1: Price time series
axes[0, 0].plot(elec_price.index, elec_price.values, color=COLORS['blue'],
                linewidth=0.4, alpha=0.8)
axes[0, 0].set_title('Electricity Price — Time Series', fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Price')

# Panel 2: Histogram
axes[0, 1].hist(elec_price, bins=80, density=True, color=COLORS['orange'],
                alpha=0.7, edgecolor='white')
axes[0, 1].axvline(elec_price.mean(), color=COLORS['red'], linestyle='--',
                    linewidth=1.5, label=f'Mean = {elec_price.mean():.2f}')
axes[0, 1].axvline(elec_price.median(), color=COLORS['green'], linestyle='--',
                    linewidth=1.5, label=f'Median = {elec_price.median():.2f}')
axes[0, 1].set_title('Price Distribution', fontweight='bold')
axes[0, 1].set_xlabel('Price')
axes[0, 1].set_ylabel('Density')
axes[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), frameon=False)

# Panel 3: Boxplot by hour (if hourly data available)
if hasattr(elec_price.index, 'hour'):
    hourly_data = elec_price.copy()
    hourly_data_df = pd.DataFrame({'price': hourly_data, 'hour': hourly_data.index.hour})
    bp = axes[1, 0].boxplot(
        [hourly_data_df[hourly_data_df['hour'] == h]['price'].values for h in range(24)],
        positions=range(24), widths=0.6, patch_artist=True,
        boxprops=dict(facecolor=COLORS['cyan'], alpha=0.6),
        medianprops=dict(color=COLORS['red'], linewidth=1.5),
        flierprops=dict(marker='.', markersize=2, alpha=0.3),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8)
    )
    axes[1, 0].set_title('Price Distribution by Hour', fontweight='bold')
    axes[1, 0].set_xlabel('Hour of Day')
    axes[1, 0].set_ylabel('Price')
    axes[1, 0].set_xticks(range(0, 24, 3))
else:
    axes[1, 0].text(0.5, 0.5, 'Hourly breakdown not available\\n(data not hourly)',
                    transform=axes[1, 0].transAxes, ha='center', va='center', fontsize=12)
    axes[1, 0].set_title('Price Distribution by Hour', fontweight='bold')

# Panel 4: Monthly averages
monthly_avg = elec_price.resample('M').mean()
axes[1, 1].bar(monthly_avg.index, monthly_avg.values, width=20, color=COLORS['purple'],
               alpha=0.7)
axes[1, 1].set_title('Monthly Average Price', fontweight='bold')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Avg Price')

fig.suptitle('Electricity Price — Exploratory Analysis', fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
save_fig(fig, 'electricity_analysis.pdf')
plt.show()
"""),
    ])


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    generators = [
        gen_ar1_simulation,
        gen_arma_comparison,
        gen_acf_pacf_identification,
        gen_arima_intro,
        gen_arima_oil,
        gen_oil_acf_pacf,
        gen_arima_diagnostics,
        gen_arima_forecast,
        gen_arima_electricity,
        gen_model_selection,
        gen_electricity_analysis,
    ]

    count = 0
    for gen_func in generators:
        gen_func()
        count += 1

    print(f"\nDone — {count} notebooks created.")
