#!/usr/bin/env python3
"""Generate ALL Lecture 1 quantlet notebooks at once."""

import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))

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

ROMANIA_CODE = '''import os

def load_romania():
    paths = [
        '../../charts/ro_de_prices_full.csv',
        '/Users/danielpele/Documents/Energy MBA/charts/ro_de_prices_full.csv',
    ]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p, parse_dates=['date'], index_col='date')
    raise FileNotFoundError("ro_de_prices_full.csv not found")

ro = load_romania()
'''

def nb(cells):
    """Create notebook JSON from list of (type, source) tuples."""
    nb_cells = []
    for ctype, src in cells:
        if ctype == 'md':
            nb_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [src]
            })
        else:
            nb_cells.append({
                "cell_type": "code",
                "metadata": {},
                "source": [src],
                "outputs": [],
                "execution_count": None
            })
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
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

# ========================================================================
# OIL / COMMODITY QUANTLETS
# ========================================================================

write_nb("EMQA_energy_overview", [
    ("md", "# EMQA_energy_overview\n\nOverview of major energy commodity prices (2020-2025).\n\n**Output:** `energy_markets_overview.pdf`"),
    ("code", STYLE_CODE),
    ("code", YFINANCE_CODE),
    ("code", '''tickers = {'Brent Oil': 'BZ=F', 'WTI Oil': 'CL=F', 'Natural Gas': 'NG=F', 'Heating Oil': 'HO=F'}
data = {k: fetch(v) for k, v in tickers.items()}
df = pd.DataFrame(data).dropna()

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
colors = [COLORS['blue'], COLORS['red'], COLORS['green'], COLORS['orange']]

for ax, (name, col), c in zip(axes.flat, df.items(), colors):
    ax.plot(df.index, col, color=c, linewidth=0.8)
    ax.axhline(col.mean(), color='gray', ls='--', alpha=0.5, lw=0.8)
    ax.set_title(name, fontweight='bold')
    ax.set_ylabel('USD')

handles = [plt.Line2D([0],[0], color=c, lw=2) for c in colors]
fig.legend(handles, list(df.columns), loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=4, frameon=False)
fig.suptitle('Energy Commodity Prices (2020-2025)', fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0.03, 1, 0.97])
save_fig(fig, 'energy_markets_overview.pdf')
plt.show()
'''),
])

write_nb("EMQA_oil_analysis", [
    ("md", "# EMQA_oil_analysis\n\nBrent crude oil historical analysis (2015-2025).\n\n**Output:** `oil_prices_historical.pdf`"),
    ("code", STYLE_CODE),
    ("code", YFINANCE_CODE),
    ("code", '''brent = fetch('BZ=F', '2015-01-01', '2025-12-31')

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(brent.index, brent.values, color=COLORS['blue'], lw=0.8)
ax.axhline(brent.mean(), color='gray', ls='--', alpha=0.5, lw=0.8)

events = [('2020-04-20', 'COVID\\nCrash', -30), ('2022-03-08', 'Ukraine\\nWar', 20)]
for date, label, offset in events:
    idx = pd.Timestamp(date)
    if idx in brent.index:
        val = brent.loc[idx]
    else:
        val = brent.iloc[brent.index.get_indexer([idx], method='nearest')[0]]
    ax.annotate(label, xy=(idx, val), xytext=(0, offset), textcoords='offset points',
                ha='center', fontsize=9, color=COLORS['red'],
                arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=1.2))

ax.set_title('Brent Crude Oil Prices (2015-2025)', fontweight='bold', fontsize=13)
ax.set_ylabel('USD/bbl')
ax.legend(['Brent Price', 'Mean'], loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)
fig.tight_layout()
save_fig(fig, 'oil_prices_historical.pdf')
plt.show()
'''),
])

write_nb("EMQA_oil_prices", [
    ("md", "# EMQA_oil_prices\n\nBrent crude oil prices (2020-2025).\n\n**Output:** `oil_prices_overview.pdf`"),
    ("code", STYLE_CODE),
    ("code", YFINANCE_CODE),
    ("code", '''brent = fetch('BZ=F')
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(brent.index, brent.values, color=COLORS['blue'], lw=0.8)
ax.axhline(brent.mean(), color=COLORS['red'], ls='--', alpha=0.6, lw=0.8, label=f'Mean: ${brent.mean():.1f}')
ax.fill_between(brent.index, brent.mean()-brent.std(), brent.mean()+brent.std(), alpha=0.08, color=COLORS['blue'])
ax.set_title('Brent Crude Oil (2020-2025)', fontweight='bold')
ax.set_ylabel('USD/bbl')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)
fig.tight_layout()
save_fig(fig, 'oil_prices_overview.pdf')
plt.show()
'''),
])

write_nb("EMQA_oil_overview", [
    ("md", "# EMQA_oil_overview\n\nBrent crude case study overview with events and MA.\n\n**Output:** `case_oil_overview.pdf`"),
    ("code", STYLE_CODE),
    ("code", YFINANCE_CODE),
    ("code", '''brent = fetch('BZ=F')
ma50 = brent.rolling(50).mean()

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(brent.index, brent, color=COLORS['blue'], lw=0.8, label='Brent Price')
ax.plot(ma50.index, ma50, color=COLORS['orange'], lw=1.2, label='50-day MA')

events = [('2020-04-20', 'COVID Crash', 25), ('2022-03-08', 'Ukraine War', -30), ('2023-06-01', 'OPEC+ Cuts', 20)]
for date, label, off in events:
    ts = pd.Timestamp(date)
    loc = brent.index.get_indexer([ts], method='nearest')[0]
    ax.annotate(label, xy=(brent.index[loc], brent.iloc[loc]), xytext=(0, off),
                textcoords='offset points', ha='center', fontsize=8, color=COLORS['red'],
                arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=1))

ax.set_title('Brent Crude Oil: Case Study (2020-2025)', fontweight='bold')
ax.set_ylabel('USD/bbl')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)
fig.tight_layout()
save_fig(fig, 'case_oil_overview.pdf')
plt.show()
'''),
])

write_nb("EMQA_oil_returns", [
    ("md", "# EMQA_oil_returns\n\nOil price levels and log returns.\n\n**Output:** `case_oil_returns.pdf`"),
    ("code", STYLE_CODE),
    ("code", YFINANCE_CODE),
    ("code", '''brent = fetch('BZ=F')
ret = np.log(brent / brent.shift(1)).dropna()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
ax1.plot(brent.index, brent, color=COLORS['blue'], lw=0.8)
ax1.set_title('Brent Crude Prices', fontweight='bold')
ax1.set_ylabel('USD/bbl')

ax2.plot(ret.index, ret, color=COLORS['red'], lw=0.5, alpha=0.8)
ax2.axhline(0, color='gray', lw=0.5)
ax2.set_title('Log Returns (Volatility Clustering)', fontweight='bold')
ax2.set_ylabel('Log Return')

fig.legend(['Price', 'Log Returns'], loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False)
fig.tight_layout()
save_fig(fig, 'case_oil_returns.pdf')
plt.show()
'''),
])

write_nb("EMQA_oil_acf", [
    ("md", "# EMQA_oil_acf\n\nACF analysis of oil returns and squared returns.\n\n**Output:** `case_oil_acf_analysis.pdf`"),
    ("code", STYLE_CODE),
    ("code", YFINANCE_CODE),
    ("code", '''from statsmodels.graphics.tsaplots import plot_acf

brent = fetch('BZ=F')
ret = np.log(brent / brent.shift(1)).dropna()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(ret, ax=ax1, lags=40, alpha=0.05)
ax1.set_title('ACF: Returns (near white noise)', fontweight='bold')

plot_acf(ret**2, ax=ax2, lags=40, alpha=0.05)
ax2.set_title('ACF: Squared Returns (GARCH effects)', fontweight='bold')

fig.legend(['ACF', '95% CI'], loc='upper center', bbox_to_anchor=(0.5, -0.03), ncol=2, frameon=False)
fig.tight_layout()
save_fig(fig, 'case_oil_acf_analysis.pdf')
plt.show()
'''),
])

write_nb("EMQA_oil_stats", [
    ("md", "# EMQA_oil_stats\n\nBrent crude oil key statistics.\n\n**Output:** `oil_statistics.pdf`"),
    ("code", STYLE_CODE),
    ("code", YFINANCE_CODE),
    ("code", '''from scipy import stats as sp_stats

brent = fetch('BZ=F')
ret = np.log(brent / brent.shift(1)).dropna()

print("=== Brent Crude Oil Statistics (2020-2025) ===")
print(f"Mean Price:    ${brent.mean():.2f}")
print(f"Median Price:  ${brent.median():.2f}")
print(f"Std Dev:       ${brent.std():.2f}")
print(f"Min:           ${brent.min():.2f}")
print(f"Max:           ${brent.max():.2f}")
print(f"Mean Return:   {ret.mean()*100:.3f}%")
print(f"Return Vol:    {ret.std()*100:.2f}%")
print(f"Skewness:      {sp_stats.skew(ret):.2f}")
print(f"Kurtosis:      {sp_stats.kurtosis(ret, fisher=False):.2f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.hist(brent, bins=40, color=COLORS['blue'], alpha=0.7, edgecolor='white')
ax1.axvline(brent.mean(), color=COLORS['red'], ls='--', label=f'Mean ${brent.mean():.0f}')
ax1.set_title('Price Distribution', fontweight='bold')
ax1.set_xlabel('USD/bbl')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)

ax2.hist(ret*100, bins=60, color=COLORS['red'], alpha=0.7, edgecolor='white', density=True)
x = np.linspace(ret.min()*100, ret.max()*100, 200)
ax2.plot(x, sp_stats.norm.pdf(x, ret.mean()*100, ret.std()*100), color=COLORS['blue'], lw=2, label='Normal fit')
ax2.set_title('Return Distribution (Fat Tails)', fontweight='bold')
ax2.set_xlabel('Daily Return (%)')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)

fig.tight_layout()
save_fig(fig, 'oil_statistics.pdf')
plt.show()
'''),
])

write_nb("EMQA_mean_reversion", [
    ("md", "# EMQA_mean_reversion\n\nMean reversion in energy commodity prices.\n\n**Output:** `mean_reversion_energy.pdf`"),
    ("code", STYLE_CODE),
    ("code", YFINANCE_CODE),
    ("code", '''brent = fetch('BZ=F')
gas = fetch('NG=F')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
ax1.plot(brent.index, brent, color=COLORS['blue'], lw=0.8)
ax1.axhline(brent.mean(), color=COLORS['red'], ls='--', lw=1, label=f'Mean ${brent.mean():.0f}')
ax1.fill_between(brent.index, brent.mean()-brent.std(), brent.mean()+brent.std(), alpha=0.08, color=COLORS['blue'])
ax1.set_title('Brent Crude: Mean Reversion', fontweight='bold')
ax1.set_ylabel('USD/bbl')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)

ax2.plot(gas.index, gas, color=COLORS['green'], lw=0.8)
ax2.axhline(gas.mean(), color=COLORS['red'], ls='--', lw=1, label=f'Mean ${gas.mean():.1f}')
ax2.fill_between(gas.index, gas.mean()-gas.std(), gas.mean()+gas.std(), alpha=0.08, color=COLORS['green'])
ax2.set_title('Natural Gas: Mean Reversion', fontweight='bold')
ax2.set_ylabel('USD/MMBtu')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)

fig.tight_layout()
save_fig(fig, 'mean_reversion_energy.pdf')
plt.show()
'''),
])

write_nb("EMQA_stationarity", [
    ("md", "# EMQA_stationarity\n\nStationarity testing for oil prices and returns.\n\n**Output:** `oil_stationarity_test.pdf`"),
    ("code", STYLE_CODE),
    ("code", YFINANCE_CODE),
    ("code", '''from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf

brent = fetch('BZ=F')
ret = np.log(brent / brent.shift(1)).dropna()

for name, series in [('Brent Price', brent), ('Brent Returns', ret)]:
    adf = adfuller(series.dropna(), autolag='AIC')
    print(f"{name}: ADF={adf[0]:.2f}, p={adf[1]:.4f}, {'Stationary' if adf[1]<0.05 else 'Non-stationary'}")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes[0,0].plot(brent.index, brent, color=COLORS['blue'], lw=0.8)
axes[0,0].set_title('Prices (Non-Stationary)', fontweight='bold')
axes[0,0].set_ylabel('USD/bbl')

axes[0,1].plot(ret.index, ret, color=COLORS['red'], lw=0.5)
axes[0,1].axhline(0, color='gray', lw=0.5)
axes[0,1].set_title('Returns (Stationary)', fontweight='bold')
axes[0,1].set_ylabel('Log Return')

plot_acf(brent.dropna(), ax=axes[1,0], lags=40, alpha=0.05)
axes[1,0].set_title('ACF: Prices (slow decay)', fontweight='bold')

plot_acf(ret, ax=axes[1,1], lags=40, alpha=0.05)
axes[1,1].set_title('ACF: Returns (quick decay)', fontweight='bold')

fig.tight_layout()
save_fig(fig, 'oil_stationarity_test.pdf')
plt.show()
'''),
])

write_nb("EMQA_stationarity_intro", [
    ("md", "# EMQA_stationarity_intro\n\nVisual comparison of stationary vs non-stationary series.\n\n**Output:** `lecture1_stationarity_comparison.pdf`"),
    ("code", STYLE_CODE),
    ("code", YFINANCE_CODE),
    ("code", '''brent = fetch('BZ=F')
ret = np.log(brent / brent.shift(1)).dropna()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
ax1.plot(brent.index, brent, color=COLORS['blue'], lw=0.8)
ax1.axhline(brent.mean(), color=COLORS['red'], ls='--', lw=0.8)
ax1.set_title('Non-Stationary: Oil Prices (mean changes over time)', fontweight='bold', color=COLORS['red'])
ax1.set_ylabel('USD/bbl')

ax2.plot(ret.index, ret, color=COLORS['green'], lw=0.5)
ax2.axhline(0, color=COLORS['red'], ls='--', lw=0.8)
ax2.fill_between(ret.index, -ret.std(), ret.std(), alpha=0.1, color=COLORS['green'])
ax2.set_title('Stationary: Oil Returns (constant mean, stable variance)', fontweight='bold', color=COLORS['green'])
ax2.set_ylabel('Log Return')

fig.legend(['Series', 'Mean'], loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False)
fig.tight_layout()
save_fig(fig, 'lecture1_stationarity_comparison.pdf')
plt.show()
'''),
])

write_nb("EMQA_stationarity_comparison", [
    ("md", "# EMQA_stationarity_comparison\n\nStationarity comparison across oil and gas.\n\n**Output:** `stationarity_comparison.pdf`"),
    ("code", STYLE_CODE),
    ("code", YFINANCE_CODE),
    ("code", '''brent = fetch('BZ=F'); gas = fetch('NG=F')
bret = np.log(brent/brent.shift(1)).dropna()
gret = np.log(gas/gas.shift(1)).dropna()

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes[0,0].plot(brent.index, brent, color=COLORS['blue'], lw=0.8)
axes[0,0].set_title('Brent Prices (Non-Stationary)', fontweight='bold')
axes[0,1].plot(bret.index, bret, color=COLORS['blue'], lw=0.5)
axes[0,1].axhline(0, color='gray', lw=0.5)
axes[0,1].set_title('Brent Returns (Stationary)', fontweight='bold')
axes[1,0].plot(gas.index, gas, color=COLORS['green'], lw=0.8)
axes[1,0].set_title('Gas Prices (Non-Stationary)', fontweight='bold')
axes[1,1].plot(gret.index, gret, color=COLORS['green'], lw=0.5)
axes[1,1].axhline(0, color='gray', lw=0.5)
axes[1,1].set_title('Gas Returns (Stationary)', fontweight='bold')

fig.tight_layout()
save_fig(fig, 'stationarity_comparison.pdf')
plt.show()
'''),
])

write_nb("EMQA_returns_distribution", [
    ("md", "# EMQA_returns_distribution\n\nFat tails in energy return distributions.\n\n**Output:** `lecture1_returns_distribution.pdf`"),
    ("code", STYLE_CODE),
    ("code", YFINANCE_CODE),
    ("code", '''from scipy import stats as sp_stats

brent = fetch('BZ=F'); gas = fetch('NG=F')
bret = np.log(brent/brent.shift(1)).dropna()
gret = np.log(gas/gas.shift(1)).dropna()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, r, name, c in [(axes[0], bret, 'Brent Oil', COLORS['blue']),
                         (axes[1], gret, 'Natural Gas', COLORS['green'])]:
    ax.hist(r*100, bins=80, density=True, color=c, alpha=0.6, edgecolor='white')
    x = np.linspace(r.min()*100, r.max()*100, 300)
    ax.plot(x, sp_stats.norm.pdf(x, r.mean()*100, r.std()*100), color=COLORS['red'], lw=2)
    kurt = sp_stats.kurtosis(r, fisher=False)
    ax.set_title(f'{name} Returns (Kurtosis: {kurt:.1f})', fontweight='bold')
    ax.set_xlabel('Daily Return (%)')

fig.legend(['Normal Distribution', 'Actual Returns'], loc='upper center',
           bbox_to_anchor=(0.5, -0.03), ncol=2, frameon=False)
fig.tight_layout()
save_fig(fig, 'lecture1_returns_distribution.pdf')
plt.show()
'''),
])

write_nb("EMQA_acf_reading", [
    ("md", "# EMQA_acf_reading\n\nHow to read ACF/PACF plots.\n\n**Output:** `oil_acf_pacf.pdf`"),
    ("code", STYLE_CODE),
    ("code", YFINANCE_CODE),
    ("code", '''from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

brent = fetch('BZ=F')
ret = np.log(brent/brent.shift(1)).dropna()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(ret, ax=ax1, lags=40, alpha=0.05)
ax1.set_title('ACF: Oil Returns', fontweight='bold')
plot_pacf(ret, ax=ax2, lags=40, alpha=0.05, method='ywm')
ax2.set_title('PACF: Oil Returns', fontweight='bold')

fig.legend(['ACF/PACF', '95% CI'], loc='upper center', bbox_to_anchor=(0.5, -0.03), ncol=2, frameon=False)
fig.tight_layout()
save_fig(fig, 'oil_acf_pacf.pdf')
plt.show()
'''),
])

write_nb("EMQA_acf_pacf", [
    ("md", "# EMQA_acf_pacf\n\nFull ACF/PACF analysis: returns and squared returns.\n\n**Output:** `oil_acf_pacf_full.pdf`"),
    ("code", STYLE_CODE),
    ("code", YFINANCE_CODE),
    ("code", '''from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

brent = fetch('BZ=F')
ret = np.log(brent/brent.shift(1)).dropna()

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
plot_acf(ret, ax=axes[0,0], lags=40, alpha=0.05); axes[0,0].set_title('ACF: Returns', fontweight='bold')
plot_pacf(ret, ax=axes[0,1], lags=40, alpha=0.05, method='ywm'); axes[0,1].set_title('PACF: Returns', fontweight='bold')
plot_acf(ret**2, ax=axes[1,0], lags=40, alpha=0.05); axes[1,0].set_title('ACF: Squared Returns', fontweight='bold')
plot_pacf(ret**2, ax=axes[1,1], lags=40, alpha=0.05, method='ywm'); axes[1,1].set_title('PACF: Squared Returns', fontweight='bold')

fig.tight_layout()
save_fig(fig, 'oil_acf_pacf_full.pdf')
plt.show()
'''),
])

write_nb("EMQA_acf_comparison", [
    ("md", "# EMQA_acf_comparison\n\nACF patterns: prices, returns, squared returns, absolute returns.\n\n**Output:** `lecture1_acf_comparison.pdf`"),
    ("code", STYLE_CODE),
    ("code", YFINANCE_CODE),
    ("code", '''from statsmodels.graphics.tsaplots import plot_acf

brent = fetch('BZ=F')
ret = np.log(brent/brent.shift(1)).dropna()

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
plot_acf(brent.dropna(), ax=axes[0,0], lags=40, alpha=0.05); axes[0,0].set_title('ACF: Prices (slow decay = non-stationary)', fontweight='bold')
plot_acf(ret, ax=axes[0,1], lags=40, alpha=0.05); axes[0,1].set_title('ACF: Returns (quick decay = stationary)', fontweight='bold')
plot_acf(ret**2, ax=axes[1,0], lags=40, alpha=0.05); axes[1,0].set_title('ACF: Squared Returns (volatility clustering)', fontweight='bold')
plot_acf(ret.abs(), ax=axes[1,1], lags=40, alpha=0.05); axes[1,1].set_title('ACF: Absolute Returns (persistence)', fontweight='bold')

fig.tight_layout()
save_fig(fig, 'lecture1_acf_comparison.pdf')
plt.show()
'''),
])

write_nb("EMQA_volatility_clustering", [
    ("md", "# EMQA_volatility_clustering\n\nVolatility clustering in oil returns.\n\n**Output:** `lecture1_volatility_clustering.pdf`"),
    ("code", STYLE_CODE),
    ("code", YFINANCE_CODE),
    ("code", '''brent = fetch('BZ=F')
ret = np.log(brent/brent.shift(1)).dropna()
rvol = ret.rolling(30).std() * np.sqrt(252) * 100

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
ax1.plot(ret.index, ret*100, color=COLORS['blue'], lw=0.4, alpha=0.8)
ax1.axhline(0, color='gray', lw=0.5)
for start, end, label in [('2020-02', '2020-06', 'COVID'), ('2022-02', '2022-06', 'Ukraine')]:
    ax1.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.15, color=COLORS['red'])
    ax1.text(pd.Timestamp(start), ax1.get_ylim()[1]*0.9, label, fontsize=8, color=COLORS['red'])
ax1.set_title('Oil Returns: "Big moves follow big moves"', fontweight='bold')
ax1.set_ylabel('Daily Return (%)')

ax2.plot(rvol.index, rvol, color=COLORS['red'], lw=1)
ax2.fill_between(rvol.index, 0, rvol, alpha=0.15, color=COLORS['red'])
ax2.set_title('30-Day Rolling Annualized Volatility', fontweight='bold')
ax2.set_ylabel('Volatility (%)')

fig.legend(['Returns', 'Volatility'], loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False)
fig.tight_layout()
save_fig(fig, 'lecture1_volatility_clustering.pdf')
plt.show()
'''),
])

write_nb("EMQA_moving_averages", [
    ("md", "# EMQA_moving_averages\n\nOil prices with moving averages.\n\n**Output:** `lecture1_moving_averages.pdf`"),
    ("code", STYLE_CODE),
    ("code", YFINANCE_CODE),
    ("code", '''brent = fetch('BZ=F', '2024-01-01', '2025-12-31')

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(brent.index, brent, color=COLORS['gray'], lw=0.6, alpha=0.7, label='Price')
for w, c, n in [(20, COLORS['green'], '20-day'), (50, COLORS['blue'], '50-day'), (100, COLORS['red'], '100-day')]:
    ma = brent.rolling(w).mean()
    ax.plot(ma.index, ma, color=c, lw=1.5, label=f'{n} MA')

ax.set_title('Brent Crude with Moving Averages (2024-2025)', fontweight='bold')
ax.set_ylabel('USD/bbl')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=4, frameon=False)
fig.tight_layout()
save_fig(fig, 'lecture1_moving_averages.pdf')
plt.show()
'''),
])

write_nb("EMQA_commodities_comparison", [
    ("md", "# EMQA_commodities_comparison\n\nComparing oil and gas prices.\n\n**Output:** `lecture1_commodities_comparison.pdf`"),
    ("code", STYLE_CODE),
    ("code", YFINANCE_CODE),
    ("code", '''brent = fetch('BZ=F'); gas = fetch('NG=F')
df = pd.DataFrame({'Brent': brent, 'Gas': gas}).dropna()
norm = df / df.iloc[0] * 100

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))
ax1.plot(norm.index, norm['Brent'], color=COLORS['blue'], lw=1, label='Brent Oil')
ax1.plot(norm.index, norm['Gas'], color=COLORS['green'], lw=1, label='Natural Gas')
ax1.axhline(100, color='gray', ls='--', lw=0.5)
ax1.set_title('Normalized Prices (Base=100)', fontweight='bold')
ax1.set_ylabel('Index')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)

bret = np.log(df['Brent']/df['Brent'].shift(1)).dropna()
gret = np.log(df['Gas']/df['Gas'].shift(1)).dropna()
rcorr = bret.rolling(60).corr(gret)
ax2.plot(rcorr.index, rcorr, color=COLORS['purple'], lw=1)
ax2.axhline(0, color='gray', ls='--', lw=0.5)
ax2.set_title('60-Day Rolling Correlation (Oil vs Gas Returns)', fontweight='bold')
ax2.set_ylabel('Correlation')
ax2.legend(['Rolling Correlation'], loc='upper center', bbox_to_anchor=(0.5, -0.08), frameon=False)

fig.tight_layout()
save_fig(fig, 'lecture1_commodities_comparison.pdf')
plt.show()
'''),
])

write_nb("EMQA_time_series_intro", [
    ("md", "# EMQA_time_series_intro\n\nWhat is a time series? Visual introduction.\n\n**Output:** `time_series_intro.pdf`"),
    ("code", STYLE_CODE),
    ("code", YFINANCE_CODE),
    ("code", '''brent = fetch('BZ=F')

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(brent.index, brent, color=COLORS['blue'], lw=0.8)
ax.set_title('A Time Series: Brent Crude Oil $\\{y_t : t = 1, ..., T\\}$', fontweight='bold', fontsize=13)
ax.set_ylabel('Price (USD/bbl)')
ax.set_xlabel('Time $t$')
ax.annotate('$y_t$ = observation\\nat time $t$', xy=(brent.index[len(brent)//2], brent.iloc[len(brent)//2]),
            xytext=(30, 30), textcoords='offset points', fontsize=11,
            arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=1.5), color=COLORS['red'])

ax.legend(['$y_t$: Daily closing price'], loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)
fig.tight_layout()
save_fig(fig, 'time_series_intro.pdf')
plt.show()
'''),
])

write_nb("EMQA_moving_avg", [
    ("md", "# EMQA_moving_avg\n\nStep-by-step moving average calculation.\n\n**Output:** `moving_average_calculation.pdf`"),
    ("code", STYLE_CODE),
    ("code", YFINANCE_CODE),
    ("code", '''brent = fetch('BZ=F', '2025-12-15', '2026-01-15')
if len(brent) < 5:
    brent = fetch('BZ=F', '2025-11-01', '2025-12-31').tail(10)
sma3 = brent.rolling(3).mean()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(brent.index, brent, 'o-', color=COLORS['blue'], lw=1.5, markersize=6, label='Price')
ax.plot(sma3.index, sma3, 's--', color=COLORS['red'], lw=1.5, markersize=5, label='3-Day SMA')
for i, (d, v) in enumerate(brent.items()):
    ax.annotate(f'${v:.2f}', xy=(d, v), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=8)

ax.set_title('Moving Average: Step-by-Step Calculation', fontweight='bold')
ax.set_ylabel('USD/bbl')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
fig.tight_layout()
save_fig(fig, 'moving_average_calculation.pdf')
plt.show()
'''),
])


# ========================================================================
# GAS QUANTLETS
# ========================================================================

write_nb("EMQA_gas_seasonality", [
    ("md", "# EMQA_gas_seasonality\n\nNatural gas prices and seasonal patterns.\n\n**Output:** `gas_prices_seasonality.pdf`"),
    ("code", STYLE_CODE),
    ("code", YFINANCE_CODE),
    ("code", '''gas = fetch('NG=F')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))
ax1.plot(gas.index, gas, color=COLORS['green'], lw=0.8)
ax1.axhline(gas.mean(), color=COLORS['red'], ls='--', lw=0.8, label=f'Mean ${gas.mean():.2f}')
ax1.set_title('Henry Hub Natural Gas Prices (2020-2025)', fontweight='bold')
ax1.set_ylabel('USD/MMBtu')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), frameon=False)

monthly = gas.groupby(gas.index.month)
bp = ax2.boxplot([monthly.get_group(m).values for m in range(1,13)], patch_artist=True, widths=0.6)
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
ax2.set_xticklabels(months)
for patch, m in zip(bp['boxes'], range(1,13)):
    patch.set_facecolor(COLORS['red'] if m in [11,12,1,2] else COLORS['blue'])
    patch.set_alpha(0.5)
ax2.set_title('Monthly Price Distribution (Seasonal Pattern)', fontweight='bold')
ax2.set_ylabel('USD/MMBtu')

fig.tight_layout()
save_fig(fig, 'gas_prices_seasonality.pdf')
plt.show()
'''),
])

write_nb("EMQA_decomposition", [
    ("md", "# EMQA_decomposition\n\nClassical seasonal decomposition of natural gas.\n\n**Output:** `gas_decomposition.pdf`"),
    ("code", STYLE_CODE),
    ("code", YFINANCE_CODE),
    ("code", '''from statsmodels.tsa.seasonal import seasonal_decompose

gas = fetch('NG=F')
monthly = gas.resample('ME').mean().dropna()
result = seasonal_decompose(monthly, model='multiplicative', period=12)

fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
axes[0].plot(monthly.index, monthly, color=COLORS['green'], lw=1); axes[0].set_title('Observed', fontweight='bold'); axes[0].set_ylabel('USD')
axes[1].plot(result.trend.index, result.trend, color=COLORS['red'], lw=1.5); axes[1].set_title('Trend', fontweight='bold'); axes[1].set_ylabel('USD')
axes[2].plot(result.seasonal.index, result.seasonal, color=COLORS['blue'], lw=1); axes[2].set_title('Seasonal', fontweight='bold'); axes[2].set_ylabel('Index')
axes[2].axhline(1, color='gray', lw=0.5)
axes[3].plot(result.resid.index, result.resid, color=COLORS['gray'], lw=1); axes[3].set_title('Residual', fontweight='bold'); axes[3].set_ylabel('Index')
axes[3].axhline(1, color='gray', lw=0.5)

fig.legend(['Component'], loc='upper center', bbox_to_anchor=(0.5, -0.01), frameon=False)
fig.tight_layout()
save_fig(fig, 'gas_decomposition.pdf')
plt.show()
'''),
])

write_nb("EMQA_stl_decomposition", [
    ("md", "# EMQA_stl_decomposition\n\nSTL decomposition of natural gas.\n\n**Output:** `lecture1_decomposition.pdf`"),
    ("code", STYLE_CODE),
    ("code", YFINANCE_CODE),
    ("code", '''from statsmodels.tsa.seasonal import STL

gas = fetch('NG=F')
monthly = gas.resample('ME').mean().dropna()
stl = STL(monthly, period=12, robust=True).fit()

fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
axes[0].plot(monthly.index, monthly, color=COLORS['green'], lw=1); axes[0].set_title('Observed', fontweight='bold'); axes[0].set_ylabel('USD')
axes[1].plot(stl.trend.index, stl.trend, color=COLORS['red'], lw=1.5); axes[1].set_title('Trend', fontweight='bold'); axes[1].set_ylabel('USD')
axes[2].plot(stl.seasonal.index, stl.seasonal, color=COLORS['blue'], lw=1); axes[2].set_title('Seasonal', fontweight='bold'); axes[2].set_ylabel('USD')
axes[2].axhline(0, color='gray', lw=0.5)
axes[3].plot(stl.resid.index, stl.resid, color=COLORS['gray'], lw=1); axes[3].set_title('Residual', fontweight='bold'); axes[3].set_ylabel('USD')
axes[3].axhline(0, color='gray', lw=0.5)

fig.tight_layout()
save_fig(fig, 'lecture1_decomposition.pdf')
plt.show()
'''),
])

write_nb("EMQA_seasonal_pattern", [
    ("md", "# EMQA_seasonal_pattern\n\nMonthly seasonal patterns in natural gas.\n\n**Output:** `lecture1_seasonal_pattern.pdf`"),
    ("code", STYLE_CODE),
    ("code", YFINANCE_CODE),
    ("code", '''gas = fetch('NG=F')
monthly = gas.resample('ME').mean()
overall_mean = monthly.mean()
seasonal_idx = monthly.groupby(monthly.index.month).mean() / overall_mean

months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
colors_bar = [COLORS['red'] if v > 1 else COLORS['blue'] for v in seasonal_idx.values]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(range(12), seasonal_idx.values, color=colors_bar, alpha=0.7, edgecolor='white')
ax.axhline(1.0, color='gray', ls='--', lw=1)
ax.set_xticks(range(12)); ax.set_xticklabels(months)
ax.set_title('Henry Hub Natural Gas: Monthly Seasonal Indices (2020-2025)', fontweight='bold')
ax.set_ylabel('Seasonal Index')
for i, v in enumerate(seasonal_idx.values):
    ax.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=9)

ax.legend(['Index = 1.0 (average)', 'Above average', 'Below average'],
          loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False)
fig.tight_layout()
save_fig(fig, 'lecture1_seasonal_pattern.pdf')
plt.show()
'''),
])

write_nb("EMQA_seasonal_index", [
    ("md", "# EMQA_seasonal_index\n\nQuarterly seasonal index calculation.\n\n**Output:** `seasonal_index_calculation.pdf`"),
    ("code", STYLE_CODE),
    ("code", YFINANCE_CODE),
    ("code", '''gas = fetch('NG=F', '2023-01-01', '2025-12-31')
quarterly = gas.resample('QE').mean()
quarterly.index = quarterly.index.to_period('Q')

# Manual approach
years = [2023, 2024, 2025]
qtrs = ['Q1','Q2','Q3','Q4']
data = {}
for y in years:
    for qi, q in enumerate(qtrs):
        mask = (gas.index.year == y) & (gas.index.quarter == qi+1)
        vals = gas[mask]
        if len(vals) > 0:
            data.setdefault(q, {})[y] = vals.mean()

print("Quarterly Average Prices (USD/MMBtu):")
print(f"{'':8s} {'2023':>8s} {'2024':>8s} {'2025':>8s} {'Avg':>8s} {'Index':>8s}")
overall_avg = np.mean([np.mean(list(data[q].values())) for q in qtrs])
for q in qtrs:
    vals = [data[q].get(y, np.nan) for y in years]
    avg = np.nanmean(vals)
    idx = avg / overall_avg
    print(f"{q:8s} {vals[0]:>8.2f} {vals[1]:>8.2f} {vals[2]:>8.2f} {avg:>8.2f} {idx:>8.2f}")
print(f"{'Overall':8s} {'':>8s} {'':>8s} {'':>8s} {overall_avg:>8.2f} {'1.00':>8s}")

indices = [np.nanmean(list(data[q].values())) / overall_avg for q in qtrs]
fig, ax = plt.subplots(figsize=(10, 5))
colors_bar = [COLORS['red'] if v > 1 else COLORS['blue'] for v in indices]
ax.bar(qtrs, indices, color=colors_bar, alpha=0.7, width=0.5)
ax.axhline(1.0, color='gray', ls='--', lw=1)
for i, v in enumerate(indices):
    ax.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=11, fontweight='bold')
ax.set_title('Quarterly Seasonal Indices: Henry Hub Gas (2023-2025)', fontweight='bold')
ax.set_ylabel('Seasonal Index')
ax.legend(['Index = 1.0'], loc='upper center', bbox_to_anchor=(0.5, -0.10), frameon=False)
fig.tight_layout()
save_fig(fig, 'seasonal_index_calculation.pdf')
plt.show()
'''),
])

write_nb("EMQA_seasonality", [
    ("md", "# EMQA_seasonality\n\nDetecting seasonality: visual and ACF methods.\n\n**Output:** `seasonality_detection.pdf`"),
    ("code", STYLE_CODE),
    ("code", YFINANCE_CODE),
    ("code", '''from statsmodels.graphics.tsaplots import plot_acf

gas = fetch('NG=F')
monthly = gas.resample('ME').mean().dropna()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for year in monthly.index.year.unique():
    subset = monthly[monthly.index.year == year]
    ax1.plot(subset.index.month, subset.values, 'o-', lw=1, markersize=4, label=str(year))
ax1.set_xticks(range(1,13))
ax1.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
ax1.set_title('Year-over-Year Overlay', fontweight='bold')
ax1.set_ylabel('USD/MMBtu')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=6, frameon=False)

plot_acf(monthly, ax=ax2, lags=36, alpha=0.05)
ax2.set_title('ACF: Monthly Gas (spikes at lag 12, 24)', fontweight='bold')
ax2.axvline(12, color=COLORS['red'], ls='--', alpha=0.5)
ax2.axvline(24, color=COLORS['red'], ls='--', alpha=0.5)

fig.tight_layout()
save_fig(fig, 'seasonality_detection.pdf')
plt.show()
'''),
])


# ========================================================================
# ELECTRICITY QUANTLETS
# ========================================================================

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
    # Fallback: generate from German data
    paths2 = [
        '../../charts/german_dayahead_prices.csv',
        '/Users/danielpele/Documents/Energy MBA/charts/german_dayahead_prices.csv',
    ]
    for p in paths2:
        if os.path.exists(p):
            return pd.read_csv(p, parse_dates=[0], index_col=0)
    raise FileNotFoundError("No electricity data found")

elec = load_elec()
# Get price column
if 'price' in elec.columns:
    price_col = 'price'
elif 'Price' in elec.columns:
    price_col = 'Price'
else:
    price_col = elec.columns[0]
elec_price = elec[price_col].dropna()
'''

write_nb("EMQA_electricity_spikes", [
    ("md", "# EMQA_electricity_spikes\n\nExtreme behavior in electricity prices.\n\n**Output:** `electricity_prices_spikes.pdf`"),
    ("code", STYLE_CODE),
    ("code", ELEC_CODE),
    ("code", '''fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(elec_price.index, elec_price, color=COLORS['blue'], lw=0.3, alpha=0.7)
ax.axhline(0, color=COLORS['red'], lw=0.5, ls='--')
ax.axhline(elec_price.mean(), color=COLORS['orange'], lw=1, ls='--')
ax.set_title('German Day-Ahead Electricity Prices', fontweight='bold')
ax.set_ylabel('EUR/MWh')
ax.legend(['Price', 'Zero', f'Mean {elec_price.mean():.0f} EUR/MWh'],
          loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False)
fig.tight_layout()
save_fig(fig, 'electricity_prices_spikes.pdf')
plt.show()
'''),
])

write_nb("EMQA_elec_overview", [
    ("md", "# EMQA_elec_overview\n\nEuropean electricity prices overview.\n\n**Output:** `case_elec_overview.pdf`"),
    ("code", STYLE_CODE),
    ("code", ELEC_CODE),
    ("code", '''fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(elec_price.index, elec_price, color=COLORS['blue'], lw=0.3, alpha=0.7)
ma30 = elec_price.rolling(24*30).mean()
ax.plot(ma30.index, ma30, color=COLORS['red'], lw=1.5)
ax.set_title('German Day-Ahead Electricity Prices (2022-2025)', fontweight='bold')
ax.set_ylabel('EUR/MWh')
ax.legend(['Hourly Price', '30-Day MA'],
          loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)
fig.tight_layout()
save_fig(fig, 'case_elec_overview.pdf')
plt.show()
'''),
])

write_nb("EMQA_elec_intraday", [
    ("md", "# EMQA_elec_intraday\n\nIntraday electricity price pattern.\n\n**Output:** `case_elec_intraday.pdf`"),
    ("code", STYLE_CODE),
    ("code", ELEC_CODE),
    ("code", '''hourly_avg = elec_price.groupby(elec_price.index.hour).mean()

fig, ax = plt.subplots(figsize=(12, 5))
colors_h = [COLORS['blue'] if h in range(7,20) else COLORS['gray'] for h in range(24)]
ax.bar(range(24), hourly_avg.values, color=colors_h, alpha=0.7)
ax.set_xticks(range(24))
ax.set_title('Average Electricity Price by Hour of Day', fontweight='bold')
ax.set_xlabel('Hour'); ax.set_ylabel('EUR/MWh')
ax.legend(['Peak hours (7-19)', 'Off-peak'],
          loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
fig.tight_layout()
save_fig(fig, 'case_elec_intraday.pdf')
plt.show()
'''),
])

write_nb("EMQA_elec_weekly", [
    ("md", "# EMQA_elec_weekly\n\nWeekly electricity price pattern.\n\n**Output:** `case_elec_weekly.pdf`"),
    ("code", STYLE_CODE),
    ("code", ELEC_CODE),
    ("code", '''weekly_avg = elec_price.groupby(elec_price.index.dayofweek).mean()
days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
colors_d = [COLORS['blue']]*5 + [COLORS['orange']]*2

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(7), weekly_avg.values, color=colors_d, alpha=0.7)
ax.set_xticks(range(7)); ax.set_xticklabels(days)
ax.set_title('Average Electricity Price by Day of Week', fontweight='bold')
ax.set_ylabel('EUR/MWh')
ax.legend(['Weekday', 'Weekend'],
          loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
fig.tight_layout()
save_fig(fig, 'case_elec_weekly.pdf')
plt.show()
'''),
])

write_nb("EMQA_elec_stats", [
    ("md", "# EMQA_elec_stats\n\nGerman electricity price statistics.\n\n**Output:** `elec_statistics.pdf`"),
    ("code", STYLE_CODE),
    ("code", ELEC_CODE),
    ("code", '''print("=== German Electricity Statistics ===")
print(f"Mean: {elec_price.mean():.2f} EUR/MWh")
print(f"Median: {elec_price.median():.2f}")
print(f"Std: {elec_price.std():.2f}")
print(f"Min: {elec_price.min():.2f}")
print(f"Max: {elec_price.max():.2f}")
print(f"Negative hours: {(elec_price<0).sum()} ({(elec_price<0).mean()*100:.1f}%)")

fig, ax = plt.subplots(figsize=(12, 5))
ax.hist(elec_price.clip(-100, 500), bins=100, color=COLORS['blue'], alpha=0.7, edgecolor='white')
ax.axvline(0, color=COLORS['red'], ls='--', lw=1.5)
ax.axvline(elec_price.mean(), color=COLORS['orange'], ls='--', lw=1.5)
ax.set_title('German Electricity Price Distribution', fontweight='bold')
ax.set_xlabel('EUR/MWh')
ax.legend(['Zero', f'Mean ({elec_price.mean():.0f})', 'Distribution'],
          loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
fig.tight_layout()
save_fig(fig, 'elec_statistics.pdf')
plt.show()
'''),
])

write_nb("EMQA_elec_seasonality", [
    ("md", "# EMQA_elec_seasonality\n\nMultiple seasonalities in electricity prices.\n\n**Output:** `electricity_seasonality.pdf`"),
    ("code", STYLE_CODE),
    ("code", ELEC_CODE),
    ("code", '''fig, axes = plt.subplots(1, 3, figsize=(16, 5))

hourly = elec_price.groupby(elec_price.index.hour).mean()
axes[0].bar(range(24), hourly.values, color=COLORS['blue'], alpha=0.7)
axes[0].set_title('Intraday Pattern', fontweight='bold')
axes[0].set_xlabel('Hour'); axes[0].set_ylabel('EUR/MWh')

daily = elec_price.groupby(elec_price.index.dayofweek).mean()
axes[1].bar(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], daily.values,
            color=[COLORS['blue']]*5+[COLORS['orange']]*2, alpha=0.7)
axes[1].set_title('Weekly Pattern', fontweight='bold')
axes[1].set_ylabel('EUR/MWh')

monthly = elec_price.groupby(elec_price.index.month).mean()
months = ['J','F','M','A','M','J','J','A','S','O','N','D']
axes[2].bar(range(12), monthly.values, color=COLORS['green'], alpha=0.7)
axes[2].set_xticks(range(12)); axes[2].set_xticklabels(months)
axes[2].set_title('Annual Pattern', fontweight='bold')
axes[2].set_ylabel('EUR/MWh')

fig.suptitle('Electricity: Multiple Seasonalities', fontweight='bold', fontsize=13)
fig.tight_layout(rect=[0,0,1,0.95])
save_fig(fig, 'electricity_seasonality.pdf')
plt.show()
'''),
])

write_nb("EMQA_load_patterns", [
    ("md", "# EMQA_load_patterns\n\nElectricity load patterns analysis.\n\n**Output:** `load_patterns.pdf`"),
    ("code", STYLE_CODE),
    ("code", ROMANIA_CODE),
    ("code", '''consumption = ro['ro_consumption'] / 1000  # MW to GW

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

daily = consumption.groupby(consumption.index.dayofweek).mean()
days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
axes[0].bar(range(7), daily.values, color=[COLORS['blue']]*5+[COLORS['orange']]*2, alpha=0.7)
axes[0].set_xticks(range(7)); axes[0].set_xticklabels(days)
axes[0].set_title('Weekly Load Pattern', fontweight='bold')
axes[0].set_ylabel('GW')

monthly = consumption.groupby(consumption.index.month).mean()
months = ['J','F','M','A','M','J','J','A','S','O','N','D']
colors_m = [COLORS['blue'] if m in [12,1,2] else COLORS['red'] if m in [6,7,8] else COLORS['gray'] for m in range(1,13)]
axes[1].bar(range(12), monthly.values, color=colors_m, alpha=0.7)
axes[1].set_xticks(range(12)); axes[1].set_xticklabels(months)
axes[1].set_title('Seasonal Load Pattern', fontweight='bold')
axes[1].set_ylabel('GW')

if 'ro_temp_mean' in ro.columns:
    axes[2].scatter(ro['ro_temp_mean'], consumption, s=5, alpha=0.3, color=COLORS['purple'])
    axes[2].set_xlabel('Temperature (C)'); axes[2].set_ylabel('GW')
    corr = consumption.corr(ro['ro_temp_mean'])
    axes[2].set_title(f'Load vs Temperature (r={corr:.2f})', fontweight='bold')
else:
    axes[2].plot(consumption.index, consumption, color=COLORS['blue'], lw=0.5)
    axes[2].set_title('Consumption Time Series', fontweight='bold')

fig.suptitle('Romanian Electricity Load Patterns', fontweight='bold', fontsize=13)
fig.tight_layout(rect=[0,0,1,0.95])
save_fig(fig, 'load_patterns.pdf')
plt.show()
'''),
])


# ========================================================================
# ROMANIA QUANTLETS
# ========================================================================

write_nb("EMQA_romania_stats", [
    ("md", "# EMQA_romania_stats\n\nRomanian electricity system statistics.\n\n**Output:** `romania_statistics.pdf`"),
    ("code", STYLE_CODE),
    ("code", ROMANIA_CODE),
    ("code", '''cols = {'Nuclear': 'ro_nuclear', 'Hydro': 'ro_hydro', 'Coal': 'ro_coal',
        'Gas': 'ro_gas', 'Wind': 'ro_wind', 'Solar': 'ro_solar'}
avgs = {k: ro[v].mean() for k, v in cols.items() if v in ro.columns}
total = sum(avgs.values())

print("=== Romanian Electricity System ===")
for k, v in avgs.items():
    print(f"  {k}: {v:.0f} MW ({v/total*100:.1f}%)")
print(f"  Demand: {ro['ro_consumption'].mean():.0f} MW")

fig, ax = plt.subplots(figsize=(10, 6))
colors_src = [COLORS['purple'], COLORS['blue'], COLORS['gray'], COLORS['orange'], COLORS['cyan'], COLORS['green']]
bars = ax.bar(list(avgs.keys()), list(avgs.values()), color=colors_src, alpha=0.7)
for b, v in zip(bars, avgs.values()):
    ax.text(b.get_x()+b.get_width()/2, v+30, f'{v:.0f} MW', ha='center', fontsize=9)
ax.axhline(ro['ro_consumption'].mean(), color=COLORS['red'], ls='--', lw=1.5, label=f"Avg Demand: {ro['ro_consumption'].mean():.0f} MW")
ax.set_title('Romanian Average Generation by Source', fontweight='bold')
ax.set_ylabel('MW')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), frameon=False)
fig.tight_layout()
save_fig(fig, 'romania_statistics.pdf')
plt.show()
'''),
])

write_nb("EMQA_romania_mix", [
    ("md", "# EMQA_romania_mix\n\nRomanian electricity generation mix over time.\n\n**Output:** `romania_energy_mix.pdf`"),
    ("code", STYLE_CODE),
    ("code", ROMANIA_CODE),
    ("code", '''cols = ['ro_nuclear','ro_hydro','ro_coal','ro_gas','ro_wind','ro_solar']
labels = ['Nuclear','Hydro','Coal','Gas','Wind','Solar']
colors_src = [COLORS['purple'], COLORS['blue'], COLORS['gray'], COLORS['orange'], COLORS['cyan'], COLORS['green']]
monthly = ro[cols].resample('ME').mean()

fig, ax = plt.subplots(figsize=(14, 6))
ax.stackplot(monthly.index, *[monthly[c].values for c in cols], labels=labels, colors=colors_src, alpha=0.8)
ax.set_title('Romanian Generation Mix (2023-2026)', fontweight='bold')
ax.set_ylabel('MW')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=6, frameon=False)
fig.tight_layout()
save_fig(fig, 'romania_energy_mix.pdf')
plt.show()
'''),
])

write_nb("EMQA_romania_pie", [
    ("md", "# EMQA_romania_pie\n\nRomanian energy mix composition pie chart.\n\n**Output:** `romania_energy_pie.pdf`"),
    ("code", STYLE_CODE),
    ("code", ROMANIA_CODE),
    ("code", '''cols = ['ro_nuclear','ro_hydro','ro_coal','ro_gas','ro_wind','ro_solar']
labels = ['Nuclear','Hydro','Coal','Gas','Wind','Solar']
colors_src = [COLORS['purple'], COLORS['blue'], COLORS['gray'], COLORS['orange'], COLORS['cyan'], COLORS['green']]
avgs = [ro[c].mean() for c in cols]

fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(avgs, labels=labels, colors=colors_src, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
for t in autotexts:
    t.set_fontsize(10)
ax.set_title('Romanian Average Generation Mix', fontweight='bold', pad=20)
fig.tight_layout()
save_fig(fig, 'romania_energy_pie.pdf')
plt.show()
'''),
])

write_nb("EMQA_romania_renew", [
    ("md", "# EMQA_romania_renew\n\nRenewable vs fossil generation in Romania.\n\n**Output:** `romania_renewable_fossil.pdf`"),
    ("code", STYLE_CODE),
    ("code", ROMANIA_CODE),
    ("code", '''renew = (ro['ro_hydro'] + ro['ro_wind'] + ro['ro_solar']).resample('W').mean()
fossil = (ro['ro_coal'] + ro['ro_gas']).resample('W').mean()

fig, ax = plt.subplots(figsize=(14, 6))
ax.fill_between(renew.index, renew, alpha=0.5, color=COLORS['green'], label='Renewable (Hydro+Wind+Solar)')
ax.fill_between(fossil.index, fossil, alpha=0.5, color=COLORS['gray'], label='Fossil (Coal+Gas)')
ax.set_title('Romania: Renewable vs Fossil Generation (Weekly Avg)', fontweight='bold')
ax.set_ylabel('MW')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)
fig.tight_layout()
save_fig(fig, 'romania_renewable_fossil.pdf')
plt.show()
'''),
])

write_nb("EMQA_romania_trade", [
    ("md", "# EMQA_romania_trade\n\nRomanian electricity import/export balance.\n\n**Output:** `romania_import_export.pdf`"),
    ("code", STYLE_CODE),
    ("code", ROMANIA_CODE),
    ("code", '''total_prod = ro[['ro_nuclear','ro_hydro','ro_coal','ro_gas','ro_wind','ro_solar']].sum(axis=1)
balance = total_prod - ro['ro_consumption']  # positive = export
weekly = balance.resample('W').mean()

fig, ax = plt.subplots(figsize=(14, 6))
ax.fill_between(weekly.index, weekly, where=weekly>=0, color=COLORS['green'], alpha=0.5, label='Export (surplus)')
ax.fill_between(weekly.index, weekly, where=weekly<0, color=COLORS['red'], alpha=0.5, label='Import (deficit)')
ax.axhline(0, color='gray', lw=0.8)
ax.set_title('Romanian Electricity: Import/Export Balance', fontweight='bold')
ax.set_ylabel('MW (+ export, - import)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)
fig.tight_layout()
save_fig(fig, 'romania_import_export.pdf')
plt.show()
'''),
])

write_nb("EMQA_romania_vres", [
    ("md", "# EMQA_romania_vres\n\nRomanian wind and solar variability.\n\n**Output:** `romania_wind_solar.pdf`"),
    ("code", STYLE_CODE),
    ("code", ROMANIA_CODE),
    ("code", '''fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
ax1.plot(ro.index, ro['ro_wind'], color=COLORS['cyan'], lw=0.5, alpha=0.7)
ax1.plot(ro['ro_wind'].rolling(30).mean().index, ro['ro_wind'].rolling(30).mean(), color=COLORS['blue'], lw=1.5)
ax1.set_title(f"Wind Production (Avg: {ro['ro_wind'].mean():.0f} MW, Max: {ro['ro_wind'].max():.0f} MW)", fontweight='bold')
ax1.set_ylabel('MW')
ax1.legend(['Daily', '30-day MA'], loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)

ax2.plot(ro.index, ro['ro_solar'], color=COLORS['amber'], lw=0.5, alpha=0.7)
ax2.plot(ro['ro_solar'].rolling(30).mean().index, ro['ro_solar'].rolling(30).mean(), color=COLORS['orange'], lw=1.5)
ax2.set_title(f"Solar Production (Avg: {ro['ro_solar'].mean():.0f} MW, Max: {ro['ro_solar'].max():.0f} MW)", fontweight='bold')
ax2.set_ylabel('MW')
ax2.legend(['Daily', '30-day MA'], loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)

fig.tight_layout()
save_fig(fig, 'romania_wind_solar.pdf')
plt.show()
'''),
])

write_nb("EMQA_romania_balance", [
    ("md", "# EMQA_romania_balance\n\nRomanian demand vs production balance.\n\n**Output:** `romania_demand_production.pdf`"),
    ("code", STYLE_CODE),
    ("code", ROMANIA_CODE),
    ("code", '''demand = ro['ro_consumption'].resample('W').mean()
prod = ro[['ro_nuclear','ro_hydro','ro_coal','ro_gas','ro_wind','ro_solar']].sum(axis=1).resample('W').mean()

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(demand.index, demand, color=COLORS['blue'], lw=1.2, label='Demand')
ax.plot(prod.index, prod, color=COLORS['green'], lw=1.2, label='Total Production')
ax.fill_between(demand.index, demand, prod, where=prod>=demand, alpha=0.2, color=COLORS['green'], label='Surplus')
ax.fill_between(demand.index, demand, prod, where=prod<demand, alpha=0.2, color=COLORS['red'], label='Deficit')
ax.set_title('Romanian Electricity: Demand vs Production', fontweight='bold')
ax.set_ylabel('MW')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=4, frameon=False)
fig.tight_layout()
save_fig(fig, 'romania_demand_production.pdf')
plt.show()
'''),
])

write_nb("EMQA_romania_monthly", [
    ("md", "# EMQA_romania_monthly\n\nRomanian electricity seasonal patterns.\n\n**Output:** `romania_monthly_patterns.pdf`"),
    ("code", STYLE_CODE),
    ("code", ROMANIA_CODE),
    ("code", '''cols = ['ro_nuclear','ro_hydro','ro_wind','ro_solar','ro_gas','ro_coal']
labels = ['Nuclear','Hydro','Wind','Solar','Gas','Coal']
colors_src = [COLORS['purple'], COLORS['blue'], COLORS['cyan'], COLORS['green'], COLORS['orange'], COLORS['gray']]
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, col, label, c in zip(axes.flat, cols, labels, colors_src):
    monthly = ro[col].groupby(ro.index.month).mean()
    ax.bar(range(12), monthly.values, color=c, alpha=0.7)
    ax.set_xticks(range(12)); ax.set_xticklabels(months, fontsize=8, rotation=45)
    ax.set_title(label, fontweight='bold')
    ax.set_ylabel('MW')

fig.suptitle('Romanian Generation by Source: Monthly Patterns', fontweight='bold', fontsize=13)
fig.tight_layout(rect=[0,0,1,0.96])
save_fig(fig, 'romania_monthly_patterns.pdf')
plt.show()
'''),
])

write_nb("EMQA_ro_demand", [
    ("md", "# EMQA_ro_demand\n\nRomanian electricity demand patterns.\n\n**Output:** `romania_demand.pdf`"),
    ("code", STYLE_CODE),
    ("code", ROMANIA_CODE),
    ("code", '''demand = ro['ro_consumption']

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(demand.index, demand, color=COLORS['blue'], lw=0.5, alpha=0.7)
ma = demand.rolling(30).mean()
ax.plot(ma.index, ma, color=COLORS['red'], lw=1.5)
ax.set_title('Romanian Electricity Demand', fontweight='bold')
ax.set_ylabel('MW')
ax.legend(['Daily', '30-day MA'], loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)
fig.tight_layout()
save_fig(fig, 'romania_demand.pdf')
plt.show()
'''),
])

write_nb("EMQA_ro_wind_solar", [
    ("md", "# EMQA_ro_wind_solar\n\nRomanian wind and solar production.\n\n**Output:** `romania_wind_solar_intro.pdf`"),
    ("code", STYLE_CODE),
    ("code", ROMANIA_CODE),
    ("code", '''fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(ro.index, ro['ro_wind'], color=COLORS['cyan'], lw=0.5, alpha=0.5, label='Wind')
ax.plot(ro.index, ro['ro_solar'], color=COLORS['amber'], lw=0.5, alpha=0.5, label='Solar')
ax.plot(ro['ro_wind'].rolling(30).mean().index, ro['ro_wind'].rolling(30).mean(), color=COLORS['blue'], lw=1.5, label='Wind 30d MA')
ax.plot(ro['ro_solar'].rolling(30).mean().index, ro['ro_solar'].rolling(30).mean(), color=COLORS['orange'], lw=1.5, label='Solar 30d MA')
ax.set_title('Romanian Wind & Solar Production', fontweight='bold')
ax.set_ylabel('MW')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=4, frameon=False)
fig.tight_layout()
save_fig(fig, 'romania_wind_solar_intro.pdf')
plt.show()
'''),
])


print("\\n" + "="*60)
print("ALL QUANTLET NOTEBOOKS GENERATED SUCCESSFULLY!")
print("="*60)

# Count
import glob
nbs = glob.glob(os.path.join(BASE, '*//*.ipynb'))
print(f"\\nTotal notebooks: {len(nbs)}")
for nb_path in sorted(nbs):
    rel = os.path.relpath(nb_path, BASE)
    print(f"  {rel}")
