#!/usr/bin/env python3
"""Generate Lecture 5 (Derivatives), Lecture 6 (ML), and missing L1 quantlet notebooks."""
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

RO_DE_CODE = '''import os

def load_ro_de():
    paths = [
        '../../charts/ro_de_prices_full.csv',
        '/Users/danielpele/Documents/Energy MBA/charts/ro_de_prices_full.csv',
    ]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p, parse_dates=['date'], index_col='date')
    raise FileNotFoundError("ro_de_prices_full.csv not found")

df = load_ro_de()
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


# ============================================================================
# Helper to build colab badge markdown
# ============================================================================
def colab_badge(name):
    url = f"{COLAB_BASE}/{name}/{name}.ipynb"
    return f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})"


# ============================================================================
# 1. EnergyDerivatives
# ============================================================================
def gen_energy_derivatives():
    NAME = "EnergyDerivatives"
    cells = [
        ('md', f"""{colab_badge(NAME)}

# {NAME}

Energy derivatives: payoff diagrams and Greeks (Delta, Gamma, Vega, Theta) using Black-Scholes.

**Output:** `deriv_payoff_diagrams.pdf`, `deriv_greeks_curves.pdf`
"""),
        ('code', STYLE_CODE),
        ('code', '''from scipy.stats import norm

# --- Black-Scholes helper functions ---

def bs_d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def bs_d2(S, K, T, r, sigma):
    return bs_d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def bs_call_price(S, K, T, r, sigma):
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = bs_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put_price(S, K, T, r, sigma):
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = bs_d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_delta_call(S, K, T, r, sigma):
    return norm.cdf(bs_d1(S, K, T, r, sigma))

def bs_delta_put(S, K, T, r, sigma):
    return bs_delta_call(S, K, T, r, sigma) - 1.0

def bs_gamma(S, K, T, r, sigma):
    d1 = bs_d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def bs_vega(S, K, T, r, sigma):
    d1 = bs_d1(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T) / 100  # per 1% move

def bs_theta_call(S, K, T, r, sigma):
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = bs_d2(S, K, T, r, sigma)
    term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
    return (term1 + term2) / 365  # per day
'''),
        ('code', '''# --- Payoff diagrams: Long Call, Long Put, Long Futures ---

K = 100
premium_call = 8
premium_put = 6

S_range = np.linspace(60, 140, 500)

# Payoffs
call_payoff = np.maximum(S_range - K, 0) - premium_call
put_payoff = np.maximum(K - S_range, 0) - premium_put
futures_payoff = S_range - K

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

titles = ['Long Call', 'Long Put', 'Long Futures']
payoffs = [call_payoff, put_payoff, futures_payoff]

for ax, title, payoff in zip(axes, titles, payoffs):
    ax.plot(S_range, payoff, color=COLORS['blue'], lw=2)
    ax.fill_between(S_range, payoff, 0, where=(payoff > 0),
                    color=COLORS['green'], alpha=0.3, label='Profit')
    ax.fill_between(S_range, payoff, 0, where=(payoff < 0),
                    color=COLORS['red'], alpha=0.3, label='Loss')
    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.axvline(K, color=COLORS['gray'], lw=0.8, ls=':', label=f'Strike K={K}')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Spot Price at Expiry')
    ax.set_ylabel('Profit / Loss')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), frameon=False, ncol=3)

fig.suptitle('Energy Derivative Payoff Diagrams', fontsize=15, fontweight='bold', y=1.02)
fig.tight_layout()
save_fig(fig, 'deriv_payoff_diagrams.pdf')
plt.show()
'''),
        ('code', '''# --- Greeks curves: Delta, Gamma, Vega, Theta ---

S = np.linspace(60, 140, 500)
K = 100
T = 0.25
r = 0.04
sigma = 0.35

delta_c = np.array([bs_delta_call(s, K, T, r, sigma) for s in S])
gamma_v = np.array([bs_gamma(s, K, T, r, sigma) for s in S])
vega_v = np.array([bs_vega(s, K, T, r, sigma) for s in S])
theta_v = np.array([bs_theta_call(s, K, T, r, sigma) for s in S])

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

greek_data = [
    (axes[0, 0], 'Delta (Call)', delta_c, COLORS['blue']),
    (axes[0, 1], 'Gamma', gamma_v, COLORS['green']),
    (axes[1, 0], 'Vega (per 1%)', vega_v, COLORS['purple']),
    (axes[1, 1], 'Theta (per day)', theta_v, COLORS['red']),
]

for ax, title, values, color in greek_data:
    ax.plot(S, values, color=color, lw=2)
    ax.fill_between(S, values, alpha=0.15, color=color)
    ax.axvline(K, color=COLORS['gray'], lw=0.8, ls=':', label=f'Strike K={K}')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Spot Price (S)')
    ax.set_ylabel(title.split('(')[0].strip())
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), frameon=False)

fig.suptitle(f'Black-Scholes Greeks  (K={K}, T={T}, r={r}, $\\sigma$={sigma})',
             fontsize=15, fontweight='bold', y=1.02)
fig.tight_layout()
save_fig(fig, 'deriv_greeks_curves.pdf')
plt.show()
'''),
    ]
    write_nb(NAME, cells)


# ============================================================================
# 2. RollingHedge
# ============================================================================
def gen_rolling_hedge():
    NAME = "RollingHedge"
    cells = [
        ('md', f"""{colab_badge(NAME)}

# {NAME}

Rolling hedge with collar strategy (cap + floor) and monthly cost comparison.

**Output:** `deriv_rolling_hedge.pdf`
"""),
        ('code', STYLE_CODE),
        ('code', '''# --- Generate realistic power price data ---
np.random.seed(42)

days = 365
t = np.arange(days)

# Mean-reverting process with seasonal component
base = 85 + 10 * np.sin(2 * np.pi * t / 365)  # seasonal
noise = np.zeros(days)
noise[0] = 0
for i in range(1, days):
    noise[i] = 0.92 * noise[i-1] + np.random.normal(0, 4)

# Add occasional spikes
spikes = np.random.choice([0, 1], size=days, p=[0.95, 0.05])
spike_mag = spikes * np.random.exponential(20, size=days)

power_price = base + noise + spike_mag
power_price = np.maximum(power_price, 30)  # floor at 30

# Collar parameters
cap = 100
floor_level = 70

# Hedged price: clipped to [floor, cap]
hedged_price = np.clip(power_price, floor_level, cap)
'''),
        ('code', '''# --- Two-panel chart: daily prices + monthly bars ---

dates = pd.date_range('2024-01-01', periods=days, freq='D')
df = pd.DataFrame({
    'unhedged': power_price,
    'hedged': hedged_price,
}, index=dates)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# (A) Daily prices
ax = axes[0]
ax.plot(df.index, df['unhedged'], color=COLORS['blue'], lw=1.2, alpha=0.8, label='Unhedged Price')
ax.plot(df.index, df['hedged'], color=COLORS['green'], lw=1.5, label='Hedged Price (Collar)')
ax.axhline(cap, color=COLORS['red'], ls='--', lw=1, label=f'Cap = {cap}')
ax.axhline(floor_level, color=COLORS['orange'], ls='--', lw=1, label=f'Floor = {floor_level}')
ax.fill_between(df.index, floor_level, cap, alpha=0.08, color=COLORS['green'], label='Collar Band')
ax.set_title('(A) Daily Power Price: Unhedged vs Collar-Hedged', fontsize=13, fontweight='bold')
ax.set_ylabel('Price (EUR/MWh)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), frameon=False, ncol=5)

# (B) Monthly average cost
monthly = df.resample('M').mean()
months = np.arange(len(monthly))
width = 0.35

ax2 = axes[1]
ax2.bar(months - width/2, monthly['unhedged'], width, color=COLORS['blue'], alpha=0.8, label='Unhedged Avg')
ax2.bar(months + width/2, monthly['hedged'], width, color=COLORS['green'], alpha=0.8, label='Hedged Avg')
ax2.set_xticks(months)
ax2.set_xticklabels([d.strftime('%b') for d in monthly.index], rotation=45)
ax2.set_title('(B) Monthly Average Cost Comparison', fontsize=13, fontweight='bold')
ax2.set_ylabel('Avg Price (EUR/MWh)')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), frameon=False, ncol=2)

# Savings
total_unhedged = df['unhedged'].sum()
total_hedged = df['hedged'].sum()
savings_pct = (total_unhedged - total_hedged) / total_unhedged * 100
print(f"Total unhedged cost index: {total_unhedged:,.0f}")
print(f"Total hedged cost index:   {total_hedged:,.0f}")
print(f"Savings from collar hedge: {savings_pct:.1f}%")

fig.suptitle('Rolling Collar Hedge Strategy', fontsize=15, fontweight='bold', y=1.02)
fig.tight_layout()
save_fig(fig, 'deriv_rolling_hedge.pdf')
plt.show()
'''),
    ]
    write_nb(NAME, cells)


# ============================================================================
# 3. EMQA_train_test_split
# ============================================================================
def gen_train_test_split():
    NAME = "EMQA_train_test_split"
    cells = [
        ('md', f"""{colab_badge(NAME)}

# {NAME}

Train/test/validation split schematic for time series data.

**Output:** `ml_train_test_split.pdf`
"""),
        ('code', STYLE_CODE),
        ('code', '''import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyArrowPatch

fig, axes = plt.subplots(2, 1, figsize=(14, 5), gridspec_kw={'height_ratios': [1, 1]})

# Common settings
bar_height = 0.5
y_center = 0.5

# --- (A) Simple 70/30 train/test split ---
ax = axes[0]
ax.add_patch(Rectangle((0, y_center - bar_height/2), 0.70, bar_height,
             facecolor=COLORS['green'], alpha=0.7, edgecolor='white', lw=2))
ax.add_patch(Rectangle((0.70, y_center - bar_height/2), 0.30, bar_height,
             facecolor=COLORS['red'], alpha=0.7, edgecolor='white', lw=2))

ax.text(0.35, y_center, 'Train (70%)', ha='center', va='center',
        fontsize=14, fontweight='bold', color='white')
ax.text(0.85, y_center, 'Test (30%)', ha='center', va='center',
        fontsize=14, fontweight='bold', color='white')

ax.set_xlim(-0.02, 1.02)
ax.set_ylim(0, 1)
ax.set_title('(A) Simple Train / Test Split', fontsize=13, fontweight='bold')
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

# Time arrow
ax.annotate('', xy=(1.0, 0.05), xytext=(0.0, 0.05),
            arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1.5))
ax.text(0.5, -0.05, 'Time', ha='center', va='top', fontsize=11, color=COLORS['gray'])

# --- (B) 70/15/15 train/val/test split ---
ax2 = axes[1]
ax2.add_patch(Rectangle((0, y_center - bar_height/2), 0.70, bar_height,
              facecolor=COLORS['green'], alpha=0.7, edgecolor='white', lw=2))
ax2.add_patch(Rectangle((0.70, y_center - bar_height/2), 0.15, bar_height,
              facecolor=COLORS['orange'], alpha=0.7, edgecolor='white', lw=2))
ax2.add_patch(Rectangle((0.85, y_center - bar_height/2), 0.15, bar_height,
              facecolor=COLORS['red'], alpha=0.7, edgecolor='white', lw=2))

ax2.text(0.35, y_center, 'Train (70%)', ha='center', va='center',
         fontsize=14, fontweight='bold', color='white')
ax2.text(0.775, y_center, 'Val (15%)', ha='center', va='center',
         fontsize=13, fontweight='bold', color='white')
ax2.text(0.925, y_center, 'Test (15%)', ha='center', va='center',
         fontsize=13, fontweight='bold', color='white')

ax2.set_xlim(-0.02, 1.02)
ax2.set_ylim(0, 1)
ax2.set_title('(B) Train / Validation / Test Split', fontsize=13, fontweight='bold')
ax2.set_xticks([])
ax2.set_yticks([])
for spine in ax2.spines.values():
    spine.set_visible(False)

# Time arrow
ax2.annotate('', xy=(1.0, 0.05), xytext=(0.0, 0.05),
             arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1.5))
ax2.text(0.5, -0.05, 'Time', ha='center', va='top', fontsize=11, color=COLORS['gray'])

fig.suptitle('Data Splitting Strategies for Time Series', fontsize=15, fontweight='bold', y=1.04)
fig.tight_layout()
save_fig(fig, 'ml_train_test_split.pdf')
plt.show()
'''),
    ]
    write_nb(NAME, cells)


# ============================================================================
# 4. EMQA_tscv
# ============================================================================
def gen_tscv():
    NAME = "EMQA_tscv"
    cells = [
        ('md', f"""{colab_badge(NAME)}

# {NAME}

Time series cross-validation visualization with expanding training window.

**Output:** `ml_tscv_visualization.pdf`
"""),
        ('code', STYLE_CODE),
        ('code', '''from matplotlib.patches import Rectangle

n_folds = 5
total_periods = 10
test_size = 1

fig, ax = plt.subplots(figsize=(14, 6))

bar_height = 0.6
y_positions = list(range(n_folds, 0, -1))

for i, y in enumerate(y_positions):
    train_end = (i + 3)  # expanding: fold 0 uses 3, fold 1 uses 4, etc.
    test_start = train_end
    test_end = test_start + test_size

    # Unused region (before training data is conceptually empty)
    # Training block
    ax.add_patch(Rectangle((0, y - bar_height/2), train_end, bar_height,
                 facecolor=COLORS['green'], alpha=0.7, edgecolor='white', lw=2))

    # Test block
    ax.add_patch(Rectangle((test_start, y - bar_height/2), test_size, bar_height,
                 facecolor=COLORS['red'], alpha=0.7, edgecolor='white', lw=2))

    # Unused (future) region
    if test_end < total_periods:
        ax.add_patch(Rectangle((test_end, y - bar_height/2), total_periods - test_end, bar_height,
                     facecolor='#E0E0E0', alpha=0.3, edgecolor='white', lw=2))

    # Labels
    ax.text(train_end / 2, y, 'Train', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')
    ax.text(test_start + test_size / 2, y, 'Test', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')
    ax.text(-0.5, y, f'Fold {i+1}', ha='right', va='center', fontsize=12, fontweight='bold')

ax.set_xlim(-1.5, total_periods + 0.5)
ax.set_ylim(0.2, n_folds + 1)
ax.set_xticks([])
ax.set_yticks([])

for spine in ax.spines.values():
    spine.set_visible(False)

# Time arrow at bottom
arrow_y = 0.4
ax.annotate('', xy=(total_periods, arrow_y), xytext=(0, arrow_y),
            arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=2))
ax.text(total_periods / 2, 0.15, 'Time', ha='center', va='top', fontsize=12, color=COLORS['gray'])

# Legend
import matplotlib.patches as mpatches
train_patch = mpatches.Patch(facecolor=COLORS['green'], alpha=0.7, label='Train')
test_patch = mpatches.Patch(facecolor=COLORS['red'], alpha=0.7, label='Test')
ax.legend(handles=[train_patch, test_patch],
          loc='upper center', bbox_to_anchor=(0.5, -0.05), frameon=False, ncol=2, fontsize=12)

ax.set_title('Time Series Cross-Validation (Expanding Window)', fontsize=15, fontweight='bold')
fig.tight_layout()
save_fig(fig, 'ml_tscv_visualization.pdf')
plt.show()
'''),
    ]
    write_nb(NAME, cells)


# ============================================================================
# 5. EMQA_learning_curves
# ============================================================================
def gen_learning_curves():
    NAME = "EMQA_learning_curves"
    cells = [
        ('md', f"""{colab_badge(NAME)}

# {NAME}

Learning curves: good fit vs overfitting (simulated data).

**Output:** `ml_learning_curves.pdf`
"""),
        ('code', STYLE_CODE),
        ('code', '''np.random.seed(42)

epochs = np.arange(1, 101)

# --- (A) Good fit ---
train_good = 1.0 * np.exp(-0.04 * epochs) + 0.15 + np.random.normal(0, 0.008, len(epochs))
val_good = 1.1 * np.exp(-0.035 * epochs) + 0.20 + np.random.normal(0, 0.012, len(epochs))

# Early stopping point
es_epoch = 65

# --- (B) Overfitting ---
train_overfit = 1.0 * np.exp(-0.05 * epochs) + 0.05 + np.random.normal(0, 0.006, len(epochs))
# Validation first decreases then increases
val_overfit_base = 1.1 * np.exp(-0.04 * epochs) + 0.20
val_overfit_rise = np.where(epochs > 30, 0.005 * (epochs - 30)**1.1, 0)
val_overfit = val_overfit_base + val_overfit_rise + np.random.normal(0, 0.01, len(epochs))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# (A) Good fit
ax = axes[0]
ax.plot(epochs, train_good, color=COLORS['green'], lw=2, label='Train Loss')
ax.plot(epochs, val_good, color=COLORS['orange'], lw=2, label='Validation Loss')
ax.axvline(es_epoch, color=COLORS['red'], ls='--', lw=1.5, label=f'Early Stopping (epoch {es_epoch})')
ax.set_title('(A) Good Fit', fontsize=13, fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), frameon=False, ncol=3)

# (B) Overfitting
ax2 = axes[1]
ax2.plot(epochs, train_overfit, color=COLORS['green'], lw=2, label='Train Loss')
ax2.plot(epochs, val_overfit, color=COLORS['orange'], lw=2, label='Validation Loss')

# Highlight overfit gap
overfit_start = 30
mask = epochs >= overfit_start
ax2.fill_between(epochs[mask], train_overfit[mask], val_overfit[mask],
                 color=COLORS['red'], alpha=0.2, label='Overfit Gap')
ax2.axvline(overfit_start, color=COLORS['gray'], ls=':', lw=1, label='Divergence Point')
ax2.set_title('(B) Overfitting', fontsize=13, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), frameon=False, ncol=4)

fig.suptitle('Learning Curves: Good Fit vs Overfitting', fontsize=15, fontweight='bold', y=1.02)
fig.tight_layout()
save_fig(fig, 'ml_learning_curves.pdf')
plt.show()
'''),
    ]
    write_nb(NAME, cells)


# ============================================================================
# Shared ML feature engineering code string
# ============================================================================
ML_FEATURE_CODE = '''from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = load_ro_de()
data = df[['ro_price', 'de_price']].dropna().copy()
data['target'] = data['ro_price']

# Lagged features
for lag in [1, 2, 7, 14, 30]:
    data[f'ro_lag_{lag}'] = data['ro_price'].shift(lag)
for lag in [1, 7]:
    data[f'de_lag_{lag}'] = data['de_price'].shift(lag)

# Rolling stats
for w in [7, 14, 30]:
    data[f'ro_ma_{w}'] = data['ro_price'].shift(1).rolling(w).mean()
    data[f'ro_std_{w}'] = data['ro_price'].shift(1).rolling(w).std()

# Temporal
data['dow'] = data.index.dayofweek
data['month'] = data.index.month
data['weekend'] = (data.index.dayofweek >= 5).astype(int)

data = data.dropna()
feature_cols = [c for c in data.columns if c not in ['target', 'ro_price', 'de_price']]

print(f"Dataset: {len(data)} rows, {len(feature_cols)} features")
print(f"Features: {feature_cols}")
'''


# ============================================================================
# 6. EMQA_model_comparison
# ============================================================================
def gen_model_comparison():
    NAME = "EMQA_model_comparison"
    cells = [
        ('md', f"""{colab_badge(NAME)}

# {NAME}

ML model comparison (Naive, Random Forest, Gradient Boosting, Ensemble) on Romanian electricity price data.

**Output:** `ml_model_comparison.pdf`
"""),
        ('code', STYLE_CODE),
        ('code', RO_DE_CODE),
        ('code', ML_FEATURE_CODE),
        ('code', '''# --- Train/test split (time series: 70/30) ---
split = int(len(data) * 0.7)
X_train, X_test = data[feature_cols].iloc[:split], data[feature_cols].iloc[split:]
y_train, y_test = data['target'].iloc[:split], data['target'].iloc[split:]

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# --- Models ---
# Naive baseline: lag-1
naive_pred = data['ro_price'].shift(1).iloc[split:].reindex(y_test.index)

# Random Forest
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = pd.Series(rf.predict(X_test), index=y_test.index)

# Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)
gb_pred = pd.Series(gb.predict(X_test), index=y_test.index)

# Simple average ensemble
ens_pred = (rf_pred + gb_pred) / 2

# --- Metrics ---
models = {
    'Naive (lag-1)': naive_pred,
    'Random Forest': rf_pred,
    'GradientBoosting': gb_pred,
    'Ensemble (Avg)': ens_pred,
}

results = {}
for name, pred in models.items():
    mask = pred.notna() & y_test.notna()
    mae = mean_absolute_error(y_test[mask], pred[mask])
    r2 = r2_score(y_test[mask], pred[mask])
    results[name] = {'MAE': mae, 'R2': r2}
    print(f"{name:20s}  MAE={mae:.2f}  R2={r2:.4f}")
'''),
        ('code', '''# --- Chart: 1x2 MAE and R2 bars ---
model_names = list(results.keys())
maes = [results[m]['MAE'] for m in model_names]
r2s = [results[m]['R2'] for m in model_names]
bar_colors = [COLORS['gray'], COLORS['green'], COLORS['purple'], COLORS['red']]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# (A) MAE
ax = axes[0]
bars = ax.bar(model_names, maes, color=bar_colors, alpha=0.8, edgecolor='white', lw=1.5)
for bar, val in zip(bars, maes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_title('(A) Mean Absolute Error (lower is better)', fontsize=13, fontweight='bold')
ax.set_ylabel('MAE (EUR/MWh)')
ax.tick_params(axis='x', rotation=20)

# (B) R2
ax2 = axes[1]
bars2 = ax2.bar(model_names, r2s, color=bar_colors, alpha=0.8, edgecolor='white', lw=1.5)
for bar, val in zip(bars2, r2s):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax2.set_title('(B) R-squared (higher is better)', fontsize=13, fontweight='bold')
ax2.set_ylabel('R$^2$')
ax2.tick_params(axis='x', rotation=20)

fig.suptitle('Model Comparison: Romanian Electricity Price Forecasting',
             fontsize=15, fontweight='bold', y=1.02)
fig.tight_layout()
save_fig(fig, 'ml_model_comparison.pdf')
plt.show()
'''),
    ]
    write_nb(NAME, cells)


# ============================================================================
# 7. EMQA_feature_importance
# ============================================================================
def gen_feature_importance():
    NAME = "EMQA_feature_importance"
    cells = [
        ('md', f"""{colab_badge(NAME)}

# {NAME}

Feature importance from a Random Forest model trained on Romanian electricity prices.

**Output:** `ml_feature_importance.pdf`
"""),
        ('code', STYLE_CODE),
        ('code', RO_DE_CODE),
        ('code', ML_FEATURE_CODE),
        ('code', '''# --- Train RF and extract feature importances ---
split = int(len(data) * 0.7)
X_train = data[feature_cols].iloc[:split]
y_train = data['target'].iloc[:split]

rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

importance = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
top12 = importance.tail(12)

# Assign colors by feature type
def feature_color(name):
    if 'de_' in name:
        return COLORS['green']   # German features
    elif 'spread' in name.lower():
        return COLORS['purple']  # Spread
    elif name in ['dow', 'month', 'weekend']:
        return COLORS['orange']  # Temporal
    else:
        return COLORS['blue']    # Romanian lags/stats

bar_colors = [feature_color(f) for f in top12.index]

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(range(len(top12)), top12.values, color=bar_colors, alpha=0.85, edgecolor='white', lw=1)

ax.set_yticks(range(len(top12)))
ax.set_yticklabels(top12.index, fontsize=11)
ax.set_xlabel('Importance', fontsize=12)
ax.set_title('Top 12 Feature Importances (Random Forest)', fontsize=15, fontweight='bold')

# Add value labels
for i, (val, bar) in enumerate(zip(top12.values, bars)):
    ax.text(val + 0.002, i, f'{val:.3f}', va='center', fontsize=10)

# Legend for color coding
import matplotlib.patches as mpatches
legend_items = [
    mpatches.Patch(color=COLORS['blue'], label='Romanian price lags/stats'),
    mpatches.Patch(color=COLORS['green'], label='German price features'),
    mpatches.Patch(color=COLORS['purple'], label='Spread features'),
    mpatches.Patch(color=COLORS['orange'], label='Temporal features'),
]
ax.legend(handles=legend_items, loc='upper center', bbox_to_anchor=(0.5, -0.10),
          frameon=False, ncol=4, fontsize=10)

fig.tight_layout()
save_fig(fig, 'ml_feature_importance.pdf')
plt.show()
'''),
    ]
    write_nb(NAME, cells)


# ============================================================================
# 8. EMQA_actual_vs_predicted
# ============================================================================
def gen_actual_vs_predicted():
    NAME = "EMQA_actual_vs_predicted"
    cells = [
        ('md', f"""{colab_badge(NAME)}

# {NAME}

Actual vs predicted evaluation: time series overlay and scatter plot with confidence intervals.

**Output:** `ml_actual_vs_predicted.pdf`
"""),
        ('code', STYLE_CODE),
        ('code', RO_DE_CODE),
        ('code', ML_FEATURE_CODE),
        ('code', '''# --- Train models and get ensemble predictions ---
split = int(len(data) * 0.7)
X_train, X_test = data[feature_cols].iloc[:split], data[feature_cols].iloc[split:]
y_train, y_test = data['target'].iloc[:split], data['target'].iloc[split:]

rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = pd.Series(rf.predict(X_test), index=y_test.index)

gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)
gb_pred = pd.Series(gb.predict(X_test), index=y_test.index)

ens_pred = (rf_pred + gb_pred) / 2

# Error stats
residuals = y_test - ens_pred
mae = mean_absolute_error(y_test, ens_pred)
r2 = r2_score(y_test, ens_pred)
std_resid = residuals.std()
print(f"Ensemble MAE: {mae:.2f}, R2: {r2:.4f}, Residual Std: {std_resid:.2f}")
'''),
        ('code', '''# --- 2-panel chart ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (A) Time series of last 60 test days
last_n = 60
y_last = y_test.iloc[-last_n:]
pred_last = ens_pred.iloc[-last_n:]

# 95% CI
ci_upper = pred_last + 1.96 * std_resid
ci_lower = pred_last - 1.96 * std_resid

ax = axes[0]
ax.plot(y_last.index, y_last.values, color=COLORS['blue'], lw=1.8, label='Actual')
ax.plot(pred_last.index, pred_last.values, color=COLORS['red'], lw=1.8, ls='--', label='Ensemble Prediction')
ax.fill_between(pred_last.index, ci_lower, ci_upper,
                color=COLORS['red'], alpha=0.12, label='95% CI')
ax.set_title('(A) Last 60 Test Days: Actual vs Predicted', fontsize=13, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Price (EUR/MWh)')
ax.tick_params(axis='x', rotation=30)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=3)

# (B) Scatter plot
ax2 = axes[1]
ax2.scatter(y_test.values, ens_pred.values, color=COLORS['blue'], alpha=0.3, s=15, edgecolors='none')

# Perfect prediction line
lims = [min(y_test.min(), ens_pred.min()), max(y_test.max(), ens_pred.max())]
ax2.plot(lims, lims, color=COLORS['red'], ls='--', lw=1.5, label='Perfect Prediction')

# Stats box
textstr = f'R$^2$ = {r2:.3f}\\nMAE = {mae:.1f} EUR/MWh'
props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor=COLORS['gray'])
ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=11,
         verticalalignment='top', bbox=props)

ax2.set_title('(B) Scatter: Actual vs Predicted', fontsize=13, fontweight='bold')
ax2.set_xlabel('Actual Price (EUR/MWh)')
ax2.set_ylabel('Predicted Price (EUR/MWh)')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), frameon=False)

fig.suptitle('Ensemble Model Evaluation', fontsize=15, fontweight='bold', y=1.02)
fig.tight_layout()
save_fig(fig, 'ml_actual_vs_predicted.pdf')
plt.show()
'''),
    ]
    write_nb(NAME, cells)


# ============================================================================
# 9. EMQA_rolling_recalibration
# ============================================================================
def gen_rolling_recalibration():
    NAME = "EMQA_rolling_recalibration"
    cells = [
        ('md', f"""{colab_badge(NAME)}

# {NAME}

Static vs monthly retrained (rolling recalibration) model comparison.

**Output:** `ml_rolling_recalibration.pdf`
"""),
        ('code', STYLE_CODE),
        ('code', RO_DE_CODE),
        ('code', ML_FEATURE_CODE),
        ('code', '''# --- Static model: trained once on first 70% ---
split = int(len(data) * 0.7)
test_start = max(split, len(data) - 120)  # last 120 days for test

X_all = data[feature_cols]
y_all = data['target']

X_train_static = X_all.iloc[:split]
y_train_static = y_all.iloc[:split]

X_test_period = X_all.iloc[test_start:]
y_test_period = y_all.iloc[test_start:]

# Static RF
rf_static = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
rf_static.fit(X_train_static, y_train_static)
static_pred = pd.Series(rf_static.predict(X_test_period), index=y_test_period.index)

# --- Rolling model: retrained every 30 days with expanding window ---
rolling_pred = pd.Series(dtype=float, index=y_test_period.index)
recalib_dates = []

test_indices = list(range(test_start, len(data)))
chunk_size = 30

for chunk_start_idx in range(0, len(test_indices), chunk_size):
    chunk_end_idx = min(chunk_start_idx + chunk_size, len(test_indices))
    idx_slice = test_indices[chunk_start_idx:chunk_end_idx]

    # Expanding training window up to current chunk start
    train_end = test_indices[chunk_start_idx]
    X_tr = X_all.iloc[:train_end]
    y_tr = y_all.iloc[:train_end]

    rf_roll = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
    rf_roll.fit(X_tr, y_tr)

    X_chunk = X_all.iloc[idx_slice]
    preds = rf_roll.predict(X_chunk)
    rolling_pred.iloc[chunk_start_idx:chunk_end_idx] = preds

    recalib_dates.append(data.index[test_indices[chunk_start_idx]])

print(f"Recalibration dates: {len(recalib_dates)}")
for d in recalib_dates:
    print(f"  {d.strftime('%Y-%m-%d')}")
'''),
        ('code', '''# --- Metrics ---
mask = rolling_pred.notna()

mae_static = mean_absolute_error(y_test_period[mask], static_pred[mask])
mae_rolling = mean_absolute_error(y_test_period[mask], rolling_pred[mask])
r2_static = r2_score(y_test_period[mask], static_pred[mask])
r2_rolling = r2_score(y_test_period[mask], rolling_pred[mask])

# Direction accuracy
def direction_accuracy(actual, predicted):
    actual_dir = actual.diff().dropna() > 0
    pred_dir = predicted.diff().dropna() > 0
    common = actual_dir.index.intersection(pred_dir.index)
    return (actual_dir[common] == pred_dir[common]).mean()

da_static = direction_accuracy(y_test_period[mask], static_pred[mask])
da_rolling = direction_accuracy(y_test_period[mask], rolling_pred[mask])

print(f"Static  - MAE: {mae_static:.2f}, R2: {r2_static:.4f}, Dir Acc: {da_static:.3f}")
print(f"Rolling - MAE: {mae_rolling:.2f}, R2: {r2_rolling:.4f}, Dir Acc: {da_rolling:.3f}")
'''),
        ('code', '''# --- 2-panel chart ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 1]})

# (A) Time series
ax = axes[0]
ax.plot(y_test_period.index, y_test_period.values, color=COLORS['blue'], lw=1.5, label='Actual')
ax.plot(static_pred.index, static_pred.values, color=COLORS['orange'], lw=1.2, ls='--',
        alpha=0.8, label='Static Model')
ax.plot(rolling_pred.index, rolling_pred.values, color=COLORS['green'], lw=1.5, ls='-',
        alpha=0.9, label='Rolling Model')

for rd in recalib_dates:
    ax.axvline(rd, color=COLORS['red'], ls=':', lw=0.8, alpha=0.6)
# One label for recalibration lines
ax.axvline(recalib_dates[0], color=COLORS['red'], ls=':', lw=0.8, alpha=0.6, label='Recalibration')

ax.set_title('(A) Actual vs Static vs Rolling Predictions', fontsize=13, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Price (EUR/MWh)')
ax.tick_params(axis='x', rotation=30)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=4)

# (B) Bar chart: MAE, R2, Direction Accuracy
ax2 = axes[1]
metrics = ['MAE', 'R$^2$', 'Dir. Acc.']
static_vals = [mae_static, r2_static, da_static]
rolling_vals = [mae_rolling, r2_rolling, da_rolling]

x = np.arange(len(metrics))
width = 0.32

bars1 = ax2.bar(x - width/2, static_vals, width, color=COLORS['orange'], alpha=0.8, label='Static')
bars2 = ax2.bar(x + width/2, rolling_vals, width, color=COLORS['green'], alpha=0.8, label='Rolling')

for bar, val in zip(bars1, static_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar, val in zip(bars2, rolling_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_xticks(x)
ax2.set_xticklabels(metrics, fontsize=11)
ax2.set_title('(B) Metric Comparison', fontsize=13, fontweight='bold')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), frameon=False, ncol=2)

fig.suptitle('Static vs Rolling Recalibration', fontsize=15, fontweight='bold', y=1.02)
fig.tight_layout()
save_fig(fig, 'ml_rolling_recalibration.pdf')
plt.show()
'''),
    ]
    write_nb(NAME, cells)


# ============================================================================
# 10. EMQA_renewables
# ============================================================================
def gen_renewables():
    NAME = "EMQA_renewables"
    cells = [
        ('md', f"""{colab_badge(NAME)}

# {NAME}

Wind and solar generation patterns from Romanian energy data.

**Output:** `renewables_patterns.pdf`
"""),
        ('code', STYLE_CODE),
        ('code', ROMANIA_CODE),
        ('code', '''# --- Identify wind and solar columns ---
# Try common column name patterns
wind_col = None
solar_col = None

for col in ro.columns:
    cl = col.lower()
    if 'wind' in cl and wind_col is None:
        wind_col = col
    if 'solar' in cl and solar_col is None:
        solar_col = col

# Fallback: if not found, simulate realistic patterns
if wind_col is None or solar_col is None:
    print("Wind/solar columns not found directly, checking available columns...")
    print(f"Available columns: {list(ro.columns)}")
    # Create simulated data based on seasonal patterns
    np.random.seed(42)
    n = len(ro)
    t = np.arange(n)

    if wind_col is None:
        # Wind: higher in winter, lower in summer, with high variability
        seasonal_wind = 1200 + 400 * np.cos(2 * np.pi * t / 365)
        noise_wind = np.random.normal(0, 300, n)
        ro['wind_gen'] = np.maximum(seasonal_wind + noise_wind, 50)
        wind_col = 'wind_gen'
        print("Using simulated wind generation data")

    if solar_col is None:
        # Solar: higher in summer, zero at night approximation via daily avg
        seasonal_solar = 500 + 450 * np.sin(2 * np.pi * (t - 80) / 365)
        seasonal_solar = np.maximum(seasonal_solar, 20)
        noise_solar = np.random.normal(0, 100, n)
        ro['solar_gen'] = np.maximum(seasonal_solar + noise_solar, 0)
        solar_col = 'solar_gen'
        print("Using simulated solar generation data")

wind = ro[wind_col]
solar = ro[solar_col]

print(f"Wind column: '{wind_col}', Solar column: '{solar_col}'")
print(f"Date range: {ro.index.min()} to {ro.index.max()}")
print(f"Wind stats: mean={wind.mean():.0f}, std={wind.std():.0f}")
print(f"Solar stats: mean={solar.mean():.0f}, std={solar.std():.0f}")
'''),
        ('code', '''# --- 2x2 chart ---
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# (A) Wind daily + 30-day MA
ax = axes[0, 0]
ax.plot(wind.index, wind.values, color=COLORS['cyan'], alpha=0.3, lw=0.5, label='Daily Wind')
wind_ma = wind.rolling(30).mean()
ax.plot(wind_ma.index, wind_ma.values, color=COLORS['blue'], lw=2, label='30-day MA')
ax.set_title('(A) Wind Generation', fontsize=13, fontweight='bold')
ax.set_ylabel('Generation (MW)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), frameon=False, ncol=2)

# (B) Solar daily + 30-day MA
ax2 = axes[0, 1]
ax2.plot(solar.index, solar.values, color=COLORS['amber'], alpha=0.3, lw=0.5, label='Daily Solar')
solar_ma = solar.rolling(30).mean()
ax2.plot(solar_ma.index, solar_ma.values, color=COLORS['orange'], lw=2, label='30-day MA')
ax2.set_title('(B) Solar Generation', fontsize=13, fontweight='bold')
ax2.set_ylabel('Generation (MW)')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), frameon=False, ncol=2)

# (C) Monthly boxplot: Wind by month
ax3 = axes[1, 0]
wind_df = pd.DataFrame({'wind': wind, 'month': wind.index.month})
wind_groups = [wind_df[wind_df['month'] == m]['wind'].values for m in range(1, 13)]
bp1 = ax3.boxplot(wind_groups, labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                   patch_artist=True, showfliers=False)
for patch in bp1['boxes']:
    patch.set_facecolor(COLORS['cyan'])
    patch.set_alpha(0.6)
for median in bp1['medians']:
    median.set_color(COLORS['blue'])
    median.set_linewidth(2)
ax3.set_title('(C) Wind by Month', fontsize=13, fontweight='bold')
ax3.set_ylabel('Generation (MW)')
ax3.tick_params(axis='x', rotation=45)

# (D) Monthly boxplot: Solar by month
ax4 = axes[1, 1]
solar_df = pd.DataFrame({'solar': solar, 'month': solar.index.month})
solar_groups = [solar_df[solar_df['month'] == m]['solar'].values for m in range(1, 13)]
bp2 = ax4.boxplot(solar_groups, labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                   patch_artist=True, showfliers=False)
for patch in bp2['boxes']:
    patch.set_facecolor(COLORS['amber'])
    patch.set_alpha(0.6)
for median in bp2['medians']:
    median.set_color(COLORS['orange'])
    median.set_linewidth(2)
ax4.set_title('(D) Solar by Month', fontsize=13, fontweight='bold')
ax4.set_ylabel('Generation (MW)')
ax4.tick_params(axis='x', rotation=45)

fig.suptitle('Romanian Wind & Solar Generation Patterns', fontsize=15, fontweight='bold', y=1.02)
fig.tight_layout()
save_fig(fig, 'renewables_patterns.pdf')
plt.show()
'''),
    ]
    write_nb(NAME, cells)


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    count = 0

    # Lecture 5 - Derivatives
    gen_energy_derivatives(); count += 1
    gen_rolling_hedge(); count += 1

    # Lecture 6 - ML
    gen_train_test_split(); count += 1
    gen_tscv(); count += 1
    gen_learning_curves(); count += 1
    gen_model_comparison(); count += 1
    gen_feature_importance(); count += 1
    gen_actual_vs_predicted(); count += 1
    gen_rolling_recalibration(); count += 1

    # Lecture 1 - Missing
    gen_renewables(); count += 1

    print(f"\nDone! Created {count} notebooks.")
