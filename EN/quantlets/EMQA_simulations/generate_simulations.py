"""
EMQA Simulations - Generate White Noise and Random Walk Charts
Energy Markets Quantitative Analysis
Harvard-quality charts: transparent background, no grid, legend outside bottom
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Set seed for reproducibility
np.random.seed(42)

# Style settings - clean, professional, no grid
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['axes.grid'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Colors
BLUE = '#1a3a6e'
RED = '#cd0000'
GREEN = '#2e7d32'
GRAY = '#808080'

# Generate data
T = 500  # Number of observations

# White noise
white_noise = np.random.normal(0, 1, T)

# Random walk (cumulative sum of white noise)
random_walk = np.cumsum(np.random.normal(0, 1, T))

# AR(1) process with phi = 0.9 (persistent)
ar1 = np.zeros(T)
phi = 0.9
for t in range(1, T):
    ar1[t] = phi * ar1[t-1] + np.random.normal(0, 1)

# Mean-reverting process (Ornstein-Uhlenbeck discretization)
mean_level = 50
theta = 0.1  # Speed of reversion
sigma = 5
mean_revert = np.zeros(T)
mean_revert[0] = mean_level
for t in range(1, T):
    mean_revert[t] = mean_revert[t-1] + theta * (mean_level - mean_revert[t-1]) + sigma * np.random.normal(0, 1)

# Time index
time = np.arange(T)

#==============================================================================
# Figure 1: White Noise
#==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Time series plot
ax1 = axes[0]
ax1.plot(time, white_noise, color=BLUE, linewidth=0.8, alpha=0.8, label='White noise')
ax1.axhline(y=0, color=GRAY, linestyle='--', linewidth=1, alpha=0.5)
ax1.axhline(y=2, color=RED, linestyle=':', linewidth=1, alpha=0.7, label='$\\pm 2\\sigma$')
ax1.axhline(y=-2, color=RED, linestyle=':', linewidth=1, alpha=0.7)
ax1.set_xlabel('Time (t)')
ax1.set_ylabel('Value')
ax1.set_title('White Noise: $\\varepsilon_t \\sim N(0,1)$')
ax1.set_xlim(0, T)
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

# ACF plot
ax2 = axes[1]
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(white_noise, ax=ax2, lags=30, alpha=0.05, color=BLUE, vlines_kwargs={'colors': BLUE})
ax2.set_title('ACF of White Noise')
ax2.set_xlabel('Lag')
ax2.set_ylabel('Autocorrelation')

plt.tight_layout()
plt.savefig('../white_noise_simulation.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.close()

#==============================================================================
# Figure 2: Random Walk
#==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Time series with confidence bands
ax1 = axes[0]
ax1.plot(time, random_walk, color=BLUE, linewidth=1, label='Random walk')
ax1.axhline(y=0, color=GRAY, linestyle='--', linewidth=1, alpha=0.5)
# Expanding confidence bands
upper_band = 2 * np.sqrt(time + 1)
lower_band = -2 * np.sqrt(time + 1)
ax1.fill_between(time, lower_band, upper_band, alpha=0.1, color=RED, label='$\\pm 2\\sigma\\sqrt{t}$')
ax1.plot(time, upper_band, color=RED, linestyle=':', linewidth=1, alpha=0.7)
ax1.plot(time, lower_band, color=RED, linestyle=':', linewidth=1, alpha=0.7)
ax1.set_xlabel('Time (t)')
ax1.set_ylabel('Value')
ax1.set_title('Random Walk: $X_t = X_{t-1} + \\varepsilon_t$')
ax1.set_xlim(0, T)
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

# ACF plot
ax2 = axes[1]
plot_acf(random_walk, ax=ax2, lags=30, alpha=0.05, color=BLUE, vlines_kwargs={'colors': BLUE})
ax2.set_title('ACF of Random Walk (Slow Decay)')
ax2.set_xlabel('Lag')
ax2.set_ylabel('Autocorrelation')

plt.tight_layout()
plt.savefig('../random_walk_simulation.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.close()

#==============================================================================
# Figure 3: AR(1) Process
#==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Time series
ax1 = axes[0]
ax1.plot(time, ar1, color=BLUE, linewidth=0.8, label=f'AR(1), $\\phi={phi}$')
ax1.axhline(y=0, color=GRAY, linestyle='--', linewidth=1, alpha=0.5)
ax1.set_xlabel('Time (t)')
ax1.set_ylabel('Value')
ax1.set_title(f'AR(1) Process: $X_t = {phi}X_{{t-1}} + \\varepsilon_t$')
ax1.set_xlim(0, T)
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

# ACF plot with theoretical decay
ax2 = axes[1]
plot_acf(ar1, ax=ax2, lags=30, alpha=0.05, color=BLUE, vlines_kwargs={'colors': BLUE})
# Add theoretical ACF
lags = np.arange(0, 31)
theoretical_acf = phi ** lags
ax2.plot(lags, theoretical_acf, 'r--', linewidth=1.5, label=f'Theoretical: $\\phi^k$')
ax2.set_title('ACF of AR(1) Process')
ax2.set_xlabel('Lag')
ax2.set_ylabel('Autocorrelation')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

plt.tight_layout()
plt.savefig('../ar1_simulation.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.close()

#==============================================================================
# Figure 4: Mean Reversion (Ornstein-Uhlenbeck)
#==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Time series
ax1 = axes[0]
ax1.plot(time, mean_revert, color=BLUE, linewidth=0.8, label='Price')
ax1.axhline(y=mean_level, color=GREEN, linestyle='--', linewidth=2, label=f'Mean = {mean_level}')
ax1.set_xlabel('Time (t)')
ax1.set_ylabel('Price')
ax1.set_title(f'Mean-Reverting Process ($\\theta = {theta}$)')
ax1.set_xlim(0, T)
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

# ACF plot
ax2 = axes[1]
plot_acf(mean_revert, ax=ax2, lags=30, alpha=0.05, color=BLUE, vlines_kwargs={'colors': BLUE})
ax2.set_title('ACF of Mean-Reverting Process')
ax2.set_xlabel('Lag')
ax2.set_ylabel('Autocorrelation')

plt.tight_layout()
plt.savefig('../mean_reversion_simulation.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.close()

#==============================================================================
# Figure 5: Comparison - Stationary vs Non-Stationary
#==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# White noise (stationary)
ax1 = axes[0, 0]
ax1.plot(time, white_noise, color=GREEN, linewidth=0.7, label='White noise')
ax1.axhline(y=0, color=GRAY, linestyle='--', linewidth=1, alpha=0.5)
ax1.set_title('White Noise (Stationary)', color=GREEN, fontweight='bold')
ax1.set_xlabel('Time')
ax1.set_ylabel('Value')
ax1.set_xlim(0, T)
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

# Random walk (non-stationary)
ax2 = axes[0, 1]
ax2.plot(time, random_walk, color=RED, linewidth=0.8, label='Random walk')
ax2.axhline(y=0, color=GRAY, linestyle='--', linewidth=1, alpha=0.5)
ax2.set_title('Random Walk (Non-Stationary)', color=RED, fontweight='bold')
ax2.set_xlabel('Time')
ax2.set_ylabel('Value')
ax2.set_xlim(0, T)
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

# AR(1) (stationary)
ax3 = axes[1, 0]
ax3.plot(time, ar1, color=GREEN, linewidth=0.7, label=f'AR(1), $\\phi={phi}$')
ax3.axhline(y=0, color=GRAY, linestyle='--', linewidth=1, alpha=0.5)
ax3.set_title('AR(1) with $\\phi=0.9$ (Stationary)', color=GREEN, fontweight='bold')
ax3.set_xlabel('Time')
ax3.set_ylabel('Value')
ax3.set_xlim(0, T)
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

# Mean-reverting (stationary)
ax4 = axes[1, 1]
ax4.plot(time, mean_revert, color=GREEN, linewidth=0.7, label='Mean-reverting')
ax4.axhline(y=mean_level, color=GRAY, linestyle='--', linewidth=1, alpha=0.5)
ax4.set_title('Mean-Reverting (Stationary)', color=GREEN, fontweight='bold')
ax4.set_xlabel('Time')
ax4.set_ylabel('Price')
ax4.set_xlim(0, T)
ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

plt.tight_layout()
plt.savefig('../stationarity_comparison_simulation.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.close()

#==============================================================================
# Figure 6: White Noise Complete (for detailed slide)
#==============================================================================
fig = plt.figure(figsize=(14, 4))

# Time series
ax1 = fig.add_subplot(131)
ax1.plot(time[:200], white_noise[:200], color=BLUE, linewidth=0.8, label='$\\varepsilon_t$')
ax1.axhline(y=0, color=GRAY, linestyle='--', linewidth=1, alpha=0.5)
ax1.set_xlabel('Time (t)')
ax1.set_ylabel('$\\varepsilon_t$')
ax1.set_title('White Noise Series')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

# ACF
ax2 = fig.add_subplot(132)
plot_acf(white_noise, ax=ax2, lags=20, alpha=0.05, color=BLUE, vlines_kwargs={'colors': BLUE})
ax2.set_title('ACF: No Autocorrelation')
ax2.set_xlabel('Lag')

# Histogram
ax3 = fig.add_subplot(133)
ax3.hist(white_noise, bins=30, density=True, color=BLUE, alpha=0.7, edgecolor='white', label='Observed')
x = np.linspace(-4, 4, 100)
ax3.plot(x, stats.norm.pdf(x, 0, 1), 'r-', linewidth=2, label='N(0,1)')
ax3.set_xlabel('Value')
ax3.set_ylabel('Density')
ax3.set_title('Distribution: Normal')
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

plt.tight_layout()
plt.savefig('../white_noise_complete.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.close()

#==============================================================================
# Figure 7: Random Walk Complete (for detailed slide)
#==============================================================================
fig = plt.figure(figsize=(14, 4))

# Random walk
ax1 = fig.add_subplot(131)
ax1.plot(time, random_walk, color=BLUE, linewidth=0.8, label='$X_t$')
ax1.set_xlabel('Time (t)')
ax1.set_ylabel('$X_t$')
ax1.set_title('Random Walk (Non-Stationary)')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

# First differences (returns)
ax2 = fig.add_subplot(132)
returns = np.diff(random_walk)
ax2.plot(np.arange(len(returns)), returns, color=GREEN, linewidth=0.7, label='$\\Delta X_t$')
ax2.axhline(y=0, color=GRAY, linestyle='--', linewidth=1, alpha=0.5)
ax2.set_xlabel('Time (t)')
ax2.set_ylabel('$\\Delta X_t$')
ax2.set_title('First Differences (Stationary)')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

# ACF of first differences
ax3 = fig.add_subplot(133)
plot_acf(returns, ax=ax3, lags=20, alpha=0.05, color=GREEN, vlines_kwargs={'colors': GREEN})
ax3.set_title('ACF of Differences: White Noise')
ax3.set_xlabel('Lag')

plt.tight_layout()
plt.savefig('../random_walk_complete.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.close()

#==============================================================================
# Figure 8: Stochastic Process Visualization
#==============================================================================
fig, ax = plt.subplots(figsize=(10, 5))

# Multiple realizations from same process
np.random.seed(42)
for i in range(5):
    path = np.cumsum(np.random.normal(0, 0.5, 200))
    if i == 0:
        ax.plot(path, color=BLUE, linewidth=1.5, label='Observed realization')
    else:
        ax.plot(path, color=GRAY, linewidth=0.8, alpha=0.5, label='Possible paths' if i == 1 else '')

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
ax.set_xlabel('Time (t)')
ax.set_ylabel('$X_t$')
ax.set_title('Stochastic Process: One Observed Path, Many Possible Paths')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)

plt.tight_layout()
plt.savefig('../stochastic_process_simulation.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.close()

#==============================================================================
# Figure 9: ACF Patterns Comparison
#==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Non-stationary (random walk) ACF
ax1 = axes[0, 0]
plot_acf(random_walk, ax=ax1, lags=30, alpha=0.05, color=RED, vlines_kwargs={'colors': RED})
ax1.set_title('Non-Stationary: Slow ACF Decay', color=RED)
ax1.set_xlabel('Lag')

# Stationary (white noise) ACF
ax2 = axes[0, 1]
plot_acf(white_noise, ax=ax2, lags=30, alpha=0.05, color=GREEN, vlines_kwargs={'colors': GREEN})
ax2.set_title('Stationary: ACF Near Zero', color=GREEN)
ax2.set_xlabel('Lag')

# AR(1) ACF
ax3 = axes[1, 0]
plot_acf(ar1, ax=ax3, lags=30, alpha=0.05, color=BLUE, vlines_kwargs={'colors': BLUE})
ax3.set_title('AR(1): Exponential Decay', color=BLUE)
ax3.set_xlabel('Lag')

# Squared returns (volatility clustering proxy)
squared_wn = white_noise ** 2
ax4 = axes[1, 1]
plot_acf(squared_wn, ax=ax4, lags=30, alpha=0.05, color=BLUE, vlines_kwargs={'colors': BLUE})
ax4.set_title('Squared Returns: Volatility Clustering', color=BLUE)
ax4.set_xlabel('Lag')

plt.tight_layout()
plt.savefig('../acf_patterns_simulation.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.close()

print("All simulation charts generated successfully!")
print("Files created in quantlets/ directory:")
print("  - white_noise_simulation.pdf")
print("  - random_walk_simulation.pdf")
print("  - ar1_simulation.pdf")
print("  - mean_reversion_simulation.pdf")
print("  - stationarity_comparison_simulation.pdf")
print("  - white_noise_complete.pdf")
print("  - random_walk_complete.pdf")
print("  - stochastic_process_simulation.pdf")
print("  - acf_patterns_simulation.pdf")
