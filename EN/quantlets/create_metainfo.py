#!/usr/bin/env python3
"""Create Metainfo.txt for all quantlet folders."""
import os

BASE = os.path.dirname(os.path.abspath(__file__))

# Quantlet name -> (description, keywords)
QUANTLETS = {
    # Lecture 1: Energy Time Series
    "EMQA_energy_overview": (
        "Overview visualization of global energy markets including oil, gas, and electricity price dynamics.",
        "energy markets, oil prices, gas prices, electricity prices, market overview"
    ),
    "EMQA_oil_prices": (
        "Oil prices overview and historical analysis with trend identification.",
        "oil prices, Brent crude, WTI, historical analysis, energy commodities"
    ),
    "EMQA_oil_analysis": (
        "Historical analysis of crude oil prices with statistical properties and stylized facts.",
        "oil prices, historical analysis, stylized facts, descriptive statistics"
    ),
    "EMQA_oil_stats": (
        "Descriptive statistics of crude oil prices including moments, quantiles, and distribution properties.",
        "oil prices, descriptive statistics, moments, distribution, energy data"
    ),
    "EMQA_oil_overview": (
        "Comprehensive overview of crude oil market dynamics and price behavior.",
        "oil market, crude oil, market dynamics, price behavior, energy overview"
    ),
    "EMQA_oil_returns": (
        "Log returns computation and analysis for crude oil prices.",
        "log returns, oil prices, return distribution, financial returns, Brent crude"
    ),
    "EMQA_oil_acf": (
        "Autocorrelation function analysis of crude oil prices and returns.",
        "ACF, autocorrelation, oil prices, serial correlation, time series"
    ),
    "EMQA_gas_seasonality": (
        "Seasonal pattern analysis of natural gas prices with seasonal index computation.",
        "natural gas, seasonality, seasonal index, Henry Hub, energy prices"
    ),
    "EMQA_electricity_spikes": (
        "Analysis of electricity price spikes and extreme events in power markets.",
        "electricity prices, price spikes, extreme events, power markets, negative prices"
    ),
    "EMQA_load_patterns": (
        "Electricity load pattern analysis showing daily, weekly, and seasonal demand cycles.",
        "electricity demand, load patterns, daily cycle, weekly pattern, energy consumption"
    ),
    "EMQA_renewables": (
        "Renewable energy production analysis including wind and solar generation patterns.",
        "renewable energy, wind power, solar energy, energy production, green energy"
    ),
    "EMQA_mean_reversion": (
        "Mean reversion analysis in energy prices with half-life estimation and spike dynamics.",
        "mean reversion, half-life, energy prices, electricity spikes, Ornstein-Uhlenbeck"
    ),
    "EMQA_time_series_intro": (
        "Introduction to time series concepts for energy markets with visual examples.",
        "time series, introduction, energy markets, stochastic processes, data analysis"
    ),
    "EMQA_simulations": (
        "Simulation of stochastic processes: white noise, random walk, and AR processes.",
        "simulation, white noise, random walk, AR process, stochastic processes"
    ),
    "EMQA_stationarity_intro": (
        "Introduction to stationarity concepts with visual comparison of stationary and non-stationary series.",
        "stationarity, non-stationary, unit root, time series, introduction"
    ),
    "EMQA_stationarity": (
        "Stationarity testing using the Augmented Dickey-Fuller (ADF) test on energy price data.",
        "stationarity, ADF test, unit root, Dickey-Fuller, hypothesis testing"
    ),
    "EMQA_stationarity_comparison": (
        "Side-by-side comparison of stationary and non-stationary energy time series.",
        "stationarity, comparison, stationary process, unit root, energy prices"
    ),
    "EMQA_returns_distribution": (
        "Distribution analysis of energy returns including normality tests and fat tails.",
        "returns distribution, fat tails, normality test, kurtosis, energy returns"
    ),
    "EMQA_acf_reading": (
        "ACF and PACF interpretation guide for model identification in energy time series.",
        "ACF, PACF, model identification, AR, MA, ARMA, autocorrelation"
    ),
    "EMQA_acf_pacf": (
        "Computation and visualization of autocorrelation and partial autocorrelation functions.",
        "ACF, PACF, autocorrelation, partial autocorrelation, correlogram"
    ),
    "EMQA_acf_comparison": (
        "ACF comparison across multiple energy commodities to identify different correlation structures.",
        "ACF, comparison, energy commodities, correlation structure, oil, gas, electricity"
    ),
    "EMQA_volatility_clustering": (
        "Volatility clustering analysis in energy markets showing persistence of large price moves.",
        "volatility clustering, GARCH, squared returns, energy volatility, stylized facts"
    ),
    "EMQA_decomposition": (
        "Classical time series decomposition of energy data into trend, seasonal, and residual components.",
        "decomposition, trend, seasonality, residual, time series components"
    ),
    "EMQA_stl_decomposition": (
        "STL (Seasonal and Trend decomposition using Loess) applied to energy time series.",
        "STL decomposition, Loess, seasonal adjustment, trend extraction, energy data"
    ),
    "EMQA_seasonal_pattern": (
        "Seasonal pattern identification and visualization in energy price data.",
        "seasonal pattern, seasonality, energy prices, periodic behavior, calendar effects"
    ),
    "EMQA_seasonality": (
        "Seasonality detection methods applied to energy market data.",
        "seasonality, detection, periodic patterns, Fourier analysis, energy markets"
    ),
    "EMQA_seasonal_index": (
        "Seasonal index calculation for energy commodities with quarterly decomposition.",
        "seasonal index, quarterly analysis, seasonal adjustment, energy prices"
    ),
    "EMQA_elec_seasonality": (
        "Electricity price seasonality analysis showing daily and weekly periodic patterns.",
        "electricity, seasonality, daily pattern, weekly pattern, power prices"
    ),
    "EMQA_elec_stats": (
        "Descriptive statistics of electricity prices including extreme value analysis.",
        "electricity prices, descriptive statistics, extreme values, power market"
    ),
    "EMQA_elec_overview": (
        "Overview of electricity market characteristics and price dynamics.",
        "electricity market, power prices, market overview, energy trading"
    ),
    "EMQA_elec_intraday": (
        "Intraday electricity price pattern analysis showing hourly demand-driven cycles.",
        "intraday, electricity prices, hourly patterns, peak demand, power market"
    ),
    "EMQA_elec_weekly": (
        "Weekly electricity price patterns comparing weekday and weekend dynamics.",
        "weekly pattern, electricity prices, weekday, weekend, power demand"
    ),
    "EMQA_ro_demand": (
        "Romania electricity demand analysis with consumption patterns and trends.",
        "Romania, electricity demand, consumption, energy balance, power system"
    ),
    "EMQA_ro_wind_solar": (
        "Romania wind and solar energy production analysis and variability.",
        "Romania, wind power, solar energy, renewable variability, energy production"
    ),
    "EMQA_romania_stats": (
        "Romania energy sector statistical overview and key indicators.",
        "Romania, energy statistics, power sector, key indicators, energy data"
    ),
    "EMQA_romania_mix": (
        "Romania energy mix analysis showing generation by source type.",
        "Romania, energy mix, generation, power sources, electricity production"
    ),
    "EMQA_romania_pie": (
        "Romania energy production composition visualized as pie chart by source.",
        "Romania, energy composition, pie chart, generation mix, power sources"
    ),
    "EMQA_romania_renew": (
        "Romania renewable vs fossil fuel energy production comparison and trends.",
        "Romania, renewable energy, fossil fuels, energy transition, green energy"
    ),
    "EMQA_romania_trade": (
        "Romania energy import and export analysis with cross-border flows.",
        "Romania, energy trade, import, export, cross-border, electricity exchange"
    ),
    "EMQA_romania_vres": (
        "Romania variable renewable energy sources (wind and solar) production analysis.",
        "Romania, VRES, wind, solar, variable renewables, intermittent generation"
    ),
    "EMQA_romania_monthly": (
        "Romania monthly energy patterns showing seasonal production and consumption cycles.",
        "Romania, monthly patterns, seasonal cycle, energy production, consumption"
    ),
    "EMQA_romania_balance": (
        "Romania electricity demand-production balance and system adequacy analysis.",
        "Romania, demand-production balance, system adequacy, power balance, grid"
    ),
    "EMQA_moving_averages": (
        "Moving average computation and visualization for energy time series smoothing.",
        "moving average, smoothing, SMA, EMA, energy time series, trend extraction"
    ),
    "EMQA_moving_avg": (
        "Moving average smoothing techniques applied to energy price data.",
        "moving average, smoothing, trend, filter, energy prices"
    ),
    "EMQA_commodities_comparison": (
        "Comparison of statistical properties across energy commodities (oil, gas, electricity).",
        "commodities comparison, oil, gas, electricity, statistical properties, energy"
    ),

    # Lecture 2: ARIMA Models
    "EMQA_AR1_simulation": (
        "AR(1) process simulation with varying autoregressive coefficients and impulse response.",
        "AR(1), simulation, autoregressive, impulse response, stochastic process"
    ),
    "EMQA_ar1_simulation": (
        "AR(1) process simulation demonstrating persistence and mean reversion dynamics.",
        "AR(1), simulation, persistence, mean reversion, autoregressive process"
    ),
    "EMQA_ARMA_comparison": (
        "Comparison of ARMA process realizations with different AR and MA specifications.",
        "ARMA, comparison, AR, MA, process simulation, model behavior"
    ),
    "EMQA_ACF_PACF_identification": (
        "ACF and PACF pattern guide for ARIMA model identification.",
        "ACF, PACF, model identification, ARIMA, Box-Jenkins, pattern recognition"
    ),
    "EMQA_arima_oil": (
        "ARIMA modeling and analysis of crude oil price time series.",
        "ARIMA, oil prices, model estimation, Brent crude, time series modeling"
    ),
    "EMQA_oil_acf_pacf": (
        "ACF and PACF analysis of oil prices for ARIMA order selection.",
        "ACF, PACF, oil prices, order selection, ARIMA identification"
    ),
    "EMQA_arima_diagnostics": (
        "ARIMA model diagnostic checking including residual analysis and Ljung-Box test.",
        "ARIMA, diagnostics, residual analysis, Ljung-Box, model adequacy"
    ),
    "EMQA_arima_forecast": (
        "ARIMA forecasting for energy prices with prediction intervals and accuracy metrics.",
        "ARIMA, forecasting, prediction intervals, RMSE, energy price forecast"
    ),
    "EMQA_arima_intro": (
        "Introduction to ARIMA models with notation, components, and energy market examples.",
        "ARIMA, introduction, notation, AR, MA, differencing, energy markets"
    ),
    "EMQA_arima_electricity": (
        "ARIMA modeling for electricity prices including seasonal patterns.",
        "ARIMA, electricity prices, seasonal ARIMA, power market, forecasting"
    ),
    "EMQA_electricity_analysis": (
        "Comprehensive statistical analysis of electricity price time series.",
        "electricity, statistical analysis, price dynamics, power market, time series"
    ),
    "EMQA_model_selection": (
        "Model selection using AIC and BIC information criteria for ARIMA models.",
        "model selection, AIC, BIC, information criteria, ARIMA, parsimony"
    ),

    # Lecture 3: GARCH Volatility
    "EMQA_vol_clustering": (
        "Volatility clustering visualization and analysis in energy commodity returns.",
        "volatility clustering, ARCH effects, energy returns, conditional variance"
    ),
    "EMQA_news_impact": (
        "News impact curve for asymmetric GARCH models showing leverage effects.",
        "news impact curve, asymmetric GARCH, leverage effect, EGARCH, GJR-GARCH"
    ),
    "EMQA_garch_oil": (
        "GARCH model estimation for crude oil return volatility.",
        "GARCH, oil volatility, conditional variance, Brent crude, volatility modeling"
    ),
    "EMQA_garch_estimation": (
        "GARCH parameter estimation and interpretation for energy time series.",
        "GARCH, parameter estimation, maximum likelihood, volatility persistence, energy"
    ),
    "EMQA_vol_dynamics": (
        "Conditional volatility dynamics visualization from GARCH model output.",
        "volatility dynamics, conditional variance, GARCH, time-varying volatility"
    ),
    "EMQA_var_calculation": (
        "Value at Risk (VaR) calculation using GARCH-based volatility estimates.",
        "Value at Risk, VaR, GARCH, risk measurement, quantile, energy risk"
    ),
    "EMQA_garch_electricity": (
        "GARCH modeling for electricity price volatility with spike dynamics.",
        "GARCH, electricity, volatility, price spikes, power market risk"
    ),

    # Lecture 4: Risk Management
    "EMQA_hedge_ratio": (
        "Optimal hedge ratio estimation using OLS regression on energy futures.",
        "hedge ratio, OLS, futures hedging, minimum variance, energy risk management"
    ),
    "EMQA_hedge_compare": (
        "Comparison of hedging strategies: no hedge, full hedge, and optimal hedge.",
        "hedging strategies, comparison, hedge effectiveness, futures, energy risk"
    ),
    "EMQA_portfolio": (
        "Portfolio optimization with energy assets using Markowitz mean-variance framework.",
        "portfolio optimization, Markowitz, efficient frontier, energy assets, diversification"
    ),
    "EMQA_crack_spread": (
        "Crack spread analysis between crude oil and refined petroleum products.",
        "crack spread, refinery margin, oil, gasoline, diesel, spread trading"
    ),
    "EMQA_spark_spread": (
        "Spark spread analysis between natural gas and electricity prices.",
        "spark spread, gas-to-power, electricity, natural gas, power generation margin"
    ),
    "EMQA_pairs_trading": (
        "Pairs trading strategy implementation for crude oil benchmarks (Brent-WTI).",
        "pairs trading, statistical arbitrage, Brent, WTI, cointegration, spread"
    ),
    "EMQA_ml_rf": (
        "Random forest model for energy price prediction with feature importance.",
        "random forest, machine learning, prediction, feature importance, energy prices"
    ),
    "EMQA_ml_compare": (
        "Machine learning model comparison for energy price forecasting tasks.",
        "model comparison, machine learning, forecasting, RMSE, energy prices"
    ),
    "EMQA_case_momentum": (
        "Momentum trading strategy case study applied to energy commodities.",
        "momentum strategy, trading, energy commodities, technical analysis, backtesting"
    ),
    "EMQA_case_perf": (
        "Trading strategy performance analysis with returns, Sharpe ratio, and drawdown metrics.",
        "strategy performance, Sharpe ratio, drawdown, backtesting, trading evaluation"
    ),

    # Lecture 5: Energy Derivatives
    "RollingHedge": (
        "Rolling hedge implementation for long-term energy exposure management.",
        "rolling hedge, futures, energy hedging, contract rollover, risk management"
    ),
    "EnergyDerivatives": (
        "Energy derivatives analysis including futures, options, and structured products.",
        "energy derivatives, futures, options, structured products, energy trading"
    ),

    # Lecture 6: Machine Learning
    "EMQA_train_test_split": (
        "Train-test split methodology for time series data respecting temporal ordering.",
        "train-test split, time series, temporal ordering, data partitioning, validation"
    ),
    "EMQA_tscv": (
        "Time series cross-validation with expanding and rolling window approaches.",
        "cross-validation, time series, rolling window, expanding window, model validation"
    ),
    "EMQA_learning_curves": (
        "Learning curves analysis for machine learning models on energy data.",
        "learning curves, bias-variance, model complexity, overfitting, underfitting"
    ),
    "EMQA_model_comparison": (
        "Comprehensive comparison of ML models for energy price forecasting.",
        "model comparison, machine learning, forecasting, benchmark, energy prices"
    ),
    "EMQA_feature_importance": (
        "Feature importance analysis for energy price prediction models.",
        "feature importance, variable selection, random forest, energy prediction"
    ),
    "EMQA_actual_vs_predicted": (
        "Actual vs predicted price comparison for energy forecasting model evaluation.",
        "actual vs predicted, forecast evaluation, model accuracy, energy prices"
    ),
    "EMQA_rolling_recalibration": (
        "Rolling recalibration of ML models to adapt to changing energy market dynamics.",
        "rolling recalibration, adaptive models, concept drift, energy markets, retraining"
    ),
}

AUTHOR = "Daniel Traian Pele"
PUBLISHED = "Energy Markets Quantitative Analysis (EMQA)"
SUBMITTED = "Wednesday, 5 February 2026"

for name, (desc, keywords) in QUANTLETS.items():
    folder = os.path.join(BASE, name)
    os.makedirs(folder, exist_ok=True)

    metainfo = f"""Name of QuantLet: '{name}'

Published in: '{PUBLISHED}'

Description: '{desc}'

Keywords: '{keywords}'

Author: '{AUTHOR}'

Submitted: '{SUBMITTED}'
"""
    filepath = os.path.join(folder, "Metainfo.txt")
    with open(filepath, "w") as f:
        f.write(metainfo)
    print(f"Created: {name}/Metainfo.txt")

print(f"\nTotal: {len(QUANTLETS)} Metainfo.txt files created.")
