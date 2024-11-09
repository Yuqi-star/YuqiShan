# Week 07 - Options and Portfolio Analysis

## Overview

This folder contains all files for Week 07's assignment, including written responses, code, and data. The assignment covers options pricing, risk measures, and a multi-factor asset pricing model.

## Folder Contents

- **Week07_answer.pdf**: Detailed written responses.
- **project_week07.py**: Well-documented Python scripts.
- **README.md**: Instructions to run the code.
- **Data Files**: 
  - `DailyPrices.csv`
  - `F-F_Research_Data_Factors_daily.CSV`
  - `F-F_Momentum_Factor_daily.CSV`
  - `problem2.csv`

## Problem Summaries

### Problem 1: Option Pricing & Greeks
- Calculate Greeks using GBSM and finite difference methods.
- Binomial tree valuation for American options with and without dividends.
- Analyze sensitivity to dividend changes.

### Problem 2: VaR & Expected Shortfall
- Simulate AAPL returns using a normal distribution for 10-day VaR and ES.
- Compare Delta-Normal and simulation results in dollar terms.

### Problem 3: Multi-Factor Model
- Fit a Fama French 3-factor + Carhart Momentum model.
- Calculate expected annual returns and covariance for 20 stocks.
- Construct a super-efficient portfolio with a 0.05 risk-free rate.

## Running the Code

1. **Setup**: Ensure Python 3.7+ is installed. Required packages: `numpy`, `pandas`, `matplotlib`, `scipy`, `statsmodels`, `yfinance`.
   - Install dependencies:
     ```bash
     pip install numpy pandas matplotlib scipy statsmodels yfinance
     ```
2. **Execution**: Navigate to `Week07` and run scripts in order. Example:
   ```bash
   python problem1.py
   ```

## Notes
- Ensure data files are in the same directory as the scripts(code).
