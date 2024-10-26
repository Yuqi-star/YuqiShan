# Week 05 - Risk Management Project

This repository contains the solutions for the Week 05 Risk Management Project. The project includes implementing a risk management library and calculating VaR and ES using various methods across three different problems.

## Directory Structure

```
Week05/
│
├── library_include_problem3/       # Custom library modules for Problem 1
│   ├── covariance.py               # Covariance estimation techniques
│   ├── correlation.py              # Non-PSD fixes for correlation matrices
│   ├── simulation.py               # Simulation methods (e.g., Monte Carlo)
│   ├── var.py                      # Value at Risk (VaR) calculation methods
│   ├── es.py                       # Expected Shortfall (ES) calculation methods
│   ├── __init__.py                 # Initialization script for the package
│   ├──DailyPrices.csv              # Data for Problem 3 (portfolio prices)
│   └──Portfolio.csv                # Data for Problem 3 (portfolio holdings)
│
├── project_week05_problem2.py      # Solution code for Problem 2
├── project_week5_answer.pdf       # Written answers for Problem 2
├── problem1.csv                    # Data for Problem 2
└── README.md                       # This file with instructions
```

## Problem 1: Custom Risk Management Library

### Description:
In this problem, we implemented a custom Python library that includes:
1. Covariance estimation techniques (sample and EWMA methods).
2. Non-PSD (non-positive semi-definite) correlation matrix fixes.
3. Simulation methods (e.g., Monte Carlo).
4. VaR calculation methods (e.g., parametric, historical).
5. ES calculation methods (parametric and historical).

### How to Run Tests:
- Navigate to the `library_include_problem3/` folder.
- Use the provided test files from the course repository to validate the functionality. For example:

```bash
python3 -m unittest test_covariance.py
```

- Repeat for other modules (e.g., `test_var.py`, `test_es.py`).

---

## Problem 2: VaR and ES Calculations

### Description:
In this problem, we calculate the VaR and ES for a dataset (`problem1.csv`) using three different methods:
1. **Exponentially Weighted Variance** with normal distribution (λ = 0.97).
2. **Maximum Likelihood Estimation** (MLE) for fitting a T-distribution.
3. **Historical Simulation**.

### How to Run:
- Execute the script `project_week05_problem2.py` by running:

```bash
python3 project_week05_problem2.py
```

- The script will output the calculated VaR and ES values for each method and compare the results.

### Data:
- The dataset `problem1.csv` contains the necessary data for calculations.

---

## Problem 3: Portfolio Risk Management with Copula

### Description:
For this problem, we calculate VaR and ES for three portfolios (A, B, and C) using the following approach:
1. **Generalized T models** are fitted to the stocks in portfolios A and B.
2. A **Normal distribution** is fitted to the stocks in portfolio C.
3. We use a **Gaussian Copula** to model the dependencies between assets in the portfolios.
4. The script calculates the total VaR and ES for all portfolios and compares the results with previous results from Week 4.

### How to Run:
- Ensure the data files `Portfolio.csv` and `DailyPrices.csv` are in the same directory as the script.
- Run the script `project_week05_problem2.py` by executing:

```bash
python3 project_week05_problem2.py
```

- The script will output the VaR and ES values for each portfolio and the total VaR and ES.

---

## Dependencies

- **Python Version**: 3.7+
- **Required Libraries**: The following libraries are required to run the code:

```bash
pip install pandas numpy scipy statsmodels scikit-learn
```

## Notes:
- Ensure all data files (`problem1.csv`, `DailyPrices.csv`, `Portfolio.csv`) are placed in the correct directory before running the scripts.
- The written explanation for the project is included in `project_week5_answer.pdf` 