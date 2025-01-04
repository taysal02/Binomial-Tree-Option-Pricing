# Binomial Tree Option Pricing

This repository provides a Python implementation of the Binomial Tree model for option pricing. It calculates the prices of European call and put options and demonstrates the use of a binomial tree to simulate the evolution of stock prices over time. The model is based on the Cox-Ross-Rubinstein binomial option pricing method and the knowledge learned from Chapter 1 "The Binomial No-Arbitrage Pricing Model" of *Stochastic Calculus for Finance I*, focusing on the binomial asset pricing model.

## Overview

The repository includes:
- A function `binomialtree` that computes the option price, the option price tree, the stock price tree, and the delta hedge tree using the binomial tree model.
- Visualization of the stock price tree and option price tree using `matplotlib`.

## Features

- Calculates European call and put options.
- Visualizes the stock price tree and option price tree at each time step.
- Computes the delta hedge for the option.
- Provides flexibility in input parameters like the risk-free rate, up and down factors, and the number of steps in the binomial tree.

## Installation

Ensure that you have the required Python libraries:

```bash
pip install numpy pandas matplotlib
