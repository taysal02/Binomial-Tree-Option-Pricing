#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 16:58:42 2025

@author: Tayyib Salawu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to calculate the binomial tree option pricing
def binomialtree(S, K, r, N, T, u, d, call_put):
    # Initialize the stock price tree with zeros
    df = pd.DataFrame(np.zeros((N+1, N+1)))
    df.iloc[0, 0] = S  # Set the initial stock price at time 0
    
    # Calculate risk-neutral probabilities
    p = (1 + r - d) / (u - d)  # Probability of going up
    q = (u - 1 - r) / (u - d)  # Probability of going down
    
    # Construct the stock price tree
    for j in range(1, N+1):  # Iterate over each time step
        for i in range(j+1):  # Iterate over possible stock prices at each time step
            if i == 0:
                df.iloc[i, j] = df.iloc[i, j-1] * u  # Up move
            else:
                df.iloc[i, j] = df.iloc[i-1, j-1] * d  # Down move
   
    # Initialize the option price tree with zeros
    option_df = pd.DataFrame(np.zeros((N+1, N+1)))
    
    # Set the terminal option prices based on the option type (call or put)
    if call_put == 'call':
        for i in range(N+1):
            option_df.iloc[i, N] = max(0, df.iloc[i, N] - K)  # Call option payoff: max(0, S - K)
    
    if call_put == 'put':
        for i in range(N+1):
            option_df.iloc[i, N] = max(0, K - df.iloc[i, N])  # Put option payoff: max(0, K - S)
    
    # Backward recursion to calculate option prices at earlier nodes
    for j in range(N-1, -1, -1):  # Start from last time step and go backward
        for i in range(j+1):  # Iterate over each node in the tree
            option_df.iloc[i, j] = (1 / (1 + r)) * ((p * option_df.iloc[i, j+1]) + (q * option_df.iloc[i+1, j+1]))  # Discount the expected option value
    
    # The option price is at the root node (time 0, stock price S)
    option_price = option_df.iloc[0, 0]
    
    # Function to visualize the stock price tree
    def showstocktree(tree):
        t = np.linspace(0, T, N+1)  # Create time points for the tree
        fig, ax = plt.subplots(figsize=(6, 4))  # Create a plot
        for j in range(len(t) - 1):  # Iterate only up to the second-to-last time step
            for i in range(j+1):
                ax.plot(t[j], tree.iloc[i, j], '.b')  # Plot the current stock price
                ax.plot([t[j], t[j+1]], [tree.iloc[i, j], tree.iloc[i, j+1]], '-b')  # Connect to the next time step (up move)
                ax.plot([t[j], t[j+1]], [tree.iloc[i, j], tree.iloc[i+1, j+1]], '-b')  # Connect to the next time step (down move)
        # Plot the final column (terminal prices)
        for i in range(len(t)):
            ax.plot(t[-1], tree.iloc[i, -1], '.b')
        ax.set_title("Binomial Tree")  # Set the title
        ax.set_ylabel("Stock Price")  # Set the y-axis label
        ax.set_xlabel("Time")  # Set the x-axis label
        plt.show()  # Show the plot
    
    # Function to visualize the option price tree
    def showoptiontree(tree):
        t = np.linspace(0, T, N+1)  # Create time points for the tree
        fig, ax = plt.subplots(figsize=(6, 4))  # Create a plot
        for j in range(len(t) - 1):  # Iterate only up to the second-to-last time step
            for i in range(j+1):
                ax.plot(t[j], tree.iloc[i, j], '.b')  # Plot the current option price
                ax.plot([t[j], t[j+1]], [tree.iloc[i, j], tree.iloc[i, j+1]], '-b')  # Connect to the next time step (up move)
                ax.plot([t[j], t[j+1]], [tree.iloc[i, j], tree.iloc[i+1, j+1]], '-b')  # Connect to the next time step (down move)
        # Plot the final column (terminal option prices)
        for i in range(len(t)):
            ax.plot(t[-1], tree.iloc[i, -1], '.b')
        ax.set_title("Option Price Tree")  # Set the title
        ax.set_ylabel("Option Price")  # Set the y-axis label
        ax.set_xlabel("Time")  # Set the x-axis label
        plt.show()  # Show the plot
    
    # Display the stock price and option price trees
    showstocktree(df)
    showoptiontree(option_df)
    
    # Calculate the delta (hedge ratio) for each node in the tree
    delta_tree = pd.DataFrame(np.zeros((N+1, N+1)))
    for j in range(N):  # Iterate through time steps
        for i in range(j+1):  # Iterate through stock prices
            delta_tree.iloc[i, j] = (option_df.iloc[i, j+1] - option_df.iloc[i+1, j+1]) / (df.iloc[i, j+1] - df.iloc[i+1, j+1])  # Calculate delta
    
    # Return the stock price tree, option price tree, option price, and delta hedge tree
    return df, option_df, option_price, delta_tree

# Example Inputs       
S = 100  # Initial stock price
K = 100  # Strike price
r = 0.05  # Risk-free rate (annualized)
T = 1     # Time to maturity (years)
N = 3     # Number of steps in the tree
u = 1.1   # Up factor
d = 0.9   # Down factor

# Test for call option
stock_tree_call, option_tree_call, option_price_call, delta_tree_call = binomialtree(S, K, r, N, T, u, d, call_put='call')
print("Stock Price Tree (Call Option):")
print(stock_tree_call)
print("\nOption Price Tree (Call Option):")
print(option_tree_call)
print(f"\nCall Option Price: {option_price_call:.2f}")
print("\nDelta Hedge Tree (Call Option):")
print(delta_tree_call)

# Test for put option
stock_tree_put, option_tree_put, option_price_put, delta_tree_put = binomialtree(S, K, r, N, T, u, d, call_put='put')
print("\nStock Price Tree (Put Option):")
print(stock_tree_put)
print("\nOption Price Tree (Put Option):")
print(option_tree_put)
print(f"\nPut Option Price: {option_price_put:.2f}")
print("\nDelta Hedge Tree (Put Option):")
print(delta_tree_put)
  
