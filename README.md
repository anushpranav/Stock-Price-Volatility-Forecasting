# Stock-Price-Volatility-Forecasting

## Overview

This project is a web application for forecasting the volatility of stock prices for various companies. It uses machine learning models (Random Forest, LSTM, Ridge Regression) and homomorphic encryption (via TenSEAL) to provide privacy-preserving financial analytics. The app is built with Streamlit for the UI and supports user authentication with a MySQL backend.

---

## Features
- User registration and login (with bcrypt password hashing)
- Upload and process historical stock data (CSV)
- Feature engineering (returns, volatility, moving averages, RSI, etc.)
- Volatility forecasting using Random Forest (default), LSTM, or Ridge Regression
- Homomorphic encryption for secure data processing
- Visualization of predicted volatility
- Model and data management (save/load)

---

## Usage
- Register a new user or log in with existing credentials.
- Select a company and forecast horizon.
- View predicted volatility and plots.

---

## Security Notes
- Passwords are hashed with bcrypt before storage.
- Homomorphic encryption (CKKS via TenSEAL) is used for privacy-preserving analytics.
- Keep your `.env` file secure and never commit it to version control.
- For production, use HTTPS and restrict access to sensitive files.

