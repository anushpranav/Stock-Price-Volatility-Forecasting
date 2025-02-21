import streamlit as st
import pandas as pd
from db_util import DatabaseManager
from enc_rf import EncryptedStockVolatilityForecaster
import os
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Initialize database manager
db_manager = DatabaseManager(
    host="localhost",
    port=3306,
    database="dpsa_db",
    user="root",
    password="mysql123"
)

# Initialize session state variables
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'login'

def load_forecaster():
    """Initialize and load the forecaster model"""
    forecaster = EncryptedStockVolatilityForecaster("dataset/")
    if os.path.exists(os.path.join("saved_model_rf", "volatility_model.pkl")):
        forecaster.load_model()
    else:
        forecaster.load_and_preprocess_data()
        forecaster.train_model()
    return forecaster

def show_login_page():
    """Display the login page"""
    st.title("Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            success, message, user = db_manager.login_user(username, password)
            if success:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.current_page = 'home'
                st.success(message)
                st.rerun()
            else:
                st.error(message)
    
    if st.button("Don't have an account? Register here"):
        st.session_state.current_page = 'register'
        st.rerun()

def show_register_page():
    """Display the registration page"""
    st.title("Register")
    
    with st.form("register_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Register")
        
        if submit:
            if password != confirm_password:
                st.error("Passwords do not match")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters long")
            else:
                success, message = db_manager.register_user(username, email, password)
                if success:
                    st.success(message)
                    st.session_state.current_page = 'login'
                    st.rerun()
                else:
                    st.error(message)
    
    if st.button("Already have an account? Login here"):
        st.session_state.current_page = 'login'
        st.rerun()

def show_home_page():
    """Display the home page with volatility forecasting functionality"""
    st.title(f"Welcome, {st.session_state.username}!")
    
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.current_page = 'login'
        st.rerun()
    
    st.header("Stock Volatility Forecaster")
    
    # Initialize forecaster
    try:
        forecaster = load_forecaster()
        
        # Get available companies
        companies = forecaster.companies
        
        # User inputs
        selected_company = st.selectbox("Select Company", companies)
        months_ahead = st.slider("Forecast Months", 1, 12, 3)
        
        if st.button("Generate Forecast"):
            with st.spinner("Generating forecast..."):
                # Get predictions
                predictions = forecaster.predict_volatility(selected_company, months_ahead)
                
                # Create interactive plot with Plotly
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=predictions['Date'],
                    y=predictions['Predicted_Volatility'],
                    mode='lines',
                    name='Predicted Volatility',
                    line=dict(color='red', width=2)
                ))
                
                fig.update_layout(
                    title=f'Stock Volatility Forecast for {selected_company} - Next {months_ahead} Month(s)',
                    xaxis_title='Date',
                    yaxis_title='Volatility',
                    hovermode='x unified',
                    showlegend=True
                )
                
                # Display the plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Display predictions table
                st.subheader("Detailed Predictions")
                st.dataframe(predictions.set_index('Date'))
                
    except Exception as e:
        st.error(f"Error: {str(e)}")

def main():
    st.set_page_config(page_title="Stock Volatility Forecaster", layout="wide")
    
    # Page routing
    if not st.session_state.logged_in:
        if st.session_state.current_page == 'register':
            show_register_page()
        else:
            show_login_page()
    else:
        show_home_page()

if __name__ == "__main__":
    main()