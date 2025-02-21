import streamlit as st
import pandas as pd
from db_util import DatabaseManager
from enc_rf import EncryptedStockVolatilityForecaster
import os
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize database manager with values from .env file
db_manager = DatabaseManager(
    host=os.getenv("DB_HOST"),
    port=int(os.getenv("DB_PORT")),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
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
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        st.title("Login")
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
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
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        st.title("Register")
        st.markdown("<br>", unsafe_allow_html=True)
        
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
    
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        st.header("Stock Volatility Forecaster")
        
        try:
            forecaster = load_forecaster()
            companies = forecaster.companies
            selected_company = st.selectbox("Select Company", companies)
            months_ahead = st.slider("Forecast Months", 1, 12, 3)
            
            if st.button("Generate Forecast"):
                with st.spinner("Generating forecast..."):
                    predictions = forecaster.predict_volatility(selected_company, months_ahead)
                    
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
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.subheader("Detailed Predictions")
                    st.dataframe(predictions.set_index('Date'))
                    
        except Exception as e:
            st.error(f"Error: {str(e)}")

def main():
    st.set_page_config(page_title="Stock Volatility Forecaster", layout="wide")
    
    # Minimal CSS just for centering titles and basic spacing
    st.markdown("""
        <style>
        /* Center titles and headers */
        .stTitle, h1, h2, h3 {
            text-align: center;
        }
        
        /* Basic spacing for form elements */
        .stForm > div {
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
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