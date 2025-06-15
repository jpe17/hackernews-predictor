import streamlit as st
# from annotated_text import annotated_text
from datetime import datetime, time, timedelta
import os
import json
import random
import time as time_module

import torch
import numpy as np

# from utils import *
# from model.architectures import *
# from model.model_loader import load_model

import requests

def load_sample_data():
    with open('frontend/data/samples.json', 'r') as f:
        data = json.load(f)
    return random.choice(data['entries'])

def load_new_sample():
    """Load new sample data and reset prediction state"""
    sample = load_sample_data()
    
    # Handle different time formats and missing time field
    if 'time' in sample:
        sample_time = datetime.strptime(sample['time'], "%Y-%m-%d %H:%M:%S")
    else:
        # Use current time as fallback if time field is missing
        sample_time = datetime.now()
    
    # Handle different author field names
    author = sample.get('by', sample.get('author', 'Unknown'))
    
    st.session_state.title = sample['title']
    st.session_state.author = author
    st.session_state.url = sample['url']
    st.session_state.date = sample_time.date()
    st.session_state.time_str = sample_time.strftime("%H:%M")
    st.session_state.true_score = sample['score']
    st.session_state.user_prediction = 0
    st.session_state.model_prediction = None
    st.session_state.prediction_made = False
    st.session_state.show_results = False

def main():
    st.set_page_config(
        page_title="Hacker News Upvote Predictor",
        page_icon="🚀",
        layout="wide"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    .score-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .win-animation {
        animation: pulse 0.5s ease-in-out infinite alternate;
    }
    @keyframes pulse {
        from { transform: scale(1); }
        to { transform: scale(1.05); }
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header"><h1>🚀 Hacker News Upvote Predictor</h1><p>Can you predict better than our AI model?</p></div>', unsafe_allow_html=True)

    # Initialize session state for form values
    if 'title' not in st.session_state:
        load_new_sample()  # Load initial sample
    
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False

    # Current post information
    st.subheader("📰 Current Post")
    
    # Display current post in a nice card format
    with st.container():
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #ff6600; margin: 1rem 0;">
            <h3 style="color: #ff6600; margin-top: 0;">{st.session_state.title}</h3>
            <p><strong>Author:</strong> {st.session_state.author}</p>
            <p><strong>URL:</strong> <a href="{st.session_state.url}" target="_blank">{st.session_state.url[:50]}{'...' if len(st.session_state.url) > 50 else ''}</a></p>
            <p><strong>Posted:</strong> {st.session_state.date} at {st.session_state.time_str}</p>
        </div>
        """, unsafe_allow_html=True)

    # Prediction input
    st.subheader("🎯 Make Your Prediction")
    
    if not st.session_state.prediction_made:
        user_prediction = st.number_input(
            "How many upvotes do you think this post will get?",
            min_value=0,
            step=1,
            value=st.session_state.get('user_prediction', 0),
            key="prediction_input"
        )
        
        # Update session state with current input
        st.session_state.user_prediction = user_prediction
        
        col_predict, col_skip = st.columns([1, 1])
        with col_predict:
            if st.button("🚀 Make Prediction", type="primary", use_container_width=True):
                if user_prediction > 0:
                    with st.spinner("Getting AI prediction..."):
                        make_prediction()
                else:
                    st.warning("Please enter a prediction greater than 0!")
        
        with col_skip:
            if st.button("⏭️ Skip This Post", use_container_width=True):
                load_new_sample()
                st.rerun()
    
    # Results section
    if st.session_state.show_results:
        show_results()

def make_prediction():
    """Handle the prediction logic"""
    try:
        # Prepare data for API
        time_str = st.session_state.time_str
        date = st.session_state.date
        combined_datetime = f"{date} {time_str}:00"
        dt = datetime.strptime(combined_datetime, "%Y-%m-%d %H:%M:%S")
        unix_timestamp = int(dt.timestamp())

        input_data = {
            "title": st.session_state.title,
            "url": st.session_state.url,
            "user": st.session_state.author,
            "timestamp": unix_timestamp
        }
        
        # Make API call
        backend_url = os.getenv("BACKEND_URL", "http://localhost:8888")
        response = requests.post(f"{backend_url}/predict", json=input_data)

        if response.status_code == 200:
            prediction = response.json()["predicted_score"]
            st.session_state.model_prediction = prediction
            st.session_state.prediction_made = True
            st.session_state.show_results = True
            st.rerun()
        else:
            st.error(f"Error from server: {response.text}")
    except requests.exceptions.ConnectionError as e:
        st.error(f"Failed to connect to FastAPI backend at {backend_url}.")
        st.error("Please ensure the backend is running and accessible.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

def show_results():
    """Display results with enhanced UI"""
    st.subheader("🎊 Results")
    
    user_prediction = st.session_state.user_prediction
    model_prediction = st.session_state.model_prediction
    true_score = st.session_state.true_score
    
    user_error = abs(true_score - user_prediction)
    model_error = abs(true_score - model_prediction)
    
    # Results cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="score-card">
            <h4>Your Prediction</h4>
            <h2>{user_prediction}</h2>
            <p>Error: {user_error}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="score-card">
            <h4>AI Prediction</h4>
            <h2>{model_prediction}</h2>
            <p>Error: {model_error}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="score-card" style="background: linear-gradient(135deg, #ff6600 0%, #ff8533 100%);">
            <h4>Actual Score</h4>
            <h2>{true_score}</h2>
            <p>Truth!</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Determine winner and show celebration
    if user_error < model_error:
        st.success("🎉 YOU WIN! You predicted better than the AI!")
        st.balloons()
        time_module.sleep(1)  # Brief pause for effect
    elif model_error < user_error:
        st.success("🤖 AI WINS! The robots are taking over prediction! 🚀")
        st.snow()
        time_module.sleep(1)  # Brief pause for effect
    else:
        st.info("🤝 IT'S A TIE! Both predictions were equally close!")
    
    # Auto-load next challenge
    if st.button("🎲 Next Challenge", type="primary", use_container_width=True):
        load_new_sample()
        st.rerun()
    
    # Auto-advance after a few seconds
    st.markdown("---")
    st.info("🔄 Loading next challenge automatically in a moment...")
    time_module.sleep(10)
    load_new_sample()
    st.rerun()

if __name__ == "__main__":
    main()