import streamlit as st
import pandas as pd
import joblib
import re
from youtube_comment_downloader import YoutubeCommentDownloader
from collections import Counter
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
import time
import random
import base64

# --- Set page config ---
st.set_page_config(
    page_title="VIBE SCOPE 😂 - YouTube Sentiment Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Download NLTK data ---
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

# --- Load Model and Vectorizer ---
@st.cache_resource
def load_model_and_vectorizer():
    try:
        model = joblib.load('sentiment_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model/vectorizer: {e}")
        return None, None

model, vectorizer = load_model_and_vectorizer()
lemmatizer = WordNetLemmatizer()

# --- Text Preprocessing ---
def clean_and_lemmatize_comment(text):
    text = re.sub(r"http\S+|[^a-zA-Z\s]", "", text).strip().lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

def predict_sentiment(text):
    if model is None or vectorizer is None:
        return "neutral"
    cleaned_text = clean_and_lemmatize_comment(text)
    X = vectorizer.transform([cleaned_text])
    try:
        return model.predict(X)[0]
    except Exception as e:
        st.error(f"Error predicting sentiment: {e}")
        return "neutral"

def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

# --- Sentiment Analysis ---
def analyze_youtube_sentiment(url, num_comments):
    video_id = extract_video_id(url)
    if not video_id:
        return "Invalid YouTube URL.", None, None, None

    downloader = YoutubeCommentDownloader()
    try:
        comments_generator = downloader.get_comments_from_url(url, sort_by=0)
    except Exception as e:
        return f"Error downloading comments: {e}", None, None, None

    sentiment_counts = Counter()
    individual_sentiments = []

    for i, comment in enumerate(comments_generator):
        if i >= num_comments:
            break
        text = comment['text']
        sentiment = predict_sentiment(text)
        sentiment_counts[sentiment] += 1
        individual_sentiments.append((text, sentiment))

    if not sentiment_counts:
        return "No comments to analyze.", None, None, None

    overall_sentiment = sentiment_counts.most_common(1)[0][0]
    summary = f"Overall sentiment is {overall_sentiment}."

    # Create vibrant color palette
    colors = {
        'positive': '#4CAF50',  # Green
        'negative': '#F44336',  # Red
        'neutral': '#2196F3'    # Blue
    }
    
    # Bar Chart with enhanced styling
    fig_bar, ax = plt.subplots(figsize=(8, 5))
    sentiment_labels = list(sentiment_counts.keys())
    sentiment_values = list(sentiment_counts.values())
    
    bars = ax.bar(
        sentiment_labels, 
        sentiment_values, 
        color=[colors.get(k, '#a8dadc') for k in sentiment_labels]
    )
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    color='white', fontweight='bold')
    
    ax.set_title("Sentiment Distribution", color='white', fontsize=14, fontweight='bold')
    ax.set_xlabel("Sentiment", color='white', fontsize=12)
    ax.set_ylabel("Count", color='white', fontsize=12)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.grid(axis='y', linestyle='--', alpha=0.7, color='gray')
    ax.set_facecolor('#1a202c')
    fig_bar.patch.set_facecolor('#0b0c2a')
    plt.tight_layout()

    # Pie Chart with enhanced styling
    fig_pie, ax_pie = plt.subplots(figsize=(8, 5))
    
    wedges, texts, autotexts = ax_pie.pie(
        sentiment_values, 
        labels=sentiment_labels, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=[colors.get(k, '#a8dadc') for k in sentiment_labels],
        wedgeprops={'edgecolor': 'white', 'linewidth': 1}
    )
    
    # Style pie chart text
    for text in texts:
        text.set_color('white')
        text.set_fontweight('bold')
    
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
    
    ax_pie.set_title("Sentiment Distribution", color='white', fontsize=14, fontweight='bold')
    fig_pie.patch.set_facecolor('#0b0c2a')
    plt.tight_layout()

    return summary, sentiment_counts, (fig_bar, fig_pie), individual_sentiments

# --- Catchphrases for sentiments ---
catchphrases = {
    "positive": [
        "Wow! The vibes are immaculate! 🚀✨",
        "Good vibes only! Everyone's loving this! 🎉",
        "Positivity is through the roof! 🌈",
        "The crowd is absolutely loving it! 💯",
        "Sunshine and rainbows everywhere! ☀"
    ],
    "negative": [
        "Yikes! Tough crowd today... 😬",
        "Houston, we have a problem! 🚨",
        "Oof, that's a lot of shade being thrown! 🌚",
        "The comments section is on fire... and not in a good way! 🔥",
        "Someone needs to bring some positive energy ASAP! 🆘"
    ],
    "neutral": [
        "The jury's still out on this one! ⚖",
        "Neither hot nor cold - just lukewarm reactions! 🤷‍♀",
        "Perfectly balanced, as all things should be! ☯",
        "The audience is keeping their cards close to their chest! 🃏",
        "Mixed feelings all around! 🔄"
    ]
}

# --- Get random catchphrase based on sentiment ---
def get_catchphrase(sentiment):
    return random.choice(catchphrases.get(sentiment, catchphrases["neutral"]))

# --- Emoji mapping for sentiments ---
sentiment_emojis = {
    "positive": "😊",
    "negative": "😡",
    "neutral": "😐"
}

# --- Welcome Screen ---
def show_welcome_screen():
    welcome_html = """
    <style>
        .welcome-container {
            text-align: center;
            padding: 20px;
            margin-top: 30px;
            animation: fadeIn 1.5s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .welcome-title {
            font-size: 4em;
            font-weight: bold;
            margin-bottom: 20px;
            background: linear-gradient(to right, #ff8a00, #e52e71, #06c3db);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: glow 2s ease-in-out infinite alternate;
        }
        
        @keyframes glow {
            from { text-shadow: 0 0 10px rgba(255, 138, 0, 0.7), 0 0 20px rgba(229, 46, 113, 0.5); }
            to { text-shadow: 0 0 20px rgba(255, 138, 0, 0.9), 0 0 30px rgba(229, 46, 113, 0.7), 0 0 40px rgba(6, 195, 219, 0.5); }
        }
        
        .welcome-emoji {
            font-size: 6em;
            margin: 30px 0;
            display: inline-block;
            animation: bounce 2s infinite alternate;
        }
        
        @keyframes bounce {
            from { transform: translateY(0); }
            to { transform: translateY(-30px); }
        }
        
        .welcome-catchphrase {
            font-size: 1.8em;
            margin: 20px 0 40px 0;
            color: #f0f8ff;
            font-style: italic;
            animation: slideIn 2s ease;
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-50px); }
            to { opacity: 1; transform: translateX(0); }
        }
        
        /* Ghibli-inspired floating elements */
        .floating-emoji {
            position: absolute;
            font-size: 2em;
            opacity: 0.7;
            animation: float 8s ease-in-out infinite;
        }
        
        @keyframes float {
            0% { transform: translateY(0) rotate(0deg); }
            50% { transform: translateY(-20px) rotate(10deg); }
            100% { transform: translateY(0) rotate(0deg); }
        }
    </style>
    
    <div class="welcome-container">
        <div class="welcome-title">VIBE SCOPE</div>
        <div class="welcome-emoji">😎 🎭 🎬</div>
        <div class="welcome-catchphrase">"Dive into the ocean of emotions behind every YouTube video!"</div>
    </div>
    
    <!-- Floating Ghibli-style emojis -->
    <div class="floating-emoji" style="top: 15%; left: 15%; animation-delay: 0s;">😊</div>
    <div class="floating-emoji" style="top: 25%; left: 85%; animation-delay: 2s;">😡</div>
    <div class="floating-emoji" style="top: 70%; left: 20%; animation-delay: 4s;">😐</div>
    <div class="floating-emoji" style="top: 80%; left: 80%; animation-delay: 1s;">🎬</div>
    <div class="floating-emoji" style="top: 40%; left: 90%; animation-delay: 3s;">🎭</div>
    <div class="floating-emoji" style="top: 60%; left: 10%; animation-delay: 5s;">🎥</div>
    """
    
    st.markdown(welcome_html, unsafe_allow_html=True)

# --- Authentication ---
def login():
    # Initialize session state for login
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    
    if "show_login_popup" not in st.session_state:
        st.session_state.show_login_popup = False
    
    # If not logged in, show welcome screen and login option
    if not st.session_state.logged_in:
        # Show welcome screen
        show_welcome_screen()
        
        # Add login button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                """
                <style>
                    .login-button {
                        background: linear-gradient(45deg, #ff8a00, #e52e71);
                        color: white !important;
                        border: none;
                        padding: 15px 30px;
                        border-radius: 50px;
                        font-size: 1.2em;
                        font-weight: bold;
                        cursor: pointer;
                        box-shadow: 0 5px 15px rgba(229, 46, 113, 0.4);
                        transition: all 0.3s ease;
                        margin: 20px auto;
                        display: block;
                        text-align: center;
                        width: 100%;
                        animation: pulse 1.5s infinite;
                    }
                    
                    @keyframes pulse {
                        0% { transform: scale(1); }
                        50% { transform: scale(1.05); }
                        100% { transform: scale(1); }
                    }
                    
                    .login-button:hover {
                        transform: translateY(-5px);
                        box-shadow: 0 8px 20px rgba(229, 46, 113, 0.6);
                    }
                </style>
                """,
                unsafe_allow_html=True
            )
            
            if st.button("Login to Get Started", key="open_login", use_container_width=True):
                st.session_state.show_login_popup = True
        
        # Show login popup if requested
        if st.session_state.show_login_popup:
            login_popup()
            
        return False
    
    return True

# --- Login Popup ---
def login_popup():
    with st.container():
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.markdown(
                """
                <style>
                    .login-popup {
                        background: linear-gradient(135deg, #1a202c, #2d3748);
                        border-radius: 15px;
                        padding: 20px;
                        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
                        animation: popIn 0.5s ease;
                        border: 2px solid #4a5568;
                        margin-top: 20px;
                        text-align: center;
                    }
                    
                    @keyframes popIn {
                        from { opacity: 0; transform: scale(0.8); }
                        to { opacity: 1; transform: scale(1); }
                    }
                    
                    .popup-title {
                        text-align: center;
                        font-size: 2em;
                        margin-bottom: 20px;
                        color: #f0f8ff;
                    }
                    
                    .popup-emoji {
                        font-size: 3em;
                        text-align: center;
                        margin-bottom: 20px;
                        animation: spin 4s linear infinite;
                    }
                    
                    @keyframes spin {
                        from { transform: rotate(0deg); }
                        to { transform: rotate(360deg); }
                    }
                </style>
                
                <div class="login-popup">
                    <div class="popup-title">Welcome Back!</div>
                    <div class="popup-emoji">🔐</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            username = st.text_input("Username", key="popup_username")
            password = st.text_input("Password", type="password", key="popup_password")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Login", key="popup_login", use_container_width=True):
                    if username == "admin" and password == "pass123":
                        st.session_state.logged_in = True
                        st.session_state.show_login_popup = False
                        st.rerun()
                    else:
                        st.error("Invalid credentials! Please try again.")
            with col2:
                if st.button("Cancel", key="popup_cancel", use_container_width=True):
                    st.session_state.show_login_popup = False
                    st.rerun()

# --- URL Input Popup ---
def youtube_url_popup():
    with st.container():
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.markdown(
                """
                <style>
                    .url-popup {
                        background: linear-gradient(135deg, #1a202c, #2d3748);
                        border-radius: 15px;
                        padding: 20px;
                        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
                        animation: popIn 0.5s ease;
                        border: 2px solid #4a5568;
                        margin-top: 20px;
                        text-align: center;
                    }
                    
                    .popup-emoji {
                        font-size: 3em;
                        text-align: center;
                        margin-bottom: 20px;
                        animation: bounce 2s infinite alternate;
                    }
                </style>
                
                <div class="url-popup">
                    <div class="popup-title">Analyze YouTube Video</div>
                    <div class="popup-emoji">📺</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            url = st.text_input("🔗 Enter the YouTube video URL:", key="url_input")
            num_comments = st.slider("🗨 Number of comments to analyze:", min_value=10, max_value=200, value=50, step=10)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Analyze", key="analyze_button", use_container_width=True):
                    if not url:
                        st.error("Please enter a YouTube URL.")
                    else:
                        st.session_state.youtube_url = url
                        st.session_state.num_comments = num_comments
                        st.session_state.show_url_popup = False
                        st.session_state.perform_analysis = True
                        st.rerun()
            with col2:
                if st.button("Cancel", key="url_cancel", use_container_width=True):
                    st.session_state.show_url_popup = False
                    st.rerun()

# --- CSS Styling ---
def load_css():
    return """
    <style>
        body {
            color: #f0f8ff;
            background: linear-gradient(120deg, #0b0c2a, #1a202c);
        }
        
        .stApp {
            background: linear-gradient(120deg, #0b0c2a, #1a202c);
        }
        
        .main .block-container {
            padding-top: 2rem;
        }
        
        h1, h2, h3 {
            color: #f0f8ff;
        }
        
        .stButton > button {
            background: linear-gradient(45deg, #4a5568, #1e3a8a);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: bold;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background: linear-gradient(45deg, #1e3a8a, #4a5568);
            transform: translateY(-3px);
            box-shadow: 0 7px 15px rgba(0,0,0,0.4);
        }
        
        /* Star background */
        .star-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            z-index: -1;
        }
        
        .star {
            position: absolute;
            background-color: #ffffff;
            border-radius: 50%;
            animation: twinkle 2s infinite;
        }
        
        @keyframes twinkle {
            0% { opacity: 0; transform: scale(1); }
            50% { opacity: 0.8; transform: scale(1.5); }
            100% { opacity: 0; transform: scale(1); }
        }
        
        /* Sentiment emoji styles */
        .big-emoji {
            font-size: 5em;
            text-align: center;
            margin: 20px auto;
            display: block;
            animation: bounce 2s infinite alternate;
        }
        
        .sentiment-emoji {
            font-size: 2em;
            margin-right: 15px;
            animation: pulse 2s infinite;
        }
        
        /* Catchphrase styles */
        .catchphrase {
            font-size: 1.5em;
            font-weight: bold;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            margin: 20px 0;
            animation: fadeInUp 1s ease, pulse 3s infinite alternate;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            background: linear-gradient(45deg, #0b0c2a, #1a202c);
            color: white;
        }
        
        /* Progress bar for sentiment display */
        .progress-bar {
            width: 100%;
            height: 10px;
            background-color: #2d3748;
            border-radius: 5px;
            margin: 5px 0;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            border-radius: 5px;
        }
        
        .positive-gradient {
            background: linear-gradient(45deg, #4CAF50, #8BC34A);
        }
        
        .negative-gradient {
            background: linear-gradient(45deg, #F44336, #FF9800);
        }
        
        .neutral-gradient {
            background: linear-gradient(45deg, #2196F3, #03A9F4);
        }
        
        /* Sentiment cards */
        .sentiment-card {
            display: flex;
            align-items: center;
            background: rgba(26, 32, 44, 0.8);
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 15px;
            border: 1px solid #4a5568;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }
        
        .sentiment-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        }
        
        /* Comment box styles */
        .comment-box {
            display: flex;
            align-items: flex-start;
            background: linear-gradient(45deg, #1a202c, #2d3748);
            color: #f0f8ff;
            padding: 18px;
            border-radius: 12px;
            margin-bottom: 20px;
            border: 1px solid #4a5568;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            transition: all 0.3s ease;
        }
        
        .comment-box:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0,0,0,0.4);
        }
        
        .comment-content {
            flex: 1;
        }
        
        /* Loading animation */
        .loading-spinner {
            text-align: center;
            margin: 20px auto;
        }
        
        .spinner-dot {
            display: inline-block;
            width: 15px;
            height: 15px;
            border-radius: 50%;
            margin: 0 5px;
            animation: bounce 1.4s ease-in-out infinite;
        }
        
        .spinner-dot:nth-child(1) {
            background-color: #4CAF50;
            animation-delay: 0s;
        }
        
        .spinner-dot:nth-child(2) {
            background-color: #2196F3;
            animation-delay: 0.2s;
        }
        
        .spinner-dot:nth-child(3) {
            background-color: #F44336;
            animation-delay: 0.4s;
        }
        
        /* Animation keyframes */
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
        
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Main content layout */
        .main-content {
            padding: 20px;
            animation: fadeIn 0.5s ease;
        }
        
        .action-card {
            background: linear-gradient(135deg, #1a202c, #2d3748);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            margin: 30px 0;
            box-shadow: 0 10px 20px rgba(0,0,0,0.3);
            border: 1px solid #4a5568;
            animation: fadeIn 1s ease;
        }
        
        .action-card-emoji {
            font-size: 4em;
            margin: 20px 0;
            animation: bounce 2s infinite alternate;
        }
        
        .action-card-title {
            font-size: 1.8em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #f0f8ff;
        }
        
        .action-card-description {
            font-size: 1.2em;
            margin-bottom: 20px;
            color: #cbd5e0;
        }
    </style>
    """

# --- Main App ---
def main():
    # Apply CSS styling
    st.markdown(load_css(), unsafe_allow_html=True)
    
    # Add starry background
    st.markdown(
        """
        <div class="star-container">
            <div class="star" style="top: 10%; left: 20%; width: 4px; height: 4px; animation-delay: 0.5s;"></div>
            <div class="star" style="top: 30%; left: 50%; width: 6px; height: 6px; animation-delay: 1.2s;"></div>
            <div class="star" style="top: 60%; left: 10%; width: 4px; height: 4px; animation-delay: 0.8s;"></div>
            <div class="star" style="top: 80%; left: 70%; width: 8px; height: 8px; animation-delay: 0.2s;"></div>
            <div class="star" style="top: 5%; left: 90%; width: 2px; height: 2px; animation-delay: 1.5s;"></div>
            <div class="star" style="top: 25%; left: 35%; width: 4px; height: 4px; animation-delay: 0.1s;"></div>
            <div class="star" style="top: 45%; left: 65%; width: 6px; height: 6px; animation-delay: 0.9s;"></div>
            <div class="star" style="top: 75%; left: 25%; width: 2px; height: 2px; animation-delay: 0.6s;"></div>
            <div class="star" style="top: 95%; left: 45%; width: 4px; height: 4px; animation-delay: 1.3s;"></div>
            <div class="star" style="top: 15%; left: 85%; width: 6px; height: 6px; animation-delay: 0.4s;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Initialize session state variables
    if "youtube_url" not in st.session_state:
        st.session_state.youtube_url = ""
    
    if "num_comments" not in st.session_state:
        st.session_state.num_comments = 50
        
    if "show_url_popup" not in st.session_state:
        st.session_state.show_url_popup = False
        
    if "perform_analysis" not in st.session_state:
        st.session_state.perform_analysis = False
    
    # Check login
    if not login():
        return
    
    # Display logged in interface
    st.sidebar.success("Logged in as admin")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()
    
    # Display main interface
    st.markdown("<h1 style='text-align: center; color: #f0f8ff;'>🎥 VIBE SCOPE 😎</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2em;'>Analyze YouTube video comments and visualize their sentiment with style!</p>", unsafe_allow_html=True)
    
    # URL input card
    if not st.session_state.show_url_popup and not st.session_state.perform_analysis:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                """
                <div class="action-card">
                    <div class="action-card-title">Ready to analyze some YouTube vibes?</div>
                    <div class="action-card-emoji">🎬 🎭 🎥</div>
                    <div class="action-card-description">
                        Enter a YouTube URL and discover what people really think about the video!
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            if st.button("Enter YouTube URL", use_container_width=True):
                st.session_state.show_url_popup = True
                st.rerun()
    
    # Show URL popup
    if st.session_state.show_url_popup:
        youtube_url_popup()
    
    # Perform analysis if requested
    if st.session_state.perform_analysis:
        youtube_url = st.session_state.youtube_url
        num_comments = st.session_state.num_comments
        
        # Show loading spinner
        st.markdown(
            """
            <div class="loading-spinner">
                <div>Analyzing YouTube comments...</div>
                <div class="spinner-dot"></div>
                <div class="spinner-dot"></div>
                <div class="spinner-dot"></div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Run analysis
        summary, sentiment_counts, (fig_bar, fig_pie), individual_sentiments = analyze_youtube_sentiment(youtube_url, num_comments)
        
        if summary:
            # Get the dominant sentiment for the catchphrase
            overall_sentiment = sentiment_counts.most_common(1)[0][0] if sentiment_counts else "neutral"
            catchphrase = get_catchphrase(overall_sentiment)
            emoji = sentiment_emojis.get(overall_sentiment, "😐")
            
            # Display results header
            st.markdown(f"<div class='big-emoji'>{emoji}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='catchphrase {overall_sentiment}-gradient'>{catchphrase}</div>", unsafe_allow_html=True)
            
            # Display summary
            st.subheader("📊 Sentiment Summary")
            st.markdown(summary)
            
            # Display sentiment breakdown
            st.subheader("📌 Sentiment Breakdown")
            
            # Create cards for each sentiment
            for sentiment, count in sentiment_counts.items():
                emoji = sentiment_emojis.get(sentiment, "😐")
                total = sum(sentiment_counts.values())
                percentage = int((count / total) * 100)
                
                st.markdown(
                    f"""
                    <div class="sentiment-card">
                        <div class="sentiment-emoji">{emoji}</div>
                        <div style="flex: 1; padding-left: 15px;">
                            <div style="font-weight: bold; font-size: 1.2em;">{sentiment.title()}</div>
                            <div class="progress-bar">
                                <div class="progress-fill {sentiment}-gradient" style="width: {percentage}%;"></div>
                            </div>
                            <div>{count} comments ({percentage}%)</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Display charts
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(fig_bar)
            with col2:
                st.pyplot(fig_pie)
            
            # Display individual comments
            st.subheader("💬 Individual Comments")
            
            for text, sentiment in individual_sentiments:
                emoji = sentiment_emojis.get(sentiment, "😐")
                st.markdown(
                    f"""
                    <div class="comment-box">
                        <div class="sentiment-emoji">{emoji}</div>
                        <div class="comment-content">
                            <strong>{sentiment.title()}</strong>
                            <p>{text}</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Add button to analyze another video
            if st.button("Analyze Another Video", use_container_width=True):
                st.session_state.perform_analysis = False
                st.session_state.youtube_url = ""
                st.rerun()
        else:
            st.error("No comments found or something went wrong.")
            if st.button("Try Again", use_container_width=True):
                st.session_state.perform_analysis = False
                st.rerun()
        
        # Reset analysis flag
        st.session_state.perform_analysis = False

if __name__ == "__main__":
    main()
