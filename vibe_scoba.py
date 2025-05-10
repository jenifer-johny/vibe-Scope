import streamlit as st
import base64

st.set_page_config(page_title="YouTube Sentiment Analyzer", layout="wide")

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import re
from youtube_comment_downloader import YoutubeCommentDownloader
from collections import Counter
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer

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
    summary = f"Overall sentiment is **{overall_sentiment}**."

    # Bar Chart
    fig_bar, ax = plt.subplots()
    ax.bar(sentiment_counts.keys(), sentiment_counts.values(), color='#a8dadc') # Light blue for bars
    ax.set_title("Sentiment Distribution (Bar Chart)", color='white')
    ax.set_xlabel("Sentiment", color='white')
    ax.set_ylabel("Count", color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.grid(axis='y', linestyle='--', alpha=0.7, color='gray')
    fig_bar.patch.set_facecolor('#0b0c2a')  # Darker background
    plt.tight_layout()

    # Pie Chart
    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct='%1.1f%%', startangle=140, textprops={'color': 'white'})
    ax_pie.set_title("Sentiment Distribution (Pie Chart)", color='white')
    fig_pie.patch.set_facecolor('#0b0c2a')  # Darker background
    plt.tight_layout()

    return summary, sentiment_counts, (fig_bar, fig_pie), individual_sentiments

# --- Authentication ---
def login():
    if "logged_in" not in st.session_state:
        st.sidebar.markdown(
            """
            <style>
                .sidebar {
                    background-color: #0b0c2a !important;
                    color: #f0f8ff;
                }
                .sidebar .st-expander {
                    background-color: #0b0c2a;
                    color: #f0f8ff;
                    border-radius: 5px;
                    margin-bottom: 10px;
                    border: 1px solid #1e3a8a;
                }
                .sidebar .st-expander-content {
                    color: #f0f8ff;
                }
                .sidebar label {
                    color: #f0f8ff;
                }
                .sidebar input[type="text"],
                .sidebar input[type="password"] {
                    background-color: #1e3a8a;
                    color: #f0f8ff;
                    border: 1px solid #4a5568;
                    border-radius: 3px;
                }
                .sidebar button {
                    background-color: #4a5568;
                    color: #f0f8ff;
                    border: none;
                    border-radius: 5px;
                    padding: 10px 20px;
                    cursor: pointer;
                    transition: background-color 0.3s ease;
                }
                .sidebar button:hover {
                    background-color: #1e3a8a;
                }
                .sidebar .st-error {
                    color: #e53e3e;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.sidebar.title("🔐 Login")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            if username == "admin" and password == "pass123":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.sidebar.error("Invalid credentials")
        return False
    return True

# --- Main App ---
def main():
    # Add background image
    def set_background(image_file):
        with open(image_file, "rb") as f:
            img_data = f.read()
        b64_encoded = base64.b64encode(img_data).decode()
        st.markdown(
            f"""
            <style>
            .main {{
                background-image: url("data:image/png;base64,{b64_encoded}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                min-height: 100vh;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    # set_background("image_2895e6.png") # Replace with your image file path
    set_background("img_emj.png")  # Replace with your image path

    st.markdown(
        """
        <style>
            body {
                color: #f0f8ff;
            }
            .st-header, .st-subheader, .st-markdown, .st-label, .st-button > button, .st-slider > div > div > div > div[data-baseweb="slider"] {
                color: #f0f8ff;
            }
            .st-title{
                color: #f0f8ff;
                text-align: center;
                font-size: 2.5em;
                letter-spacing: 0.1em;
                text-shadow: 2px 2px 8px rgba(0,0,0,0.7);
            }
            .st-button > button {
                background-color: #4a5568;
                border-color: #718096;
                transition: background-color 0.3s ease, transform 0.2s ease;
            }
            .st-button > button:hover {
                background-color: #1e3a8a;
                border-color: #2d3748;
                transform: scale(1.05);
            }
            .st-slider > div > div > div > div[data-baseweb="slider"] {
                background-color: #2d3748;
            }
            .st-slider > div > div > div > div[data-baseweb="slider-thumb"] {
                background-color: #f0f8ff;
                border-color: #cbd5e0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.3);
            }
            .css-1c0ab9f {
                background-color: #2d3748;
                color: #f0f8ff;
                border-color: #4a5568;
                transition: border-color 0.3s ease;
            }
            .css-1c0ab9f:focus {
                border-color: #5bc0de;
                box-shadow: 0 0 5px rgba(91, 192, 222, 0.5);
            }
            .comment-box {
                background-color: #1a202c !important;
                color: #f0f8ff !important;
                padding: 15px;
                border-radius: 12px;
                margin-bottom: 15px;
                border: 1px solid #4a5568;
                box-shadow: 0 2px 5px rgba(0,0,0,0.3);
                transition: transform 0.2s ease;
            }
            .comment-box:hover {
                transform: translateY(-2px);
            }
            .sidebar .sidebar-content {
                background-color: #0b0c2a;
            }
            /* Star Animation */
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
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Add stars to the background
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

    if not login():
        st.warning("**Welcome to VIBE SCOPE😂**. Please log in to continue.")
        return

    st.sidebar.success("Logged in as admin")
    if st.sidebar.button("Logout"):
        del st.session_state["logged_in"]
        st.rerun()

    st.title("🎥 Welcome to VIBE SCOPE😂")
    st.markdown("Analyze YouTube video comments and visualize their sentiment!")

    youtube_url = st.text_input("🔗 Enter the YouTube video URL:")
    num_comments = st.slider("🗨️ Number of comments to analyze:", min_value=10, max_value=200, value=50, step=10)

    if st.button("🔍 Analyze"):
        if not youtube_url:
            st.error("Please enter a YouTube URL.")
            return

        if model is None or vectorizer is None:
            st.error("Model not loaded. Please check the logs.")
            return

        with st.spinner("🔄 Analyzing comments..."):
            summary, sentiment_counts, (fig_bar, fig_pie), individual_sentiments = analyze_youtube_sentiment(youtube_url, num_comments)

        if summary:
            st.subheader("📊 Analysis Summary")
            st.markdown(summary)

        if sentiment_counts:
            st.subheader("📌 Sentiment Breakdown")
            for sentiment, count in sentiment_counts.items():
                st.write(f"**{sentiment.title()}**: {count}")

            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(fig_bar)
            with col2:
                st.pyplot(fig_pie)

            st.subheader("💬 Individual Comments and Sentiments")
            for text, sentiment in individual_sentiments:
                st.markdown(f"""
                <div class="comment-box">
                    <strong>Sentiment:</strong> {sentiment.title()}<br>
                    <strong>Comment:</strong> {text}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("No comments found or something went wrong.")

if __name__ == "__main__":
    main()

