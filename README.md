# 🎥 YouTube Sentiment Analyzer

The **YouTube Sentiment Analyzer** is a Streamlit web application that allows users to analyze the sentiment of comments from any public YouTube video. It uses sentiment analysis models to classify comments as **positive**, **negative**, or **neutral**, and visualizes the results with interactive charts.

## 🚀 Features

- 🔗 Accepts any valid YouTube video URL.
- 💬 Fetches and analyzes YouTube comments.
- 🎯 Classifies comments using sentiment analysis.
- 📊 Displays sentiment distribution via bar and pie charts.
- 🌌 Animated star background for an engaging UI.
- 🎛️ Interactive controls to choose the number of comments and sorting options.
- 🔒 Custom login screen for user access control.

## 🛠️ Tech Stack

- **Frontend/UI**: Streamlit
- **Backend**: Python
- **Libraries**:
  - `streamlit`
  - `google-api-python-client`
  - `textblob` or `transformers`
  - `matplotlib` or `plotly`
  - `pandas`, `numpy`

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/youtube-sentiment-analyzer.git
   cd youtube-sentiment-analyzer
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Set up Google API:

Go to Google Developer Console.

Enable the YouTube Data API v3.

Generate an API key.

Add it to your .env file or configure it securely in the code.

Run the app:

bash
Copy
Edit
streamlit run app.py
🔐 Login Credentials
You can set your own login credentials in the login.py or configuration section. For testing, default values might be:

Username: admin

Password: admin123

⚠️ Use environment variables or encrypted storage for production-level apps.

📁 Project Structure
bash
Copy
Edit
youtube-sentiment-analyzer/
│
├── app.py                # Main Streamlit application
├── login.py              # User login handling
├── sentiment.py          # Sentiment analysis logic
├── youtube_api.py        # YouTube comment fetching
├── utils.py              # Helper functions
├── styles.css            # Custom CSS styles (optional)
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (API keys)
└── README.md             # This file
📸 Screenshots
Home Page	Sentiment Results

✨ Future Improvements
OAuth-based Google login

Multi-language sentiment support

Comment filtering by keywords

Model customization (BERT, RoBERTa, etc.)

