🎯 VibeScope: YouTube Comment Sentiment Analyzer

VibeScope is a Streamlit web application that analyzes sentiments from YouTube video comments using a machine learning model trained on social media sentiment data. It classifies comments into positive, negative, or neutral, and provides visual insights such as comment breakdowns and a sentiment distribution chart.

📌 FEATURES

🧠 Sentiment Classification: Classifies comments as Positive, Negative, or Neutral.

📊 Visual Insights: Sentiment pie chart and categorized comment display.

🔄 Real-Time Comment Analysis: Fetches comments from YouTube videos.

🎨 Interactive UI: Built with Streamlit for a smooth user experience.

📈 Detailed Analytics: View total comment count, sentiment percentages, and more.

📱 Responsive Design: Works well on both desktop and mobile.

🧰 TECH STACK

Frontend: Streamlit

Machine Learning: Custom ML model trained on social media sentiment data

Libraries: Pandas, Matplotlib, scikit-learn, Streamlit, NLTK

APIs: YouTube Data API v3 (for comment fetching)

Deployment Options: Streamlit Community Cloud, Heroku, AWS

🚀 GETTING STARTED

Clone the Repository
bash Copy Edit git clone https://github.com/KaviyasreeK/vibe-scope.git
cd vibe-scope 2. Install Dependencies

bash Copy Edit pip install -r requirements.txt

Train the Model Ensure social_sentiments.csv is in the root folder. Run:
bash Copy Edit python train_model.py This will:

Train a TF-IDF vectorizer and a Logistic Regression classifier

Save the model as sentiment_model.pkl and vectorizer.pkl

Run the App
bash Copy Edit streamlit run vibe_scope.py or python -m streamlit run vibe_scope.py Then go to: http://localhost:8501

(Optional) Set Up a Virtual Environment
bash Copy Edit python -m venv venv 8. Activate the Environment

On Windows: venv\Scripts\activate

On macOS/Linux: source venv/bin/activate

📁 FOLDER STRUCTURE

sql Copy Edit vibe-scope/ ├── vibe_scope.py → Main Streamlit application
├── train_model.py → Model training script
├── sentiment_model.pkl → Trained sentiment classifier
├── vectorizer.pkl → TF-IDF vectorizer
├── social_sentiments.csv → Dataset
├── requirements.txt → Dependencies list
├── README.md → Project documentation
📦 DEPENDENCIES

streamlit

pandas

scikit-learn

nltk

matplotlib

joblib

textblob

google-api-python-client

youtube_transcript_api

🔮 FUTURE ENHANCEMENTS

Real-time integration with YouTube Data API (fully automated)

Emotion-level tagging (happy, sad, angry, etc.)

User login and analysis history

Exportable/downloadable reports
