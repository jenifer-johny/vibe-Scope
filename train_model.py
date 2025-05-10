import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
import re

# Download necessary NLTK data (run this once)
try:
    nltk.data.find('wordnet')
except LookupError:
    try:
        nltk.download('wordnet')
    except Exception as e:
        print(f"Error downloading wordnet: {e}")
        exit()

try:
    nltk.data.find('omw-1.4')
except LookupError:
    try:
        nltk.download('omw-1.4')
    except Exception as e:
        print(f"Error downloading omw-1.4: {e}")
        exit()

print("Loading dataset...")
# Load Sentiment140 dataset
try:
    df = pd.read_csv('social_sentiments.csv', encoding='latin-1', header=None, names=['sentiment', 'id', 'date', 'query', 'user', 'text'])
except FileNotFoundError:
    print("Error: small_sentiment_data.csv not found. Please ensure the file is downloaded and placed in the same directory as this script.")
    exit()

# Drop rows with any NaN values
print(f"Number of rows before dropping NaNs: {len(df)}")
df.dropna(inplace=True)
print(f"Number of rows after dropping NaNs: {len(df)}")

# Map sentiment labels (handle potential errors if unexpected values exist)
label_map = {'0': 'negative', '2': 'neutral', '4': 'positive'}
df['sentiment'] = df['sentiment'].astype(str).map(label_map)
df.dropna(subset=['sentiment'], inplace=True) # Ensure no NaN sentiment after mapping

# Lemmatization function
lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text, re.I|re.A) # Keep only alphabetic characters and spaces
    words = text.lower().split()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

print("Preprocessing text...")
df['text_processed'] = df['text'].apply(lemmatize_text)

print("Vectorizing text...")
# TF-IDF Vectorization with n-grams
vectorizer = TfidfVectorizer(max_features=20000, stop_words='english', ngram_range=(1, 2)) # Increased max_features and added n-grams
X = vectorizer.fit_transform(df['text_processed'])
y = df['sentiment']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training model...")
# Train Logistic Regression with GridSearchCV for hyperparameter tuning
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced'), param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1, error_score='raise') # Added error_score='raise'
grid_search.fit(X_train, y_train)

# Get the best model
model = grid_search.best_estimator_
print("Best hyperparameters:", grid_search.best_params_)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("Model and vectorizer saved successfully!")