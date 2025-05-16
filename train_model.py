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
        # Keep running even if download fails, though model training might fail later

try:
    nltk.data.find('omw-1.4')
except LookupError:
    try:
        nltk.download('omw-1.4')
    except Exception as e:
        print(f"Error downloading omw-1.4: {e}")
        # Keep running even if download fails

print("Loading dataset...")
# Load Sentiment140 dataset
try:
    # Assuming the CSV file is named 'social_sentiment.csv' based on previous interaction
    df = pd.read_csv('social_sentiments.csv', encoding='latin-1', header=None, names=['sentiment', 'id', 'date', 'query', 'user', 'text'])
except FileNotFoundError:
    print("Error: social_sentiment.csv not found. Please ensure the file is downloaded and placed in the same directory as this script.")
    exit()
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()


# Drop rows with any NaN values
print(f"Number of rows before dropping NaNs: {len(df)}")
df.dropna(inplace=True)
print(f"Number of rows after dropping NaNs: {len(df)}")

# --- Added diagnostic print statement ---
print(f"Unique values in 'sentiment' column before mapping: {df['sentiment'].unique()}")
# ----------------------------------------

# Map sentiment labels (handle potential errors if unexpected values exist)
label_map = {'0': 'negative', '2': 'neutral', '4': 'positive'}
# Convert sentiment column to string before mapping to avoid errors with mixed types
df['sentiment'] = df['sentiment'].astype(str).map(label_map)
df.dropna(subset=['sentiment'], inplace=True) # Ensure no NaN sentiment after mapping
print(f"Number of rows after mapping sentiment: {len(df)}")


# Lemmatization function
lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    if not isinstance(text, str): # Ensure the input is a string
        return ""
    # Remove non-alphabetic characters and spaces, convert to lower case
    text = re.sub(r"[^a-zA-Z\s]", "", text, re.I|re.A).lower()
    words = text.split()
    # Lemmatize words
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

print("Preprocessing text...")
# Apply lemmatization and handle potential non-string types in the 'text' column
df['text_processed'] = df['text'].apply(lemmatize_text)

# Remove rows where text_processed might have become empty after cleaning
df.replace('', pd.NA, inplace=True)
df.dropna(subset=['text_processed'], inplace=True)
print(f"Number of rows after text preprocessing and dropping empty texts: {len(df)}")


print("Vectorizing text...")
# TF-IDF Vectorization with n-grams
vectorizer = TfidfVectorizer(max_features=20000, stop_words='english', ngram_range=(1, 2))

try:
    X = vectorizer.fit_transform(df['text_processed'])
    y = df['sentiment']

    # Check if the vocabulary is empty after fitting
    if not vectorizer.vocabulary_:
        raise ValueError("Empty vocabulary after TF-IDF. This could be due to a very small dataset or aggressive preprocessing removing all meaningful words.")

except ValueError as e:
    print(f"Error during vectorization: {e}")
    print("\nSuggestions:")
    print("- Ensure your dataset is large enough and contains diverse language.")
    print("- Review your text preprocessing steps (lemmatization, removing characters) to ensure they are not too aggressive.")
    print("- Consider adjusting the 'stop_words' parameter in TfidfVectorizer or removing fewer characters.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during vectorization: {e}")
    exit()


# Split dataset
# Check if there are enough samples to split
if X.shape[0] < 2:
    print("Error: Not enough samples left after preprocessing to split the dataset.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check if train and test sets are not empty
if X_train.shape[0] == 0 or X_test.shape[0] == 0:
     print("Error: Train or test set is empty after splitting. This might be due to a very small dataset.")
     exit()


print("Training model...")
# Train Logistic Regression with GridSearchCV for hyperparameter tuning
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

# Check if there are at least 5 samples for 5-fold cross-validation
if X_train.shape[0] < 5:
    print(f"Warning: Not enough samples ({X_train.shape[0]}) for 5-fold cross-validation. Reducing folds.")
    cv_folds = X_train.shape[0] if X_train.shape[0] > 0 else 1
else:
    cv_folds = 5


try:
    grid_search = GridSearchCV(LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced'), param_grid, cv=cv_folds, scoring='accuracy', verbose=1, n_jobs=-1, error_score='raise')
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

except ValueError as e:
    print(f"Error during model training: {e}")
    print("\nSuggestions:")
    print("- Check if your target variable 'y' contains valid sentiment labels ('negative', 'neutral', 'positive').")
    print("- Ensure there are enough samples for cross-validation.")
except Exception as e:
    print(f"An unexpected error occurred during model training: {e}")
