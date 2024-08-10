import csv
import string
import nltk
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')  # Download tokenizer
nltk.download('wordnet')  # Download lemmatizer (optional)
nltk.download('stopwords') # Download stopwords
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Optional: Import Lemmatizer from nltk.stem import WordNetLemmatizer

# Streamlit application title
st.image('https://infobeat.com/wp-content/uploads/2018/08/featured-image.png')
st.title("Plagiarism Checker")

# Function to preprocess text
def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text.lower())

    # Optional: Lemmatization (consider for improved text comparison)
    # lemmatizer = WordNetLemmatizer()
    # filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Consider keeping relevant stop words depending on content type
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]

    preprocessed_text = ' '.join(filtered_tokens)
    return preprocessed_text

# Load your text documents (modify as needed)
@st.cache_data
def load_documents(file_path):
    documents = []
    with open(file_path, 'r') as file:  # Replace with your file path
        for line in file:
            documents.append(preprocess_text(line.strip()))  # Preprocess each line
    return documents

# File path
file_path = '/home/muhammad/Downloads/AI Plag_Check/train_snli.txt'

# Load documents from specified file path
documents = load_documents(file_path)
st.success("Documents loaded and preprocessed successfully!")

# User input
user_text = st.text_area("Enter text for plagiarism check:")
if user_text:
    preprocessed_user_text = preprocess_text(user_text)

    # Choose a similarity metric (consider trying different options)
    vectorizer = TfidfVectorizer()  # One option for comparison
    # vectorizer = CountVectorizer()  # Another option

    tfidf_vectors = vectorizer.fit_transform(documents)
    user_tfidf_vector = vectorizer.transform([preprocessed_user_text])

    # Calculate similarity scores
    similarities = cosine_similarity(user_tfidf_vector, tfidf_vectors)

    # Report potential plagiarism (modify as needed)
    st.subheader("Similarity Results")
    max_similarity = max(similarities[0])
    
    if max_similarity > 0.7:  # Adjust threshold based on your needs
        st.write(f"The entered text is {max_similarity*100:.2f}% similar to the document, indicating potential plagiarism.")
    else:
        uniqueness_percentage = (1 - max_similarity) * 100
        st.write(f"The entered text is {uniqueness_percentage:.2f}% unique.")

