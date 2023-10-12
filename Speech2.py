import speech_recognition as sr
from audio_recorder_streamlit import audio_recorder
from io import BytesIO
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import wordnet
from nltk import pos_tag, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Load dataset
df = pd.read_csv("API_EGY_DS2_en_csv_v2_5883235.csv", skiprows=4)


# Normalization
def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9]', " ", text)
    tokens = word_tokenize(text)
    lemmatizer = wordnet.WordNetLemmatizer()

    tagged_tokens = pos_tag(tokens, tagset=None)
    token_lemmas = []

    for (token, pos_token) in tagged_tokens:
        if pos_token.startswith("V"):  # verb
            pos_val = "v"
        elif pos_token.startswith("J"):  # adjective
            pos_val = "a"
        elif pos_token.startswith("R"):  # adverb
            pos_val = "r"
        else:
            pos_val = 'n'  # noun
        token_lemmas.append(lemmatizer.lemmatize(token, pos_val))

    return " ".join(token_lemmas)


df["normalized_text"] = df["Indicator Name"].apply(normalize_text)


# Remove stopwords
def remove_stopwords(text):
    stop = stopwords.words("english")
    text = [word for word in text.split() if word not in stop]
    return " ".join(text)


# Bag of Words
bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(df["normalized_text"]).toarray()
bow_features = bow_vectorizer.get_feature_names_out()
bow_df = pd.DataFrame(bow_matrix, columns=bow_features)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df["normalized_text"]).toarray()
tfidf_df = pd.DataFrame(tfidf_matrix, columns=tfidf_vectorizer.get_feature_names_out())


# Cosine Similarity using BOW
def chat_bow(user_input):
    normalized_input = normalize_text(user_input)
    without_stopwords = remove_stopwords(normalized_input)
    bow_input = bow_vectorizer.transform([without_stopwords]).toarray()
    cosine_similarity = cosine_similarity(bow_df, bow_input)
    index_of_most_similar = cosine_similarity.argmax()
    return df["Indicator Name"].loc[index_of_most_similar]


# Cosine Similarity using TF-IDF
def chat_tfidf(user_input):
    normalized_input = normalize_text(user_input)
    tfidf_input = tfidf_vectorizer.transform([normalized_input]).toarray()
    cosine_similarity = cosine_similarity(tfidf_df, tfidf_input)
    index_of_most_similar = cosine_similarity.argmax()
    return df["Indicator Name"].loc[index_of_most_similar]


# Speech recognition function
def transcribe_audio(audio_data):
    r = sr.Recognizer()

    try:
        audio_io = BytesIO(audio_data)

        with sr.AudioFile(audio_io) as source:
            audio = r.record(source)
            return r.recognize_google(audio)

    except sr.UnknownValueError as e:
        print(f"Error: Sorry, the model did not understand. {str(e)}")
    except sr.RequestError as e:
        print(f"Error connecting to the speech recognition service: {str(e)}")


# Main function
def main():
    print("Welcome to the Speech-Enabled Chatbot!")
    print("Type your message or use the microphone to speak.")

    # Initialize user_input
    user_input = ""

    while True:
        text_input = input("You: ")

        if text_input.lower() == "save text":
            user_input = text_input

        if text_input.lower() == "save audio":
            audio_data = audio_recorder()
            if audio_data:
                transcribed_text = transcribe_audio(audio_data)
                user_input = transcribed_text

        print("Input Text:", user_input)

        if text_input.lower() == "submit":
            bow_response = chat_bow(user_input)
            tfidf_response = chat_tfidf(user_input)

            print("BOW Response:", bow_response)
            print("TF-IDF Response:", tfidf_response)
            break


if __name__ == "__main__":
    main()