import nltk
import string
import streamlit as st
import speech_recognition as sr
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")
nltk.download('wordnet')



# Load the text file and preprocess the data

with open("tom sawyer.txt", "r", encoding="utf-8") as f:
    data = f.read().replace("\n", " ")

# Tokenize the text into sentences

sentences = sent_tokenize(data)


# Define a function to preprocess each sentence

def preprocess(sentence):
    # Tokenize the sentence into words

    words = word_tokenize(sentence)

    # Remove stopwords and punctuation

    words = [word.lower() for word in words if
             word.lower() not in stopwords.words("english") and word not in string.punctuation]

    # Lemmatize the words

    lemmatizer = WordNetLemmatizer()

    words = [lemmatizer.lemmatize(word) for word in words]

    return words


# Preprocess each sentence in the text

corpus = [preprocess(sentence) for sentence in sentences]


# Define a function to find the most relevant sentence given a query

def get_most_relevant_sentence(query):
    # Preprocess the query

    query = preprocess(query)

    # Compute the similarity between the query and each sentence in the text

    max_similarity = 0

    most_relevant_sentence = ""

    for sentence in corpus:
        similarity = len(set(query).intersection(sentence)) / float(len(set(query).union(sentence)))

        if similarity > max_similarity:
            max_similarity = similarity
            most_relevant_sentence = " ".join(sentence)

    return most_relevant_sentence


def chatbot(question):
    # Find the most relevant sentence

    most_relevant_sentence = get_most_relevant_sentence(question)

    # Return the answer

    return most_relevant_sentence


def transcribe_speech_srec():
    # initialize recognizer class
    r = sr.Recognizer()

    # Reading with microphone as source

    with sr.Microphone() as source:
        st.info("Speak now...")

        # Listen for speech and store in audio_text variable
        audio_text = r.listen(source)

        st.info("Transcribing...")

        try:

            # using google speech recognition
            text = r.recognize_google(audio_text)

            return text

        except:

            return "Sorry I did not get that click the 'Start Record' button and try again"


def main():
    st.title("Chatbot")
    st.write("Hello! I'm a chatbot. Ask me anything about the topic in the text file.")
    format = st.selectbox("Choose the format of your question", ("AUDIO", "TEXT"))

    if format == "TEXT":
        # Get the users question

        question = st.text_input("You:")

        # Create a button to submit the question

        if st.button("Submit"):
            # Call the chatbot function with the question and display the response

            response = chatbot(question)

            st.write("Chatbot: " + response)

    elif format == "AUDIO":
        if st.button("Start Recording"):
            text = transcribe_speech_srec()
            st.write("You said: " + text)
            response = chatbot(text)
            st.info("Searching")
            st.write("Chatbot: " + response)


if __name__ == "__main__":
    main()
