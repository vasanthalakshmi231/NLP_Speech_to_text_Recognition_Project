import streamlit as st
import joblib
import re
import nltk
import speech_recognition as sr

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# 2. CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

    section[data-testid="stSidebar"] {
        background-color: #f8f9fc;
        border-right: 1px solid #e1e4e8;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
        padding-left: 0.8rem;
        padding-right: 0.8rem;
    }

    .sidebar-title {
        font-size: 15px;
        font-weight: 600;
        color: #374151;
        margin-bottom: 8px;
    }

    .sidebar-divider {
        height: 1px;
        background-color: #dfe3e8;
        margin: 18px 0 14px 0;
    }

    div[data-testid="stRadio"] label {
        font-size: 13px;
        color: #374151;
    }

    div[data-testid="stExpander"] {
        border: 1px solid #d9dde5;
        border-radius: 8px;
        background-color: #f8f9fc;
        margin-bottom: 10px;
    }

    div[data-testid="stExpander"] summary {
        font-size: 13px;
        color: #374151;
    }

    .main-title {
        font-size: 32px;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 5px;
    }

    .main-subtitle {
        font-size: 15px;
        color: #6b7280;
        margin-bottom: 25px;
    }

    .result-card {
        padding: 22px;
        border-radius: 12px;
        border: 1px solid #e1e5eb;
        background-color: #f8f9fc;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .result-title {
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .positive {
        color: #15803d;
        font-size: 26px;
        font-weight: 700;
    }

    .negative {
        color: #dc2626;
        font-size: 26px;
        font-weight: 700;
    }

    .positive-message {
        color: #166534;
        font-size: 15px;
        margin-top: 8px;
        margin-bottom: 8px;
    }

    .negative-message {
        color: #991b1b;
        font-size: 15px;
        margin-top: 8px;
        margin-bottom: 8px;
    }

    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 12px;
        margin-top: 40px;
        padding: 15px;
    }

</style>
""", unsafe_allow_html=True)


# ==========================================================
# 3. LOAD NLTK RESOURCES
# ==========================================================

@st.cache_resource
def load_nltk_resources():

    nltk.download(
        "punkt",
        quiet=True
    )

    nltk.download(
        "punkt_tab",
        quiet=True
    )

    nltk.download(
        "stopwords",
        quiet=True
    )

    nltk.download(
        "wordnet",
        quiet=True
    )


load_nltk_resources()


# ==========================================================
# 4. LOAD MODEL AND VECTORIZER
# ==========================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "models/Logistic_model.pkl"
    )

    vectorizer = joblib.load(
        "models/vectorizer.pkl"
    )

    return model, vectorizer


model, vectorizer = load_model()


# ==========================================================
# 5. TEXT PREPROCESSING
# ==========================================================

stop_words = set(
    stopwords.words("english")
)


# Keep important negation words
negation_words = {
    "not",
    "no",
    "never",
    "nor"
}


stop_words = (
    stop_words - negation_words
)


lemmatizer = WordNetLemmatizer()


def clean(doc):

    # ------------------------------------------------------
    # Remove special characters
    # ------------------------------------------------------

    doc = re.sub(
        "[^a-zA-Z]",
        " ",
        doc
    )

    # ------------------------------------------------------
    # Lowercase
    # ------------------------------------------------------

    doc = doc.lower()

    # ------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------

    tokens = nltk.word_tokenize(
        doc
    )

    # ------------------------------------------------------
    # Stopword removal
    # ------------------------------------------------------

    filtered_tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    # ------------------------------------------------------
    # Lemmatization
    # ------------------------------------------------------

    lemmatized_tokens = [
        lemmatizer.lemmatize(token)
        for token in filtered_tokens
    ]

    return " ".join(
        lemmatized_tokens
    )


# ==========================================================
# 6. SENTIMENT PREDICTION FUNCTION
# ==========================================================

def predict_sentiment(text):

    processed_text = clean(
        text
    )

    text_vector = vectorizer.transform(
        [processed_text]
    )

    prediction = model.predict(
        text_vector
    )[0]

    probability = model.predict_proba(
        text_vector
    )

    confidence = (
        probability.max() * 100
    )

    return (
        prediction,
        confidence
    )


# ==========================================================
# 7. SESSION STATE
# ==========================================================

if "speech_text" not in st.session_state:

    st.session_state[
        "speech_text"
    ] = ""


if "prediction" not in st.session_state:

    st.session_state[
        "prediction"
    ] = None


if "confidence" not in st.session_state:

    st.session_state[
        "confidence"
    ] = None


# ==========================================================
# 8. SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">'
        'Navigation'
        '</div>',
        unsafe_allow_html=True
    )


    page = st.radio(

        "",

        [
            "Analyze Review",
            "Speech-to-Text",
            "About Project"
        ],

        index=0,

        label_visibility="collapsed"
    )


    st.markdown(
        '<div class="sidebar-divider"></div>',
        unsafe_allow_html=True
    )


    # ------------------------------------------------------
    # AI MODEL
    # ------------------------------------------------------

    with st.expander(
        "🤖 AI Model",
        expanded=False
    ):

        st.write(
            "**Algorithm:** Logistic Regression"
        )

        st.write(
            "**Feature Extraction:** TF-IDF"
        )

        st.write(
            "**NLP:** NLTK"
        )

        st.write(
            "**Task:** Binary Sentiment Classification"
        )


    # ------------------------------------------------------
    # SENTIMENT CLASSES
    # ------------------------------------------------------

    with st.expander(
        "😊 Sentiment Classes",
        expanded=False
    ):

        st.write(
            "🟢 Positive"
        )

        st.write(
            "🔴 Negative"
        )


    # ------------------------------------------------------
    # NLP PIPELINE
    # ------------------------------------------------------

    with st.expander(
        "🧠 NLP Pipeline",
        expanded=False
    ):

        st.write(
            "1. Text Cleaning"
        )

        st.write(
            "2. Lowercase Conversion"
        )

        st.write(
            "3. Tokenization"
        )

        st.write(
            "4. Stopword Removal"
        )

        st.write(
            "5. Lemmatization"
        )

        st.write(
            "6. TF-IDF Vectorization"
        )

        st.write(
            "7. Logistic Regression"
        )


    # ------------------------------------------------------
    # SYSTEM STATUS
    # ------------------------------------------------------

    with st.expander(
        "📊 System Status",
        expanded=False
    ):

        st.success(
            "Model Loaded"
        )

        st.success(
            "Vectorizer Loaded"
        )

        st.success(
            "NLP Pipeline Ready"
        )

        st.success(
            "Speech Recognition Ready"
        )


    # ------------------------------------------------------
    # FUTURE ROADMAP
    # ------------------------------------------------------

    with st.expander(
        "🚀 Future Roadmap",
        expanded=False
    ):

        st.write(
            "• BERT / Transformers"
        )

        st.write(
            "• Multi-class sentiment"
        )

        st.write(
            "• Aspect-based sentiment"
        )

        st.write(
            "• Cloud deployment"
        )

        st.write(
            "• Larger movie-review datasets"
        )


    # ------------------------------------------------------
    # DEVELOPER
    # ------------------------------------------------------

    with st.expander(
        "👨‍💻 Developer",
        expanded=False
    ):

        st.write(
            "IMDB Movie Review Sentiment Analysis"
        )

        st.write(
            "Machine Learning + NLP + Speech Recognition"
        )

        st.write(
            "Built with Python & Streamlit"
        )


# ==========================================================
# 9. MAIN HEADER
# ==========================================================

st.markdown(
    '<div class="main-title">'
    '🎬 IMDB Movie Review Sentiment Analysis'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'Analyze movie reviews using Natural Language Processing '
    'and Speech Recognition.'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================================
# 10. ANALYZE REVIEW
# ==========================================================

if page == "Analyze Review":

    st.subheader(
        "✍️ Analyze Movie Review"
    )

    st.write(
        "Enter an IMDB movie review below and "
        "the AI model will predict its sentiment."
    )


    typed_review = st.text_area(

        "Movie Review",

        height=180,

        placeholder=(
            "Example: "
            "This movie was absolutely fantastic. "
            "The acting and story were excellent!"
        )
    )


    st.write("")


    if st.button(
        "🔍 Predict Sentiment",
        use_container_width=True
    ):

        if typed_review.strip() == "":

            st.warning(
                "Please enter a movie review."
            )

        else:

            prediction, confidence = (
                predict_sentiment(
                    typed_review
                )
            )


            st.session_state[
                "prediction"
            ] = prediction

            st.session_state[
                "confidence"
            ] = confidence


    # ------------------------------------------------------
    # DISPLAY RESULT
    # ------------------------------------------------------

    if st.session_state[
        "prediction"
    ] is not None:

        prediction = (
            st.session_state[
                "prediction"
            ]
        )

        confidence = (
            st.session_state[
                "confidence"
            ]
        )


        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )


        st.markdown(
            '<div class="result-title">'
            'Prediction Result'
            '</div>',
            unsafe_allow_html=True
        )


        st.write(
            "**Input Review:**"
        )

        st.write(
            typed_review
        )


        if str(
            prediction
        ).lower() == "positive":

            st.markdown(
                '<div class="positive">'
                '😊👍 Positive Sentiment'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="positive-message">'
                '🎉 Great! You seem to have enjoyed this movie! '
                '🍿🎬 ⭐'
                '</div>',
                unsafe_allow_html=True
            )


        else:

            st.markdown(
                '<div class="negative">'
                '😞👎 Negative Sentiment'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="negative-message">'
                '😕 Looks like this movie was not your favorite! '
                '🎬🍿'
                '</div>',
                unsafe_allow_html=True
            )


        st.write(
            f"**Confidence:** "
            f"{confidence:.2f}%"
        )

        st.progress(
            int(confidence)
        )


        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


# ==========================================================
# 11. SPEECH-TO-TEXT
# ==========================================================

elif page == "Speech-to-Text":

    st.subheader(
        "🎤 Speech-to-Text"
    )

    st.write(
        "Speak your movie review and convert "
        "your voice into text."
    )


    # ------------------------------------------------------
    # START RECORDING
    # ------------------------------------------------------

    if st.button(
        "🎙️ Start Recording",
        use_container_width=True
    ):

        recognizer = sr.Recognizer()


        try:

            with sr.Microphone() as source:

                st.info(
                    "Listening... Please speak now."
                )


                recognizer.adjust_for_ambient_noise(
                    source,
                    duration=1
                )


                audio = recognizer.listen(
                    source,
                    timeout=10,
                    phrase_time_limit=30
                )


            # ------------------------------------------------
            # GOOGLE SPEECH RECOGNITION
            # ------------------------------------------------

            recognized_text = (
                recognizer.recognize_google(
                    audio
                )
            )


            st.session_state[
                "speech_text"
            ] = recognized_text


            # Reset old prediction
            st.session_state[
                "prediction"
            ] = None

            st.session_state[
                "confidence"
            ] = None


            st.success(
                "Speech recognized successfully!"
            )


        except sr.WaitTimeoutError:

            st.error(
                "No speech detected. "
                "Please try again."
            )


        except sr.UnknownValueError:

            st.error(
                "Sorry, I could not understand "
                "the speech."
            )


        except sr.RequestError:

            st.error(
                "Google Speech Recognition "
                "service is unavailable."
            )


        except Exception as e:

            st.error(
                f"Microphone error: {e}"
            )


    # ------------------------------------------------------
    # DISPLAY SPEECH
    # ------------------------------------------------------

    if st.session_state[
        "speech_text"
    ]:

        speech_review = (
            st.session_state[
                "speech_text"
            ]
        )


        st.subheader(
            "📝 Recognized Review"
        )


        st.text_area(

            "Speech Result",

            value=speech_review,

            height=150,

            disabled=True
        )


        # --------------------------------------------------
        # SENTIMENT BUTTON
        # --------------------------------------------------

        if st.button(
            "🔍 Predict Speech Sentiment",
            use_container_width=True
        ):

            prediction, confidence = (
                predict_sentiment(
                    speech_review
                )
            )


            st.session_state[
                "prediction"
            ] = prediction

            st.session_state[
                "confidence"
            ] = confidence


        # --------------------------------------------------
        # DISPLAY SPEECH SENTIMENT
        # --------------------------------------------------

        if st.session_state[
            "prediction"
        ] is not None:

            prediction = (
                st.session_state[
                    "prediction"
                ]
            )

            confidence = (
                st.session_state[
                    "confidence"
                ]
            )


            st.markdown(
                '<div class="result-card">',
                unsafe_allow_html=True
            )


            st.markdown(
                '<div class="result-title">'
                'Speech Sentiment'
                '</div>',
                unsafe_allow_html=True
            )


            if str(
                prediction
            ).lower() == "positive":

                st.markdown(
                    '<div class="positive">'
                    '😊👍 Positive Sentiment'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="positive-message">'
                    '🎉 Your voice review sounds positive! '
                    'You seem to have enjoyed the movie! 🍿🎬 ⭐'
                    '</div>',
                    unsafe_allow_html=True
                )


            else:

                st.markdown(
                    '<div class="negative">'
                    '😞👎 Negative Sentiment'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="negative-message">'
                    '😕 Your voice review sounds negative. '
                    'Looks like this movie was not your favorite! '
                    '🎬🍿'
                    '</div>',
                    unsafe_allow_html=True
                )


            st.write(
                f"**Confidence:** "
                f"{confidence:.2f}%"
            )


            st.progress(
                int(confidence)
            )


            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


# ==========================================================
# 12. ABOUT PROJECT
# ==========================================================

elif page == "About Project":

    st.subheader(
        "📘 About the Project"
    )


    st.markdown(
        """
        ### 🎬 IMDB Movie Review Sentiment Analysis

        This application combines:

        - Natural Language Processing
        - Machine Learning
        - Speech Recognition

        ---

        ### 1️⃣ Analyze Review

        Movie Review

        ↓

        Text Cleaning

        ↓

        Tokenization

        ↓

        Stopword Removal

        ↓

        Lemmatization

        ↓

        TF-IDF

        ↓

        Logistic Regression

        ↓

        🟢 Positive / 🔴 Negative

        ---

        ### 2️⃣ Speech-to-Text

        🎤 Microphone

        ↓

        SpeechRecognition

        ↓

        Google Speech Recognition

        ↓

        📝 Recognized Text

        ↓

        TF-IDF

        ↓

        Logistic Regression

        ↓

        🟢 Positive / 🔴 Negative

        ---

        ### 🧠 Technologies Used

        - Python
        - Streamlit
        - NLTK
        - Scikit-learn
        - TF-IDF
        - Logistic Regression
        - SpeechRecognition
        - Google Speech Recognition
        """
    )


# ==========================================================
# 13. FOOTER
# ==========================================================

st.markdown(
    '<div class="footer">'
    '🎬 IMDB Sentiment Analysis | '
    'NLP + Machine Learning + Speech Recognition'
    '</div>',
    unsafe_allow_html=True
)