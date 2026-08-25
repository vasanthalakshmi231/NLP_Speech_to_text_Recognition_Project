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

    /* ================================
       SIDEBAR
       ================================ */

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

    /* Radio buttons */

    div[data-testid="stRadio"] label {
        font-size: 13px;
        color: #374151;
    }

    /* Expanders */

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


    /* ================================
       MAIN CONTENT
       ================================ */

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


    /* ================================
       RESULT CARDS
       ================================ */

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


    /* ================================
       RESULT MESSAGE
       ================================ */

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


    /* ================================
       INFO CARDS
       ================================ */

    .info-card {
        padding: 18px;
        border-radius: 10px;
        border: 1px solid #e1e5eb;
        background-color: #ffffff;
        margin-bottom: 12px;
    }

    .info-card-title {
        font-size: 16px;
        font-weight: 600;
        color: #374151;
    }

    .info-card-text {
        font-size: 14px;
        color: #6b7280;
    }


    /* ================================
       FOOTER
       ================================ */

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

stop_words = stop_words - negation_words

lemmatizer = WordNetLemmatizer()


def clean(doc):

    # Remove special characters and numbers
    doc = re.sub(
        "[^a-zA-Z]",
        " ",
        doc
    )

    # Lowercase
    doc = doc.lower()

    # Tokenization
    tokens = nltk.word_tokenize(doc)

    # Stopword removal
    filtered_tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    # Lemmatization
    lemmatized_tokens = [
        lemmatizer.lemmatize(token)
        for token in filtered_tokens
    ]

    # Join
    return " ".join(
        lemmatized_tokens
    )


# ==========================================================
# 6. SESSION STATE
# ==========================================================

if "speech_text" not in st.session_state:

    st.session_state["speech_text"] = ""


if "prediction" not in st.session_state:

    st.session_state["prediction"] = None


if "confidence" not in st.session_state:

    st.session_state["confidence"] = None


# ==========================================================
# 7. SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">Navigation</div>',
        unsafe_allow_html=True
    )


    # ------------------------------------------------------
    # Navigation
    # ------------------------------------------------------

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
    # AI Model
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
    # Sentiment Classes
    # ------------------------------------------------------

    with st.expander(
        "😊 Sentiment Classes",
        expanded=False
    ):

        st.write("🟢 Positive")

        st.write("🔴 Negative")


    # ------------------------------------------------------
    # NLP Pipeline
    # ------------------------------------------------------

    with st.expander(
        "🧠 NLP Pipeline",
        expanded=False
    ):

        st.write("1. Text Cleaning")

        st.write("2. Lowercase Conversion")

        st.write("3. Tokenization")

        st.write("4. Stopword Removal")

        st.write("5. Lemmatization")

        st.write("6. TF-IDF Vectorization")

        st.write("7. Logistic Regression")


    # ------------------------------------------------------
    # System Status
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


    # ------------------------------------------------------
    # Future Roadmap
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
    # Developer
    # ------------------------------------------------------

    with st.expander(
        "👨‍💻 Developer",
        expanded=False
    ):

        st.write(
            "IMDB Movie Review Sentiment Analysis"
        )

        st.write(
            "Machine Learning + NLP"
        )

        st.write(
            "Built with Python & Streamlit"
        )


# ==========================================================
# 8. MAIN HEADER
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
    'and Machine Learning.'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================================
# 9. ANALYZE REVIEW PAGE
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

            # Clean text
            processed_text = clean(
                typed_review
            )

            # TF-IDF
            text_vector = vectorizer.transform(
                [processed_text]
            )

            # Prediction
            prediction = model.predict(
                text_vector
            )[0]

            # Probability
            probability = model.predict_proba(
                text_vector
            )

            confidence = (
                probability.max() * 100
            )


            # Store result
            st.session_state[
                "prediction"
            ] = prediction

            st.session_state[
                "confidence"
            ] = confidence


    # ------------------------------------------------------
    # Display Prediction
    # ------------------------------------------------------

    if st.session_state["prediction"] is not None:

        prediction = st.session_state[
            "prediction"
        ]

        confidence = st.session_state[
            "confidence"
        ]


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


        # ==================================================
        # POSITIVE RESULT
        # ==================================================

        if str(prediction).lower() == "positive":

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


        # ==================================================
        # NEGATIVE RESULT
        # ==================================================

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
            f"**Confidence:** {confidence:.2f}%"
        )

        st.progress(
            int(confidence)
        )


        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


# ==========================================================
# 10. SPEECH-TO-TEXT PAGE
# ==========================================================

elif page == "Speech-to-Text":

    st.subheader(
        "🎤 Speech-to-Text"
    )

    st.write(
        "Speak your movie review and convert "
        "your voice into text."
    )


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


            recognized_text = (
                recognizer.recognize_google(
                    audio
                )
            )


            st.session_state[
                "speech_text"
            ] = recognized_text


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
                "Speech recognition service "
                "is unavailable."
            )


    # ------------------------------------------------------
    # Display recognized speech
    # ------------------------------------------------------

    if st.session_state["speech_text"]:

        speech_review = st.session_state[
            "speech_text"
        ]


        st.subheader(
            "📝 Recognized Review"
        )

        st.text_area(
            "Speech Result",
            value=speech_review,
            height=150,
            disabled=True
        )


        if st.button(
            "🔍 Predict Speech Sentiment",
            use_container_width=True
        ):

            processed_text = clean(
                speech_review
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


            # ==================================================
            # POSITIVE SPEECH RESULT
            # ==================================================

            if str(prediction).lower() == "positive":

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


            # ==================================================
            # NEGATIVE SPEECH RESULT
            # ==================================================

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
                f"**Confidence:** {confidence:.2f}%"
            )


            st.progress(
                int(confidence)
            )


            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


# ==========================================================
# 11. ABOUT PROJECT PAGE
# ==========================================================

elif page == "About Project":

    st.subheader(
        "📘 About the Project"
    )


    st.markdown(
        """
        ### 🎬 IMDB Movie Review Sentiment Analysis

        This application uses **Natural Language Processing
        (NLP)** and **Machine Learning** to classify movie
        reviews into:

        - 🟢 Positive
        - 🔴 Negative

        ### 🔄 Processing Pipeline

        **Movie Review**

        ↓

        **Text Cleaning**

        ↓

        **Tokenization**

        ↓

        **Stopword Removal**

        ↓

        **Lemmatization**

        ↓

        **TF-IDF Vectorization**

        ↓

        **Logistic Regression**

        ↓

        **Sentiment Prediction**

        ### 🧠 Technologies Used

        - Python
        - NLTK
        - Scikit-learn
        - TF-IDF
        - Logistic Regression
        - SpeechRecognition
        - Streamlit
        """
    )


# ==========================================================
# 12. FOOTER
# ==========================================================

st.markdown(
    '<div class="footer">'
    '🎬 IMDB Sentiment Analysis | '
    'NLP + Machine Learning + Streamlit'
    '</div>',
    unsafe_allow_html=True
)