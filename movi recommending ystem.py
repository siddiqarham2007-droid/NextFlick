import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="NextFlick 🎬",
    page_icon="🍿",
    layout="wide"
)

# ---------------- DARK THEME CSS ----------------
st.markdown("""
<style>

.stApp {
    background-color: #0f1117;
    color: white;
}

h1, h2, h3, h4 {
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: #161a23;
}

.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.6rem 1rem;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #ff2e2e;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_resource
def load_data():

    All_movies = pickle.load(open("movies.pkl", "rb"))
    vectors = pickle.load(open("vectors2.pkl", "rb"))

    similarity = cosine_similarity(vectors)

    return All_movies, similarity

All_movies, similarity = load_data()

# ---------------- TITLE ----------------
st.title("🍿 NextFlick")
st.markdown("### What's on your mind today? 🎬")

st.write(
    "Choose your vibe and let me find the perfect movie for you ✨"
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🎭 Choose Your Preferences")

# GENRE
all_genres = sorted(
    list(set(
        genre
        for sublist in All_movies['genres']
        for genre in sublist
    ))
)
all_genres.insert(0,'Any Genre')

selected_genre = st.sidebar.selectbox(
    "🎬 Genre",
    all_genres
)

# LANGUAGE
language_map = {
    "en": "English",
    "hi": "Hindi",
    "fr": "French",
    "es": "Spanish",
    "zh": "Chinese",
    "cn": "Chinese",
    "de": "German",
    "ja": "Japanese",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "pt": "Portuguese",
    "da": "Danish",
    "sv": "Swedish",
    "nl": "Dutch",
    "fa": "Persian"
}
language_map.insert(0,'Any Language')

selected_language = st.sidebar.selectbox(
    "🌍 Language",
    list(language_map.keys())
)

# ACTOR
all_actors = sorted(
    list(set(
        actor
        for sublist in All_movies['cast']
        for actor in sublist
    ))
)

all_actors.insert(0, "Any Actor")

selected_actor = st.sidebar.selectbox(
    "⭐ Favorite Actor",
    all_actors
)

# ---------------- ADVANCED FEATURES ----------------
with st.sidebar.expander("⚡ Advanced Features"):

    all_directors = sorted(
        list(set(
            director
            for sublist in All_movies['crew']
            for director in sublist
        ))
    )

    all_directors.insert(0, "Any Director")

    selected_director = st.selectbox(
        "🎥 Director",
        all_directors
    )

# ---------------- RECOMMEND FUNCTION ----------------
def recommend_movies():

    filtered = movies.copy()

    # Genre filter
    filtered = filtered[
        filtered['genres'].apply(
            lambda x: selected_genre in x
        )
    ]

    # Actor filter
    if selected_actor != "Any Actor":

        filtered = filtered[
            filtered['cast'].apply(
                lambda x: selected_actor in x
            )
        ]

    # Director filter
    if selected_director != "Any Director":

        filtered = filtered[
            filtered['crew'].apply(
                lambda x: selected_director in x
            )
        ]

    # Language filter
    lang_code = language_map[selected_language]

    lang_filtered = filtered[
        filtered['original_language'] == lang_code
    ]

    # fallback
    if len(lang_filtered) > 0:
        filtered = lang_filtered

    # ranking
    filtered = filtered.sort_values(
        by=['popularity', 'vote_average'],
        ascending=False
    )

    return filtered.head(12)

# ---------------- BUTTON ----------------
if st.button("✨ Recommend Movies"):

    with st.spinner("Finding movies for your mood 🍿..."):

        recommendations = recommend_movies()

    if len(recommendations) == 0:

        st.warning(
            "Couldn't find exact match 😕 Try changing filters!"
        )

    else:

        st.success("🍿 Here are some movies for you!")

        cols = st.columns(4)

        for idx, (_, movie) in enumerate(recommendations.iterrows()):

            with cols[idx % 4]:

                poster_path = movie.get('poster_path')

                if pd.notna(poster_path):

                    poster_url = (
                        "https://image.tmdb.org/t/p/w500"
                        + poster_path
                    )

                    st.image(
                        poster_url,
                        use_container_width=True
                    )

                st.markdown(
                    f"### 🎬 {movie['title']}"
                )

                st.caption(
                    f"⭐ {movie['vote_average']}"
                )
