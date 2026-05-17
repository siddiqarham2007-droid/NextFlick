from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from movie_recommendation_system.py import load_data

def ml_fallback(All_movies, vectors,cv,
                genre="Any Genre",
                actor="Any Actor",
                director="Any Director",
                top_n=10):

    cv = CountVectorizer(max_features=5000, stop_words='english')

    query = ""

    if genre != "Any Genre":
        query += genre + " "

    if actor != "Any Actor":
        query += actor.replace(" ", "") + " "

    if director != "Any Director":
        query += director.replace(" ", "") + " "

    query = query.lower().strip()

    # if empty query → return top movies
    if query == "":
        return All_movies.sort_values(
            by=['vote_average','popularity'],
            ascending=False
        )['title'].head(top_n).tolist()

    # vectorize query
    query_vec = cv.transform([query]).toarray()

    # similarity with all movies
    sim_scores = cosine_similarity(query_vec, vectors)[0]

    # sort
    movies_list = sorted(
        list(enumerate(sim_scores)),
        key=lambda x: x[1],
        reverse=Trues
    )

    return  All_movies.iloc[[i[0] for i in movies_list[:top_n]]]



def hybrid_recommend(All_movies, similarity, vectors,
                     genre="Any Genre",
                     actor="Any Actor",
                     language="Any Language",
                     director="Any Director",
                     movie=None,
                     top_n=10):

    filtered = All_movies.copy()

    # 🔹 FILTERS
    if genre != "Any Genre":
        g = genre.replace(" ", "").lower()
        filtered = filtered[
            filtered['genres'].apply(
                lambda x: g in [i.lower() for i in x]
            )
        ]

    if actor != "Any Actor":
        a = actor.replace(" ", "").lower()
        filtered = filtered[
            filtered['cast'].apply(
                lambda x: a in [i.lower() for i in x]
            )
        ]

    if language != "Any Language":
        filtered = filtered[
            filtered['original_language'].str.lower()
            == language.lower()
        ]

    if director != "Any Director":
        d = director.replace(" ", "").lower()
        filtered = filtered[
            filtered['crew'].apply(
                lambda x: d in [i.lower() for i in x]
            )
        ]

    # 🔥 CASE 1: Filtered results exist → use ML similarity
    if filtered.shape[0] > 0:

        # choose reference movie
        if movie and movie.lower() in All_movies['title'].str.lower().values:
            ref_index = All_movies[
                All_movies['title'].str.lower() == movie.lower()
            ].index[0]

        else:
            ref_index = filtered.index[0]

        distances = similarity[ref_index]

        sim_list = list(enumerate(distances))
        filtered_indices = filtered.index.tolist()

        sim_filtered = [i for i in sim_list if i[0] in filtered_indices]

        sim_filtered = sorted(
            sim_filtered,
            key=lambda x: x[1],
            reverse=True
        )

        final_indices = [i[0] for i in sim_filtered]

        final_df = All_movies.iloc[final_indices]

        # 🚨 ONLY CHANGE: removed rating-based sorting

        return final_df.head(top_n)

    # 🔥 CASE 2: No results → ML fallback
    else:
        return ml_fallback(All_movies, vectors, genre, actor, director, top_n)
