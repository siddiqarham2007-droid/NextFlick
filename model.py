from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


cv = pickle.load(open("cv.pkl", "rb"))

# vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(All_movies['tags'])

# similarity matrix
similarity = cosine_similarity(vectors)
def ml_fallback(All_movies, cv, vectors,
                genre="Any Genre",
                actor="Any Actor",
                director="Any Director",
                top_n=10):

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

    return [All_movies.iloc[i[0]].title for i in movies_list[:top_n]]



def hybrid_recommend(All_movies, similarity, cv, vectors,
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
            filtered['original_language'].str.lower() == language.lower()
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
            ref_index = All_movies[All_movies['title'].str.lower() == movie.lower()].index[0]
        else:
            ref_index = filtered.sort_values(
                by=['vote_average','popularity'],
                ascending=False
            ).index[0]

        distances = similarity[ref_index]

        sim_list = list(enumerate(distances))
        filtered_indices = filtered.index.tolist()

        sim_filtered = [i for i in sim_list if i[0] in filtered_indices]
        sim_filtered = sorted(sim_filtered, key=lambda x: x[1], reverse=True)

        final_indices = [i[0] for i in sim_filtered]
        final_df = All_movies.iloc[final_indices]

        final_df = final_df.sort_values(
            by=['vote_average','popularity'],
            ascending=False
        )

        return final_df['title'].head(top_n).tolist()

    # 🔥 CASE 2: No results → ML fallback
    else:
        return ml_fallback(All_movies, cv, vectors, genre, actor, director, top_n)



def recommend(genre, actor="Any Actor", director="Any Director", language=None):
    
    filtered = All_movies.copy()

    # genre filter
    filtered = filtered[
        filtered['genres'].apply(lambda x: genre in x)
    ]

    # actor filter
    if actor != "Any Actor":
        filtered = filtered[
            filtered['cast'].apply(lambda x: actor in x)
        ]

    # director filter
    if director != "Any Director":
        filtered = filtered[
            filtered['crew'].apply(lambda x: director in x)
        ]

    # language filter
    if language:
        temp = filtered[
            filtered['original_language'] == language
        ]

        if len(temp) > 0:
            filtered = temp

    # ranking
    filtered = filtered.sort_values(
        by=['popularity', 'vote_average'],
        ascending=False
    )

    return filtered['title'].head(10)
