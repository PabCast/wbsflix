import streamlit as st
import pandas as pd

# Loading DataFrames
url_movies = 'https://drive.google.com/uc?id=1WB7QQGjulU_ODpfIAkey5ZO5dYDStkc_'
movies_df = pd.read_csv(url_movies)

url_ratings = 'https://drive.google.com/uc?id=1sQl_yG4sv_AKIcO2Z1d_nw2XTQ7FWaTt'
ratings_df = pd.read_csv(url_ratings)

movies_ratings = movies_df.merge(ratings_df)

def pop_movies(df_rate, df_movies, rate_tresh, range_of_days: float):
    df_merge = (df_rate
                .merge(df_movies, on='movieId')
                .assign(time_count = round(abs(df_rate['timestamp'] - max(df_rate['timestamp'])) / 60 / 60 / 24))
                .sort_values('time_count')
                .filter(['userId', 'rating', 'title', 'time_count'])
                .query('time_count<=@range_of_days')
                )
    df = df_merge.groupby('title').agg(rate_count=('rating', 'count'), rate_mean=('rating', 'mean')).query(
        'rate_count >= @rate_tresh').sort_values('rate_mean', ascending=False)
    return df.index.to_list()

def get_fav_movie(df, userId):
    user_top_movies = (movies_ratings[movies_ratings.userId == userId]
         .sort_values('rating', ascending=False)
         .head(5)
         .reset_index()
    )
    return user_top_movies

def get_recommended_movies(df, top_movie, userId):
    list_of_movies = []
    
    # Drop columns with zero variance or only one unique value
    df = df.dropna(axis='columns', thresh=2)
    df = df.loc[:, (df != df.iloc[0]).any()]

    corr = df.corrwith(df[top_movie])
    lists = corr.sort_values(ascending=False).iloc[:50].index.to_list()
    for movie in lists:
        if did_already_watch(df, movie, userId) == 0:
           list_of_movies.append(movie)
        if len(list_of_movies) == 5:
           break
    return list_of_movies

def did_already_watch(df, movie, userId):
    if df[df.index == userId][movie].isnull().iloc[0]:
        return 0
    else:
        return 1

def chat(df, userId, n_movies):
    movies_ratings_pivot = pd.pivot_table(df, values='rating', columns='title', index='userId')
    st.write(f'''Hi {userId}! I am your personal recommender.
    Here are {n_movies} popular movies based on your previous ratings:''')
    
    fav_movie = get_fav_movie(df, userId)['title'].iloc[0]
    list_of_movies = get_recommended_movies(movies_ratings_pivot, fav_movie, userId)[:n_movies]
    
    for movie in list_of_movies:
        genre = movies_df[movies_df['title'] == movie]['genres'].iloc[0]
        st.write(f'''Title: {movie}, Genre: {genre}''')

# Streamlit Interface
user_id = st.number_input("Enter your user ID:", min_value=1, max_value=movies_ratings['userId'].max(), step=1)
n_movies = st.number_input("Enter the number of movie recommendations you wish:", min_value=1, max_value=50, step=1)

if st.button("Get Recommendations"):
    chat(movies_ratings, user_id, n_movies)