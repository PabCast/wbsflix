import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import random
import sys

# Function that takes as an input a user id and outputs the top n movies of the user:
def top_movies(df_rate, df_movies, name, range_of_days: float, n):
    top_movies_for_id = []
    df_merge = (df_rate
                .merge(df_movies, on='movieId')
                .assign(time_count = round(abs(ratings['timestamp']-max(ratings['timestamp']))/60/60/24))
                .sort_values('time_count')
                .filter(['userId', 'rating', 'title', 'time_count'])
                .query('time_count<=@range_of_days')
               )
    df_pivot = pd.pivot_table(df_merge, values='rating', columns='title', index='userId')
    df_movie_list = df_pivot.columns.values.tolist()
    random.shuffle(df_movie_list)
    for value in df_movie_list:
        if df_pivot.loc[name][value] >= 4:
            top_movies_for_id.append(value)
            if len(top_movies_for_id) == n:
                break
    return top_movies_for_id

# Function to get the most popular movies:
def pop_movies(df_rate, df_movies, rate_tresh, range_of_days: float):  # Added colon here
    df_merge = (df_rate
               .merge(df_movies, on='movieId')
               .assign(time_count = round(abs(df_rate['timestamp']-max(df_rate['timestamp']))/60/60/24))  # Fixed reference to 'ratings' here
               .sort_values('time_count')
               .filter(['userId', 'rating', 'title', 'time_count'])
               .query('time_count<=@range_of_days')
              )
    df = df_merge.groupby('title').agg(rate_count=('rating','count'), rate_mean=('rating','mean')).query('rate_count >= @rate_tresh').sort_values('rate_mean', ascending=False)
    return df.index.to_list()  # Fixed indentation here


# Item-based Collaborative Filtering: Function which outputs the top n most similar movies to top rated movies of a user:
def item_based_recommender(df_rate, df_movies, top_movies, range_of_days: float, tresh_n):
    recommend_movies = []
    df_merge = (df_rate
                .merge(df_movies, on='movieId')
                .assign(time_count=round(abs(df_rate['timestamp'] - max(df_rate['timestamp'])) / 60 / 60 / 24)) # Replaced 'ratings' with 'df_rate'
                .sort_values('time_count')
                .filter(['userId', 'rating', 'title', 'time_count'])
                .query('time_count<=@range_of_days')
               )
    corr = pd.pivot_table(df_merge, values='rating', columns='title', index='userId').dropna(axis='index', thresh=tresh_n).corr()
    top_corr = corr.filter(top_movies).round(1)
    num_list = [0.9, 1.0]
    for i in num_list:
        for index, value in top_corr.iterrows():
            if i in value.values:
                recommend_movies.append(index)
    return recommend_movies

# Function for user-based recommendation:
def dense_df_preparation(data: pd.DataFrame):
    if len(data.columns) != 3:
        print("""
            Be sure to have added a dataframe with only the columns: 
                users, movies and ratings
            They have to be sorted in the same way!
        """)
        return False
    data.columns = ['userId', 'movies', 'rating']
    return data


def sparse_df_preparation(data: pd.DataFrame):
    dense_df = dense_df_preparation(data)
    # in case there is a problem, stop the recommender
    if dense_df is False: 
        return False
    sparse_df = dense_df.pivot('userId','movies','rating')
    return sparse_df

def train_test_creation(data, random_state=1, train_size=.8):
    sparse_df = sparse_df_preparation(data)
    ratings_pos = pd.DataFrame(np.argwhere(~np.isnan(np.array(sparse_df))))
    # np.argwhere(a) is almost the same as np.transpose(np.nonzero(a)), but produces a result of the correct shape for a 0D array.
    train_pos, test_pos = train_test_split(ratings_pos, random_state=random_state, train_size=train_size)
    # creating an empty dataframe full of 0, with the same shape as the sparse_df data
    train = np.zeros(sparse_df.shape)
    # filling the set with the sparse_df ratings based on the train positions
    for pos in train_pos.values: 
        index = pos[0]
        col = pos[1]
        train[index, col] = sparse_df.iloc[index, col]
    train = pd.DataFrame(
        train, 
        columns=sparse_df.columns, 
        index=sparse_df.index
    )
    test = np.zeros(sparse_df.shape)
    for pos in test_pos.values: 
        index = pos[0]
        col = pos[1]
        test[index, col] = sparse_df.iloc[index, col]
    test = pd.DataFrame(
        test, 
        columns=sparse_df.columns, 
        index=sparse_df.index
        )
    return train, test, train_pos, test_pos

# Reading data
url = 'https://drive.google.com/file/d/1WB7QQGjulU_ODpfIAkey5ZO5dYDStkc_/view?usp=sharing'
path = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
movies = pd.read_csv(path)

url = 'https://drive.google.com/file/d/1sQl_yG4sv_AKIcO2Z1d_nw2XTQ7FWaTt/view?usp=sharing'
path = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
ratings = pd.read_csv(path)

dictionary = movies.filter(['movieId', 'title']).drop_duplicates()

# Title for the app
st.title('Personalized Movie Recommender: A Multi-Method Approach')

st.write("""
Choose the recommendation method you prefer, and let's find some great movies for you!
""")

# Sidebar with options
method = st.sidebar.selectbox(
    "Choose a recommendation method:",
    ("Top Movies", "Popular Movies", "Item-Based Collaborative Filtering", "User-Based Recommendations")
)

user_id = st.sidebar.number_input("Enter your user ID:", min_value=1, step=1)
n_movies = st.sidebar.number_input("Enter the number of movie recommendations you wish:", min_value=1, max_value=50, step=1)
range_of_days = st.sidebar.number_input("Enter the range of days:", min_value=1, max_value=365, step=1)
rate_tresh = st.sidebar.number_input("Enter the rate threshold:", min_value=1, max_value=5, step=1)

if st.sidebar.button("Get Recommendations"):
    if method == "Top Movies":
        top_movies_list = top_movies(ratings, movies, user_id, range_of_days, n_movies)
        st.write(f"Here are your top {n_movies} movies:")
        for movie in top_movies_list:
            st.write(movie)
    elif method == "Popular Movies":
        pop_movies_list = pop_movies(ratings, movies, rate_tresh, range_of_days)
        st.write(f"Here are the most popular movies:")
        for movie in pop_movies_list[:n_movies]:
            st.write(movie)
    elif method == "Item-Based Collaborative Filtering":
        top_movies_list = top_movies(ratings, movies, user_id, range_of_days, n_movies)
        recommended_movies = item_based_recommender(ratings, movies, top_movies_list, range_of_days, rate_tresh)
        st.write(f"Here are {n_movies} similar movies based on your top-rated movies:")
        for movie in recommended_movies[:n_movies]:
            st.write(movie)
    elif method == "User-Based Recommendations":
        recommendation = get_user_based_recommendations(ratings.drop(columns='timestamp'), user_id=user_id, top_n=n_movies)
        recommended_movies = item_to_movie_title(recommendation)
        st.write(f"You will probably like these movies:")
        for movie in recommended_movies:
            st.write(movie)