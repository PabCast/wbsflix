import streamlit as st
import pandas as pd


url = 'https://drive.google.com/file/d/1WB7QQGjulU_ODpfIAkey5ZO5dYDStkc_/view?usp=sharing'
path = 'https://drive.google.com/uc?id='+url.split('/')[-2]
movies = pd.read_csv(path)

url = 'https://drive.google.com/file/d/1sQl_yG4sv_AKIcO2Z1d_nw2XTQ7FWaTt/view?usp=sharing'
path = 'https://drive.google.com/uc?id='+url.split('/')[-2]
ratings = pd.read_csv(path)

movies_ratings = movies.merge(ratings)

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
def chat(df, userId):
    movies_ratings_pivot = pd.pivot_table(df, values='rating', columns='title', index='userId')
    st.write(f'''Hi {userId}! I am your personal recommender.
    Would you like me to recommend you some popular movies based on your previous ratings?''')
    x = st.selectbox("Type:", options=["y", "n"])

    if x == 'y':
        fav_movie = get_fav_movie(df, userId)['title'].iloc[0]
        list_of_movies = get_recommended_movies(movies_ratings_pivot, fav_movie, userId)
        for movie in list_of_movies:
            st.write(f'''{movie}''')
    else:
        st.write(f'''Goodbye''')

user_id = st.number_input("Enter your user ID:", min_value=1, max_value=movies_ratings['userId'].max(), step=1)
chat(movies_ratings, user_id)