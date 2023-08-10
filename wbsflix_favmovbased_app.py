import streamlit as st
import pandas as pd

# Set the title of the app
st.title('Movie Recommendation Engine')

url = 'https://drive.google.com/file/d/1WB7QQGjulU_ODpfIAkey5ZO5dYDStkc_/view?usp=sharing'
path = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
movies = pd.read_csv(path)

url = 'https://drive.google.com/file/d/1sQl_yG4sv_AKIcO2Z1d_nw2XTQ7FWaTt/view?usp=sharing'
path = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
ratings = pd.read_csv(path)

movies_ratings = movies.merge(ratings)

def get_genre(movie_title):
    return movies[movies['title'] == movie_title]['genres'].iloc[0]

def get_recommended_movies(df, favorite_movie, num_recommendations):
    list_of_movies = []

    # Drop columns with zero variance or only one unique value
    df = df.dropna(axis='columns', thresh=2)
    df = df.loc[:, (df != df.iloc[0]).any()]

    corr = df.corrwith(df[favorite_movie])
    lists = corr.sort_values(ascending=False).iloc[1:num_recommendations+1].index.to_list() # Exclude the favorite movie itself
    for movie in lists:
        list_of_movies.append((movie, get_genre(movie)))

    return list_of_movies

def chat():
    movies_ratings_pivot = pd.pivot_table(movies_ratings, values='rating', columns='title', index='userId')
    st.write(f'''Hi there! I am your personal recommender.
    Please enter your favorite movie, and I'll recommend some similar ones that you might enjoy.''')
    
    favorite_movie = st.selectbox("Please select your favorite movie:", options=movies['title'].tolist())
    num_recommendations = st.number_input("How many similar movies would you like?", min_value=1, max_value=10, step=1)

    list_of_movies = get_recommended_movies(movies_ratings_pivot, favorite_movie, num_recommendations)

    # Create a DataFrame to display the recommended movies and their genres
    recommended_movies_df = pd.DataFrame(list_of_movies, columns=['Title', 'Genre'])
    st.write("Here are some movies similar to your favorite:")
    st.write(recommended_movies_df) # Displaying the DataFrame

chat()