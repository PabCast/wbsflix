import streamlit as st
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
import requests
import numpy as np

# Load data
movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')

# Combine the data into one dataset
data_combined = pd.merge(movies, ratings)

# Split into training and test set
trainset, testset = train_test_split(data_combined, test_size=0.2)

# Prepare data for training
R_df = trainset.pivot(index='userId', columns='movieId', values='rating').fillna(0)
R = R_df.values

# Train SVD model
svd = TruncatedSVD(n_components=200)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
R_transformed = lsa.fit_transform(R)

# Get popular movies
popular_movies = data_combined.groupby(['movieId', 'title', 'genres'])['rating'].mean().reset_index()
popular_movies = popular_movies.sort_values(by='rating', ascending=False)
popular_movies = popular_movies.rename(columns={'rating': 'average_rating'})

# Add image URLs to popular movies DataFrame
image_urls = []
for movie_id in popular_movies.head(10)['movieId']:
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=YOUR_API_KEY&append_to_response=images"
    response = requests.get(url)
    images = response.json().get('images', {})
    image_url = images['posters'][0]['file_path'] if images.get('posters') else None
    image_urls.append(f"https://image.tmdb.org/t/p/w200{image_url}" if image_url else None)

popular_movies['image_url'] = image_urls

# Display popular movies
st.title('Personalized Movie Recommender')
st.subheader('Most Popular Movies')
st.dataframe(popular_movies.head(10))

st.write("""
Choose a user ID and find some great movie recommendations!
""")

# Sidebar with user input
user_id = st.sidebar.number_input("Enter your user ID:", min_value=1, max_value=ratings['userId'].max(), step=1)
n_movies = st.sidebar.number_input("Enter the number of movie recommendations you wish:", min_value=1, max_value=10, step=1)

# Button to generate recommendations
if st.sidebar.button("Get Recommendations"):
    # Get user recommendation
    user_vector = R_transformed[user_id - 1, :]
    predicted_ratings = np.dot(user_vector, svd.components_)
    recommended_movie_ids = np.argsort(predicted_ratings)[::-1][:n_movies]

    recommendations_df = pd.DataFrame(columns=['title', 'image_url'])
    for movie_id in recommended_movie_ids:
        movie_details = movies.loc[movies['movieId'] == movie_id].iloc[0]
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=da33e359269762e4cfacfd0eec2816a3&append_to_response=images"
        response = requests.get(url)
        images = response.json().get('images', {})
        image_url = images['posters'][0]['file_path'] if images.get('posters') else None
        recommendations_df = recommendations_df.append({
            'title': movie_details['title'],
            'image_url': f"https://image.tmdb.org/t/p/w200{image_url}" if image_url else None
        }, ignore_index=True)

    st.dataframe(recommendations_df)

    st.write(f"Here are the top {n_movies} movie recommendations for user {user_id}:")
    for movie_id in recommended_movie_ids:
        movie_details = movies.loc[movies['movieId'] == movie_id].iloc[0]
        st.write(movie_details['title'])
