import streamlit as st
import pandas as pd
import requests
from surprise import Reader, Dataset, SVD

# Load data
movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')

# Combine the data into one dataset
data_combined = pd.merge(movies, ratings)

# Calculate most popular movies by average rating
popular_movies = data_combined.groupby(['movieId', 'title', 'genres'])['rating'].mean().reset_index()
popular_movies = popular_movies.sort_values(by='rating', ascending=False)
popular_movies = popular_movies.rename(columns={'rating': 'average_rating'})

# Get top 10 popular movies
top_10_popular_movies = popular_movies.head(10).copy()

# Add image URLs to popular movies DataFrame
image_urls = []
for movie_id in top_10_popular_movies['movieId']:
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=da33e359269762e4cfacfd0eec2816a3&append_to_response=images"
    response = requests.get(url)
    images = response.json().get('images', {})
    image_url = images['posters'][0]['file_path'] if images.get('posters') else None
    image_urls.append(f"https://image.tmdb.org/t/p/w200{image_url}" if image_url else None)

top_10_popular_movies['image_url'] = image_urls

# Display popular movies
st.title('Personalized Movie Recommender')
st.subheader('Most Popular Movies')
for index, row in top_10_popular_movies.iterrows():
    st.image(row['image_url'], width=200)
    st.write(f"Title: {row['title']}")
    st.write(f"Genres: {row['genres']}")
    st.write(f"Average Rating: {row['average_rating']}")

# Read data using Surprise library
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(data_combined[['userId', 'movieId', 'rating']], reader)

# Split into training and test set
trainset = data.build_full_trainset()

# Train SVD algorithm
algo = SVD()
algo.fit(trainset)

# Sidebar with user input
user_id = st.sidebar.number_input("Enter your user ID:", min_value=1, max_value=ratings['userId'].max(), step=1)
n_movies = st.sidebar.number_input("Enter the number of movie recommendations you wish:", min_value=1, max_value=10, step=1)

# Button to generate recommendations
if st.sidebar.button("Get Recommendations"):
    # Build anti-testset
    testset = trainset.build_anti_testset()
    testset = [x for x in testset if x[0] == user_id]

    # Predict ratings
    predictions = algo.test(testset)

    # Get top n recommendations
    top_n_recs = sorted(predictions, key=lambda x: x.est, reverse=True)[:n_movies]
    movie_recs = [movies.loc[movies['movieId'] == x.iid].iloc[0]['title'] for x in top_n_recs]

    # Display recommendations
    rec_df = pd.DataFrame({'Movie Recommendations': movie_recs})
    st.dataframe(rec_df)

