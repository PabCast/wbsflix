import streamlit as st
import pandas as pd
from surprise import Reader, Dataset, SVD
from sklearn.model_selection import train_test_split

# Load data
movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')

# Combine the data into one dataset
data_combined = pd.merge(movies, ratings)

# Calculate most popular movies by average rating
popular_movies = data_combined.groupby(['movieId', 'title', 'genres'])['rating'].mean().reset_index()
popular_movies = popular_movies.sort_values(by='rating', ascending=False)
popular_movies = popular_movies.rename(columns={'rating': 'average_rating'})

# Display the top 10 popular movies
st.title('Personalized Movie Recommender')
st.subheader('Most Popular Movies')
st.write(popular_movies[['title', 'genres', 'average_rating']].head(10))

st.write("""
Choose a user ID and find some great movie recommendations!
""")

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

    st.write(f"Here are the top {n_movies} movie recommendations for user {user_id}:")
    for movie in movie_recs:
        st.write(movie)
