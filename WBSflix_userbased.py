import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
url = 'https://drive.google.com/file/d/1WB7QQGjulU_ODpfIAkey5ZO5dYDStkc_/view?usp=sharing'
path = 'https://drive.google.com/uc?id='+url.split('/')[-2]
movies = pd.read_csv(path)

url = 'https://drive.google.com/file/d/1sQl_yG4sv_AKIcO2Z1d_nw2XTQ7FWaTt/view?usp=sharing'
path = 'https://drive.google.com/uc?id='+url.split('/')[-2]
df_rate = pd.read_csv(path)
ratings = movies.merge(ratings)

# Title and Introduction
st.title('Personal Movie Recommender')
st.write("Hey there! I can help you discover some cool movies based on your previous ratings. Let's get started!")

# User ID input
user_id = st.number_input('Enter your user ID:', min_value=1, max_value=movies_ratings['userId'].max(), value=4, step=1)

def get_recommended_movies(df, userId):
    movies_ratings_pivot = pd.pivot_table(df, values='rating', columns='title', index='userId')
    movies_ratings_pivot = movies_ratings_pivot.fillna(0)
    
    list_of_movies = []
    sample_similitude = pd.DataFrame(cosine_similarity(movies_ratings_pivot), 
                columns=movies_ratings_pivot.index, index=movies_ratings_pivot.index)
    predictions = pd.DataFrame(columns=['title','prediction'])
    for movie in movies_ratings_pivot.columns:
        if movies_ratings_pivot[movie][userId] == 0:
            temp = pd.DataFrame({'ratings': movies_ratings_pivot[movie],
                                 'simility': sample_similitude[userId]})
            temp = temp[temp.ratings != 0]
            if sum(temp['simility']) != 0:
                pred = sum(temp['ratings'] * temp['simility']) / sum(temp['simility'])
                predictions = predictions.append({'title': movie, 'prediction': pred}, ignore_index=True)
    list_of_movies = predictions.sort_values('prediction', ascending=False)['title'].head(5).to_list()

    return list_of_movies

# Recommend Button
if st.button('Recommend Movies!'):
    recommended_movies = get_recommended_movies(movies_ratings, user_id)
    st.header('Your Movie Recommendations:')
    for movie in recommended_movies:
        st.write(movie)
else:
    st.write("Enter your user ID and hit the button to get recommendations!")
