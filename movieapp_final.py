from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import pandas as pd
from surprise import Reader, Dataset, KNNBasic, accuracy
from surprise.model_selection import train_test_split
from surprise import SVD
import movieposters as mp
st.title("WBSFLIX")
st.header("Popularity ranking")

url2 = "/content/drive/MyDrive/Colab Notebooks/Week 11/movies.csv"
movies_df = pd.read_csv(url2)

url3 = "/content/drive/MyDrive/Colab Notebooks/Week 11/ratings.csv"
ratings_df = pd.read_csv(url3)



# Sidebar with user input
selected_movie_name = st.sidebar.selectbox(
    "Select a movie from the dropdown",
    movies_df['title'].values)

#user_id
n = 5
user_id = st.sidebar.number_input("Enter your user ID:", min_value=1, max_value=ratings_df['userId'].max(), step=1)

def movie_rec(n):
    rating_count_df = ratings_df.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
    rating_count_df["score"] = ((rating_count_df["mean"] + rating_count_df["count"]) * 0.1).round(2)

    result_df_combined = []

    for i in range(1, n + 1):
        highest_rated = rating_count_df.nlargest(i, 'score')['movieId'].values[i - 1]

        highest_rated_isbn_mask = movies_df['movieId'] == highest_rated
        book_info_columns = ['movieId', 'title']

        result_df = movies_df.loc[highest_rated_isbn_mask, book_info_columns].drop_duplicates()
        result_df_combined.append(result_df)

    final_result_df = pd.concat(result_df_combined)

    return final_result_df

rec_movies = movie_rec(5)

col1, col2, col3, col4, col5 = st.columns(5)
col_list = [col1, col2, col3, col4, col5]

for col, (_, row) in zip(col_list, rec_movies.iterrows()):
    poster_url = mp.get_poster(title=row['title'])

    with col:
        st.image(poster_url, caption=row['title'], width=150, output_format='JPEG')



st.header("Based on your pick ")
def movies_cosines_rating(id, n):

  # Calculate movie cosine similarity
  movie_cosines_df = pd.DataFrame(movies_cosines_matrix[id])
  movie_cosines_df = movie_cosines_df.rename(columns={id: 'cosine'})
  movie_cosines_df = movie_cosines_df[movie_cosines_df.index != id]
  movie_cosines_df = movie_cosines_df.sort_values(by="cosine", ascending=False)

  # Calculate number of users who rated both movies
  no_of_users_rated_both_movies = [sum((user_movies_matrix[id] > 0) & (user_movies_matrix[movieId] > 0)) for movieId in  movie_cosines_df.index]

  # Add user count and filter movies
  movie_cosines_df['users_who_rated_both_movies'] = no_of_users_rated_both_movies
  movie_cosines_df = movie_cosines_df[movie_cosines_df["users_who_rated_both_movies"] > 10]


  # Retrieve top rated movies with cosine similarity
  top_cosine_movies = (movie_cosines_df
                              .head(n)
                              .reset_index()
                              .merge(movies_df.drop_duplicates(subset='movieId'),
                                     on='movieId',
                                     how='left')
                              [['movieId', 'title', 'cosine',	'users_who_rated_both_movies']]
                              )
  return top_cosine_movies

user_movies_matrix = pd.pivot_table(data=ratings_df,
                                  values='rating',
                                  index='userId',
                                  columns='movieId',
                                  fill_value=0)

movies_cosines_matrix = pd.DataFrame(cosine_similarity(user_movies_matrix.T),
                                    columns=user_movies_matrix.columns,
                                    index=user_movies_matrix.columns)




# Lookup the selected movie's ID based on the name
selected_movie_id = movies_df.loc[movies_df['title'] == selected_movie_name, 'movieId'].iloc[0]

# Call the recommendation function with the selected movie's ID
rec_movies_selected = movies_cosines_rating(selected_movie_id, 5)

col1, col2, col3, col4, col5 = st.columns(5)
col_list = [col1, col2, col3, col4, col5]

for col, (_, row) in zip(col_list, rec_movies_selected.iterrows()):
    poster_url = mp.get_poster(title=row['title'])

    with col:
        st.image(poster_url, caption=row['title'], width=150, output_format='JPEG')



st.header("What others are watching")

data = ratings_df[['userId', 'movieId', 'rating']]
watcher = Reader(rating_scale=(1, 5.0))
data = Dataset.load_from_df(data, watcher)

trainset, testset = train_test_split(data, test_size=0.2, random_state=142)

full_train = data.build_full_trainset()
algo = SVD(n_factors=150, n_epochs=30, lr_all=0.01, reg_all=0.05)
algo.fit(trainset)

testset = trainset.build_anti_testset()
predictions = algo.test(testset)

def get_top_n(predictions, user_id, n=5):

  user_recommendations = []

  for uid, iid, true_r, est, _ in predictions:
    if user_id == uid:
      user_recommendations.append((iid, est))
    else:
      continue

  ordered_recommendations = sorted(user_recommendations, key=lambda x: x[1], reverse=True)

  ordered_recommendations_top_n = ordered_recommendations[:n]

  return ordered_recommendations_top_n


def get_top_n_movies(predictions, user_id, n):
    top_n = get_top_n(predictions, user_id, n)

    tuples_df = pd.DataFrame(top_n, columns=["movieId", "estimated_rating"])

    reduced_df = movies_df.drop_duplicates(subset='movieId').copy()

    tuples_df_expanded = tuples_df.merge(reduced_df, on="movieId", how='left')

    tuples_df_expanded = tuples_df_expanded[['movieId', 'title']]

    return tuples_df_expanded



top_n_movies = get_top_n_movies(predictions, user_id, n)

col1, col2, col3, col4, col5 = st.columns(5)
col_list = [col1, col2, col3, col4, col5]

for col, (_, row) in zip(col_list, top_n_movies.iterrows()):
    poster_url = mp.get_poster(title=row['title'])

    with col:
        st.image(poster_url, caption=row['title'], width=150, output_format='JPEG')
