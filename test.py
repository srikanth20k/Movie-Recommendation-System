import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np

overview_tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb') )


def recommend_movies(title, df,top_n=6): 
    
    # Check if the movie  title exists in the dataset
    if title not in df['Series_Title'].values:
        # If not, return a message indicating that the movie was not found
        return "Movie not found in the database."
    
    
    # Get the cluster label for the given movie title
    cluster_label = df[df['Series_Title'] == title]['cluster'].values[0]
    
    # Filter movies that belong to the same cluster
    cluster_movies = df[df['cluster'] == cluster_label]
    
    # Get the TF-IDF vector for the given movie title
    movie_cluster = overview_tfidf[df[df['Series_Title'] == title].index[0]]
    
    # Calculate cosine similarities between the given movie and other movies in the same cluster
    similarities = cosine_similarity(movie_cluster, overview_tfidf[cluster_movies.index]).flatten()
    
    # Get the indices of the top N most similar movies (excluding the given movie itself)
    similart_indices =similarities.argsort()[-(top_n+1):-1][::-1]
    
    # Return the top N most similar movies
    recommendation = cluster_movies.iloc[similart_indices][['Series_Title','Overview', 'IMDB_Rating','Poster_Link']]
    
    return  recommendation.reset_index(drop=True)

# recommend_movies('Inception', df)
    
    



st.set_page_config(page_title="Movie Recommendation System", layout="wide")
st.title(" 🎬Movie Recommendation System")
df = pd.read_csv('movie_recommendation_with_clusters.csv')


movies_list = df['Series_Title'].sort_values().unique()
movie_name = st.selectbox("Select a movie", movies_list)

if st.button("Recommend"):
    output  = recommend_movies(movie_name, df)
    
    
    if isinstance(output, str):
        st.error(output)
    else:
        for i in range(0,len(output),3):
            cols = st.columns(3)
            for j in range(3):
                if i+j < len(output):
                    row = output.iloc[i+j]
                    with cols[j]:
                        st.subheader(row['Series_Title'])
                        st.image(row['Poster_Link'])
                        st.write(f"IMDB Rating: {row['IMDB_Rating']}")
                        st.write(row['Overview'])