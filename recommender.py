import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_csv('movies.csv')
# Load movie data
movies = pd.read_csv('movies.csv')
print(movies.head())
# Combine important columns into one text column
movies['combined_features'] = movies['genre'] + ' ' + movies['cast'] + ' ' + movies['description']

# Convert text into numbers using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])

print(tfidf_matrix.shape)
# Find similarity between all movies
similarity = cosine_similarity(tfidf_matrix)

print(similarity.shape)
print(similarity[0])
def recommend(movie_title, top_n=5):
    # Find the index of the movie that matches the title
    idx = movies[movies['title'] == movie_title].index[0]
    
    # Get similarity scores for that movie with all others
    scores = list(enumerate(similarity[idx]))
    
    # Sort movies by similarity score (highest first)
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    # Skip the first one (it's the movie itself) and take top_n
    scores = scores[1:top_n+1]
    
    # Get the movie titles from these indices
    recommended_movies = [movies.iloc[i[0]]['title'] for i in scores]
    
    return recommended_movies

# Test the function
print(recommend('Kalki 2898 AD'))