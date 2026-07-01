from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load and prepare data (same as recommender.py)
movies = pd.read_csv('TeluguMovies_dataset.csv')
movies['combined_features'] = movies['Genre'].fillna('') + ' ' + movies['Overview'].fillna('')

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])
similarity = cosine_similarity(tfidf_matrix)

def recommend(movie_title, top_n=5):
    idx = movies[movies['Movie'] == movie_title].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1:top_n+1]
    recommended_movies = [movies.iloc[i[0]]['Movie'] for i in scores]
    return recommended_movies

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    selected_movie = None
    if request.method == 'POST':
        selected_movie = request.form['movie']
        recommendations = recommend(selected_movie)
    movie_list = movies['Movie'].tolist()
    return render_template('index.html', movies=movie_list, recommendations=recommendations, selected_movie=selected_movie)

if __name__ == '__main__':
    app.run(debug=True)