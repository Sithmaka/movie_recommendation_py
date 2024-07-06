from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

# Initialize Flask app
app = Flask(__name__)

# Load and prepare data
df = pd.read_csv('IMDb_Dataset.csv')
df_cleaned = df.dropna(subset=['Genre', 'Title', 'Director'])

# Create combined features
df_cleaned['combined_features'] = df_cleaned.apply(lambda row: f"{row['Genre']} {row['Director']} {row['Title']}",
                                                   axis=1)

# Vectorize the combined features
tfidf = TfidfVectorizer(stop_words='english')
combined_features_matrix = tfidf.fit_transform(df_cleaned['combined_features'].fillna(''))
similarity_matrix = cosine_similarity(combined_features_matrix, combined_features_matrix)


def get_recommendations_by_title(title, similarity_matrix, features, top_n=5):
    try:
        matched_title = process.extractOne(title, features['Title'])[0]
        idx = features[features['Title'] == matched_title].index[0]
        sim_scores = list(enumerate(similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_sim_scores = sim_scores[1:top_n + 1]
        movie_indices = [i[0] for i in top_sim_scores]
        recommended_movies = features.iloc[movie_indices][
            ['Title', 'IMDb Rating', 'Year', 'Director', 'Star Cast', 'Genre']]
        return matched_title, recommended_movies
    except IndexError:
        return None


def get_recommendations_by_director(director, features, top_n=5):
    try:
        movies_by_director = features[features['Director'].str.contains(director, case=False, na=False)]
        if not movies_by_director.empty:
            return movies_by_director[['Title', 'IMDb Rating', 'Year', 'Director', 'Star Cast', 'Genre']].head(top_n)
        else:
            return None
    except IndexError:
        return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    search_type = request.form['searchType']
    query = request.form['query']

    if search_type == 'title':
        result = get_recommendations_by_title(query, similarity_matrix, df_cleaned)
        if result is not None:
            matched_title, recommendations = result
            movie_details = df_cleaned[df_cleaned['Title'] == matched_title].iloc[0]
            return render_template('results.html', movie=movie_details, recommendations=recommendations.values)
        else:
            return render_template('not_found.html', movie=query)
    elif search_type == 'director':
        recommendations = get_recommendations_by_director(query, df_cleaned)
        if recommendations is not None:
            return render_template('results.html', movie={'Title': query, 'Director': query},
                                   recommendations=recommendations.values)
        else:
            return render_template('not_found.html', movie=query)


if __name__ == '__main__':
    app.run(debug=True)
