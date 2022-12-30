from django.shortcuts import render, redirect
import spotipy
import pandas as pd
import numpy as np
import os
from spotipy.oauth2 import SpotifyOAuth
from django.conf import settings
from .models import User, Track
from django.template.context_processors import csrf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble._forest import RandomForestClassifier
from scipy.sparse import hstack
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style='white')
from sklearn.manifold import TSNE
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def connect_spotify(request):
    sp_oauth = SpotifyOAuth(
        client_id=settings.SPOTIPY_CLIENT_ID,
        client_secret=settings.SPOTIPY_CLIENT_SECRET,
        redirect_uri=settings.SPOTIPY_REDIRECT_URI,
        scope='user-read-private user-read-email user-library-read user-follow-read user-top-read',
    )
    login_url = sp_oauth.get_authorize_url()
    return redirect(login_url)

def callback(request):
    authorization_code = request.GET.get('code')
    sp_oauth = SpotifyOAuth(
        client_id=settings.SPOTIPY_CLIENT_ID,
        client_secret=settings.SPOTIPY_CLIENT_SECRET,
        redirect_uri=settings.SPOTIPY_REDIRECT_URI,
    )
    token_info = sp_oauth.get_access_token(authorization_code)
    access_token = token_info['access_token']
    refresh_token = token_info['refresh_token']
    sp = spotipy.Spotify(auth=access_token)
    user = sp.current_user()
    spotify_id = user['id']
    display_name = user['display_name']
    # Check if the city field is present in the data returned by the Spotify API
    city = user.get('city', '')
    # Check if a User object already exists for this user
    try:
        user = User.objects.get(spotify_id=spotify_id)
        # Update the user's access token
        user.access_token = access_token
        user.save()
    except User.DoesNotExist:
        # Create a new User object
        user = User.objects.create(
            spotify_id=spotify_id,
            display_name=display_name,
            city=city,
            access_token=access_token,
            refresh_token=refresh_token,
        )
    # Store the user's ID in a session or cookie for use in subsequent requests
    request.session['user_id'] = spotify_id
    return redirect('/playlists')


def playlists(request):
    # Get the user's ID from the session or cookie
    user_id = request.session.get('user_id')
    if user_id is None:
        return redirect('/connect')
    
    # Get the User object for this user
    user = User.objects.get(spotify_id=user_id)
    
    # Set up Spotipy with the user's access token
    sp = spotipy.Spotify(auth=user.access_token)
    
    # Get the user's playlists
    playlists = sp.user_playlists(user=user.spotify_id)
    
    # Render the template with the playlists data
    return render(request, "playlists.html", {"playlists": playlists})


def export_playlist(request):
    # Generate the CSRF token
    csrf_token = csrf(request)

    # Get the user's ID from the session or cookie
    user_id = request.session.get('user_id')
    if user_id is None:
        return redirect('/connect')
    
    # Get the User object for this user
    user = User.objects.get(spotify_id=user_id)
    
    # Set up Spotipy with the user's access token
    sp = spotipy.Spotify(auth=user.access_token)
    
    # Get the selected playlist ID from the form submission
    playlist_id = request.POST.get('playlist')
    
    # Get the playlist object for the selected playlist
    playlist = sp.user_playlist(user_id, playlist_id)
    
    # Get the list of songs in the playlist
    tracks = playlist["tracks"]
    songs = tracks["items"]
    
    # Get the track IDs and names
    track_ids = []
    track_names = []
    for i in range(0, len(songs)):
        if songs[i]['track']['id'] != None: # Removes the local tracks in your playlist if there is any
            track_ids.append(songs[i]['track']['id'])
            track_names.append(songs[i]['track']['name'])

    # Get the audio features for each track
    features = []
    for i in range(0,len(track_ids)):
        audio_features = sp.audio_features(track_ids[i])
        for track in audio_features:
            if track is None:
                features.append({'danceability': 0, 'energy': 0, 'key': 0, 'loudness': 0, 'mode': 0, 'speechiness': 0, 'acousticness': 0, 'instrumentalness': 0, 'liveness': 0, 'valence': 0, 'tempo': 0, 'type': 'audio_features', 'id': '00000', 'uri': 'spotify:track:0', 'track_href': 'https://api.spotify.com/', 'analysis_url': 'https://api.spotify.com/', 'duration_ms': 0, 'time_signature': 0})
            else:
                features.append(track)
    
    # Create the dataframe from the audio features
    playlist_df = pd.DataFrame(features, index=track_names)
    
    # Keep only the specified columns in the dataframe
    playlist_df = playlist_df.filter(["id", "acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "key", "liveness", "loudness", "mode", "speechiness", "tempo", "valence"])

    if request.method == 'POST':
        # Read the selected playlist and export the data to a CSV file
        playlist_df.to_csv('playlist.csv')
        return redirect('/playlist_data')
    else:
        # Render the playlists template
        return render(request, 'playlists.html')


def playlist_data(request):
    # Read the CSV file and create a dataframe
    df = pd.read_csv('playlist.csv')
    
    # Render the template with the dataframe
    return render(request, "playlist_data.html", {"df": df})

def vectorize_names(request):
    # Read the CSV file and create a dataframe
    df = pd.read_csv('playlist.csv')
    
    # Print the list of column names
    print(df.columns)
    
    # Get the track names from the dataframe
    track_names = df['Unnamed: 0']  # Change 'name' to the correct column name
    
    # Vectorize the track names
    v = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 6), max_features=10000)
    X_names_sparse = v.fit_transform(track_names)
    
    # Render the template with the sparse matrix
    return render(request, "vectorize_names.html", {"X_names_sparse": X_names_sparse})

def playlist_dataset(request):
    # Read the CSV file and create a dataframe
    df = pd.read_csv('playlist.csv')
    
    # Print the list of column names
    df['ratings']=[5,8,5,9,10,5,5,9,9,9,7,6,9,10,10,8,9,7,6,5,8,5,9,10,5,5,9,9,9,7,6,9,10,10,8,9,7,6,9,10,10,8,9,7,6,10]
    
    # Get the track names from the dataframe
    df.head()

    # Print the list of column names
    print(df.columns)
    
    # Get the track names from the dataframe
    track_names = df['Unnamed: 0']  # Change 'name' to the correct column name
    
    # Vectorize the track names
    v = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 6), max_features=10000)
    X_names_sparse = v.fit_transform(track_names)
    
    # Save the dataframe to a subdirectory of the root directory
    df.to_csv(os.path.join(settings.BASE_DIR, "Spotify_playlist_Dataset.csv"))

    # Read the updated dataframe
    df2 = pd.read_csv(os.path.join(settings.BASE_DIR, "Spotify_playlist_Dataset.csv"))


    # Get the features and labels from the dataframe
    X = df2.drop(['id', 'ratings', 'Unnamed: 0'], axis=1)
    y = df2['ratings']

    # Scale the features
    X_scaled = StandardScaler().fit_transform(X)

    # Apply TSNE
    tsne = TSNE(random_state=17, perplexity=12)
    X_tsne = tsne.fit_transform(X_scaled)
    
    # Combine the PCA or TSNE transformed data with the sparse matrix of song names
    X_train_last = csr_matrix(hstack([X_tsne, X_names_sparse])) 

    warnings.filterwarnings('ignore')

    # Initialize a stratified split for the validation process
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    tree = DecisionTreeClassifier()

    tree_params = {'max_depth': range(1,11), 'max_features': range(4,19)}

    tree_grid = GridSearchCV(tree, tree_params, cv=skf, n_jobs=-1, verbose=True)

    tree_grid.fit(X_train_last, y)
    tree_best_estimator = tree_grid.best_estimator_
    tree_best_score = tree_grid.best_score_

    # Fit the random forest classifier
    forest = RandomForestClassifier(random_state=42, max_depth=5, max_features=13)
    forest.fit(X, y)

    parameters = {'max_features': [4, 7, 8, 10], 'min_samples_leaf': [1, 3, 5, 8], 'max_depth': [3, 5, 8]}
    rfc = RandomForestClassifier(n_estimators=100, random_state=42, 
                                n_jobs=-1, oob_score=True)
    gcv1 = GridSearchCV(rfc, parameters, n_jobs=-1, cv=skf, verbose=1)
    gcv1.fit(X_train_last, y)
    rfc_best_estimator = gcv1.best_estimator_
    rfc_best_score = gcv1.best_score_

    # Perform KNN classification and grid search for best parameters
    knn_params = {'n_neighbors': range(1, 10)}
    knn = KNeighborsClassifier(n_jobs=-1)
    knn_grid = GridSearchCV(knn, knn_params, cv=skf, n_jobs=-1, verbose=True)
    knn_grid.fit(X_train_last, y)
    best_params = knn_grid.best_params_
    best_score = knn_grid.best_score_


    # Get the feature importances
    importances = forest.feature_importances_

    # Get the feature names
    feature_names = X.columns

    # Combine the feature names and importances into a list of tuples
    feature_importances = [(feature, importance) for feature, importance in zip(feature_names, importances)]

    # Sort the list of tuples by importance
    feature_importances.sort(key=lambda x: x[1], reverse=True)

    #TEST
    X_test = df2[['Unnamed: 0.1', 'acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence']]
    y_test = df2['ratings']

    # Predict the ratings for the test set using the model
    predictions = forest.predict(X_test)

    # Create a dataframe for the recommended tracks
    recommended_tracks_df = pd.DataFrame(columns=['track_name', 'artist_name', 'predicted_rating'])

    # Add the track names and artist names as columns in the dataframe
    recommended_tracks_df['track_name'] = df2['Unnamed: 0']
    recommended_tracks_df['artist_name'] = df2['id']

    # Add the predicted ratings to the dataframe
    recommended_tracks_df['predicted_rating'] = predictions

    # Sort the dataframe by the predicted rating column
    recommended_tracks_df = recommended_tracks_df.sort_values(by='predicted_rating', ascending=False)


    # Render the template with the dataframe and feature importances data
    return render(request, "playlist_dataset.html", {"df": df2, "feature_importances": feature_importances, 
    "X_tsne": X_tsne, "y": y, 
    "tree_best_estimator": tree_best_estimator, "tree_best_score": tree_best_score, 
    "rfc_best_estimator": rfc_best_estimator, "rfc_best_score": rfc_best_score,
    "best_estimator": gcv1.best_estimator_, "gcv_best_score": gcv1.best_score_,
    "best_params": knn_grid.best_params_, "best_score": knn_grid.best_score_})

    # Render the template with the dataframe
    # return render(request, "playlist_dataset.html", {"df": df2})
