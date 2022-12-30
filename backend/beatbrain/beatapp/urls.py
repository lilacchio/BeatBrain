from django.urls import path
from . import views

urlpatterns = [
    path("connect", views.connect_spotify, name="connect_spotify"),
    path('export_playlist/', views.export_playlist, name='export_playlist'),
    path('playlist_data/', views.playlist_data, name='playlist_data'),
    path('vectorize_names/', views.vectorize_names, name='vectorize_names'),
    path('playlist_dataset/', views.playlist_dataset, name='playlist_dataset'),
    path("callback", views.callback, name="callback"),
    path("playlists", views.playlists, name="playlists"),
]