from django.db import models

class User(models.Model):
    spotify_id = models.CharField(max_length=255, unique=True)
    display_name = models.CharField(max_length=255)
    city = models.CharField(max_length=255)
    access_token = models.CharField(max_length=255)
    refresh_token = models.CharField(max_length=255)

    class Meta:
        app_label = 'beatapp'

class Track(models.Model):
    track_id = models.CharField(max_length=255)
    acousticness = models.FloatField()
    danceability = models.FloatField()
    duration_ms = models.IntegerField()
    energy = models.FloatField()
    instrumentalness = models.FloatField()
    key = models.IntegerField()
    liveness = models.FloatField()
    loudness = models.FloatField()
    mode = models.IntegerField()
    speechiness = models.FloatField()
    tempo = models.FloatField()
    valence = models.FloatField()
