# Generated by Django 4.1.4 on 2022-12-28 14:54

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('beatapp', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('spotify_id', models.CharField(max_length=255, unique=True)),
                ('display_name', models.CharField(max_length=255)),
                ('city', models.CharField(max_length=255)),
                ('access_token', models.CharField(max_length=255)),
                ('refresh_token', models.CharField(max_length=255)),
            ],
        ),
        migrations.DeleteModel(
            name='Token',
        ),
    ]
