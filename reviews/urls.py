from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_sentiment, name='predict_sentiment'),
]
