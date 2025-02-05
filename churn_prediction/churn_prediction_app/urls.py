# churn_prediction_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_churn, name='predict_churn'),
    path('predict', views.predict_churn, name='predict_churn'),
]
