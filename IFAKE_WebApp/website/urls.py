from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name="index"),
    path('runAnalysis', views.runAnalysis),
    path('runVideoAnalysis', views.runVideoAnalysis),
    path('getImages', views.getImages),
    path('video', views.video, name="video"),
    path('image', views.image, name="image"),
    path('pdf', views.pdf, name="pdf"),
    path('runPdf2image', views.runPdf2image),

]

