from django.db import models


class Report(models.Model):
    id = models.BigAutoField(primary_key=True)
    mood = models.CharField(max_length=1)
    reported_on = models.DateTimeField(auto_now_add=True)
    location = models.CharField(max_length=255)
    site = models.CharField(max_length=255)

