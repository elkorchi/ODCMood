from django.http import JsonResponse
from .models import Report


# Create your views here.
def report(request):
    print(request.GET)
    mood = request.GET.get('mood')
    location = request.GET.get('location')
    site = request.GET.get('site')
    reported = Report()
    reported.mood = mood
    reported.location = location
    reported.site = site
    reported.save()
    return JsonResponse({"blah": "Hello, world. You're at the polls index."})
