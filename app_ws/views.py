import datetime

from django.http import JsonResponse, HttpResponse
from ai_ws.models import Report
from django.template import loader


def index(request):
    template = loader.get_template('index.html')
    happyHourlyReport = {}
    otherReport = {}

    for i in range(8, 18):
        happyHourlyReport[i] = 0
        otherReport[i] = 0

    for item in Report.objects.all():
        if item.reported_on.year == datetime.datetime.now().year and item.reported_on.month == datetime.datetime.now().month and item.reported_on.day == datetime.datetime.now().day:
            hour = item.reported_on.hour
            if hour < 8:
                hour = 8
            if hour > 17:
                hour = 17
            if item.mood == 'H':
                happyHourlyReport[hour] += 1
            otherReport[hour] += 1

    for item in happyHourlyReport.keys():
        if otherReport[item] > 0:
            happyHourlyReport[item] = (happyHourlyReport[item] / (otherReport[item])) * 100
    data = [{'x': k, 'y': v, 'label': str(k) + "H"} for k, v in happyHourlyReport.items()]

    context = {'happyHourlyReport': data}
    return HttpResponse(template.render(context, request))


# Create your views here.
def graph(request):
    happyHourlyReport = {}
    otherReport = {}

    for i in range(8, 18):
        happyHourlyReport[i] = 0
        otherReport[i] = 0

    for item in Report.objects.all():
        if item.reported_on.year == datetime.datetime.now().year and item.reported_on.month == datetime.datetime.now().month and item.reported_on.day == datetime.datetime.now().day:
            hour = item.reported_on.hour
            if hour < 8:
                hour = 8
            if hour > 17:
                hour = 17
            if item.mood == 'H':
                happyHourlyReport[hour] += 1
            otherReport[hour] += 1

    for item in happyHourlyReport.keys():
        if otherReport[item] > 0:
            happyHourlyReport[item] = (happyHourlyReport[item] / (otherReport[item])) * 100
    data = [{'x': k, 'y': v, 'label': str(k) + "H"} for k, v in happyHourlyReport.items()]

    return JsonResponse(data, safe=False)
