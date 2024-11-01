from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import Channel, User, Comment, Video, PromptFilter, FilterPrediction

admin.site.register(Channel)
admin.site.register(User)
admin.site.register(Comment)
admin.site.register(Video)
admin.site.register(PromptFilter)
admin.site.register(FilterPrediction)
 