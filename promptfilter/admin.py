from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import Channel, User, Comment, Video, PromptFilter, FilterPrediction, UserLog

admin.site.register(Channel)
admin.site.register(User)
admin.site.register(Comment)
admin.site.register(Video)
admin.site.register(PromptFilter)
admin.site.register(FilterPrediction)

@admin.register(UserLog)
class UserLogAdmin(admin.ModelAdmin):
    # --- list page ---
    list_display  = ("timestamp", "user", "action", "short_details")
    list_filter   = ("user", "action")
    search_fields = ("user__username", "action", "details")
    ordering      = ("-timestamp",)

    # make every field read-only
    readonly_fields = [f.name for f in UserLog._meta.fields]

    # disable add / edit / delete
    def has_add_permission   (self, request):            return False
    def has_change_permission(self, request, obj=None):  return False
    def has_delete_permission(self, request, obj=None):  return False

    # helper to keep the list page tidy
    def short_details(self, obj):
        if obj.details and len(obj.details) > 75:
            return f"{obj.details[:75]}…"
        return obj.details
    short_details.short_description = "details"