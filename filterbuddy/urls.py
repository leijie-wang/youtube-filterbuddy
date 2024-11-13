"""
URL configuration for filterbuddy project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from promptfilter import views as promptviews

urlpatterns = [
    path('admin/', admin.site.urls),
    path('get-csrf-token/', promptviews.csrf_token_view),
    path('authorize_users/', promptviews.authorize_user),
    path('oauth2callback/', promptviews.oauth2_callback),
    path('logout/', promptviews.logout_user),
    path('synchronize/', promptviews.synchronize_youtube),

    path('request/filters/', promptviews.request_filters),
    path('request/comments/', promptviews.request_comments),
    path('request/tasks/', promptviews.poll_tasks),
    path('request/comment_info/', promptviews.request_comment_info),
    path('request/user/', promptviews.request_user),

    path('prompt/refresh/', promptviews.refresh_predictions),

    path('prompt/initialize/', promptviews.initialize_prompt),
    path('prompt/explore/', promptviews.explore_prompt),
    path('prompt/save/', promptviews.save_prompt),
    path('prompt/delete/', promptviews.delete_prompt),
    path('prompt/improve/', promptviews.improve_prompt),
    path('prompt/clarify/', promptviews.clarify_prompt),
    path('prompt/refine/', promptviews.refine_prompt),

    path('prompt/explain/', promptviews.explain_prediction),
    path('prompt/revert/', promptviews.revert_prediction),
]
