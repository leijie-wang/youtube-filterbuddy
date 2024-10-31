import random
import pandas as pd
from datetime import timedelta
from django.utils import timezone
from promptfilter.models import *

def random_time(zerotime=None):
    if zerotime is None:
        zerotime = timezone.now() - timedelta(days=30)
    days_after = random.randint(0, 30)
    random_time = zerotime + timedelta(days=days_after, hours=random.randint(0, 23), minutes=random.randint(0, 59))
    return random_time

def populate_test_users():
    user = User(username='Ryan')
    user.save()
    print(f"User {user.username} has been successfully created!")
    print(f"There are {User.objects.all().count()} users in the database.")

    channel = Channel(owner=user, name='Ryan\'s Channel')
    channel.save()
    print(f"Channel {channel.name} has been successfully created!")
    print(f"There are {Channel.objects.all().count()} channels in the database.")

    video_info = [
        {"title": "Conspiracy Theorists Can't Answer This One Simple Question", "video_link": "https://www.example.com/watch?v=conspiracy-theorists-question"},
        {"title": "Climate Change Deniers Walk Out After Seeing the Facts", "video_link": "https://www.example.com/watch?v=climate-change-deniers-facts"},
        {"title": "The Earth is Flat and Here's Why", "video_link": "https://www.example.com/watch?v=flat-earth-reasons"},
        {"title": "Anti-Vaxxers Lose It Over Simple Science Questions", "video_link": "https://www.example.com/watch?v=anti-vaxxers-science"},
    ]
    for info in video_info:
        video = Video(channel=channel, title=info['title'], video_link=info['video_link'], published_at=random_time())
        video.save()
    print(f"{len(video_info)} videos have been successfully created!")
    print(f"There are {Video.objects.all().count()} videos in the database.")

    prompt_filters = [
        {"name": "Sexually Explicit Content", "description": "Comments that contain sexually explicit or inappropriate content not suitable for public viewing."},
        {"name": "Spam", "description": "Comments that are repetitive, irrelevant, or promotional in nature."},
        {"name": "Off-Topic", "description": "Comments that are unrelated to the video content or discussion."},
    ]
    for info in prompt_filters:
        filter = PromptFilter(channel=channel, name=info['name'], description=info['description'])
        filter.save()
    print(f"{len(prompt_filters)} prompt filters have been successfully created!")
    print(f"There are {PromptFilter.objects.all().count()} prompt filters in the database.")

def populate_test_comments(file_path, field="comment"):
    # file_path = 'datasets/test.csv'
    comments_df = pd.read_csv(file_path)
    print(f"Successfully loaded {len(comments_df)} test comments from {file_path}.")

    for _, row in comments_df.iterrows():
        user = User.objects.filter(username='Ryan').first()

        # Randomly select one of the user's videos
        videos = Video.objects.filter(channel__owner=user).all()
        if videos.exists():
            # Pick one random video
            video = random.choice(videos)  
            likes = random.randint(0, 100)
            dislikes = random.randint(0, 100)
            
            # randomly generate a username as five random alphanumerics as the commenter
            commentor = User(username=''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=5)))
            commentor.save()

            # Create a new Comment object
            comment = Comment(
                video=video,
                user=commentor,
                content=row[field],
                posted_at=random_time(video.published_at),
                likes=likes,
                dislikes=dislikes
            )

            # Save the comment to the database
            comment.save()
        else:
            print("No videos found for the user. Please create a video first.")

    print(f"{len(Comment.objects.all())} test comments have been successfully created!")

def populate_test_predictions():
    comments = Comment.objects.all()
    filters = PromptFilter.objects.all()
    for comment in comments:
        # generate a binary prediction for each filter
        comment.groundtruth = random.choice([CommentStatus.PUBLISHED, CommentStatus.DELETED, None])
        predictions = [random.choice([True, False]) for _ in filters]
        for prediction, filter in zip(predictions, filters):
            match = FilterPrediction(filter=filter, comment=comment, prediction=prediction)
            match.save()
        comment.status = comment.determine_status()
        # randomly generate a groundtruth status as either published, deleted, or None
        comment.save()
    print(f"{FilterPrediction.objects.all().count()} test predictions have been successfully created!")
    
