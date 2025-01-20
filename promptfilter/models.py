from venv import logger
from django.db import models

class CommentStatus:
    PUBLISHED = 'published'
    DELETED = 'deleted'


class User(models.Model):
    username = models.CharField(max_length=255, unique=True, primary_key=True)
    avatar = models.URLField(blank=True, null=True)
    oauth_credentials = models.JSONField(blank=True, null=True)
    second_last_sync = models.DateTimeField(blank=True, null=True)
    last_sync = models.DateTimeField(blank=True, null=True)

    def __str__(self):
        return self.username

class Channel(models.Model):
    DEFAULT_MODERATION = [
        (CommentStatus.DELETED, 'Disallow Comments'),
        (CommentStatus.PUBLISHED, 'Approve Comments'),
    ]
    id = models.CharField(max_length=255, unique=True, primary_key=True)
    owner = models.OneToOneField(User, on_delete=models.CASCADE, related_name='channel')
    name = models.CharField(max_length=255)
    default_moderation = models.CharField(max_length=10, choices=DEFAULT_MODERATION, default=CommentStatus.PUBLISHED)
    
    def __str__(self):
        return self.name

class Video(models.Model):
    channel = models.ForeignKey(Channel, on_delete=models.CASCADE, related_name='videos')
    id = models.CharField(max_length=255, unique=True, primary_key=True)
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    video_link = models.URLField(unique=True)
    thumbnail = models.URLField(blank=True, null=True)
    posted_at = models.DateTimeField()

    def __str__(self):
        return self.title
    
    def serialize(self):
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'video_link': self.video_link,
            'thumbnail': self.thumbnail,
            'posted_at': self.posted_at,
        }

class Comment(models.Model):
    COMMENT_STATUS = [
        (CommentStatus.DELETED, 'Deleted'),
        (CommentStatus.PUBLISHED, 'Published'),   
    ]    
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='comments', )
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='comments')
    content = models.TextField()
    id = models.CharField(max_length=255, unique=True, primary_key=True)
    posted_at = models.DateTimeField()
    likes = models.PositiveIntegerField(default=0)
    dislikes = models.PositiveIntegerField(default=0)
    parent = models.ForeignKey('self', null=True, blank=True, on_delete=models.SET_NULL, related_name='replies')
    total_replies = models.PositiveIntegerField(default=0)
    
    status = models.CharField(max_length=10, choices=COMMENT_STATUS, default=CommentStatus.PUBLISHED)

    def __str__(self):
        return f'Comment by {self.user.username} on {self.video.title}'

    @property
    def is_reply(self):
        return self.parent is not None
    
    def determine_status(self):
        # find all filter predictions for this comment
        predictions = self.predictions.all()
        if not predictions.exists():
            default_moderation = self.video.channel.default_moderation
            return default_moderation
        else:
            actions = [prediction.filter.action for prediction in predictions]
            # as long as one of them is deleted, the comment is deleted
            if CommentStatus.DELETED in actions:
                return CommentStatus.DELETED
            else:
                return CommentStatus.PUBLISHED
            
    def serialize(self):
        return {
            'id': self.id,
            'content': self.content,
            'user': self.user.username,
            'avatar': self.user.avatar,
            'video': self.video.title,
            'posted_at': self.posted_at,
            'likes': self.likes,
            'dislikes': self.dislikes,
            'status': self.status,
            'totalReplies': self.total_replies,
        }

class Example(models.Model):
    content = models.TextField()
    groundtruth = models.BooleanField()

    def serialize(self):
        return {
            'content': self.content,
            'groundtruth': self.groundtruth
        }

class PromptRubric(models.Model):
    rubric = models.TextField()
    is_positive = models.BooleanField(default=True)
    examples = models.ManyToManyField('Example', blank=True, related_name='rubrics')

    def serialize(self):
        return {
            'rubric': self.rubric,
            'examples': [example.serialize() for example in self.examples.all()]
        }

class PromptFilter(models.Model):
    FILTER_ACTIONS = [
        ('delete', 'Delete Comments'),
        ('publish', 'Publish Comments'),   
        ('nothing', 'Do Nothing'),
        ('reply', 'Reply to Comments'),
    ]

    name = models.CharField(max_length=255)
    description = models.TextField()
    rubrics = models.ManyToManyField(PromptRubric, blank=True, related_name='filters')
    examples = models.ManyToManyField('Example', blank=True, related_name='filters')
    few_shot_examples = models.JSONField(default=list, blank=True)

    channel = models.ForeignKey(Channel, on_delete=models.CASCADE, related_name='filters')
    action = models.CharField(max_length=10, choices=FILTER_ACTIONS, default='nothing')
    reply_message = models.TextField(blank=True)
    last_run = models.DateTimeField(blank=True, null=True)

    def __str__(self):
        return f'{self.name} ({self.channel.name})'
    
    def serialize(self):
        positive_rubrics = self.rubrics.filter(is_positive=True)
        negative_rubrics = self.rubrics.filter(is_positive=False)
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'positives': [rubric.serialize() for rubric in positive_rubrics],
            'negatives': [rubric.serialize() for rubric in negative_rubrics],
            'examples': [example.serialize() for example in self.examples.all()],
            'fewShotExamples': self.few_shot_examples,
            'action': self.action,
            'channel': self.channel.name,
            'replyMessage': self.reply_message,

        }
    
    def delete_prompt(self):
        # delete predictions associated with this filter
        FilterPrediction.objects.filter(filter=self).delete()
        PromptFilter.objects.filter(id=self.id).delete()

    def retrieve_update_comments(self, mode):
        comments = Comment.objects.filter(video__channel=self.channel).order_by('posted_at')
        if mode == 'new' and self.last_run:
            # select comments that appear after the last run
            comments = list(comments.filter(posted_at__gt=self.last_run).order_by('posted_at'))
        else:
            comments = list(comments.all())
        return comments

class FilterPrediction(models.Model):
    """A prediction of whether a comment matches a filter. 
    
    By default we only store positive predictions.
    But as some participants might curate false negatives, we can store negative predictions as well.
    """
    filter = models.ForeignKey(PromptFilter, on_delete=models.CASCADE, related_name='matches')
    comment = models.ForeignKey(Comment, on_delete=models.CASCADE, related_name='predictions')
    prediction = models.BooleanField(blank=True, null=True)
    groundtruth = models.BooleanField(blank=True, null=True)
    explanation = models.TextField(blank=True, null=True, help_text='Optional explanation for why the comment matched/don\'t match the filter')

    def __str__(self):
        return f'Prediction of {self.filter.name} on {self.comment.id} at {self.matched_at}: {self.prediction} versus the groundtruth {self.groundtruth}'
    
    def serialize(self):
        comment = self.comment.serialize()
        comment['prediction'] = self.prediction
        comment['explanation'] = self.explanation
        comment['groundtruth'] = self.groundtruth
        return comment

    class Meta:
        unique_together = ('filter', 'comment')  # Enforces unique constraint on the filter-comment combination