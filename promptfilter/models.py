from venv import logger
from django.db import models
from .llm_filter import LLMFilter

class CommentStatus:
    PUBLISHED = 'published'
    DELETED = 'deleted'


class User(models.Model):
    username = models.CharField(max_length=255, unique=True)
    oauth_credentials = models.JSONField(blank=True, null=True)

    def __str__(self):
        return self.username


class Channel(models.Model):
    DEFAULT_MODERATION = [
        (CommentStatus.DELETED, 'Disallow Comments'),
        (CommentStatus.PUBLISHED, 'Approve Comments'),
    ]
    owner = models.OneToOneField(User, on_delete=models.CASCADE, related_name='channel')
    name = models.CharField(max_length=255)
    default_moderation = models.CharField(max_length=10, choices=DEFAULT_MODERATION, default=CommentStatus.PUBLISHED)
    
    def __str__(self):
        return self.name

class Video(models.Model):
    channel = models.ForeignKey(Channel, on_delete=models.CASCADE, related_name='videos')
    title = models.CharField(max_length=255)
    video_link = models.URLField(unique=True)
    published_at = models.DateTimeField()

    def __str__(self):
        return self.title

class Comment(models.Model):
    COMMENT_STATUS = [
        (CommentStatus.DELETED, 'Deleted'),
        (CommentStatus.PUBLISHED, 'Published'),   
    ]    
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='comments')
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='comments')
    content = models.TextField()
    posted_at = models.DateTimeField()
    likes = models.PositiveIntegerField(default=0)
    dislikes = models.PositiveIntegerField(default=0)
    parent = models.ForeignKey('self', null=True, blank=True, on_delete=models.SET_NULL, related_name='replies')
    
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
            'video': self.video.title,
            'posted_at': self.posted_at,
            'likes': self.likes,
            'dislikes': self.dislikes,
            'status': self.status,
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
    prompt = models.TextField(blank=True)
    positive_examples = models.TextField(blank=True, help_text='List of positive examples (optional)')
    negative_examples = models.TextField(blank=True, help_text='List of negative examples (optional)')
    channel = models.ForeignKey(Channel, on_delete=models.CASCADE, related_name='filters')
    action = models.CharField(max_length=10, choices=FILTER_ACTIONS, default='nothing')
    last_run = models.DateTimeField(blank=True, null=True)

    def __str__(self):
        return f'{self.name} ({self.channel.name})'
    
    def serialize(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'action': self.action,
            'channel': self.channel.name,
        }
    
    def update_predictions(self, mode):
        # randomly sample comments from the database to begin with
        comments = Comment.objects.filter(video__channel=self.channel).order_by('posted_at')
        if not comments.exists():
            return None
        logger.info(f'Filter {self.name} has {comments.count()} comments.')
        if mode == 'new' and self.last_run:
            # select comments that appear after the last run
            comments = comments.filter(posted_at__gt=self.last_run)
        elif mode == 'all':
            # we randomly sample 100 comments for testing purposes
            comments = list(comments.all())

        datasets = [comment.content for comment in comments]
        
        llm_filter = LLMFilter({
            'name': self.name,
            'description': self.description,
            }, debug=False)
        predictions = llm_filter.predict(datasets)

        # summarize the predictions
        positive_num = sum(predictions)
        negative_num = len(predictions) - positive_num
        print(f'There are {positive_num} positive predictions and {negative_num} negative predictions.')

        comments = [comment.serialize() for comment in comments]
        for index, comment in enumerate(comments):
            comment['prediction'] = predictions[index]
            # update the prediction in the database
            FilterPrediction.objects.update_or_create(
                filter=self,
                comment_id=comment['id'],
                defaults={'prediction': comment['prediction']}
            )
        return comments
        
    def delete_prompt(self):
        # delete predictions associated with this filter
        FilterPrediction.objects.filter(filter=self).delete()
        PromptFilter.objects.filter(id=self.id).delete()

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