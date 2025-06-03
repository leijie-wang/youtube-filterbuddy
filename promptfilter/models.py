
import random
from venv import logger

from django.db import models

class CommentStatus:
    PUBLISHED = 'published'
    DELETED = 'deleted'
    REVIEW = 'reviewing'


class User(models.Model):
    username = models.CharField(max_length=255, unique=True, primary_key=True)
    avatar = models.URLField(blank=True, null=True)
    oauth_credentials = models.JSONField(blank=True, null=True)
    # used to determine whether our tool has access to moderation features
    moderation_access = models.BooleanField(default=False)
    second_last_sync = models.DateTimeField(blank=True, null=True)
    last_sync = models.DateTimeField(blank=True, null=True)

    def __str__(self):
        return self.username

class Channel(models.Model):
    DEFAULT_MODERATION = [
        (CommentStatus.DELETED, 'Disallow Comments'),
        (CommentStatus.PUBLISHED, 'Approve Comments'),
        (CommentStatus.REVIEW, 'Hold Comments for Review'),
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
        (CommentStatus.REVIEW, 'Under Review'),
    ]    
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='comments')
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
            elif CommentStatus.REVIEW in actions:
                # if any of the predictions is under review, we return under review
                return CommentStatus.REVIEW
            elif CommentStatus.PUBLISHED in actions:
                return CommentStatus.PUBLISHED
            else:
                return self.video.channel.default_moderation
            
    def serialize(self, as_prediction=False):
        comment = {
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
            'link': f'https://www.youtube.com/watch?v={self.video.id}&t=1s&lc={self.id}',
        }
        if as_prediction:
            comment['prediction'] = None
            comment['explanation'] = ''
            comment['groundtruth'] = None
        return comment

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
    filter = models.ForeignKey('PromptFilter', on_delete=models.CASCADE, related_name='rubrics')

    def serialize(self):
        return {
            'id': self.id,
            'rubric': self.rubric,
            'examples': [example.serialize() for example in self.examples.all()]
        }

class PromptFilter(models.Model):
    FILTER_ACTIONS = [
        (CommentStatus.DELETED, 'Delete Comments'),
        (CommentStatus.PUBLISHED, 'Publish Comments'),
        (CommentStatus.REVIEW, 'Hold Comments for Review'), 
        ('nothing', 'Do Nothing'),
        # ('reply', 'Reply to Comments'),
    ]

    name = models.CharField(max_length=255)
    description = models.TextField()

    # examples = models.ManyToManyField('Example', blank=True, related_name='filters')
    few_shot_examples = models.ManyToManyField('Example', blank=True, related_name='filters')

    channel = models.ForeignKey(Channel, on_delete=models.CASCADE, related_name='filters')
    action = models.CharField(max_length=10, choices=FILTER_ACTIONS, default='nothing')
    reply_message = models.TextField(blank=True)
    last_run = models.DateTimeField(blank=True, null=True)

    def __str__(self):
        return f'{self.name} ({self.channel.name})'
    
    def serialize(self, view=False):
        positive_rubrics = self.rubrics.filter(is_positive=True)
        negative_rubrics = self.rubrics.filter(is_positive=False)
        groundtruths = self.matches.filter(groundtruth__isnull=False)

        serialized_filter = {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'positives': [rubric.serialize() for rubric in positive_rubrics],
            'negatives': [rubric.serialize() for rubric in negative_rubrics],
            'lastRun': self.last_run,
            'action': self.action,
            'channelName': self.channel.name,
        }
        if not view:
            serialized_filter['examples'] = [groundtruth.serialize() for groundtruth in groundtruths]
            serialized_filter['fewShotExamples'] = [example.serialize() for example in self.few_shot_examples.all()]
            serialized_filter['channelId'] = self.channel.id
            serialized_filter['replyMessage'] = self.reply_message
        
        return serialized_filter

    def whether_changed(self, new_filter):
        # Check direct fields first
        if self.description != new_filter.get('description', ''):
            return True

        # Helper function to process rubric comparisons
        def rubrics_changed(old_rubrics, new_rubrics):
            # Create dictionaries for quick lookup {id: rubric_text}
            old_dict = {str(r['id']): r['rubric'] for r in old_rubrics}
            revised_dict = {str(r['id']): r['rubric'] for r in new_rubrics if r.get('id', None) is not None}
            new_list = [r['rubric'] for r in new_rubrics if r.get('id', None) is None]

            # Check for additions (new IDs not in old)
            if len(new_list) > 0:
                return True

            # Check for deletions (old IDs not in new)
            if any(r_id not in revised_dict for r_id in old_dict):
                return True

            # Check for content changes in existing rubrics
            for r_id, content in revised_dict.items():
                if old_dict.get(r_id) != content:
                    return True

            return False

        # Check positive rubrics
        current_positives = [r.serialize() for r in self.rubrics.filter(is_positive=True)]
        if rubrics_changed(current_positives, new_filter.get('positives', [])):
            return True

        # Check negative rubrics
        current_negatives = [r.serialize() for r in self.rubrics.filter(is_positive=False)]
        if rubrics_changed(current_negatives, new_filter.get('negatives', [])):
            return True

        # If all checks passed
        return False

    def update_filter(self, new_filter):
        """
            Update a PromptFilter instance with new data in a serialized format.
        
        """
        self.description = new_filter.get('description', self.description)
        self.action = new_filter.get('action', self.action)
        # Helper function to process rubric updates
        def process_rubrics(new_rubrics, is_positive):
            updated_rubrics = []
            
            for rubric_data in new_rubrics:
                rubric_text = rubric_data.get('rubric', '').strip()
                if not rubric_text:  # Skip empty rubrics
                    continue
                    
                if rubric_data.get('id', None) is not None:
                    try:
                        rubric = PromptRubric.objects.get(id=rubric_data['id'], filter=self)
                        # Update existing rubric if text changed
                        if rubric.rubric != rubric_text:
                            rubric.rubric = rubric_text
                            rubric.save()

                        updated_rubrics.append(rubric)
                    except PromptRubric.DoesNotExist:
                        logger.warning(f'Rubric with ID {rubric_data["id"]} not found.')
                else:
                    # Create new rubric
                    new_rubric = PromptRubric.objects.create(
                        rubric=rubric_text,
                        is_positive=is_positive,
                        filter=self
                    )
                    updated_rubrics.append(new_rubric)
            
            return updated_rubrics

        # Process positive and negative rubrics
        new_positives = process_rubrics(new_filter.get('positives', []), True)
        new_negatives = process_rubrics(new_filter.get('negatives', []), False)

        # Update M2M relationships
        existing_rubrics = PromptRubric.objects.filter(filter=self)
        updated_rubrics = new_positives + new_negatives
        
        # Remove deleted rubrics
        for rubric in existing_rubrics:
            if rubric not in updated_rubrics:
                rubric.delete()  # Delete rubrics that are no longer relevant

        for few_shot_example in new_filter.get('fewShotExamples', []):
            if 'content' not in few_shot_example or 'groundtruth' not in few_shot_example:
                logger.warning(f'Few-shot example {few_shot_example} is missing required fields.')
                continue
            new_example = Example.objects.create(
                content=few_shot_example['content'],
                groundtruth=few_shot_example['groundtruth']
            )
            new_example.save()
            self.few_shot_examples.add(new_example)
        # Save final state
        self.save()
        return self

    def delete_prompt(self):
        # delete predictions associated with this filter
        FilterPrediction.objects.filter(filter=self).delete()
        PromptFilter.objects.filter(id=self.id).delete()

    def retrieve_update_comments(self, mode, start_date=None):
        comments = Comment.objects.filter(video__channel=self.channel).order_by('posted_at')
        if mode == 'new' and self.last_run:
            # select comments that appear after the last run
            comments = list(comments.filter(posted_at__gt=self.last_run).order_by('posted_at'))
        elif mode == 'all':
            if start_date:
                comments = list(comments.filter(posted_at__gt=start_date).order_by('posted_at'))
            else:
                comments = list(comments.order_by('posted_at'))
        elif mode == 'initialize':
            # we randomly sample 200 comments because users might still quickly iterate on the filter
            # and we want to avoid wasting too many API calls
            comments = list(comments.all())
            comments = random.sample(comments, min(200, len(comments)))
            logger.info(f'Initializing filter {self.name} with {len(comments)} comments.')
        elif mode == 'iteration':
            # we only select comments with groundtruths; they must have corresponding predictions
            comments = self.matches.filter(groundtruth__isnull=False)   
            logger.info(f'Iterating filter {self.name} with {len(comments)} comments.')
        elif mode == 'refresh':
            # only select comments that have predictions
            comments = self.matches.all()
            logger.info(f'Refreshing filter {self.name} with {len(comments)} comments.')
        return comments

class MistakeCluster(models.Model):

    filter = models.ForeignKey(PromptFilter, on_delete=models.CASCADE, related_name='clusters')
    predictions = models.ManyToManyField('FilterPrediction', blank=True, related_name='clusters')
    summary = models.TextField(blank=True, null=True)

    # represents which kind of rubrics this cluster indicates refinement for.
    kind = models.CharField(max_length=10, choices=[('positive', 'Positive'), ('negative', 'Negative')], default='positive')
    # represents the action that this cluster wants to take in order to refine the filter.
    action = models.CharField(max_length=10, choices=[('add', 'Add'), ('edit', 'Edit')], default='add')
    # represents the rubric that this cluster wants to refine, if any.
    rubric = models.ForeignKey('PromptRubric', on_delete=models.CASCADE, blank=True, null=True, related_name='clusters')
    # reprensents whether this rubric is still active.
    active = models.BooleanField(default=True)

    def __str__(self):
        return f'Mistake cluster for {self.filter.name} on {len(self.predictions)} mistakes: {self.summary}'

    def serialize(self):
        return {
            'id': self.id,
            'filter': self.filter.name,
            'cluster': [prediction.serialize() for prediction in self.predictions.all()],
            'summary': self.summary,
            'kind': self.kind,
            'action': self.action,
            'rubric': self.rubric.serialize() if self.rubric else None,
            'active': self.active
        } 
    
    @staticmethod
    def create_cluster(filter, cluster):
        
        summary = cluster.get('summary', None)

        kind = cluster.get('kind', None)
        action = cluster.get('action', None)  # Defaults to 'add'
        if kind is None or action is None:
            logger.error(f'We expect a valid kind {kind} or action {action}.')
            return None

        rubric_content = cluster.get('rubric', None)
        if rubric_content is not None:
            # search for this particular rubric given the filter and the content
            rubric_instance = PromptRubric.objects.filter(
                filter=filter,
                is_positive=(kind=='positive'),
                rubric=rubric_content
            ).first()

            if not rubric_instance:
                logger.error(f'Rubric "{rubric_content}" not found for filter {filter.name}.')
                return None
        else:
            rubric_instance = None

        # Create and save the MistakeCluster instance
        cluster_instance = MistakeCluster.objects.create(
            filter=filter,
            summary=summary,
            kind=kind,
            action=action,
            rubric=rubric_instance,
            active=True  # Default value
        )

        predictions_data = cluster.get('cluster', [])
        comment_ids = [p['id'] for p in predictions_data if 'id' in p]
        predictions = FilterPrediction.objects.filter(filter=filter, comment_id__in=comment_ids)
        cluster_instance.predictions.set(predictions)  # Assign ManyToMany predictions. no save is needed for ManyToMany updates.

        return cluster_instance

class FilterPrediction(models.Model):
    """A prediction of whether a comment matches a filter. 
    
    By default we only store positive predictions.
    But as some participants might curate false negatives, we can store negative predictions as well.
    """
    filter = models.ForeignKey(PromptFilter, on_delete=models.CASCADE, related_name='matches')
    comment = models.ForeignKey(Comment, on_delete=models.CASCADE, related_name='predictions')
    prediction = models.BooleanField(blank=True, null=True)
    confidence = models.FloatField(blank=True, null=True)
    groundtruth = models.BooleanField(blank=True, null=True)
    explanation = models.TextField(blank=True, null=True, help_text='Optional explanation for why the comment matched/don\'t match the filter')
    reflection = models.TextField(blank=True, null=True, help_text='Failure reasons if it is a mistake')

    # TODO: think of whether we should store the mitake reflection
    # the concern is that we need to track whether this reflection is no longer applicable because of future iterations.

    def __str__(self):
        return f'Prediction of {self.filter.name} on {self.comment.id} at {self.matched_at}: {self.prediction} versus the groundtruth {self.groundtruth}'
    
    def serialize(self):
        comment = self.comment.serialize()
        comment['prediction'] = self.prediction
        comment['confidence'] = self.confidence
        comment['explanation'] = self.explanation
        comment['groundtruth'] = self.groundtruth
        comment['reflection'] = self.reflection
        return comment

    class Meta:
        unique_together = ('filter', 'comment')  # Enforces unique constraint on the filter-comment combination