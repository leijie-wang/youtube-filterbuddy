# Generated by Django 5.1.2 on 2024-11-11 05:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('promptfilter', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='comment',
            name='total_replies',
            field=models.PositiveIntegerField(default=0),
        ),
    ]
