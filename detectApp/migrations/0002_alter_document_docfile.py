# Generated by Django 3.2.5 on 2021-07-31 09:54

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('detectApp', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='document',
            name='docfile',
            field=models.FileField(upload_to='documents/'),
        ),
    ]
