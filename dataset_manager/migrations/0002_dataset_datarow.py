# Generated by Django 5.1 on 2024-09-12 23:10

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dataset_manager', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='DataSet',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('dataset_type', models.CharField(max_length=50)),
                ('start_timestamp', models.DateTimeField()),
                ('end_timestamp', models.DateTimeField()),
                ('features', models.JSONField()),
                ('metadata', models.JSONField()),
            ],
        ),
        migrations.CreateModel(
            name='DataRow',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('timestamp', models.DateTimeField()),
                ('features', models.JSONField()),
                ('dataset', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='dataset_manager.dataset')),
            ],
        ),
    ]