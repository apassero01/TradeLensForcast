#!/bin/bash

python manage.py test \
    dataset_manager.tests.FeatureFactoryServiceTest \
    dataset_manager.tests.DatasetManagerServiceTest \
    dataset_manager.tests.MovingAverageFeatureFactoryTest \
    dataset_manager.tests.TestTargetFeatureFactory \
    dataset_manager.tests.TestMomentumFeatureFactory \
    dataset_manager.tests.TestBandFeatureFactory \
    dataset_manager.tests.TestOHLCVFeatureFactory \
    dataset_manager.tests.TestDatasetTrackerService \
    dataset_manager.tests.TestFeatureTrackerService

python manage.py test sequenceset_manager.tests.SequencesetManagerServiceTest