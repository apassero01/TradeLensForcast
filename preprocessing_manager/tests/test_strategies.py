from copy import deepcopy

import numpy as np
import pandas as pd
from django.test import TestCase

from preprocessing_manager.strategies import Create3dArraySequenceSetStrategy, TrainTestSplitDateStrategy, \
    ScaleByFeaturesStrategy, CombineDataSetsStrategy
from sequenceset_manager.models import SequenceSet
from sequenceset_manager.services import SequenceSetService
from training_session.models import ModelSet
from training_session.strategy import CreateFeatureSetsStrategy


class Create3dArraySequenceSetStrategyTestCase(TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'open': [1, 2, 3, 4, 5], 'high': [2, 1.5, 2.5, 4, 5], 'close+1': [2, 3, 4, 5, 6]})
        self.sequence_set_1 = SequenceSet.objects.create(dataset_type='stock', sequence_length=2,
                                                       start_timestamp='2020-01-01', end_timestamp='2020-01-05',
                                                       feature_dict={'open': 0, 'high': 1, 'close+1': 2},
                                                       metadata={'ticker': 'AAPL'})
        self.stock_sequences = SequenceSetService.create_sequence_objects(self.sequence_set_1, self.df)
        self.stock_sequences = sorted(self.stock_sequences, key=lambda x: x.start_timestamp)
        self.sequence_set_1.sequences = self.stock_sequences

        self.df = pd.DataFrame({'high': [2, 1.5, 2.5, 4, 5],'open': [1, 2, 3, 4, 5], 'close+1': [2, 3, 4, 5, 6]})
        self.sequence_set_2 = SequenceSet.objects.create(dataset_type='stock', sequence_length=2,
                                                       start_timestamp='2020-01-01', end_timestamp='2020-01-05',
                                                       feature_dict={'open': 0, 'high': 1, 'close+1': 2},
                                                       metadata={'ticker': 'AAPL'})
        self.stock_sequences = SequenceSetService.create_sequence_objects(self.sequence_set_2, self.df)
        self.stock_sequences = sorted(self.stock_sequences, key=lambda x: x.start_timestamp)
        self.sequence_set_2.sequences = self.stock_sequences

        config = {
            'step_number': 1,
            'm_service': 'preprocessing_manager',
            'parent_strategy': 'ModelSetsStrategy',
            'type': 'Create3dArraySequenceSetStrategy',

        }
        self.strategy = Create3dArraySequenceSetStrategy(config)

        model_set_1 = ModelSet()
        model_set_1.data_set = self.sequence_set_1
        model_set_1.X_features = ['open', 'high']
        model_set_1.y_features = ['close+1']

        model_set_2 = ModelSet()
        model_set_2.data_set = self.sequence_set_2
        model_set_2.X_features = ['open', 'high']
        model_set_2.y_features = ['close+1']

        self.model_sets = [model_set_1, model_set_2]

    def test_apply(self):
        model_sets = self.strategy.apply(self.model_sets)

        for model_set in model_sets:
            self.assertEqual(model_set.X.shape, (4, 2, 2))
            self.assertEqual(model_set.y.shape, (4, 1, 1))

            expected_X = np.array([[[1, 2], [2, 1.5]], [[2, 1.5], [3, 2.5]], [[3, 2.5], [4, 4]], [[4, 4], [5, 5]]])
            expected_y = np.array([[[3]], [[4]], [[5]], [[6]]])
            np.testing.assert_almost_equal(model_set.X, expected_X)
            np.testing.assert_almost_equal(model_set.y, expected_y)

class TrainTestSplitDateStrategyTestCase(TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'open': [1, 2, 3, 4, 5], 'high': [2, 1.5, 2.5, 4, 5], 'close+1': [2, 3, 4, 5, 6]})
        self.sequence_set_1 = SequenceSet.objects.create(dataset_type='stock', sequence_length=2,
                                                       start_timestamp='2020-01-01', end_timestamp='2020-01-05',
                                                       feature_dict={'open': 0, 'high': 1, 'close+1': 2},
                                                       metadata={'ticker': 'AAPL'})
        self.stock_sequences = SequenceSetService.create_sequence_objects(self.sequence_set_1, self.df)
        self.stock_sequences = sorted(self.stock_sequences, key=lambda x: x.start_timestamp)
        self.sequence_set_1.sequences = self.stock_sequences

        self.df = pd.DataFrame({'high': [2, 1.5, 2.5, 4, 5],'open': [1, 2, 3, 4, 5], 'close+1': [2, 3, 4, 5, 6]})
        self.sequence_set_2 = SequenceSet.objects.create(dataset_type='stock', sequence_length=2,
                                                       start_timestamp='2020-01-01', end_timestamp='2020-01-05',
                                                       feature_dict={'open': 1, 'high': 0, 'close+1': 2},
                                                       metadata={'ticker': 'AAPL'})
        self.stock_sequences = SequenceSetService.create_sequence_objects(self.sequence_set_2, self.df)
        self.stock_sequences = sorted(self.stock_sequences, key=lambda x: x.start_timestamp)
        self.sequence_set_2.sequences = self.stock_sequences

        config = {
            'parent_strategy': 'ModelSetsStrategy',
            'm_service': 'preprocessing_manager',
            'type': 'TrainTestSplitDateStrategy',
            'step_number': 1,
            'split_date': '2020-01-04',
        }
        model_set_1 = ModelSet()
        model_set_1.data_set = self.sequence_set_1
        model_set_1.X_features = ['open', 'high']
        model_set_1.y_features = ['close+1']
        model_set_1.X = np.array([[[1, 2], [2, 1.5]], [[2, 1.5], [3, 2.5]], [[3, 2.5], [4, 4]], [[4, 4], [5, 5]]])
        model_set_1.y = np.array([[[3]], [[4]], [[5]], [[6]]])
        model_set_1.row_ids = [1, 2, 3, 4, 5]

        model_set_2 = ModelSet()
        model_set_2.data_set = self.sequence_set_2
        model_set_2.X_features = ['open', 'high']
        model_set_2.y_features = ['close+1']
        model_set_2.X = np.array([[[1, 2], [2, 1.5]], [[2, 1.5], [3, 2.5]], [[3, 2.5], [4, 4]], [[4, 4], [5, 5]]])
        model_set_2.y = np.array([[[3]], [[4]], [[5]], [[6]]])
        model_set_2.row_ids = [1, 2, 3, 4, 5]

        self.model_sets = [model_set_1, model_set_2]
        self.strategy = TrainTestSplitDateStrategy(config)

    def test_apply(self):
        model_sets = self.strategy.apply(self.model_sets)

        for model_set in model_sets:
            self.assertEqual(model_set.X_train.shape, (3, 2, 2))
            self.assertEqual(model_set.y_train.shape, (3, 1, 1))
            self.assertEqual(model_set.X_test.shape, (1, 2, 2))
            self.assertEqual(model_set.y_test.shape, (1, 1, 1))

            expected_X_train = np.array([[[1, 2], [2, 1.5]], [[2, 1.5], [3, 2.5]], [[3, 2.5], [4, 4]]])
            expected_y_train = np.array([[[3]], [[4]], [[5]]])
            expected_X_test = np.array([[[4, 4], [5, 5]]])
            expected_y_test = np.array([[[6]]])

            np.testing.assert_almost_equal(model_set.X_train, expected_X_train)
            np.testing.assert_almost_equal(model_set.y_train, expected_y_train)
            np.testing.assert_almost_equal(model_set.X_test, expected_X_test)
            np.testing.assert_almost_equal(model_set.y_test, expected_y_test)


class ScaleByFeaturesStrategyTestCase(TestCase):
    def setUp(self):
        model_set1 = ModelSet()
        model_set1.X_train = np.array([[[1, 2], [2, 1.5]], [[2, 1.5], [3, 2.5]], [[3, 2.5], [4, 4]], [[4, 4], [5, 5]]])
        model_set1.X_test = np.array([[[4, 4], [5, 5]]])

        model_set1.y_train = np.array([[[3],[4]], [[4],[5]], [[5],[6]],[[6],[7]]])
        model_set1.y_test = np.array([[[6],[7]]])

        print(model_set1.X_train.shape, model_set1.y_train.shape)
        print(model_set1.X_test.shape, model_set1.y_test.shape)

        model_set1.X_feature_dict = {'open': 0, 'high': 1}
        model_set1.y_feature_dict = {'close+1': 0}
        self.model_sets = [model_set1]


    def test_applyXFeatureType1FeatureSet(self):
        feature_set_config = {
            'scaler_config':{
                'scaler_name': 'MEAN_VARIANCE_SCALER_3D'
            },
            'feature_list': ['open', 'high'],
            'do_fit_test' : False,
            'feature_set_type': 'X',
        }

        feature_set_strategy_config = {
            'm_service': 'training_session',
            'type': 'FeatureSetStrategy',
            'parent_strategy': 'ModelSetsStrategy',
            'step_number': 1,
            'feature_set_configs' : [feature_set_config],
        }

        feature_set_strategy = CreateFeatureSetsStrategy(feature_set_strategy_config)
        model_sets = feature_set_strategy.apply(self.model_sets)

        config = {
            'step_number': 1,
            'm_service': 'preprocessing_manager',
            'parent_strategy': 'ModelSetsStrategy',
            'type': 'ScaleByFeaturesStrategy',
            'X_feature_dict': {'open': 0, 'high': 1},
            'y_feature_dict': {'close': 0}
        }

        strategy = ScaleByFeaturesStrategy(config)
        X_train_orig = deepcopy(model_sets[0].X_train)
        X_test_orig = deepcopy(model_sets[0].X_test)

        model_sets = strategy.apply(model_sets)

        model_set = model_sets[0]
        scaler = model_set.X_feature_sets[0].scaler

        X_train_scaled = scaler.fit_transform(X_train_orig)
        X_test_scaled = scaler.transform(X_test_orig)

        np.testing.assert_almost_equal(model_set.X_train_scaled, X_train_scaled)
        np.testing.assert_almost_equal(model_set.X_test_scaled, X_test_scaled)

    def test_applyYFeatureType1FeatureSet(self):
        feature_set_config = {
            'scaler_config':{
                'scaler_name': 'TIME_STEP_SCALER_3D'
            },
            'feature_list': ['close+1'],
            'do_fit_test' : False,
            'feature_set_type': 'y',
        }

        feature_set_strategy_config = {
            'm_service': 'training_session',
            'type': 'FeatureSetStrategy',
            'parent_strategy': 'ModelSetsStrategy',
            'step_number': 1,
            'feature_set_configs' : [feature_set_config]
        }

        feature_set_strategy = CreateFeatureSetsStrategy(feature_set_strategy_config)
        model_sets = feature_set_strategy.apply(self.model_sets)

        config = {
            'step_number': 1,
            'parent_strategy': 'ModelSetsStrategy',
            'm_service': 'preprocessing_manager',
            'type': 'ScaleByFeaturesStrategy',
            'X_feature_dict': {'open': 0, 'high': 1},
            'y_feature_dict': {'close+1': 0}
        }

        strategy = ScaleByFeaturesStrategy(config)
        y_train_orig = deepcopy(model_sets[0].y_train)
        y_test_orig = deepcopy(model_sets[0].y_test)

        model_sets = strategy.apply(model_sets)

        model_set = model_sets[0]
        scaler = model_set.y_feature_sets[0].scaler

        y_train_scaled = scaler.fit_transform(y_train_orig)
        y_test_scaled = scaler.transform(y_test_orig)

        np.testing.assert_almost_equal(model_set.y_train_scaled, y_train_scaled)
        np.testing.assert_almost_equal(model_set.y_test_scaled, y_test_scaled)


    def test_applyXFeatureType2FeatureSets(self):
        feauture_set_config1 = {
            'scaler_config':{
                'scaler_name': 'MEAN_VARIANCE_SCALER_3D'
            },
            'feature_list': ['open'],
            'do_fit_test' : False,
            'feature_set_type': 'X',
        }
        feauture_set_config2 = {
            'scaler_config':{
                'scaler_name': 'MEAN_VARIANCE_SCALER_3D'
            },
            'feature_list': ['high'],
            'do_fit_test' : False,
            'feature_set_type': 'X',
        }

        feature_set_strategy_config = {
            'm_service': 'training_session',
            'parent_strategy': 'ModelSetsStrategy',
            'type': 'FeatureSetStrategy',
            'step_number': 1,
            'feature_set_configs' : [feauture_set_config1, feauture_set_config2]
        }

        feature_set_strategy = CreateFeatureSetsStrategy(feature_set_strategy_config)
        model_sets = feature_set_strategy.apply(self.model_sets)

        config = {
            'step_number': 1,
            'm_service': 'preprocessing_manager',
            'parent_strategy': 'ModelSetsStrategy',
            'type': 'ScaleByFeaturesStrategy',
            'X_feature_dict': {'open': 0, 'high': 1},
            'y_feature_dict': {'close+1': 0}
        }

        strategy = ScaleByFeaturesStrategy(config)
        X_train_orig = deepcopy(model_sets[0].X_train)
        X_test_orig = deepcopy(model_sets[0].X_test)

        model_sets = strategy.apply(model_sets)

        model_set = model_sets[0]
        scaler1 = model_set.X_feature_sets[0].scaler
        scaler2 = model_set.X_feature_sets[1].scaler

        X_train_scaled1 = scaler1.fit_transform(X_train_orig[:, :, 0:1])
        X_train_scaled2 = scaler2.fit_transform(X_train_orig[:, :, 1:2])

        X_test_scaled1 = scaler1.transform(X_test_orig[:, :, 0:1])
        X_test_scaled2 = scaler2.transform(X_test_orig[:, :, 1:2])

        np.testing.assert_almost_equal(model_set.X_train_scaled[:, :, 0:1], X_train_scaled1)
        np.testing.assert_almost_equal(model_set.X_train_scaled[:, :, 1:2], X_train_scaled2)
        np.testing.assert_almost_equal(model_set.X_test_scaled[:, :, 0:1], X_test_scaled1)
        np.testing.assert_almost_equal(model_set.X_test_scaled[:, :, 1:2], X_test_scaled2)


    def test_applyXyFeatureType1FeatureSet(self):
        # Prepare feature set configuration
        feature_set_config = {
            'scaler_config': {
                'scaler_name': 'MIN_MAX_SEQ_BY_SEQ_2D'
            },
            'feature_list': ['open', 'high', 'close+1'],
            'secondary_feature_list': ['close+1'],
            'do_fit_test': False,
            'feature_set_type': 'Xy',
        }

        # Prepare feature set strategy configuration
        feature_set_strategy_config = {
            'm_service': 'training_session',
            'type': 'FeatureSetStrategy',
            'parent_strategy': 'ModelSetsStrategy',
            'step_number': 1,
            'feature_set_configs': [feature_set_config]
        }

        # Create feature sets strategy and apply it to the model sets
        feature_set_strategy = CreateFeatureSetsStrategy(feature_set_strategy_config)
        model_sets = feature_set_strategy.apply(self.model_sets)

        # Prepare the configuration for ScaleByFeaturesStrategy
        config = {
            'step_number': 1,
            'parent_strategy': 'ModelSetsStrategy',
            'm_service': 'preprocessing_manager',
            'type': 'ScaleByFeaturesStrategy',
            'X_feature_dict': {'open': 0, 'high': 1},
            'y_feature_dict': {'close+1': 0}
        }

        # Create the scaling strategy
        strategy = ScaleByFeaturesStrategy(config)

        # Deep copy the original data for comparison
        X_test_orig = deepcopy(model_sets[0].X_test)

        # Apply the scaling strategy to the model sets
        model_sets = strategy.apply(model_sets)

        # Retrieve the scaled data
        model_set = model_sets[0]
        X_test_scaled = model_set.X_test_scaled

        # Retrieve the scaler
        scaler = model_set.Xy_feature_sets[0].scaler

        # Verify that the shape is unchanged
        self.assertEqual(X_test_scaled.shape, X_test_orig.shape)

        # Prepare feature indices
        arr1_feature_dict = config['X_feature_dict']
        feature_set = model_set.Xy_feature_sets[0]
        arr1_feature_indices = [
            arr1_feature_dict[feature]
            for feature in feature_set.feature_list
            if feature in arr1_feature_dict
        ]

        # Extract features
        arr1_features = X_test_orig[:, :, arr1_feature_indices]  # Shape: (samples, time_steps, features)
        arr1_scaled_features = X_test_scaled[:, :, arr1_feature_indices]

        arr1_scaled_features_reshaped = arr1_scaled_features.reshape(arr1_scaled_features.shape[0], -1)

        # Check inverse transformation for arr1
        arr1_inverse_flat = scaler.inverse_transform(arr1_scaled_features_reshaped)
        arr1_inverse = arr1_inverse_flat.reshape(arr1_features.shape)

        # Verify that the inverse transformed data matches the original
        np.testing.assert_array_almost_equal(arr1_features, arr1_inverse, decimal=6)


class CombineDataSetsStrategyTestCase(TestCase):
    def setUp(self):
        model_set1 = ModelSet()
        model_set1.X_train_scaled = np.array([[[1, 2], [2, 1.5]], [[2, 1.5], [3, 2.5]], [[3, 2.5], [4, 4]], [[4, 4], [5, 5]]])
        model_set1.X_test_scaled = np.array([[[4, 4], [5, 5]]])

        model_set1.y_train_scaled = np.array([[[3],[4]], [[4],[5]], [[5],[6]],[[6],[7]]])
        model_set1.y_test_scaled = np.array([[[6],[7]]])

        model_set1.train_row_ids = [1, 2, 3, 4]
        model_set1.test_row_ids = [9]

        self.model_sets = [model_set1]

        model_set2 = ModelSet()
        model_set2.X_train_scaled = np.array([[[1, 2], [2, 1.5]], [[2, 1.5], [3, 2.5]], [[3, 2.5], [4, 4]], [[4, 4], [5, 5]]])
        model_set2.X_test_scaled = np.array([[[4, 4], [5, 5]]])

        model_set2.y_train_scaled = np.array([[[3],[4]], [[4],[5]], [[5],[6]],[[6],[7]]])
        model_set2.y_test_scaled = np.array([[[6],[7]]])

        model_set2.train_row_ids = [5, 6, 7, 8]
        model_set2.test_row_ids = [10]


        self.model_sets.append(model_set2)

    def test_apply(self):
        config = {
            'parent_strategy': 'ModelSetsStrategy',
            'step_number': 1,
            'm_service' : 'preprocessing_manager',
            'type': 'CombineDataSetsStrategy',
        }
        strategy = CombineDataSetsStrategy(config)
        X_train, X_test, y_train, y_test, train_row_ids, test_row_ids = strategy.apply(self.model_sets)

        self.assertEqual(X_train.shape, (8, 2, 2))
        self.assertEqual(X_test.shape, (2, 2, 2))
        self.assertEqual(y_train.shape, (8, 2, 1))
        self.assertEqual(y_test.shape, (2, 2, 1))

        expected_X_train = np.array([[[1, 2], [2, 1.5]], [[2, 1.5], [3, 2.5]], [[3, 2.5], [4, 4]], [[4, 4], [5, 5]], [[1, 2], [2, 1.5]], [[2, 1.5], [3, 2.5]], [[3, 2.5], [4, 4]], [[4, 4], [5, 5]]])
        expected_y_train = np.array([[[3],[4]], [[4],[5]], [[5],[6]],[[6],[7]], [[3],[4]], [[4],[5]], [[5],[6]],[[6],[7]]])
        expected_X_test = np.array([[[4, 4], [5, 5]], [[4, 4], [5, 5]]])
        expected_y_test = np.array([[[6],[7]], [[6],[7]]])

        np.testing.assert_almost_equal(X_train, expected_X_train)
        np.testing.assert_almost_equal(y_train, expected_y_train)
        np.testing.assert_almost_equal(X_test, expected_X_test)
        np.testing.assert_almost_equal(y_test, expected_y_test)

        expected_train_row_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        expected_test_row_ids = np.array([9, 10])

        np.testing.assert_almost_equal(train_row_ids, expected_train_row_ids)









