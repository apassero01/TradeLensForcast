import numpy as np
import pandas as pd

from data_bundle_manager.entities.DataBundleEntity import DataBundleEntity
from data_bundle_manager.entities.services.FeatureSetEntityService import FeatureSetEntityService
from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.strategy.BaseStrategy import Strategy


class DataBundleStrategy(Strategy):
    def __init__(self, strategy_executor, strategy_request):
        super().__init__(strategy_executor, strategy_request)

    def apply(self, data_bundle):
        NotImplementedError("Child classes must implement the 'apply' method.")

    def verify_executable(self, entity, strategy_request):
        raise NotImplementedError("Child classes must implement the 'verify_executable' method.")

    @staticmethod
    def get_request_config():
        return {}


class LoadDataBundleDataStrategy(DataBundleStrategy):
    name = "LoadDataBundleData"

    def __init__(self, strategy_executor, strategy_request):
        super().__init__(strategy_executor, strategy_request)
        self.strategy_request = strategy_request
        self.strategy_executor = strategy_executor

    def apply(self, data_bundle):
        param_config = self.strategy_request.param_config
        data_bundle.set_dataset(param_config)

    def verify_executable(self, entity, strategy_request):
        raise NotImplementedError("Child classes must implement the 'verify_executable' method.")

    @staticmethod
    def get_request_config():
        return {
            'strategy_name': LoadDataBundleDataStrategy.__name__,
            'strategy_path': None,
            'param_config': {}
        }

class CreateFeatureSetsStrategy(DataBundleStrategy):
    name = "CreateFeatureSets"

    def __init__(self, strategy_executor, strategy_request):
        super().__init__(strategy_executor, strategy_request)
        self.feature_set_entity_service = FeatureSetEntityService()

    def apply(self, data_bundle):
        param_config = self.strategy_request.param_config
        feature_sets = []
        for config in param_config['feature_set_configs']:
            feature_set = self.feature_set_entity_service.create_feature_set(config)
            feature_sets.append(feature_set)

        data_bundle.set_entity_map({EntityEnum.FEATURE_SETS.value: feature_sets})


    def verify_executable(self, entity, strategy_request):
        param_config = strategy_request.param_config
        if 'feature_set_configs' not in param_config.keys():
            raise ValueError("Missing feature_set_configs in config")
        for feature_set_config in param_config['feature_set_configs']:
            if 'feature_set_type' not in feature_set_config.keys():
                raise ValueError("Missing feature_set_type in feature_set_config")
            if 'feature_list' not in feature_set_config.keys():
                raise ValueError("Missing feature_list in feature_set_config")
            if 'scaler_config' not in feature_set_config.keys():
                raise ValueError("Missing scaler_config in feature_set_config")
            if 'do_fit_test' not in feature_set_config.keys():
                raise ValueError("Missing do_fit_test in feature_set_config")

    @staticmethod
    def get_request_config():
        return {
            'strategy_name': CreateFeatureSetsStrategy.__name__,
            'strategy_path': None,
            'param_config': {
                'feature_set_configs': [
                    {
                        'scaler_config': {
                            'scaler_name': 'MEAN_VARIANCE_SCALER_3D'
                        },
                        'feature_list': [],
                        'feature_set_type': 'X',
                        'do_fit_test': False,
                        'secondary_feature_list': None

                    }
                ]
            }
        }


class SplitBundleDateStrategy(DataBundleStrategy):
    name = "SplitBundleDate"
    def __init__(self, strategy_executor, strategy_request):
        super().__init__(strategy_executor, strategy_request)

    def apply(self, data_bundle):
        self.verify_executable(data_bundle, self.strategy_request)
        param_config = self.strategy_request.param_config
        X_train, X_test, y_train, y_test, train_row_ids, test_row_ids = self.train_test_split(data_bundle, param_config['split_date'], param_config['date_list'])
        data_bundle.set_dataset({'X_train': X_train,
                                 'X_test': X_test,
                                 'y_train': y_train,
                                 'y_test': y_test,
                                 'train_row_ids': train_row_ids,
                                 'test_row_ids': test_row_ids})

    def train_test_split(self, data_bundle, split_date, date_list):
        X = data_bundle.dataset['X']
        y = data_bundle.dataset['y']
        row_ids = data_bundle.dataset['row_ids']

        split_date = pd.to_datetime(split_date).tz_localize(None)
        date_list = [pd.to_datetime(date).tz_localize(None) for date in date_list]

        if split_date not in date_list:
            split_date = min(date_list, key=lambda x: abs(x - split_date))

        if len(date_list) != len(X):
            raise ValueError("Dates and X must be the same")

        split_index = date_list.index(split_date)

        X_train, X_test = X[:split_index], X[split_index:]
        train_row_ids = row_ids[:split_index]
        test_row_ids = row_ids[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        return X_train, X_test, y_train, y_test, train_row_ids, test_row_ids

    def verify_executable(self, entity, strategy_request):
        param_config = strategy_request.param_config
        if 'split_date' not in param_config.keys():
            raise ValueError("Missing split_date in config")
        if 'date_list' not in param_config.keys():
            raise ValueError("Missing date_list in config")
        if 'X' not in entity.dataset.keys():
            raise ValueError("Missing X in dataset")
        if 'y' not in entity.dataset.keys():
            raise ValueError("Missing y in dataset")
        if 'row_ids' not in entity.dataset.keys():
            raise ValueError("Missing row_ids in dataset")

    @staticmethod
    def get_request_config():
        return {
            'strategy_name': SplitBundleDateStrategy.__name__,
            'strategy_path': None,
            'param_config': {
                'split_date': None,
                'date_list': None
            }
        }


class ScaleByFeatureSetsStrategy(DataBundleStrategy):
    name = "ScaleByFeatureSets"
    def __init__(self, strategy_executor, strategy_request):
        super().__init__(strategy_executor, strategy_request)

    def apply(self, data_bundle):
        self.verify_executable(data_bundle, self.strategy_request)
        param_config = self.strategy_request.param_config
        X_feature_dict = data_bundle.dataset['X_feature_dict']
        y_feature_dict = data_bundle.dataset['y_feature_dict']
        feature_sets = data_bundle.get_entity(EntityEnum.FEATURE_SETS.value)

        X_feature_sets = [feature_set for feature_set in feature_sets if feature_set.feature_set_type == 'X']
        y_feature_sets = [feature_set for feature_set in feature_sets if feature_set.feature_set_type == 'y']
        Xy_feature_sets = [feature_set for feature_set in feature_sets if feature_set.feature_set_type == 'Xy']

        if len(X_feature_sets) > 0:
            data_bundle.dataset['X_train_scaled'], data_bundle.dataset['X_test_scaled'] = self.scale_X_by_features(X_feature_sets, data_bundle.dataset['X_train'], data_bundle.dataset['X_test'], X_feature_dict)
        if len(y_feature_sets) > 0:
            data_bundle.dataset['y_train_scaled'], data_bundle.dataset['y_test_scaled'] = self.scale_y_by_features(y_feature_sets, data_bundle.dataset['y_train'], data_bundle.dataset['y_test'], y_feature_dict)
        if len(Xy_feature_sets) > 0:
            data_bundle.dataset['X_train_scaled'], data_bundle.dataset['y_train_scaled'] = self.scale_Xy_by_features(Xy_feature_sets, data_bundle.dataset['X_train'], data_bundle.dataset['y_train'], X_feature_dict, y_feature_dict)
            data_bundle.dataset['X_test_scaled'], data_bundle.dataset['y_test_scaled'] = self.scale_Xy_by_features(Xy_feature_sets, data_bundle.dataset['X_test'], data_bundle.dataset['y_test'], X_feature_dict, y_feature_dict)

    def scale_X_by_features(self, feature_sets, arr1, arr2, arr1_feature_dict):
        arr1_scaled = np.zeros(arr1.shape)
        arr2_scaled = np.zeros(arr2.shape)

        for feature_set in feature_sets:
            do_fit_test = feature_set.do_fit_test
            scaler = feature_set.scaler
            arr1_feature_indices = [arr1_feature_dict[feature] for feature in feature_set.feature_list]
            arr1_scaled[:, :, arr1_feature_indices] = scaler.fit_transform(arr1[:, :, arr1_feature_indices])

            if do_fit_test:
                arr2_scaled[:, :, arr1_feature_indices] = scaler.fit_transform(arr2[:, :, arr1_feature_indices])
            else:
                arr2_scaled[:, :, arr1_feature_indices] = scaler.transform(arr2[:, :, arr1_feature_indices])

        return arr1_scaled, arr2_scaled

    def scale_y_by_features(self, feature_sets, arr1, arr2, arr1_feature_dict):
        '''
        Scale the y_feature sets. In practice, we only have a single y_feature_set so here we do not filter features by feature_dict
        but in the future, if for some reason we had y with different valued we would need to rework this.
        '''

        arr1_scaled = np.zeros(arr1.shape)
        arr2_scaled = np.zeros(arr2.shape)

        for feature_set in feature_sets:
            do_fit_test = feature_set.do_fit_test
            scaler = feature_set.scaler
            # Irrelivent for y features
            arr1_feature_indices = [arr1_feature_dict[feature] for feature in feature_set.feature_list]
            arr1_scaled = scaler.fit_transform(arr1)

            if do_fit_test:
                arr2_scaled = scaler.fit_transform(arr2)
            else:
                arr2_scaled = scaler.transform(arr2)

        return arr1_scaled, arr2_scaled

    def scale_Xy_by_features(self, feature_sets, arr1, arr2, arr1_feature_dict, arr2_feature_dict):
        arr1_scaled = np.copy(arr1)
        arr2_scaled = np.copy(arr2)

        for feature_set in feature_sets:
            do_fit_test = feature_set.do_fit_test
            scaler = feature_set.scaler

            # Extract feature indices for arr1 and arr2
            arr1_feature_indices = [
                arr1_feature_dict[feature]
                for feature in feature_set.feature_list
                if feature in arr1_feature_dict
            ]
            arr2_feature_indices = [
                arr2_feature_dict[feature]
                for feature in feature_set.feature_list
                if feature in arr2_feature_dict
            ]

            # Extract features from arr1 and arr2
            arr1_features = arr1[:, :, arr1_feature_indices]  # Shape: (samples, time_steps, features)
            arr2_features = arr2[:, arr2_feature_indices, :]  # Shape: (samples, features, time_steps)

            # Flatten arr1 and arr2 to 2D arrays
            arr1_reshaped = arr1_features.reshape(arr1_features.shape[0], -1)
            arr2_reshaped = arr2_features.reshape(arr2_features.shape[0], -1)

            # Fit and transform arr1, and transform arr2
            arr1_scaled_flat = scaler.fit_transform(arr1_reshaped)
            arr2_scaled_flat = scaler.transform(arr2_reshaped)

            # Reshape scaled data back to original 3D shapes
            arr1_scaled_features = arr1_scaled_flat.reshape(arr1_features.shape)
            arr2_scaled_features = arr2_scaled_flat.reshape(arr2_features.shape)

            # Update scaled arrays
            arr1_scaled[:, :, arr1_feature_indices] = arr1_scaled_features
            arr2_scaled[:, arr2_feature_indices, :] = arr2_scaled_features

        return arr1_scaled, arr2_scaled


    def verify_executable(self, entity, strategy_request):
        param_config = strategy_request.param_config
        if 'X_feature_dict' not in entity.dataset.keys():
            raise ValueError("Missing X_feature_dict in dataset")
        if 'y_feature_dict' not in entity.dataset.keys():
            raise ValueError("Missing y_feature_dict in dataset")
        if 'X_train' not in entity.dataset.keys():
            raise ValueError("Missing X in dataset")
        if 'y_train' not in entity.dataset.keys():
            raise ValueError("Missing y in dataset")
        if 'X_test' not in entity.dataset.keys():
            raise ValueError("Missing X_test in dataset")
        if 'y_test' not in entity.dataset.keys():
            raise ValueError("Missing y_test in dataset")

        if entity.get_entity(EntityEnum.FEATURE_SETS.value) is None:
            raise ValueError("FeatureSets not found in DataBundleEntity")

    @staticmethod
    def get_request_config():
        return {
            'strategy_name': ScaleByFeatureSetsStrategy.__name__,
            'strategy_path': None,
            'param_config': {}
        }


class CombineDataBundlesStrategy(DataBundleStrategy):
    name = "CombineDataBundles"
    def __init__(self, strategy_executor, strategy_request):
        super().__init__(strategy_executor, strategy_request)

    def apply(self, data_bundles):
        self.verify_executable(data_bundles, self.strategy_request)

        # confirm they have the same keys
        keys = data_bundles[0].dataset.keys()
        for data_bundle in data_bundles[1:]:
            if data_bundle.dataset.keys() != keys:
                raise ValueError("DataBundles do not have the same keys")

        # combine the datasets
        combined_dataset = {}
        if 'X_feature_dict' in data_bundles[0].dataset.keys():
            combined_dataset['X_feature_dict'] = data_bundles[0].dataset['X_feature_dict']
        if 'y_feature_dict' in data_bundles[0].dataset.keys():
            combined_dataset['y_feature_dict'] = data_bundles[0].dataset['y_feature_dict']

        for key in keys:
            if key in ['X', 'y','row_ids', 'X_train', 'y_train', 'X_test', 'y_test']:
                combined_dataset[key] = np.concatenate([data_bundle.dataset[key] for data_bundle in data_bundles], axis=0)
            elif key in ['train_row_ids', 'test_row_ids', 'row_ids']:
                continue

        new_bundle = DataBundleEntity()
        new_bundle.set_dataset(combined_dataset)
        old_entity_map = data_bundles[0].entity_map
        new_bundle.set_entity_map(old_entity_map)

        self.strategy_request.ret_val[EntityEnum.DATA_BUNDLE.value] = new_bundle

        return self.strategy_request


    def verify_executable(self, entity, strategy_request):
        param_config = strategy_request.param_config

    @staticmethod
    def get_request_config():
        return {
            'strategy_name': CombineDataBundlesStrategy.__name__,
            'strategy_path': None,
            'param_config': {
            }
        }
