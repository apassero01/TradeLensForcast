from django.db import models

class FeatureSet:
    def __init__(self, scaler_config, feature_list, do_fit_test, secondary_feature_list=None):
        self.scaler_config = scaler_config
        self.feature_list = feature_list
        self.secondary_feature_list = secondary_feature_list
        self.do_fit_test = do_fit_test

class TrainingSession(models.Model):
    X_features = models.JSONField()
    y_features = models.JSONField()
    X_feature_dict = models.JSONField()
    y_feature_dict = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.IntegerField()
    ordered_model_set_strategies = models.JSONField()
    model_set_configs = models.JSONField(default=list)
    start_date = models.DateField(null=True)
    end_date = models.DateField(null=True)


class ModelSet():
    X_features = None
    y_features = None
    X_feature_dict = None
    y_feature_dict = None
    X = None
    y = None
    data_set = None
    row_ids = None
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    X_train_scaled = None
    X_test_scaled = None
    y_train_scaled = None
    y_test_scaled = None
    train_row_ids = None
    test_row_ids = None