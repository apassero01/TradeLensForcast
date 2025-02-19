from shared_utils.entities.EnityEnum import EntityEnum
from importlib import import_module

class StrategyDirectory:
    def __init__(self):
        self.directory = {
            EntityEnum.ENTITY.value: [
                "shared_utils.strategy.BaseStrategy.AssignAttributesStrategy",
                "shared_utils.strategy.BaseStrategy.CreateEntityStrategy",
                "shared_utils.strategy.BaseStrategy.RemoveEntityStrategy",
                "shared_utils.strategy.BaseStrategy.MergeEntitiesStrategy",
                "shared_utils.strategy.BaseStrategy.ClusterStrategy",
                "shared_utils.strategy.BaseStrategy.RetreiveSequencesStrategy",
                "shared_utils.strategy.BaseStrategy.ExecuteCodeStrategy",
                "shared_utils.strategy.BaseStrategy.GetAttributesStrategy",
                "shared_utils.strategy.BaseStrategy.RemoveChildStrategy",
                "shared_utils.strategy.BaseStrategy.GetEntityStrategy",
                "shared_utils.strategy.BaseStrategy.AddChildStrategy",
                "data_bundle_manager.strategy.DataBundleStrategy.InverseScaleByFeatureSetsStrategy",
            ],
            EntityEnum.TRAINING_SESSION.value: [
                "training_session.strategy.TrainingSessionStrategy.GetSequenceSetsStrategy",
            ],
            EntityEnum.DATA_BUNDLE.value: [
                "data_bundle_manager.strategy.DataBundleStrategy.CreateFeatureSetsStrategy",
                "data_bundle_manager.strategy.DataBundleStrategy.SplitBundleDateStrategy",
                "data_bundle_manager.strategy.DataBundleStrategy.ScaleByFeatureSetsStrategy",
                "data_bundle_manager.strategy.DataBundleStrategy.CombineDataBundlesStrategy",
            ],
            EntityEnum.SEQUENCE_SET.value: [
                "sequenceset_manager.strategy.SequenceSetStrategy.PopulateDataBundleStrategy",
            ],
            EntityEnum.VISUALIZATION.value: [
                "shared_utils.strategy.VisualizationStrategy.HistogramStrategy",
                "shared_utils.strategy.VisualizationStrategy.LineGraphStrategy",
                "shared_utils.strategy.VisualizationStrategy.VisualizationStrategy",
            ],
            EntityEnum.MODEL_STAGE.value: [
                "model_stage.strategy.ModelStageStrategy.CreateModelStrategy",
                "model_stage.strategy.ModelStageStrategy.ConfigureModelStageStrategy",
                "model_stage.strategy.ModelStageStrategy.FitModelStrategy",
                "model_stage.strategy.ModelStageStrategy.EvaluateModelStrategy",
                "model_stage.strategy.ModelStageStrategy.PredictModelStrategy",
                "model_stage.strategy.ModelStageStrategy.ComparePredictionsStrategy",
            ],
            EntityEnum.DOCUMENT.value: [
                "shared_utils.entities.document_entities.strategy.FileTreeStrategies.ScrapeFilePathStrategy",
                "shared_utils.entities.document_entities.strategy.FileTreeStrategies.RecursiveFileScrapeStrategy",
            ],  
            EntityEnum.API_MODEL.value: [
                "shared_utils.entities.api_model.strategy.ApiModelStrategy.ConfigureApiModelStrategy",
                "shared_utils.entities.api_model.strategy.ApiModelStrategy.CallApiModelStrategy",
                "shared_utils.entities.api_model.strategy.ApiModelStrategy.CallDeepSeekApiStrategy",
                "shared_utils.entities.api_model.strategy.ApiModelStrategy.GenerateEmbeddingsStrategy",
            ],
            EntityEnum.STRATEGY_REQUEST.value: [
            ],
        }

    def get_strategy_classes(self):
        """
        Lazily load strategy classes and return them in the same format as the directory.
        """
        strategies_dict = {}
        for entity_type, strategy_paths in self.directory.items():
            strategies = []
            for strategy_path in strategy_paths:
                module_path, class_name = strategy_path.rsplit('.', 1)
                module = import_module(module_path)
                strategies.append(getattr(module, class_name))
            strategies_dict[entity_type] = strategies
        return strategies_dict