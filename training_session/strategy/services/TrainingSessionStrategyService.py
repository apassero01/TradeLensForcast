
class TrainingSessionStrategyService:

    @staticmethod
    def get_available_strategies():
        available_stragies = []
        for strategy in TrainingSessionStrategyService.registered_strategies:
            available_stragies.append({'name': strategy.name, 'config': strategy.get_request_config()})
        return available_stragies
