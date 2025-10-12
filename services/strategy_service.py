
from database.strategy_db import get_strategy, get_active_user_strategies
from restx_api.schemas import StrategySchema


def serialize_strategy(strategy):
    schema = StrategySchema()
    return schema.dump(strategy)

def get_strategy_by_id(strategy_id):
    strategy = get_strategy(strategy_id)
    return serialize_strategy(strategy)

def get_active_strategies_for_user(user_id):
    strategies = get_active_user_strategies(user_id)
    return [serialize_strategy(s) for s in strategies]
