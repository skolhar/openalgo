from enum import unique
from json import loads
from zoneinfo import ZoneInfo
from flask_restx import Namespace, Resource
from flask import request, jsonify, make_response
from marshmallow import ValidationError
import time
from datetime import datetime, timedelta, time as dtime
import numpy as np
import pandas as pd
from py_vollib_vectorized import price_dataframe, vectorized_black_scholes
from blueprints import strategy
from broker.pocketful.api.pocketfulwebsocket import on_open
from database.auth_db import verify_api_key
from services.strategy_service import get_strategy_by_id, get_active_strategies_for_user
from services.quotes_service import get_quotes
from services.websocket_service import (
    get_websocket_status,
    subscribe_to_symbols,
    unsubscribe_from_symbols,
    get_market_data
)
from services.market_data_service import (
    get_market_data_service,
    get_ltp,
    get_quote,
    subscribe_to_market_updates
)

from .account_schema import ActiveStrategiesSchema, StrategyDetailsRequestSchema
from restx_api.schemas import StrategySchema, StrategyDetailsResponseSchema
from utils.logging import get_logger

api = Namespace('strategy', description='Strategy API')

# Initialize logger
logger = get_logger(__name__)

# Initialize schema
activestrategy_schema = ActiveStrategiesSchema()
strategydetailsrequest_schema = StrategyDetailsRequestSchema()

# Initialize strategies_df
strategies_df = None

def refresh_strategies_df(data):
    logger.info("Refreshing strategies_df with latest market data")
    global strategies_df
    
    if strategies_df is None:
        return

    if data.get('type') != 'market_data':
        return

    logger.info("Updating strategies_df with new market data")
    # Update ltp for all symbols in strategies_df
    strategies_df['spot'] = strategies_df['index'].apply(lambda index: get_ltp(symbol=index, exchange='NSE_INDEX')[0])
    strategies_df['ltp'] = strategies_df['symbol'].apply(lambda symbol: get_ltp(symbol=symbol, exchange='NFO')[0])
    price_dataframe(strategies_df, flag_col='type', underlying_price_col='spot',  price_col='ltp',
                        strike_col='strike', annualized_tte_col='expiry', riskfree_rate_col='riskfree_rate', inplace=True)

def update_strategies_df(api_key, strategies=None):
    global strategies_df

    user_id = verify_api_key(api_key)
    if strategies is None:
        strategies = get_active_strategies_for_user(user_id)

    # create a dataframe with all strategies symbol_mappings as rows
    strategies_df = pd.DataFrame(strategies).explode('symbol_mappings')
    strategies_df = strategies_df['symbol_mappings'].apply(pd.Series)
    # strategies_df['id'] = strategies_df['id'].astype(str) + '.' + strategies_df['strategy_id'].astype(str)

    # Split symbol into index, expiry, strike, type using regex
    symbol_pattern = r'^(?P<index>[A-Z]+)(?P<expiry>\d{2}[A-Z]{3}\d{2})(?P<strike>\d+)(?P<type>[A-Z]+)$'
    symbol_parts = strategies_df['symbol'].str.extract(symbol_pattern)
    strategies_df = pd.concat([strategies_df, symbol_parts], axis=1)

    strategies_df['type'] = strategies_df['type'].map(lambda x: 'p' if str(x).lower() == 'pe' else ('c' if str(x).lower() == 'ce' else str(x).lower()))
    strategies_df['expiry'] = strategies_df['expiry'].apply(expiry_to_tte)
    strategies_df['riskfree_rate'] = 0.07

    strategies_df['spot'] = strategies_df['index'].apply(lambda index: get_quotes(symbol=index, exchange='NSE_INDEX', api_key=api_key)[1].get('data', {}).get('ltp', 0.0))
    strategies_df['ltp'] = strategies_df['symbol'].apply(lambda symbol: get_quotes(symbol=symbol, exchange='NFO', api_key=api_key)[1].get('data', {}).get('ltp', 0.0))

    # Call price_dataframe to compute IV and Greeks
    price_dataframe(strategies_df, flag_col='type', underlying_price_col='spot',  price_col='ltp',
                        strike_col='strike', annualized_tte_col='expiry', riskfree_rate_col='riskfree_rate', inplace=True)
    
    symbols = [{"symbol": index, "exchange": "NSE_INDEX"} for index in strategies_df['index'].unique().tolist()]
    symbol_keys = ("NSE_INDEX:"+strategies_df['index']).unique().tolist()
    symbol_keys.extend(("NFO:"+strategies_df['symbol']).unique().tolist())
    subscriber_id = subscribe_to_market_updates('ltp', refresh_strategies_df, symbol_keys)
    # Register user callback to start receiving data
    market_service = get_market_data_service()
    market_service.register_user_callback(user_id)
   

# Convert 'expiry' column from 'DDMMMYY' to annualized fraction to 3:30pm on expiry date
def expiry_to_tte(expiry):
    try:
        # Parse expiry string (e.g., '14OCT25')
        expiry_dt = datetime.strptime(expiry, '%d%b%y')
        expiry_dt = expiry_dt.replace(hour=15, minute=30, second=0)
        now = datetime.now()
        # If expiry is in the past, return 0
        if expiry_dt < now:
            return 0.0
        # Calculate fraction of year
        year_frac = (expiry_dt - now).total_seconds() / ((365.0 - 104.0) * 24 * 3600)
        return np.round(year_frac, 6)
    except Exception:
        return None
    
def expiry_pnl(x, strike, option_type):
    if option_type == 'c':
        return np.maximum(0, x - strike)
    elif option_type == 'p':
        return np.maximum(0, strike - x)
    else:
        return np.zeros_like(x)

@api.route('/active', strict_slashes=False)
class ActiveStrategies(Resource):
    def post(self):
        """Get all active strategies for a user"""
        try:
            # Validate request data
            activestrategy_data = activestrategy_schema.load(request.json)

            api_key = activestrategy_data['apikey']
            user_id = verify_api_key(api_key)
            if not user_id:
                return {"error": "Missing user_id parameter"}, 400

            strategies = get_active_strategies_for_user(user_id)

            global strategies_df
            if strategies_df is None:
                update_strategies_df(api_key, strategies)

            for s in strategies:
                s['index_spot'] = strategies_df[strategies_df['strategy_id'] == s['id']]['spot'].iloc[0]
                s['symbol_mappings'] = loads(strategies_df[strategies_df['strategy_id'] == s['id']].to_json(orient='records'))

            schema = StrategySchema(many=True)
            return {"strategies": schema.dump(strategies)}, 200

        except ValidationError as err:
            return make_response(jsonify({
                'status': 'error',
                'message': err.messages
            }), 400)
        except Exception as e:
            logger.exception(f"Unexpected error in positionbook endpoint: {e}")
            return make_response(jsonify({
                'status': 'error',
                'message': 'An unexpected error occurred'
            }), 500)
        
@api.route('/details', strict_slashes=False)
class StrategyDetails(Resource):
    def post(self):
        """Get strategy details by strategy ID"""
        try:
            # Validate request data
            strategydetails_data = strategydetailsrequest_schema.load(request.json)

            api_key = strategydetails_data['apikey']
            strategy_id = strategydetails_data['strategy_id']

            user_id = verify_api_key(api_key)
            if not user_id:
                return {"error": "Missing user_id parameter"}, 400

            strategy = get_strategy_by_id(strategy_id)
            
            global strategies_df
            if strategies_df is None:
                strategies = get_active_strategies_for_user(user_id)
                update_strategies_df(api_key, strategies)

            symbols_df = strategies_df[strategies_df['strategy_id'] == strategy['id']]

            strategy['index_spot'] = symbols_df['spot'].iloc[0]
            strategy['symbol_mappings'] = loads(symbols_df.to_json(orient='records'))

            lower_bound = min(symbols_df['strike'].astype(int)) - 500
            upper_bound = max(symbols_df['strike'].astype(int)) + 600

            x_range = np.arange(lower_bound, upper_bound, 100)
            on_expiry = np.zeros_like(x_range, dtype='float64')
            on_open = np.zeros_like(x_range, dtype='float64')
        
            next_mkt_open = datetime.now(tz=ZoneInfo('Asia/Kolkata')).replace(hour=9, minute=15, second=0, microsecond=0) + timedelta(hours=24)
            open_tte = (next_mkt_open - datetime.now(tz=ZoneInfo('Asia/Kolkata'))).total_seconds() / ((365.0 - 104.0) * 24 * 3600)

            strategy_expiry = symbols_df['expiry'].iloc[0]
            for pos in strategy['symbol_mappings']:
                if pos['expiry'] == strategy_expiry:
                    on_expiry += (expiry_pnl(x_range, int(pos['strike']), pos['type']) - float(pos['price'])) * int(pos['quantity'])
                else:
                    expiry_tte = expiry_to_tte(pos['expiry'])
                    expiry_price = vectorized_black_scholes(pos['type'], x_range, int(pos['strike']), expiry_tte,
                                                                    float(pos['riskfree_rate']), float(pos['IV']), return_as='numpy')
                    on_expiry += (expiry_price - float(pos['price'])) * int(pos['quantity'])

                open_price = vectorized_black_scholes(pos['type'], x_range, int(pos['strike']), open_tte,
                                                          float(pos['riskfree_rate']), float(pos['IV']), return_as='numpy')
                on_open += (open_price - float(pos['price'])) * int(pos['quantity'])

            on_expiry += strategy['realized_pnl']
            on_open += strategy['realized_pnl']

            strategy['breakeven_points'] = []
            for i in range(1, len(on_expiry)):
                if on_expiry[i-1] * on_expiry[i] < 0:
                    # Linear interpolation for more accurate breakpoint
                    x0, x1 = x_range[i-1], x_range[i]
                    y0, y1 = on_expiry[i-1], on_expiry[i]
                    x_break = x0 - y0 * (x1 - x0) / (y1 - y0)
                    strategy['breakeven_points'].append(x_break)

            strategy['plot_data'] = [{'index_value': int(x), 'on_expiry': float(np.round(y_exp, 2)), 'on_open': float(np.round(y_open, 2))} for x, y_exp, y_open in zip(x_range, on_expiry, on_open)]

            schema = StrategyDetailsResponseSchema()
            return {"details": schema.dump(strategy)}, 200

        except ValidationError as err:
            return make_response(jsonify({
                'status': 'error',
                'message': err.messages
            }), 400)
        except Exception as e:
            logger.exception(f"Unexpected error in positionbook endpoint: {e}")
            return make_response(jsonify({
                'status': 'error',
                'message': 'An unexpected error occurred'
            }), 500)
