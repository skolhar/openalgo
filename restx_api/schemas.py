from marshmallow import Schema, fields, validate

class OrderSchema(Schema):
    apikey = fields.Str(required=True)
    strategy = fields.Str(required=True)
    exchange = fields.Str(required=True)
    symbol = fields.Str(required=True)
    action = fields.Str(required=True, validate=validate.OneOf(["BUY", "SELL", "buy", "sell"]))
    quantity = fields.Int(required=True, validate=validate.Range(min=1, error="Quantity must be a positive integer."))
    pricetype = fields.Str(missing='MARKET', validate=validate.OneOf(["MARKET", "LIMIT", "SL", "SL-M"]))
    product = fields.Str(missing='MIS', validate=validate.OneOf(["MIS", "NRML", "CNC"]))
    price = fields.Float(missing=0.0, validate=validate.Range(min=0, error="Price must be a non-negative number."))
    trigger_price = fields.Float(missing=0.0, validate=validate.Range(min=0, error="Trigger price must be a non-negative number."))
    disclosed_quantity = fields.Int(missing=0, validate=validate.Range(min=0, error="Disclosed quantity must be a non-negative integer."))

class SmartOrderSchema(Schema):
    apikey = fields.Str(required=True)
    strategy = fields.Str(required=True)
    exchange = fields.Str(required=True)
    symbol = fields.Str(required=True)
    action = fields.Str(required=True, validate=validate.OneOf(["BUY", "SELL", "buy", "sell"]))
    quantity = fields.Int(required=True, validate=validate.Range(min=0, error="Quantity must be a non-negative integer."))
    position_size = fields.Int(required=True)
    pricetype = fields.Str(missing='MARKET', validate=validate.OneOf(["MARKET", "LIMIT", "SL", "SL-M"]))
    product = fields.Str(missing='MIS', validate=validate.OneOf(["MIS", "NRML", "CNC"]))
    price = fields.Float(missing=0.0, validate=validate.Range(min=0, error="Price must be a non-negative number."))
    trigger_price = fields.Float(missing=0.0, validate=validate.Range(min=0, error="Trigger price must be a non-negative number."))
    disclosed_quantity = fields.Int(missing=0, validate=validate.Range(min=0, error="Disclosed quantity must be a non-negative integer."))

class ModifyOrderSchema(Schema):
    apikey = fields.Str(required=True)
    strategy = fields.Str(required=True)
    exchange = fields.Str(required=True)
    symbol = fields.Str(required=True)
    orderid = fields.Str(required=True)
    action = fields.Str(required=True, validate=validate.OneOf(["BUY", "SELL", "buy", "sell"]))
    product = fields.Str(required=True, validate=validate.OneOf(["MIS", "NRML", "CNC"]))
    pricetype = fields.Str(required=True, validate=validate.OneOf(["MARKET", "LIMIT", "SL", "SL-M"]))
    price = fields.Float(required=True, validate=validate.Range(min=0, error="Price must be a non-negative number."))
    quantity = fields.Int(required=True, validate=validate.Range(min=1, error="Quantity must be a positive integer."))
    disclosed_quantity = fields.Int(required=True, validate=validate.Range(min=0, error="Disclosed quantity must be a non-negative integer."))
    trigger_price = fields.Float(required=True, validate=validate.Range(min=0, error="Trigger price must be a non-negative number."))

class CancelOrderSchema(Schema):
    apikey = fields.Str(required=True)
    strategy = fields.Str(required=True)
    orderid = fields.Str(required=True)

class ClosePositionSchema(Schema):
    apikey = fields.Str(required=True)
    strategy = fields.Str(required=True)

class CancelAllOrderSchema(Schema):
    apikey = fields.Str(required=True)
    strategy = fields.Str(required=True)

class BasketOrderItemSchema(Schema):
    exchange = fields.Str(required=True)
    symbol = fields.Str(required=True)
    action = fields.Str(required=True, validate=validate.OneOf(["BUY", "SELL", "buy", "sell"]))
    quantity = fields.Int(required=True, validate=validate.Range(min=1, error="Quantity must be a positive integer."))
    pricetype = fields.Str(missing='MARKET', validate=validate.OneOf(["MARKET", "LIMIT", "SL", "SL-M"]))
    product = fields.Str(missing='MIS', validate=validate.OneOf(["MIS", "NRML", "CNC"]))
    price = fields.Float(missing=0.0, validate=validate.Range(min=0, error="Price must be a non-negative number."))
    trigger_price = fields.Float(missing=0.0, validate=validate.Range(min=0, error="Trigger price must be a non-negative number."))
    disclosed_quantity = fields.Int(missing=0, validate=validate.Range(min=0, error="Disclosed quantity must be a non-negative integer."))

class BasketOrderSchema(Schema):
    apikey = fields.Str(required=True)
    strategy = fields.Str(required=True)
    orders = fields.List(fields.Nested(BasketOrderItemSchema), required=True)  # List of order details

class SplitOrderSchema(Schema):
    apikey = fields.Str(required=True)
    strategy = fields.Str(required=True)
    exchange = fields.Str(required=True)
    symbol = fields.Str(required=True)
    action = fields.Str(required=True, validate=validate.OneOf(["BUY", "SELL", "buy", "sell"]))
    quantity = fields.Int(required=True, validate=validate.Range(min=1, error="Total quantity must be a positive integer."))  # Total quantity to split
    splitsize = fields.Int(required=True, validate=validate.Range(min=1, error="Split size must be a positive integer."))  # Size of each split
    pricetype = fields.Str(missing='MARKET', validate=validate.OneOf(["MARKET", "LIMIT", "SL", "SL-M"]))
    product = fields.Str(missing='MIS', validate=validate.OneOf(["MIS", "NRML", "CNC"]))
    price = fields.Float(missing=0.0, validate=validate.Range(min=0, error="Price must be a non-negative number."))
    trigger_price = fields.Float(missing=0.0, validate=validate.Range(min=0, error="Trigger price must be a non-negative number."))
    disclosed_quantity = fields.Int(missing=0, validate=validate.Range(min=0, error="Disclosed quantity must be a non-negative integer."))

class StrategySymbolMappingSchema(Schema):
    id = fields.Int()
    strategy_id = fields.Int()
    symbol = fields.Str()
    quantity = fields.Int()
    price = fields.Float()
    index = fields.Str()
    expiry = fields.Str()
    strike = fields.Str()
    type = fields.Str()
    delta = fields.Float()
    gamma = fields.Float()
    theta = fields.Float()
    vega = fields.Float()
    rho = fields.Float()
    IV = fields.Float()

class StrategySchema(Schema):
    id = fields.Int()
    name = fields.Str()
    is_active = fields.Bool()
    index_spot = fields.Float()
    realized_pnl = fields.Float()
    symbol_mappings = fields.List(fields.Nested(StrategySymbolMappingSchema))


class PlotPoint(Schema):
    index_value = fields.Int()
    on_expiry = fields.Float()
    on_open = fields.Float()


class StrategyDetailsResponseSchema(StrategySchema):
    plot_data = fields.List(fields.Nested(PlotPoint))
    breakeven_points = fields.List(fields.Float())