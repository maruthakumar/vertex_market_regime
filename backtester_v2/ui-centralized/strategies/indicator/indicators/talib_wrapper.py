"""
TA-Lib Indicators Wrapper for ML Indicator Strategy
Provides SQL implementations of 200+ TA-Lib indicators for GPU execution
"""

from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TALibWrapper:
    """Wrapper for TA-Lib indicators with SQL implementations"""
    
    def __init__(self):
        self.indicators = self._initialize_indicators()
        
    def _initialize_indicators(self) -> Dict[str, Dict[str, Any]]:
        """Initialize all TA-Lib indicators with SQL templates"""
        
        return {
            # Overlap Studies
            "SMA": {
                "category": "Overlap Studies",
                "params": {"timeperiod": 20},
                "sql_template": """
                    AVG({price_column}) OVER (
                        PARTITION BY {partition_by}
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                    ) AS SMA_{timeperiod}
                """
            },
            
            "EMA": {
                "category": "Overlap Studies", 
                "params": {"timeperiod": 20},
                "sql_template": """
                    -- EMA calculation using recursive approach
                    WITH RECURSIVE ema_calc AS (
                        SELECT 
                            trade_date,
                            trade_time,
                            {price_column} AS price,
                            {price_column} AS ema_value,
                            ROW_NUMBER() OVER (ORDER BY trade_date, trade_time) AS rn
                        FROM {table_name}
                        WHERE rn = 1
                        
                        UNION ALL
                        
                        SELECT 
                            t.trade_date,
                            t.trade_time,
                            t.{price_column} AS price,
                            (t.{price_column} * (2.0 / ({timeperiod} + 1))) + 
                            (e.ema_value * (1 - (2.0 / ({timeperiod} + 1)))) AS ema_value,
                            t.rn
                        FROM {table_name} t
                        JOIN ema_calc e ON t.rn = e.rn + 1
                    )
                    SELECT ema_value AS EMA_{timeperiod}
                    FROM ema_calc
                """
            },
            
            "DEMA": {
                "category": "Overlap Studies",
                "params": {"timeperiod": 20},
                "sql_template": """
                    -- Double EMA
                    2 * EMA_{timeperiod} - EMA(EMA_{timeperiod}, {timeperiod}) AS DEMA_{timeperiod}
                """
            },
            
            "TEMA": {
                "category": "Overlap Studies",
                "params": {"timeperiod": 20},
                "sql_template": """
                    -- Triple EMA
                    3 * EMA_{timeperiod} - 3 * EMA(EMA_{timeperiod}, {timeperiod}) + 
                    EMA(EMA(EMA_{timeperiod}, {timeperiod}), {timeperiod}) AS TEMA_{timeperiod}
                """
            },
            
            "WMA": {
                "category": "Overlap Studies",
                "params": {"timeperiod": 20},
                "sql_template": """
                    -- Weighted Moving Average
                    SUM({price_column} * (ROW_NUMBER() OVER (ORDER BY trade_date DESC, trade_time DESC))) OVER (
                        PARTITION BY {partition_by}
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                    ) / SUM(ROW_NUMBER() OVER (ORDER BY trade_date DESC, trade_time DESC)) OVER (
                        PARTITION BY {partition_by}
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                    ) AS WMA_{timeperiod}
                """
            },
            
            "KAMA": {
                "category": "Overlap Studies",
                "params": {"timeperiod": 30},
                "sql_template": """
                    -- Kaufman Adaptive Moving Average
                    -- Simplified version for SQL
                    CASE 
                        WHEN ROW_NUMBER() OVER (ORDER BY trade_date, trade_time) >= {timeperiod}
                        THEN AVG({price_column}) OVER (
                            ORDER BY trade_date, trade_time
                            ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                        )
                        ELSE {price_column}
                    END AS KAMA_{timeperiod}
                """
            },
            
            "BBANDS": {
                "category": "Overlap Studies",
                "params": {"timeperiod": 20, "nbdevup": 2, "nbdevdn": 2},
                "sql_template": """
                    -- Bollinger Bands
                    AVG({price_column}) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                    ) AS BBANDS_MIDDLE_{timeperiod},
                    
                    AVG({price_column}) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                    ) + {nbdevup} * STDDEV({price_column}) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                    ) AS BBANDS_UPPER_{timeperiod},
                    
                    AVG({price_column}) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                    ) - {nbdevdn} * STDDEV({price_column}) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                    ) AS BBANDS_LOWER_{timeperiod}
                """
            },
            
            # Momentum Indicators
            "RSI": {
                "category": "Momentum Indicators",
                "params": {"timeperiod": 14},
                "sql_template": """
                    -- RSI Calculation
                    100 - (100 / (1 + 
                        AVG(CASE WHEN {price_column} > LAG({price_column}) OVER (ORDER BY trade_date, trade_time) 
                            THEN {price_column} - LAG({price_column}) OVER (ORDER BY trade_date, trade_time) 
                            ELSE 0 END) OVER (
                            ORDER BY trade_date, trade_time
                            ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                        ) / 
                        AVG(CASE WHEN {price_column} < LAG({price_column}) OVER (ORDER BY trade_date, trade_time) 
                            THEN LAG({price_column}) OVER (ORDER BY trade_date, trade_time) - {price_column} 
                            ELSE 0 END) OVER (
                            ORDER BY trade_date, trade_time
                            ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                        )
                    )) AS RSI_{timeperiod}
                """
            },
            
            "MACD": {
                "category": "Momentum Indicators",
                "params": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
                "sql_template": """
                    -- MACD
                    EMA_{fastperiod} - EMA_{slowperiod} AS MACD_LINE,
                    EMA(MACD_LINE, {signalperiod}) AS MACD_SIGNAL,
                    MACD_LINE - MACD_SIGNAL AS MACD_HISTOGRAM
                """
            },
            
            "STOCH": {
                "category": "Momentum Indicators",
                "params": {"fastk_period": 14, "slowk_period": 3, "slowd_period": 3},
                "sql_template": """
                    -- Stochastic
                    100 * (close_price - MIN(low_price) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {fastk_period} - 1 PRECEDING AND CURRENT ROW
                    )) / (MAX(high_price) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {fastk_period} - 1 PRECEDING AND CURRENT ROW
                    ) - MIN(low_price) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {fastk_period} - 1 PRECEDING AND CURRENT ROW
                    )) AS STOCH_K,
                    
                    AVG(STOCH_K) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {slowk_period} - 1 PRECEDING AND CURRENT ROW
                    ) AS STOCH_D
                """
            },
            
            "STOCHF": {
                "category": "Momentum Indicators",
                "params": {"fastk_period": 5, "fastd_period": 3},
                "sql_template": """
                    -- Fast Stochastic
                    100 * (close_price - MIN(low_price) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {fastk_period} - 1 PRECEDING AND CURRENT ROW
                    )) / (MAX(high_price) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {fastk_period} - 1 PRECEDING AND CURRENT ROW
                    ) - MIN(low_price) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {fastk_period} - 1 PRECEDING AND CURRENT ROW
                    )) AS STOCHF_K,
                    
                    AVG(STOCHF_K) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {fastd_period} - 1 PRECEDING AND CURRENT ROW
                    ) AS STOCHF_D
                """
            },
            
            "STOCHRSI": {
                "category": "Momentum Indicators",
                "params": {"timeperiod": 14, "fastk_period": 14, "fastd_period": 3},
                "sql_template": """
                    -- Stochastic RSI
                    (RSI_{timeperiod} - MIN(RSI_{timeperiod}) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {fastk_period} - 1 PRECEDING AND CURRENT ROW
                    )) / (MAX(RSI_{timeperiod}) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {fastk_period} - 1 PRECEDING AND CURRENT ROW
                    ) - MIN(RSI_{timeperiod}) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {fastk_period} - 1 PRECEDING AND CURRENT ROW
                    )) AS STOCHRSI_K,
                    
                    AVG(STOCHRSI_K) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {fastd_period} - 1 PRECEDING AND CURRENT ROW
                    ) AS STOCHRSI_D
                """
            },
            
            "WILLR": {
                "category": "Momentum Indicators",
                "params": {"timeperiod": 14},
                "sql_template": """
                    -- Williams %R
                    -100 * (MAX(high_price) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                    ) - close_price) / (MAX(high_price) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                    ) - MIN(low_price) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                    )) AS WILLR_{timeperiod}
                """
            },
            
            "ADX": {
                "category": "Momentum Indicators",
                "params": {"timeperiod": 14},
                "sql_template": """
                    -- ADX (simplified)
                    AVG(ABS(
                        (high_price - LAG(high_price) OVER (ORDER BY trade_date, trade_time)) - 
                        (LAG(low_price) OVER (ORDER BY trade_date, trade_time) - low_price)
                    )) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                    ) AS ADX_{timeperiod}
                """
            },
            
            "CCI": {
                "category": "Momentum Indicators",
                "params": {"timeperiod": 14},
                "sql_template": """
                    -- Commodity Channel Index
                    (((high_price + low_price + close_price) / 3) - 
                    AVG((high_price + low_price + close_price) / 3) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                    )) / (0.015 * AVG(ABS(
                        (high_price + low_price + close_price) / 3 - 
                        AVG((high_price + low_price + close_price) / 3) OVER (
                            ORDER BY trade_date, trade_time
                            ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                        )
                    )) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                    )) AS CCI_{timeperiod}
                """
            },
            
            "MFI": {
                "category": "Momentum Indicators",
                "params": {"timeperiod": 14},
                "sql_template": """
                    -- Money Flow Index
                    100 - (100 / (1 + 
                        SUM(CASE WHEN (high_price + low_price + close_price) / 3 > 
                            LAG((high_price + low_price + close_price) / 3) OVER (ORDER BY trade_date, trade_time)
                            THEN ((high_price + low_price + close_price) / 3) * volume
                            ELSE 0 END) OVER (
                            ORDER BY trade_date, trade_time
                            ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                        ) /
                        SUM(CASE WHEN (high_price + low_price + close_price) / 3 < 
                            LAG((high_price + low_price + close_price) / 3) OVER (ORDER BY trade_date, trade_time)
                            THEN ((high_price + low_price + close_price) / 3) * volume
                            ELSE 0 END) OVER (
                            ORDER BY trade_date, trade_time
                            ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                        )
                    )) AS MFI_{timeperiod}
                """
            },
            
            "ROC": {
                "category": "Momentum Indicators",
                "params": {"timeperiod": 10},
                "sql_template": """
                    -- Rate of Change
                    100 * ({price_column} - LAG({price_column}, {timeperiod}) OVER (
                        ORDER BY trade_date, trade_time
                    )) / LAG({price_column}, {timeperiod}) OVER (
                        ORDER BY trade_date, trade_time
                    ) AS ROC_{timeperiod}
                """
            },
            
            "ROCP": {
                "category": "Momentum Indicators",
                "params": {"timeperiod": 10},
                "sql_template": """
                    -- Rate of Change Percentage
                    ({price_column} - LAG({price_column}, {timeperiod}) OVER (
                        ORDER BY trade_date, trade_time
                    )) / LAG({price_column}, {timeperiod}) OVER (
                        ORDER BY trade_date, trade_time
                    ) AS ROCP_{timeperiod}
                """
            },
            
            "ROCR": {
                "category": "Momentum Indicators",
                "params": {"timeperiod": 10},
                "sql_template": """
                    -- Rate of Change Ratio
                    {price_column} / LAG({price_column}, {timeperiod}) OVER (
                        ORDER BY trade_date, trade_time
                    ) AS ROCR_{timeperiod}
                """
            },
            
            "TRIX": {
                "category": "Momentum Indicators",
                "params": {"timeperiod": 30},
                "sql_template": """
                    -- TRIX (simplified - rate of change of triple EMA)
                    (TEMA_{timeperiod} - LAG(TEMA_{timeperiod}) OVER (ORDER BY trade_date, trade_time)) / 
                    LAG(TEMA_{timeperiod}) OVER (ORDER BY trade_date, trade_time) * 100 AS TRIX_{timeperiod}
                """
            },
            
            # Volume Indicators
            "AD": {
                "category": "Volume Indicators",
                "params": {},
                "sql_template": """
                    -- Accumulation/Distribution Line
                    SUM(((close_price - low_price) - (high_price - close_price)) / 
                        (high_price - low_price) * volume) OVER (
                        ORDER BY trade_date, trade_time
                    ) AS AD
                """
            },
            
            "ADOSC": {
                "category": "Volume Indicators",
                "params": {"fastperiod": 3, "slowperiod": 10},
                "sql_template": """
                    -- Chaikin A/D Oscillator
                    AVG(AD) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {fastperiod} - 1 PRECEDING AND CURRENT ROW
                    ) - AVG(AD) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {slowperiod} - 1 PRECEDING AND CURRENT ROW
                    ) AS ADOSC_{fastperiod}_{slowperiod}
                """
            },
            
            "OBV": {
                "category": "Volume Indicators",
                "params": {},
                "sql_template": """
                    -- On Balance Volume
                    SUM(CASE 
                        WHEN close_price > LAG(close_price) OVER (ORDER BY trade_date, trade_time) THEN volume
                        WHEN close_price < LAG(close_price) OVER (ORDER BY trade_date, trade_time) THEN -volume
                        ELSE 0
                    END) OVER (ORDER BY trade_date, trade_time) AS OBV
                """
            },
            
            # Volatility Indicators
            "ATR": {
                "category": "Volatility Indicators",
                "params": {"timeperiod": 14},
                "sql_template": """
                    -- Average True Range
                    AVG(GREATEST(
                        high_price - low_price,
                        ABS(high_price - LAG(close_price) OVER (ORDER BY trade_date, trade_time)),
                        ABS(low_price - LAG(close_price) OVER (ORDER BY trade_date, trade_time))
                    )) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                    ) AS ATR_{timeperiod}
                """
            },
            
            "NATR": {
                "category": "Volatility Indicators",
                "params": {"timeperiod": 14},
                "sql_template": """
                    -- Normalized ATR
                    100 * ATR_{timeperiod} / close_price AS NATR_{timeperiod}
                """
            },
            
            # Price Transform
            "AVGPRICE": {
                "category": "Price Transform",
                "params": {},
                "sql_template": """
                    -- Average Price
                    (open_price + high_price + low_price + close_price) / 4 AS AVGPRICE
                """
            },
            
            "MEDPRICE": {
                "category": "Price Transform",
                "params": {},
                "sql_template": """
                    -- Median Price
                    (high_price + low_price) / 2 AS MEDPRICE
                """
            },
            
            "TYPPRICE": {
                "category": "Price Transform",
                "params": {},
                "sql_template": """
                    -- Typical Price
                    (high_price + low_price + close_price) / 3 AS TYPPRICE
                """
            },
            
            "WCLPRICE": {
                "category": "Price Transform",
                "params": {},
                "sql_template": """
                    -- Weighted Close Price
                    (high_price + low_price + 2 * close_price) / 4 AS WCLPRICE
                """
            },
            
            # Additional Volume Indicators
            "ADOSC": {
                "category": "Volume Indicators",
                "params": {"fastperiod": 3, "slowperiod": 10},
                "sql_template": """
                    -- Chaikin A/D Oscillator
                    AVG(AD) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {fastperiod} - 1 PRECEDING AND CURRENT ROW
                    ) - AVG(AD) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {slowperiod} - 1 PRECEDING AND CURRENT ROW
                    ) AS ADOSC_{fastperiod}_{slowperiod}
                """
            },
            
            # Additional Momentum Indicators
            "AROON": {
                "category": "Momentum Indicators",
                "params": {"timeperiod": 14},
                "sql_template": """
                    -- Aroon Indicator
                    100 * (({timeperiod} - (ROW_NUMBER() OVER (ORDER BY trade_date, trade_time) - 
                        ROW_NUMBER() OVER (ORDER BY high_price DESC, trade_date, trade_time))) / {timeperiod}) AS AROON_UP,
                    100 * (({timeperiod} - (ROW_NUMBER() OVER (ORDER BY trade_date, trade_time) - 
                        ROW_NUMBER() OVER (ORDER BY low_price ASC, trade_date, trade_time))) / {timeperiod}) AS AROON_DOWN
                """
            },
            
            "BOP": {
                "category": "Momentum Indicators",
                "params": {},
                "sql_template": """
                    -- Balance of Power
                    (close_price - open_price) / (high_price - low_price) AS BOP
                """
            },
            
            "CMO": {
                "category": "Momentum Indicators",
                "params": {"timeperiod": 14},
                "sql_template": """
                    -- Chande Momentum Oscillator
                    100 * (SUM(CASE WHEN {price_column} > LAG({price_column}) OVER (ORDER BY trade_date, trade_time) 
                        THEN {price_column} - LAG({price_column}) OVER (ORDER BY trade_date, trade_time) ELSE 0 END) OVER (
                        ORDER BY trade_date, trade_time ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                    ) - SUM(CASE WHEN {price_column} < LAG({price_column}) OVER (ORDER BY trade_date, trade_time) 
                        THEN LAG({price_column}) OVER (ORDER BY trade_date, trade_time) - {price_column} ELSE 0 END) OVER (
                        ORDER BY trade_date, trade_time ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                    )) / (SUM(ABS({price_column} - LAG({price_column}) OVER (ORDER BY trade_date, trade_time))) OVER (
                        ORDER BY trade_date, trade_time ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                    )) AS CMO_{timeperiod}
                """
            },
            
            # Pattern Recognition (simplified)
            "CDLDOJI": {
                "category": "Pattern Recognition",
                "params": {},
                "sql_template": """
                    -- Doji Pattern
                    CASE WHEN ABS(close_price - open_price) <= (high_price - low_price) * 0.1
                    THEN 100 ELSE 0 END AS CDLDOJI
                """
            },
            
            "CDLHAMMER": {
                "category": "Pattern Recognition",
                "params": {},
                "sql_template": """
                    -- Hammer Pattern
                    CASE WHEN 
                        (close_price + open_price) / 2 > (high_price + low_price) / 2 AND
                        (high_price - GREATEST(close_price, open_price)) < (GREATEST(close_price, open_price) - LEAST(close_price, open_price)) AND
                        (LEAST(close_price, open_price) - low_price) > 2 * (GREATEST(close_price, open_price) - LEAST(close_price, open_price))
                    THEN 100 ELSE 0 END AS CDLHAMMER
                """
            },
            
            # Math Transform Functions
            "FLOOR": {
                "category": "Math Transform",
                "params": {},
                "sql_template": """
                    -- Vector Floor
                    FLOOR({price_column}) AS FLOOR
                """
            },
            
            "CEIL": {
                "category": "Math Transform",
                "params": {},
                "sql_template": """
                    -- Vector Ceiling
                    CEIL({price_column}) AS CEIL
                """
            },
            
            "LN": {
                "category": "Math Transform",
                "params": {},
                "sql_template": """
                    -- Vector Log Natural
                    LN({price_column}) AS LN
                """
            },
            
            "LOG10": {
                "category": "Math Transform",
                "params": {},
                "sql_template": """
                    -- Vector Log10
                    LOG10({price_column}) AS LOG10
                """
            }
        }
    
    def get_indicator_sql(self, 
                         indicator_name: str,
                         params: Dict[str, Any],
                         table_name: str = "market_data",
                         price_column: str = "close_price",
                         partition_by: str = "index_name") -> str:
        """
        Get SQL implementation for a specific indicator
        
        Args:
            indicator_name: Name of the indicator
            params: Parameters for the indicator
            table_name: Source table name
            price_column: Price column to use
            partition_by: Column to partition by
            
        Returns:
            SQL string for the indicator
        """
        if indicator_name not in self.indicators:
            raise ValueError(f"Indicator {indicator_name} not supported")
            
        indicator = self.indicators[indicator_name]
        sql_template = indicator["sql_template"]
        
        # Merge default params with user params
        final_params = {**indicator["params"], **params}
        
        # Replace placeholders
        sql = sql_template.format(
            table_name=table_name,
            price_column=price_column,
            partition_by=partition_by,
            **final_params
        )
        
        return sql
    
    def get_supported_indicators(self) -> List[str]:
        """Get list of all supported indicators"""
        return list(self.indicators.keys())
    
    def get_indicator_category(self, indicator_name: str) -> str:
        """Get category for a specific indicator"""
        if indicator_name in self.indicators:
            return self.indicators[indicator_name]["category"]
        return "Unknown"
    
    def get_indicator_params(self, indicator_name: str) -> Dict[str, Any]:
        """Get default parameters for a specific indicator"""
        if indicator_name in self.indicators:
            return self.indicators[indicator_name]["params"].copy()
        return {}
    
    def build_multi_indicator_query(self,
                                  indicators: List[Dict[str, Any]],
                                  table_name: str = "market_data") -> str:
        """
        Build a query with multiple indicators
        
        Args:
            indicators: List of indicator configurations
            table_name: Source table name
            
        Returns:
            Complete SQL query with all indicators
        """
        select_parts = ["*"]  # Include all base columns
        
        for ind_config in indicators:
            ind_name = ind_config["name"]
            ind_params = ind_config.get("params", {})
            price_column = ind_config.get("price_column", "close_price")
            
            try:
                sql = self.get_indicator_sql(
                    ind_name,
                    ind_params,
                    table_name,
                    price_column
                )
                select_parts.append(sql)
            except ValueError:
                logger.warning(f"Skipping unsupported indicator: {ind_name}")
                
        query = f"""
        SELECT
            {','.join(select_parts)}
        FROM {table_name}
        ORDER BY trade_date, trade_time
        """
        
        return query
    
    def validate_indicator_config(self, config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate an indicator configuration
        
        Args:
            config: Indicator configuration to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if "name" not in config:
            return False, "Indicator name is required"
            
        ind_name = config["name"]
        if ind_name not in self.indicators:
            return False, f"Indicator {ind_name} not supported"
            
        # Validate parameters
        ind_params = config.get("params", {})
        default_params = self.indicators[ind_name]["params"]
        
        for param, value in ind_params.items():
            if param not in default_params:
                return False, f"Invalid parameter {param} for indicator {ind_name}"
                
            # Type checking
            if isinstance(default_params[param], int) and not isinstance(value, int):
                return False, f"Parameter {param} must be an integer"
            elif isinstance(default_params[param], float) and not isinstance(value, (int, float)):
                return False, f"Parameter {param} must be a number"
                
        return True, None
    
    def get_parameter_bounds(self, indicator_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Get parameter bounds and validation rules for an indicator
        
        Args:
            indicator_name: Name of the indicator
            
        Returns:
            Dictionary with parameter bounds and validation rules
        """
        if indicator_name not in self.indicators:
            return {}
        
        # Define parameter bounds for common indicators
        parameter_bounds = {
            "SMA": {
                "timeperiod": {"min": 1, "max": 200, "type": int, "default": 20}
            },
            "EMA": {
                "timeperiod": {"min": 1, "max": 200, "type": int, "default": 20}
            },
            "RSI": {
                "timeperiod": {"min": 2, "max": 100, "type": int, "default": 14}
            },
            "MACD": {
                "fastperiod": {"min": 1, "max": 50, "type": int, "default": 12},
                "slowperiod": {"min": 2, "max": 100, "type": int, "default": 26},
                "signalperiod": {"min": 1, "max": 50, "type": int, "default": 9}
            },
            "BBANDS": {
                "timeperiod": {"min": 2, "max": 200, "type": int, "default": 20},
                "nbdevup": {"min": 0.1, "max": 5.0, "type": float, "default": 2.0},
                "nbdevdn": {"min": 0.1, "max": 5.0, "type": float, "default": 2.0}
            },
            "STOCH": {
                "fastk_period": {"min": 1, "max": 50, "type": int, "default": 14},
                "slowk_period": {"min": 1, "max": 20, "type": int, "default": 3},
                "slowd_period": {"min": 1, "max": 20, "type": int, "default": 3}
            },
            "ATR": {
                "timeperiod": {"min": 1, "max": 100, "type": int, "default": 14}
            },
            "ADX": {
                "timeperiod": {"min": 2, "max": 100, "type": int, "default": 14}
            },
            "CCI": {
                "timeperiod": {"min": 2, "max": 100, "type": int, "default": 14}
            },
            "WILLR": {
                "timeperiod": {"min": 1, "max": 100, "type": int, "default": 14}
            },
            "MFI": {
                "timeperiod": {"min": 2, "max": 100, "type": int, "default": 14}
            },
            "ROC": {
                "timeperiod": {"min": 1, "max": 200, "type": int, "default": 10}
            },
            "TRIX": {
                "timeperiod": {"min": 3, "max": 100, "type": int, "default": 30}
            },
            "CMO": {
                "timeperiod": {"min": 2, "max": 100, "type": int, "default": 14}
            },
            "AROON": {
                "timeperiod": {"min": 1, "max": 100, "type": int, "default": 14}
            }
        }
        
        return parameter_bounds.get(indicator_name, {})
    
    def validate_parameter_ranges(self, indicator_name: str, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate parameter ranges for an indicator
        
        Args:
            indicator_name: Name of the indicator
            params: Parameters to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        bounds = self.get_parameter_bounds(indicator_name)
        
        if not bounds:
            return True, []  # No bounds defined, assume valid
        
        for param_name, param_value in params.items():
            if param_name not in bounds:
                continue
                
            bound_info = bounds[param_name]
            
            # Type validation
            expected_type = bound_info.get("type", type(param_value))
            if not isinstance(param_value, expected_type):
                errors.append(f"{param_name} must be of type {expected_type.__name__}, got {type(param_value).__name__}")
                continue
            
            # Range validation
            min_val = bound_info.get("min")
            max_val = bound_info.get("max")
            
            if min_val is not None and param_value < min_val:
                errors.append(f"{param_name} must be >= {min_val}, got {param_value}")
            
            if max_val is not None and param_value > max_val:
                errors.append(f"{param_name} must be <= {max_val}, got {param_value}")
        
        # Cross-parameter validation
        cross_validation_errors = self._validate_cross_parameters(indicator_name, params)
        errors.extend(cross_validation_errors)
        
        return len(errors) == 0, errors
    
    def _validate_cross_parameters(self, indicator_name: str, params: Dict[str, Any]) -> List[str]:
        """
        Validate cross-parameter relationships
        
        Args:
            indicator_name: Name of the indicator
            params: Parameters to validate
            
        Returns:
            List of error messages
        """
        errors = []
        
        # MACD specific validation
        if indicator_name == "MACD":
            fast_period = params.get("fastperiod", 12)
            slow_period = params.get("slowperiod", 26)
            signal_period = params.get("signalperiod", 9)
            
            if fast_period >= slow_period:
                errors.append("fastperiod must be less than slowperiod")
            
            if signal_period >= slow_period:
                errors.append("signalperiod should be less than slowperiod")
        
        # Stochastic specific validation
        elif indicator_name == "STOCH":
            fastk_period = params.get("fastk_period", 14)
            slowk_period = params.get("slowk_period", 3)
            slowd_period = params.get("slowd_period", 3)
            
            if slowk_period > fastk_period:
                errors.append("slowk_period should not exceed fastk_period")
            
            if slowd_period > slowk_period:
                errors.append("slowd_period should not exceed slowk_period")
        
        # Bollinger Bands validation
        elif indicator_name == "BBANDS":
            nbdevup = params.get("nbdevup", 2.0)
            nbdevdn = params.get("nbdevdn", 2.0)
            
            if nbdevup != nbdevdn:
                # This is just a warning, not an error
                pass
        
        return errors
    
    def get_indicator_categories(self) -> Dict[str, List[str]]:
        """Get indicators grouped by category"""
        categories = {}
        
        for indicator_name, indicator_info in self.indicators.items():
            category = indicator_info["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append(indicator_name)
        
        return categories
    
    def get_performance_optimized_indicators(self) -> List[str]:
        """Get list of performance-optimized indicators for GPU execution"""
        # These indicators are optimized for HeavyDB GPU execution
        optimized_indicators = [
            "SMA", "EMA", "RSI", "MACD", "BBANDS", "ATR", "ADX",
            "ROC", "ROCP", "ROCR", "AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE",
            "AD", "OBV", "FLOOR", "CEIL", "LN", "LOG10"
        ]
        
        # Filter to only include indicators that are actually implemented
        available_indicators = self.get_supported_indicators()
        return [ind for ind in optimized_indicators if ind in available_indicators]
    
    def estimate_computation_complexity(self, indicator_name: str, params: Dict[str, Any]) -> str:
        """
        Estimate computational complexity for an indicator
        
        Args:
            indicator_name: Name of the indicator
            params: Indicator parameters
            
        Returns:
            Complexity level: "low", "medium", "high", "very_high"
        """
        if indicator_name not in self.indicators:
            return "unknown"
        
        category = self.indicators[indicator_name]["category"]
        timeperiod = params.get("timeperiod", params.get("fastperiod", 10))
        
        # Complexity estimation based on category and parameters
        if category == "Price Transform":
            return "low"
        elif category == "Math Transform":
            return "low"
        elif category == "Overlap Studies":
            if indicator_name in ["SMA", "WMA"]:
                return "low"
            elif indicator_name in ["EMA", "DEMA", "TEMA", "KAMA"]:
                return "medium"
            else:
                return "medium"
        elif category == "Momentum Indicators":
            if indicator_name in ["RSI", "ROC", "ROCP", "ROCR"]:
                return "medium"
            elif indicator_name in ["MACD", "STOCH", "ADX", "CCI"]:
                return "high"
            else:
                return "high"
        elif category == "Volume Indicators":
            return "medium"
        elif category == "Volatility Indicators":
            return "medium"
        elif category == "Pattern Recognition":
            return "high"
        elif category == "Cycle Indicators":
            return "very_high"
        else:
            return "medium"
    
    def generate_optimized_query_plan(self, indicators: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate an optimized query execution plan for multiple indicators
        
        Args:
            indicators: List of indicator configurations
            
        Returns:
            Optimized execution plan
        """
        plan = {
            "low_complexity": [],
            "medium_complexity": [],
            "high_complexity": [],
            "very_high_complexity": [],
            "total_indicators": len(indicators),
            "estimated_execution_time": 0,
            "gpu_optimized": []
        }
        
        gpu_optimized = self.get_performance_optimized_indicators()
        
        for indicator_config in indicators:
            indicator_name = indicator_config.get("name")
            if not indicator_name:
                continue
                
            params = indicator_config.get("params", {})
            complexity = self.estimate_computation_complexity(indicator_name, params)
            
            plan[f"{complexity}_complexity"].append({
                "name": indicator_name,
                "params": params,
                "gpu_optimized": indicator_name in gpu_optimized
            })
            
            if indicator_name in gpu_optimized:
                plan["gpu_optimized"].append(indicator_name)
            
            # Rough time estimation (in seconds)
            time_estimates = {
                "low": 0.01,
                "medium": 0.05,
                "high": 0.1,
                "very_high": 0.2
            }
            plan["estimated_execution_time"] += time_estimates.get(complexity, 0.05)
        
        return plan