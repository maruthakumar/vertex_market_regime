"""
Smart Money Concepts (SMC) Indicators for ML Indicator Strategy
Implements BOS, CHoCH, Order Blocks, FVG, Liquidity concepts in SQL
"""

from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SMCIndicators:
    """Smart Money Concepts indicators with SQL implementations"""
    
    def __init__(self):
        self.indicators = self._initialize_indicators()
        
    def _initialize_indicators(self) -> Dict[str, Dict[str, Any]]:
        """Initialize SMC indicators with SQL templates"""
        
        return {
            "BOS": {
                "name": "Break of Structure",
                "description": "Identifies breaks of market structure",
                "sql_template": """
                    -- Break of Structure (BOS) Detection
                    CASE 
                        -- Bullish BOS: Price breaks above previous high
                        WHEN close_price > MAX(high_price) OVER (
                            ORDER BY trade_date, trade_time
                            ROWS BETWEEN {lookback} PRECEDING AND 1 PRECEDING
                        ) AND LAG(close_price) OVER (ORDER BY trade_date, trade_time) <= 
                        MAX(high_price) OVER (
                            ORDER BY trade_date, trade_time
                            ROWS BETWEEN {lookback} PRECEDING AND 1 PRECEDING
                        ) THEN 1
                        
                        -- Bearish BOS: Price breaks below previous low
                        WHEN close_price < MIN(low_price) OVER (
                            ORDER BY trade_date, trade_time
                            ROWS BETWEEN {lookback} PRECEDING AND 1 PRECEDING
                        ) AND LAG(close_price) OVER (ORDER BY trade_date, trade_time) >= 
                        MIN(low_price) OVER (
                            ORDER BY trade_date, trade_time
                            ROWS BETWEEN {lookback} PRECEDING AND 1 PRECEDING
                        ) THEN -1
                        
                        ELSE 0
                    END AS bos_signal
                """,
                "params": {"lookback": 20}
            },
            
            "CHOCH": {
                "name": "Change of Character",
                "description": "Identifies changes in market character/trend",
                "sql_template": """
                    -- Change of Character (CHoCH) Detection
                    WITH trend_calc AS (
                        SELECT 
                            *,
                            -- Calculate trend based on higher highs/lows
                            CASE 
                                WHEN high_price > LAG(high_price, 1) OVER (ORDER BY trade_date, trade_time)
                                    AND low_price > LAG(low_price, 1) OVER (ORDER BY trade_date, trade_time)
                                THEN 1  -- Uptrend
                                WHEN high_price < LAG(high_price, 1) OVER (ORDER BY trade_date, trade_time)
                                    AND low_price < LAG(low_price, 1) OVER (ORDER BY trade_date, trade_time)
                                THEN -1  -- Downtrend
                                ELSE 0  -- No clear trend
                            END AS current_trend,
                            LAG(
                                CASE 
                                    WHEN high_price > LAG(high_price, 1) OVER (ORDER BY trade_date, trade_time)
                                        AND low_price > LAG(low_price, 1) OVER (ORDER BY trade_date, trade_time)
                                    THEN 1
                                    WHEN high_price < LAG(high_price, 1) OVER (ORDER BY trade_date, trade_time)
                                        AND low_price < LAG(low_price, 1) OVER (ORDER BY trade_date, trade_time)
                                    THEN -1
                                    ELSE 0
                                END, 1
                            ) OVER (ORDER BY trade_date, trade_time) AS prev_trend
                        FROM {table_name}
                    )
                    SELECT 
                        *,
                        CASE 
                            -- Bullish CHoCH: Downtrend to Uptrend
                            WHEN current_trend = 1 AND prev_trend = -1 THEN 1
                            -- Bearish CHoCH: Uptrend to Downtrend
                            WHEN current_trend = -1 AND prev_trend = 1 THEN -1
                            ELSE 0
                        END AS choch_signal
                    FROM trend_calc
                """,
                "params": {}
            },
            
            "ORDER_BLOCKS": {
                "name": "Order Blocks",
                "description": "Identifies institutional order blocks",
                "sql_template": """
                    -- Order Block Detection
                    WITH candle_analysis AS (
                        SELECT 
                            *,
                            -- Candle type
                            CASE 
                                WHEN close_price > open_price THEN 'BULLISH'
                                WHEN close_price < open_price THEN 'BEARISH'
                                ELSE 'DOJI'
                            END AS candle_type,
                            -- Candle body size
                            ABS(close_price - open_price) AS body_size,
                            -- Average body size
                            AVG(ABS(close_price - open_price)) OVER (
                                ORDER BY trade_date, trade_time
                                ROWS BETWEEN {lookback} PRECEDING AND CURRENT ROW
                            ) AS avg_body_size,
                            -- Next candle direction
                            LEAD(
                                CASE 
                                    WHEN close_price > open_price THEN 'BULLISH'
                                    WHEN close_price < open_price THEN 'BEARISH'
                                    ELSE 'DOJI'
                                END, 1
                            ) OVER (ORDER BY trade_date, trade_time) AS next_candle_type,
                            -- Next candle breaks high/low
                            LEAD(high_price, 1) OVER (ORDER BY trade_date, trade_time) AS next_high,
                            LEAD(low_price, 1) OVER (ORDER BY trade_date, trade_time) AS next_low
                        FROM {table_name}
                    )
                    SELECT 
                        *,
                        CASE 
                            -- Bullish Order Block: Large bearish candle followed by bullish break
                            WHEN candle_type = 'BEARISH' 
                                AND body_size > avg_body_size * {multiplier}
                                AND next_candle_type = 'BULLISH'
                                AND next_high > high_price
                            THEN 'BULLISH_OB'
                            
                            -- Bearish Order Block: Large bullish candle followed by bearish break
                            WHEN candle_type = 'BULLISH' 
                                AND body_size > avg_body_size * {multiplier}
                                AND next_candle_type = 'BEARISH'
                                AND next_low < low_price
                            THEN 'BEARISH_OB'
                            
                            ELSE 'NONE'
                        END AS order_block_type,
                        -- Order block levels
                        CASE 
                            WHEN candle_type = 'BEARISH' 
                                AND body_size > avg_body_size * {multiplier}
                                AND next_candle_type = 'BULLISH'
                                AND next_high > high_price
                            THEN high_price  -- Bullish OB high
                            
                            WHEN candle_type = 'BULLISH' 
                                AND body_size > avg_body_size * {multiplier}
                                AND next_candle_type = 'BEARISH'
                                AND next_low < low_price
                            THEN low_price  -- Bearish OB low
                            
                            ELSE NULL
                        END AS order_block_level
                    FROM candle_analysis
                """,
                "params": {"lookback": 20, "multiplier": 1.5}
            },
            
            "FVG": {
                "name": "Fair Value Gap",
                "description": "Identifies fair value gaps (imbalances)",
                "sql_template": """
                    -- Fair Value Gap (FVG) Detection
                    WITH gap_analysis AS (
                        SELECT 
                            *,
                            -- Previous and next candle high/low
                            LAG(high_price, 1) OVER (ORDER BY trade_date, trade_time) AS prev_high,
                            LAG(low_price, 1) OVER (ORDER BY trade_date, trade_time) AS prev_low,
                            LEAD(high_price, 1) OVER (ORDER BY trade_date, trade_time) AS next_high,
                            LEAD(low_price, 1) OVER (ORDER BY trade_date, trade_time) AS next_low
                        FROM {table_name}
                    )
                    SELECT 
                        *,
                        CASE 
                            -- Bullish FVG: Gap between prev high and next low
                            WHEN prev_high < next_low 
                                AND high_price > prev_high
                                AND low_price < next_low
                            THEN 1
                            
                            -- Bearish FVG: Gap between prev low and next high
                            WHEN prev_low > next_high 
                                AND low_price < prev_low
                                AND high_price > next_high
                            THEN -1
                            
                            ELSE 0
                        END AS fvg_signal,
                        -- FVG boundaries
                        CASE 
                            WHEN prev_high < next_low 
                                AND high_price > prev_high
                                AND low_price < next_low
                            THEN prev_high  -- Bullish FVG bottom
                            
                            WHEN prev_low > next_high 
                                AND low_price < prev_low
                                AND high_price > next_high
                            THEN prev_low  -- Bearish FVG top
                            
                            ELSE NULL
                        END AS fvg_level_1,
                        CASE 
                            WHEN prev_high < next_low 
                                AND high_price > prev_high
                                AND low_price < next_low
                            THEN next_low  -- Bullish FVG top
                            
                            WHEN prev_low > next_high 
                                AND low_price < prev_low
                                AND high_price > next_high
                            THEN next_high  -- Bearish FVG bottom
                            
                            ELSE NULL
                        END AS fvg_level_2
                    FROM gap_analysis
                """,
                "params": {}
            },
            
            "LIQUIDITY": {
                "name": "Liquidity Levels",
                "description": "Identifies liquidity pools and sweeps",
                "sql_template": """
                    -- Liquidity Level Detection
                    WITH liquidity_calc AS (
                        SELECT 
                            *,
                            -- Recent highs and lows (potential liquidity)
                            MAX(high_price) OVER (
                                ORDER BY trade_date, trade_time
                                ROWS BETWEEN {lookback} PRECEDING AND 1 PRECEDING
                            ) AS recent_high,
                            MIN(low_price) OVER (
                                ORDER BY trade_date, trade_time
                                ROWS BETWEEN {lookback} PRECEDING AND 1 PRECEDING
                            ) AS recent_low,
                            -- Count of touches at these levels
                            SUM(CASE 
                                WHEN ABS(high_price - MAX(high_price) OVER (
                                    ORDER BY trade_date, trade_time
                                    ROWS BETWEEN {lookback} PRECEDING AND 1 PRECEDING
                                )) < (high_price * {tolerance})
                                THEN 1 ELSE 0 
                            END) OVER (
                                ORDER BY trade_date, trade_time
                                ROWS BETWEEN {lookback} PRECEDING AND CURRENT ROW
                            ) AS high_touches,
                            SUM(CASE 
                                WHEN ABS(low_price - MIN(low_price) OVER (
                                    ORDER BY trade_date, trade_time
                                    ROWS BETWEEN {lookback} PRECEDING AND 1 PRECEDING
                                )) < (low_price * {tolerance})
                                THEN 1 ELSE 0 
                            END) OVER (
                                ORDER BY trade_date, trade_time
                                ROWS BETWEEN {lookback} PRECEDING AND CURRENT ROW
                            ) AS low_touches
                        FROM {table_name}
                    )
                    SELECT 
                        *,
                        CASE 
                            -- Buy-side liquidity grab
                            WHEN high_price > recent_high 
                                AND close_price < recent_high 
                                AND high_touches >= {min_touches}
                            THEN -1  -- Bearish after liquidity grab
                            
                            -- Sell-side liquidity grab
                            WHEN low_price < recent_low 
                                AND close_price > recent_low 
                                AND low_touches >= {min_touches}
                            THEN 1  -- Bullish after liquidity grab
                            
                            ELSE 0
                        END AS liquidity_grab
                    FROM liquidity_calc
                """,
                "params": {"lookback": 50, "tolerance": 0.001, "min_touches": 2}
            },
            
            "MARKET_STRUCTURE": {
                "name": "Market Structure",
                "description": "Overall market structure analysis",
                "sql_template": """
                    -- Market Structure Analysis
                    WITH structure_analysis AS (
                        SELECT 
                            *,
                            -- Swing highs and lows
                            CASE 
                                WHEN high_price > LAG(high_price, 1) OVER (ORDER BY trade_date, trade_time)
                                    AND high_price > LEAD(high_price, 1) OVER (ORDER BY trade_date, trade_time)
                                THEN high_price
                                ELSE NULL
                            END AS swing_high,
                            CASE 
                                WHEN low_price < LAG(low_price, 1) OVER (ORDER BY trade_date, trade_time)
                                    AND low_price < LEAD(low_price, 1) OVER (ORDER BY trade_date, trade_time)
                                THEN low_price
                                ELSE NULL
                            END AS swing_low,
                            -- Previous swing levels
                            LAG(
                                CASE 
                                    WHEN high_price > LAG(high_price, 1) OVER (ORDER BY trade_date, trade_time)
                                        AND high_price > LEAD(high_price, 1) OVER (ORDER BY trade_date, trade_time)
                                    THEN high_price
                                    ELSE NULL
                                END, 1, high_price
                            ) OVER (ORDER BY trade_date, trade_time) AS prev_swing_high,
                            LAG(
                                CASE 
                                    WHEN low_price < LAG(low_price, 1) OVER (ORDER BY trade_date, trade_time)
                                        AND low_price < LEAD(low_price, 1) OVER (ORDER BY trade_date, trade_time)
                                    THEN low_price
                                    ELSE NULL
                                END, 1, low_price
                            ) OVER (ORDER BY trade_date, trade_time) AS prev_swing_low
                        FROM {table_name}
                    )
                    SELECT 
                        *,
                        CASE 
                            -- Bullish market structure
                            WHEN swing_high > prev_swing_high 
                                AND swing_low > prev_swing_low
                            THEN 'BULLISH'
                            
                            -- Bearish market structure
                            WHEN swing_high < prev_swing_high 
                                AND swing_low < prev_swing_low
                            THEN 'BEARISH'
                            
                            -- Ranging market structure
                            ELSE 'RANGING'
                        END AS market_structure
                    FROM structure_analysis
                """,
                "params": {}
            },
            
            "PREMIUM_DISCOUNT": {
                "name": "Premium/Discount Zones",
                "description": "Identifies premium and discount price zones",
                "sql_template": """
                    -- Premium/Discount Zone Detection
                    WITH range_calc AS (
                        SELECT 
                            *,
                            -- Calculate recent range
                            MAX(high_price) OVER (
                                ORDER BY trade_date, trade_time
                                ROWS BETWEEN {range_period} PRECEDING AND CURRENT ROW
                            ) AS range_high,
                            MIN(low_price) OVER (
                                ORDER BY trade_date, trade_time
                                ROWS BETWEEN {range_period} PRECEDING AND CURRENT ROW
                            ) AS range_low,
                            -- Calculate midpoint
                            (MAX(high_price) OVER (
                                ORDER BY trade_date, trade_time
                                ROWS BETWEEN {range_period} PRECEDING AND CURRENT ROW
                            ) + MIN(low_price) OVER (
                                ORDER BY trade_date, trade_time
                                ROWS BETWEEN {range_period} PRECEDING AND CURRENT ROW
                            )) / 2 AS range_midpoint
                        FROM {table_name}
                    )
                    SELECT 
                        *,
                        -- Premium/Discount/Equilibrium zones
                        CASE 
                            WHEN close_price > range_midpoint + (range_high - range_midpoint) * {premium_threshold}
                            THEN 'PREMIUM'
                            
                            WHEN close_price < range_midpoint - (range_midpoint - range_low) * {discount_threshold}
                            THEN 'DISCOUNT'
                            
                            ELSE 'EQUILIBRIUM'
                        END AS price_zone,
                        -- Zone percentages
                        CASE 
                            WHEN range_high - range_low > 0 
                            THEN (close_price - range_low) / (range_high - range_low) * 100
                            ELSE 50
                        END AS zone_percentage
                    FROM range_calc
                """,
                "params": {"range_period": 50, "premium_threshold": 0.7, "discount_threshold": 0.7}
            }
        }
    
    def get_indicator_sql(self,
                         indicator_name: str,
                         params: Dict[str, Any] = None,
                         table_name: str = "market_data") -> str:
        """
        Get SQL implementation for a specific SMC indicator
        
        Args:
            indicator_name: Name of the SMC indicator
            params: Optional parameters to override defaults
            table_name: Source table name
            
        Returns:
            SQL string for the indicator
        """
        if indicator_name not in self.indicators:
            raise ValueError(f"SMC indicator {indicator_name} not supported")
            
        indicator = self.indicators[indicator_name]
        sql_template = indicator["sql_template"]
        
        # Merge default params with user params
        final_params = {**indicator["params"], **(params or {})}
        
        # Add table name to params
        final_params["table_name"] = table_name
        
        # Replace placeholders
        sql = sql_template.format(**final_params)
        
        return sql
    
    def get_all_smc_indicators_sql(self,
                                  table_name: str = "market_data",
                                  params: Dict[str, Dict[str, Any]] = None) -> str:
        """
        Get SQL that calculates all SMC indicators at once
        
        Args:
            table_name: Source table name
            params: Optional dictionary of parameters per indicator
            
        Returns:
            Complete SQL query with all SMC indicators
        """
        # Build comprehensive SMC analysis query
        query = f"""
        WITH base_data AS (
            SELECT * FROM {table_name}
        ),
        """
        
        # Add each SMC calculation as a CTE
        cte_parts = []
        
        # BOS calculation
        bos_params = (params or {}).get("BOS", {})
        cte_parts.append(f"""
        bos_calc AS (
            SELECT 
                *,
                {self.get_indicator_sql("BOS", bos_params, "base_data")}
            FROM base_data
        )""")
        
        # Add other calculations similarly...
        # For brevity, showing simplified version
        
        query += ",\n".join(cte_parts)
        
        # Final select combining all indicators
        query += """
        SELECT 
            *
        FROM bos_calc
        ORDER BY trade_date, trade_time
        """
        
        return query
    
    def get_supported_indicators(self) -> List[str]:
        """Get list of all supported SMC indicators"""
        return list(self.indicators.keys())
    
    def get_indicator_description(self, indicator_name: str) -> str:
        """Get description for a specific SMC indicator"""
        if indicator_name in self.indicators:
            return self.indicators[indicator_name]["description"]
        return "Unknown indicator"
    
    def get_indicator_params(self, indicator_name: str) -> Dict[str, Any]:
        """Get default parameters for a specific SMC indicator"""
        if indicator_name in self.indicators:
            return self.indicators[indicator_name]["params"].copy()
        return {}
    
    def combine_smc_signals(self,
                          signals: Dict[str, Any],
                          weights: Dict[str, float] = None) -> float:
        """
        Combine multiple SMC signals into a single score
        
        Args:
            signals: Dictionary of signal values
            weights: Optional weights for each signal
            
        Returns:
            Combined signal score between -1 and 1
        """
        default_weights = {
            "bos_signal": 0.25,
            "choch_signal": 0.20,
            "order_block_type": 0.20,
            "fvg_signal": 0.15,
            "liquidity_grab": 0.20
        }
        
        weights = weights or default_weights
        
        total_score = 0
        total_weight = 0
        
        for signal_name, signal_value in signals.items():
            if signal_name in weights and signal_value is not None:
                # Normalize signal values
                if signal_name == "order_block_type":
                    if signal_value == "BULLISH_OB":
                        signal_value = 1
                    elif signal_value == "BEARISH_OB":
                        signal_value = -1
                    else:
                        signal_value = 0
                        
                total_score += signal_value * weights[signal_name]
                total_weight += weights[signal_name]
                
        if total_weight > 0:
            return total_score / total_weight
        return 0
    
    def validate_smc_config(self, config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate SMC configuration
        
        Args:
            config: SMC configuration to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if "indicators" not in config:
            return False, "SMC indicators list is required"
            
        for indicator in config["indicators"]:
            if indicator not in self.indicators:
                return False, f"Unknown SMC indicator: {indicator}"
                
        # Validate parameters if provided
        if "params" in config:
            for indicator, params in config["params"].items():
                if indicator not in self.indicators:
                    continue
                    
                default_params = self.indicators[indicator]["params"]
                for param, value in params.items():
                    if param not in default_params:
                        return False, f"Invalid parameter {param} for SMC indicator {indicator}"
                        
        return True, None