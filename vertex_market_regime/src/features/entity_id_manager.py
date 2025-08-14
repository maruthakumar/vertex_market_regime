"""
Entity ID Manager for Market Regime Feature Store
Story 2.6: Minimal Online Feature Registration - Subtask 1.2

Manages entity ID format: ${symbol}_${yyyymmddHHMM}_${dte}
Example: NIFTY_202508141430_7

Provides utilities for:
- Entity ID generation and validation
- Timestamp parsing and formatting
- Symbol and DTE extraction
"""

import re
from datetime import datetime, timedelta
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class EntityIdManager:
    """
    Manages entity ID format for Feature Store operations.
    
    Entity ID Format: ${symbol}_${yyyymmddHHMM}_${dte}
    - symbol: Market symbol (e.g., NIFTY, BANKNIFTY)
    - yyyymmddHHMM: Timestamp in format YYYYMMDDHHMM (e.g., 202508141430)
    - dte: Days to expiry (e.g., 7, 14, 21)
    
    Examples:
    - NIFTY_202508141430_7
    - BANKNIFTY_202508141500_14
    - FINNIFTY_202508141515_21
    """
    
    # Entity ID pattern: ${symbol}_${yyyymmddHHMM}_${dte}
    ENTITY_ID_PATTERN = re.compile(r'^([A-Z]+)_(\d{12})_(\d+)$')
    
    # Valid symbols for market regime analysis
    VALID_SYMBOLS = {
        'NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'SENSEX'
    }
    
    # Valid DTE ranges (days to expiry)
    MIN_DTE = 0
    MAX_DTE = 45
    
    def __init__(self):
        """Initialize Entity ID Manager"""
        logger.info("Entity ID Manager initialized")
    
    def generate_entity_id(self, symbol: str, timestamp: datetime, dte: int) -> str:
        """
        Generate entity ID from components.
        
        Args:
            symbol: Market symbol (e.g., 'NIFTY')
            timestamp: Datetime object for the market data point
            dte: Days to expiry
            
        Returns:
            str: Formatted entity ID
            
        Raises:
            ValueError: If any component is invalid
        """
        # Validate inputs
        self._validate_symbol(symbol)
        self._validate_dte(dte)
        self._validate_timestamp(timestamp)
        
        # Format timestamp as yyyymmddHHMM
        timestamp_str = timestamp.strftime("%Y%m%d%H%M")
        
        # Generate entity ID
        entity_id = f"{symbol.upper()}_{timestamp_str}_{dte}"
        
        logger.debug(f"Generated entity ID: {entity_id}")
        return entity_id
    
    def parse_entity_id(self, entity_id: str) -> Tuple[str, datetime, int]:
        """
        Parse entity ID into its components.
        
        Args:
            entity_id: Entity ID string to parse
            
        Returns:
            Tuple[str, datetime, int]: (symbol, timestamp, dte)
            
        Raises:
            ValueError: If entity ID format is invalid
        """
        match = self.ENTITY_ID_PATTERN.match(entity_id)
        if not match:
            raise ValueError(f"Invalid entity ID format: {entity_id}")
        
        symbol, timestamp_str, dte_str = match.groups()
        
        # Parse timestamp
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d%H%M")
        except ValueError as e:
            raise ValueError(f"Invalid timestamp in entity ID {entity_id}: {e}")
        
        # Parse DTE
        try:
            dte = int(dte_str)
        except ValueError as e:
            raise ValueError(f"Invalid DTE in entity ID {entity_id}: {e}")
        
        # Validate components
        self._validate_symbol(symbol)
        self._validate_dte(dte)
        
        logger.debug(f"Parsed entity ID {entity_id}: symbol={symbol}, timestamp={timestamp}, dte={dte}")
        return symbol, timestamp, dte
    
    def validate_entity_id(self, entity_id: str) -> bool:
        """
        Validate entity ID format and components.
        
        Args:
            entity_id: Entity ID to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            self.parse_entity_id(entity_id)
            return True
        except ValueError as e:
            logger.warning(f"Entity ID validation failed for {entity_id}: {e}")
            return False
    
    def generate_minute_range_entity_ids(
        self, 
        symbol: str, 
        start_time: datetime, 
        end_time: datetime, 
        dte: int
    ) -> List[str]:
        """
        Generate entity IDs for a minute-level time range.
        
        Args:
            symbol: Market symbol
            start_time: Start datetime (inclusive)
            end_time: End datetime (inclusive)
            dte: Days to expiry
            
        Returns:
            List[str]: List of entity IDs for each minute in the range
        """
        entity_ids = []
        current_time = start_time
        
        while current_time <= end_time:
            entity_id = self.generate_entity_id(symbol, current_time, dte)
            entity_ids.append(entity_id)
            current_time += timedelta(minutes=1)
        
        logger.info(f"Generated {len(entity_ids)} entity IDs for {symbol} from {start_time} to {end_time}")
        return entity_ids
    
    def generate_daily_entity_id(self, symbol: str, date: datetime, dte: int) -> str:
        """
        Generate entity ID for daily aggregation (using 09:15 as standard market open).
        
        Args:
            symbol: Market symbol
            date: Date for the entity (time will be set to 09:15)
            dte: Days to expiry
            
        Returns:
            str: Entity ID for daily aggregation
        """
        # Set time to market open (09:15 AM IST)
        daily_time = date.replace(hour=9, minute=15, second=0, microsecond=0)
        return self.generate_entity_id(symbol, daily_time, dte)
    
    def extract_trading_session_entity_ids(
        self, 
        symbol: str, 
        date: datetime, 
        dte: int
    ) -> List[str]:
        """
        Generate entity IDs for a full trading session (09:15 to 15:30).
        
        Args:
            symbol: Market symbol
            date: Trading date
            dte: Days to expiry
            
        Returns:
            List[str]: Entity IDs for the entire trading session
        """
        # Market hours: 09:15 to 15:30 IST
        start_time = date.replace(hour=9, minute=15, second=0, microsecond=0)
        end_time = date.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return self.generate_minute_range_entity_ids(symbol, start_time, end_time, dte)
    
    def get_entity_id_components_example(self) -> dict:
        """
        Get example entity ID components for documentation.
        
        Returns:
            dict: Example components and formatted entity ID
        """
        example_symbol = "NIFTY"
        example_timestamp = datetime(2025, 8, 14, 14, 30, 0)  # 2:30 PM
        example_dte = 7
        
        entity_id = self.generate_entity_id(example_symbol, example_timestamp, example_dte)
        
        return {
            'symbol': example_symbol,
            'timestamp': example_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            'timestamp_format': example_timestamp.strftime("%Y%m%d%H%M"),
            'dte': example_dte,
            'entity_id': entity_id,
            'pattern': '${symbol}_${yyyymmddHHMM}_${dte}'
        }
    
    def _validate_symbol(self, symbol: str) -> None:
        """Validate market symbol"""
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        symbol_upper = symbol.upper()
        if symbol_upper not in self.VALID_SYMBOLS:
            logger.warning(f"Symbol {symbol_upper} not in predefined valid symbols: {self.VALID_SYMBOLS}")
            # Allow non-predefined symbols but log warning
        
        if not symbol_upper.isalpha():
            raise ValueError("Symbol must contain only alphabetic characters")
    
    def _validate_dte(self, dte: int) -> None:
        """Validate days to expiry"""
        if not isinstance(dte, int):
            raise ValueError("DTE must be an integer")
        
        if dte < self.MIN_DTE or dte > self.MAX_DTE:
            raise ValueError(f"DTE must be between {self.MIN_DTE} and {self.MAX_DTE}, got {dte}")
    
    def _validate_timestamp(self, timestamp: datetime) -> None:
        """Validate timestamp"""
        if not isinstance(timestamp, datetime):
            raise ValueError("Timestamp must be a datetime object")
        
        # Check if timestamp is reasonable (not too far in past/future)
        now = datetime.now()
        min_date = now - timedelta(days=365)  # 1 year in past
        max_date = now + timedelta(days=365)  # 1 year in future
        
        if timestamp < min_date or timestamp > max_date:
            logger.warning(f"Timestamp {timestamp} is outside reasonable range ({min_date} to {max_date})")
    
    def get_valid_symbols(self) -> set:
        """Get set of valid symbols"""
        return self.VALID_SYMBOLS.copy()
    
    def get_dte_range(self) -> Tuple[int, int]:
        """Get valid DTE range"""
        return self.MIN_DTE, self.MAX_DTE