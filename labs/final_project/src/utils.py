"""
Utilities module for LCA tool.
Contains helper functions and constants.
"""

import pandas as pd
from typing import Dict, List, Union
from pathlib import Path

# Unit conversion factors
UNIT_CONVERSIONS = {
    'kg': {
        'g': 1000,
        'ton': 0.001,
        'lb': 2.20462
    },
    'L': {
        'mL': 1000,
        'm3': 0.001,
        'gal': 0.264172
    },
    'MJ': {
        'kJ': 1000,
        'kWh': 0.277778,
        'BTU': 947.817
    }
}

def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert values between different units using a two-step base conversion.
    
    Args:
        value: Value to convert
        from_unit: Source unit
        to_unit: Target unit
        
    Returns:
        Converted value
        
    Raises:
        ValueError: If units are not supported or are from different categories
    """
    if from_unit == to_unit:
        return value

    from_base_unit = None
    to_base_unit = None

    """
    Find the base unit category for both the 'from' and 'to' units.
    This is necessary to ensure we are not trying to convert between incompatible types (e.g., kg to L).
    """
    for base_unit, conversions in UNIT_CONVERSIONS.items():
        if from_unit == base_unit or from_unit in conversions:
            from_base_unit = base_unit
        if to_unit == base_unit or to_unit in conversions:
            to_base_unit = base_unit

    if not from_base_unit or not to_base_unit:
        raise ValueError(f"Unsupported unit provided: {from_unit} or {to_unit}")

    if from_base_unit != to_base_unit:
        raise ValueError(f"Cannot convert between different unit types: {from_unit} to {to_unit}")
    
    "Convert the initial value TO the base unit of its category."
    value_in_base_unit = value
    if from_unit != from_base_unit:
        "We divide when converting from a smaller unit to the larger base unit."
        value_in_base_unit = value / UNIT_CONVERSIONS[from_base_unit][from_unit]

    "Convert the value FROM the base unit TO the target unit."
    final_value = value_in_base_unit
    if to_unit != from_base_unit:
        "We multiply when converting from the larger base unit to a smaller unit."
        final_value = value_in_base_unit * UNIT_CONVERSIONS[from_base_unit][to_unit]
        
    return final_value


def save_results(data: pd.DataFrame, file_path: Union[str, Path], 
                format: str = 'csv') -> None:
    """
    Save analysis results to file.
    
    Args:
        data: DataFrame to save
        file_path: Path to save file
        format: File format ('csv', 'xlsx', or 'json')
        
    Raises:
        ValueError: If format is not supported
    """
    file_path = Path(file_path)
    
    if format == 'csv':
        data.to_csv(file_path, index=False)
    elif format == 'xlsx':
        data.to_excel(file_path, index=False)
    elif format == 'json':
        data.to_json(file_path, orient='records')
    else:
        raise ValueError(f"Unsupported format: {format}")

def load_impact_factors(file_path: Union[str, Path]) -> Dict:
    """
    Load impact factors from a JSON file.
    
    Args:
        file_path: Path to impact factors file
        
    Returns:
        Dictionary of impact factors
        
    Raises:
        FileNotFoundError: If file does not exist
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Impact factors file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        return pd.read_json(f).to_dict() 