import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

def validate_data(df):
    """
    Validate that the input data has the required columns for flood prediction
    
    Args:
        df (DataFrame): Input dataframe
        
    Returns:
        dict: Validation result with 'valid' flag and error 'message' if invalid
    """
    # Define essential columns that should be present
    essential_numeric_columns = [
        'Rainfall (mm)', 'Temperature (°C)', 'Humidity (%)', 
        'Water Level (m)', 'Elevation (m)'
    ]
    
    essential_categorical_columns = [
        'Land Cover', 'Soil Type'
    ]
    
    # Check for target column
    if 'Flood Occurred' not in df.columns:
        return {
            'valid': False,
            'message': "Dataset must contain 'Flood Occurred' column as target variable"
        }
    
    # Check for numeric columns
    missing_numeric = [col for col in essential_numeric_columns if col not in df.columns]
    if missing_numeric:
        return {
            'valid': False,
            'message': f"Missing essential numeric columns: {', '.join(missing_numeric)}"
        }
    
    # Check for categorical columns
    missing_categorical = [col for col in essential_categorical_columns if col not in df.columns]
    if missing_categorical:
        return {
            'valid': False,
            'message': f"Missing essential categorical columns: {', '.join(missing_categorical)}"
        }
    
    # Check for location data (optional, but needed for map visualization)
    if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        logger.warning("Dataset does not contain latitude/longitude data, map visualization will be limited")
    
    # Check target column values (must be binary)
    target_values = df['Flood Occurred'].unique()
    if not all(val in [0, 1] for val in target_values):
        return {
            'valid': False,
            'message': "'Flood Occurred' column must contain only binary values (0 or 1)"
        }
    
    # Check for missing values in essential columns
    essential_columns = essential_numeric_columns + essential_categorical_columns + ['Flood Occurred']
    missing_counts = df[essential_columns].isnull().sum()
    if missing_counts.sum() > 0:
        columns_with_missing = missing_counts[missing_counts > 0].index.tolist()
        return {
            'valid': False,
            'message': f"Missing values in columns: {', '.join(columns_with_missing)}"
        }
    
    return {'valid': True, 'message': "Data validation successful"}

def preprocess_data(df):
    """
    Preprocess the dataframe for flood prediction
    
    Args:
        df (DataFrame): Input dataframe
        
    Returns:
        dict: Dictionary containing preprocessed data
    """
    try:
        # Create a copy of the dataframe to avoid modifying the original
        processed_df = df.copy()
        
        # Encode categorical variables
        categorical_columns = ['Land Cover', 'Soil Type']
        for col in categorical_columns:
            if col in processed_df.columns:
                processed_df = pd.get_dummies(processed_df, columns=[col])
        
        # Save latitude and longitude if available
        lat_lon = {}
        if 'Latitude' in processed_df.columns:
            lat_lon['latitude'] = processed_df['Latitude'].values
        if 'Longitude' in processed_df.columns:
            lat_lon['longitude'] = processed_df['Longitude'].values
        
        # Prepare features and target
        X = processed_df.drop(columns=['Flood Occurred'] + 
                            [col for col in ['Latitude', 'Longitude'] if col in processed_df.columns])
        y = processed_df['Flood Occurred']
        
        # Return preprocessed data
        result = {
            'X': X,
            'y': y,
            'feature_names': X.columns.tolist(),
            'target_name': 'Flood Occurred'
        }
        
        # Add latitude and longitude if available
        result.update(lat_lon)
        
        return result
    
    except Exception as e:
        logger.error(f"Error in preprocess_data: {str(e)}")
        raise
