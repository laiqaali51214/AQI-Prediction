"""EPA AQI calculation module based on US EPA standards."""
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EPAAQICalculator:
    """Calculate AQI using US EPA methodology."""
    
    # EPA AQI breakpoints (concentration ranges in µg/m³)
    # Format: {pollutant: [(low_conc, high_conc, low_aqi, high_aqi), ...]}
    BREAKPOINTS = {
        'pm25': [
            (0.0, 12.0, 0, 50),      # Good
            (12.1, 35.4, 51, 100),    # Moderate
            (35.5, 55.4, 101, 150),    # Unhealthy for Sensitive Groups
            (55.5, 150.4, 151, 200),  # Unhealthy
            (150.5, 250.4, 201, 300), # Very Unhealthy
            (250.5, 500.4, 301, 500)  # Hazardous
        ],
        'pm10': [
            (0.0, 54.0, 0, 50),
            (55.0, 154.0, 51, 100),
            (155.0, 254.0, 101, 150),
            (255.0, 354.0, 151, 200),
            (355.0, 424.0, 201, 300),
            (425.0, 604.0, 301, 500)
        ],
        'o3': [  # 8-hour average in ppb (parts per billion)
            (0.0, 54.0, 0, 50),
            (55.0, 70.0, 51, 100),
            (71.0, 85.0, 101, 150),
            (86.0, 105.0, 151, 200),
            (106.0, 200.0, 201, 300),
            (201.0, 500.0, 301, 500)
        ],
        'no2': [  # 1-hour average in ppb
            (0.0, 53.0, 0, 50),
            (54.0, 100.0, 51, 100),
            (101.0, 360.0, 101, 150),
            (361.0, 649.0, 151, 200),
            (650.0, 1249.0, 201, 300),
            (1250.0, 2049.0, 301, 500)
        ],
        'co': [  # 8-hour average in ppm (parts per million)
            (0.0, 4.4, 0, 50),
            (4.5, 9.4, 51, 100),
            (9.5, 12.4, 101, 150),
            (12.5, 15.4, 151, 200),
            (15.5, 30.4, 201, 300),
            (30.5, 50.4, 301, 500)
        ],
        'so2': [  # 1-hour average in ppb
            (0.0, 35.0, 0, 50),
            (36.0, 75.0, 51, 100),
            (76.0, 185.0, 101, 150),
            (186.0, 304.0, 151, 200),
            (305.0, 604.0, 201, 300),
            (605.0, 1004.0, 301, 500)
        ]
    }
    
    # Conversion factors (if needed)
    CONVERSIONS = {
        'o3_ppb_to_ugm3': 1.96,  # ppb to µg/m³
        'no2_ppb_to_ugm3': 1.88,
        'co_ppm_to_ppb': 1000,    # ppm to ppb
        'so2_ppb_to_ugm3': 2.62
    }
    
    @staticmethod
    def calculate_sub_index(concentration: float, breakpoints: list) -> float:
        """
        Calculate AQI sub-index for a pollutant using EPA formula.
        
        Formula: I = ((I_high - I_low) / (C_high - C_low)) * (C - C_low) + I_low
        
        Args:
            concentration: Pollutant concentration
            breakpoints: List of (low_conc, high_conc, low_aqi, high_aqi) tuples
        
        Returns:
            AQI sub-index value
        """
        if pd.isna(concentration) or concentration < 0:
            return np.nan
        
        # Find the appropriate breakpoint range
        for low_conc, high_conc, low_aqi, high_aqi in breakpoints:
            if low_conc <= concentration <= high_conc:
                # Linear interpolation
                if high_conc == low_conc:
                    return low_aqi
                aqi = ((high_aqi - low_aqi) / (high_conc - low_conc)) * (concentration - low_conc) + low_aqi
                return round(aqi)
        
        # If concentration exceeds highest breakpoint
        if concentration > breakpoints[-1][1]:
            return 500  # Maximum AQI
        
        return np.nan
    
    def calculate_pollutant_aqi(self, pollutant: str, concentration: float) -> float:
        """
        Calculate AQI for a specific pollutant.
        
        Args:
            pollutant: Pollutant name (pm25, pm10, o3, no2, co, so2)
            concentration: Pollutant concentration
        
        Returns:
            AQI sub-index for the pollutant
        """
        pollutant = pollutant.lower()
        
        if pollutant not in self.BREAKPOINTS:
            logger.warning(f"Unknown pollutant: {pollutant}")
            return np.nan
        
        breakpoints = self.BREAKPOINTS[pollutant]
        return self.calculate_sub_index(concentration, breakpoints)
    
    def calculate_overall_aqi(self, pollutants: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate overall AQI from multiple pollutants.
        
        Overall AQI = max(all sub-indices)
        
        Args:
            pollutants: Dictionary of {pollutant_name: concentration}
        
        Returns:
            Dictionary with sub-indices and overall AQI
        """
        sub_indices = {}
        
        for pollutant, concentration in pollutants.items():
            if pd.notna(concentration):
                sub_indices[pollutant] = self.calculate_pollutant_aqi(pollutant, concentration)
        
        if not sub_indices:
            return {'overall_aqi': np.nan, 'dominant_pollutant': None, **sub_indices}
        
        # Overall AQI is the maximum of all sub-indices
        overall_aqi = max(sub_indices.values())
        dominant_pollutant = max(sub_indices.items(), key=lambda x: x[1])[0]
        
        return {
            'overall_aqi': overall_aqi,
            'dominant_pollutant': dominant_pollutant,
            **sub_indices
        }
    
    def calculate_aqi_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate AQI for all rows in a DataFrame.
        
        Args:
            df: DataFrame with pollutant columns (pm25, pm10, o3, no2, co, so2)
        
        Returns:
            DataFrame with AQI columns added
        """
        df = df.copy()
        
        pollutant_cols = ['pm25', 'pm10', 'o3', 'no2', 'co', 'so2']
        available_pollutants = [col for col in pollutant_cols if col in df.columns]
        
        if not available_pollutants:
            logger.warning("No pollutant columns found in DataFrame")
            return df
        
        # Calculate sub-indices for each pollutant
        for pollutant in available_pollutants:
            if pollutant in df.columns:
                df[f'{pollutant}_aqi'] = df[pollutant].apply(
                    lambda x: self.calculate_pollutant_aqi(pollutant, x)
                )
        
        # Calculate overall AQI
        aqi_results = []
        for idx, row in df.iterrows():
            pollutants = {p: row[p] for p in available_pollutants if pd.notna(row.get(p))}
            result = self.calculate_overall_aqi(pollutants)
            aqi_results.append(result)
        
        # Add results to DataFrame
        aqi_df = pd.DataFrame(aqi_results)
        df['aqi_calculated'] = aqi_df['overall_aqi']
        df['dominant_pollutant'] = aqi_df['dominant_pollutant']
        
        # Add sub-indices
        for pollutant in available_pollutants:
            if f'{pollutant}_aqi' in aqi_df.columns:
                df[f'{pollutant}_aqi'] = aqi_df[f'{pollutant}_aqi']
        
        return df
    
    def get_aqi_category(self, aqi: float) -> str:
        """
        Get AQI category name from AQI value.
        
        Args:
            aqi: AQI value
        
        Returns:
            Category name
        """
        if pd.isna(aqi):
            return "Unknown"
        
        if aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Moderate"
        elif aqi <= 150:
            return "Unhealthy for Sensitive Groups"
        elif aqi <= 200:
            return "Unhealthy"
        elif aqi <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"


if __name__ == "__main__":
    # Example usage
    calculator = EPAAQICalculator()
    
    # Test with sample data
    test_data = {
        'pm25': 25.0,
        'pm10': 60.0,
        'o3': 70.0,
        'no2': 80.0
    }
    
    result = calculator.calculate_overall_aqi(test_data)
    print("AQI Calculation Result:")
    print(f"Overall AQI: {result['overall_aqi']}")
    print(f"Dominant Pollutant: {result['dominant_pollutant']}")
    print(f"Category: {calculator.get_aqi_category(result['overall_aqi'])}")
    print("\nSub-indices:")
    for pollutant, aqi in result.items():
        if pollutant not in ['overall_aqi', 'dominant_pollutant']:
            print(f"  {pollutant}: {aqi}")
