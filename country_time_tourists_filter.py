
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def train_country_time_tourists_model(df):
    """Train a model for country, time and tourists based filtering."""
    try:
        # Prepare the data
        model_df = df.copy()
        
        # Create encoders
        country_encoder = LabelEncoder()
        time_encoder = LabelEncoder()
        
        # Encode features
        model_df['country_encoded'] = country_encoder.fit_transform(model_df['country'])
        model_df['time_encoded'] = time_encoder.fit_transform(model_df['best_time_to_visit'])
        
        # Train model
        X = model_df[['country_encoded', 'time_encoded', 'approximate_annual_tourists']]
        y = model_df.index  # Use index as target
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        return {
            'model': model,
            'country_encoder': country_encoder,
            'time_encoder': time_encoder
        }
        
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return None

def recommend_by_country_time_tourists(model_info, df, country, best_time, tourists_level):
    """Get recommendations based on country, time and tourist preferences."""
    try:
        if model_info is None:
            return pd.DataFrame(), "Model not available"
            
        if country == "All" and best_time == "All" and tourists_level == "All":
            return df.copy(), None
            
        # Filter based on selections
        filtered_df = df.copy()
        
        if country != "All":
            filtered_df = filtered_df[filtered_df['country'] == country]
            
        if best_time != "All":
            filtered_df = filtered_df[filtered_df['best_time_to_visit'] == best_time]
            
        if tourists_level != "All":
            tourist_thresholds = {
                'Low': (0, 1000000),
                'Medium': (1000000, 5000000),
                'High': (5000000, float('inf'))
            }
            if tourists_level in tourist_thresholds:
                min_tourists, max_tourists = tourist_thresholds[tourists_level]
                filtered_df = filtered_df[
                    (filtered_df['approximate_annual_tourists'] >= min_tourists) &
                    (filtered_df['approximate_annual_tourists'] < max_tourists)
                ]
        
        if filtered_df.empty:
            return pd.DataFrame(), "No destinations match your criteria"
            
        return filtered_df, None
        
    except Exception as e:
        return pd.DataFrame(), f"Error finding recommendations: {str(e)}"
