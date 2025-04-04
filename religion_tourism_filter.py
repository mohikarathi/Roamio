
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import re

def train_religion_tourism_model(df):
    """Train a model to cluster destinations based on religion and tourist numbers."""
    try:
        clustering_df = df.copy()
        
        # Features for clustering
        features = ['approximate_annual_tourists', 'majority_religion_encoded']
        
        # Encode religion
        le = LabelEncoder()
        clustering_df['majority_religion_encoded'] = le.fit_transform(clustering_df['majority_religion'])
        
        # Make sure tourists is numeric
        clustering_df['approximate_annual_tourists'] = pd.to_numeric(clustering_df['approximate_annual_tourists'], errors='coerce')
        clustering_df['approximate_annual_tourists'].fillna(clustering_df['approximate_annual_tourists'].median(), inplace=True)
        
        # Standardize numerical features
        scaler = StandardScaler()
        clustering_df[features] = scaler.fit_transform(clustering_df[features])
        
        # Apply K-Means
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clustering_df['tourism_cluster'] = kmeans.fit_predict(clustering_df[features])
        
        # Map clusters to tourism categories based on average tourist numbers
        tourism_means = clustering_df.groupby('tourism_cluster')['approximate_annual_tourists'].mean()
        sorted_tourism_clusters = tourism_means.sort_values().index
        
        cluster_tourism_map = {
            sorted_tourism_clusters[0]: 'Low',
            sorted_tourism_clusters[1]: 'Medium',
            sorted_tourism_clusters[2]: 'High'
        }
        
        model_info = {
            'kmeans': kmeans,
            'scaler': scaler,
            'label_encoder': le,
            'features': features,
            'cluster_tourism_map': cluster_tourism_map
        }
        
        return model_info
        
    except Exception as e:
        print(f"Error training religion tourism model: {str(e)}")
        return None

def recommend_by_religion_and_tourism(model_info, df, religion, tourism_level):
    """Recommend destinations based on religion and tourism levels."""
    if model_info is None:
        return pd.DataFrame(), "Model not available. Please try again later."
    
    try:
        # If both are "All", return all destinations
        if religion == "All" and tourism_level == "All":
            return df.copy(), None
            
        # Get model components
        kmeans = model_info['kmeans']
        scaler = model_info['scaler']
        le = model_info['label_encoder']
        features = model_info['features']
        cluster_tourism_map = model_info['cluster_tourism_map']
        
        # Prepare data for prediction
        prediction_df = df.copy()
        
        # Create the necessary columns
        prediction_df['majority_religion_encoded'] = le.transform(prediction_df['majority_religion'])
        
        # Make sure tourists is numeric
        prediction_df['approximate_annual_tourists'] = pd.to_numeric(prediction_df['approximate_annual_tourists'], errors='coerce')
        prediction_df['approximate_annual_tourists'].fillna(prediction_df['approximate_annual_tourists'].median(), inplace=True)
        
        # Scale features
        prediction_df[features] = scaler.transform(prediction_df[features])
        
        # Predict clusters
        prediction_df['tourism_cluster'] = kmeans.predict(prediction_df[features])
        
        # Map clusters to categories
        prediction_df['tourism_level'] = prediction_df['tourism_cluster'].map(cluster_tourism_map)
        
        # Filter based on user selections
        filtered_destinations = pd.DataFrame()
        message = None
        
        if religion != "All" and tourism_level != "All":
            filtered_destinations = prediction_df[
                (prediction_df['majority_religion'] == religion) & 
                (prediction_df['tourism_level'] == tourism_level)
            ].copy()
        elif religion != "All":
            filtered_destinations = prediction_df[prediction_df['majority_religion'] == religion].copy()
        elif tourism_level != "All":
            filtered_destinations = prediction_df[prediction_df['tourism_level'] == tourism_level].copy()
        
        if filtered_destinations.empty:
            message = f"No destinations match your criteria. Try different selections."
            
        return filtered_destinations, message
        
    except Exception as e:
        return pd.DataFrame(), f"Error finding recommendations: {str(e)}"
