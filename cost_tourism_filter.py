import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

def train_cost_tourism_model(df):
    """Train a model to cluster destinations based on cost of living and tourist numbers."""
    try:
        clustering_df = df.copy()

        # Features for clustering
        features = ['approximate_annual_tourists', 'safety_numeric', 'category_encoded']

        # Create numeric safety values
        clustering_df['safety_numeric'] = clustering_df['safety'].apply(lambda x: 1 if x == 'High' else 0)

        # Encode category
        le = LabelEncoder()
        clustering_df['category_encoded'] = le.fit_transform(clustering_df['category'])

        # Make sure tourists is numeric
        clustering_df['approximate_annual_tourists'] = pd.to_numeric(clustering_df['approximate_annual_tourists'], errors='coerce')
        clustering_df['approximate_annual_tourists'].fillna(clustering_df['approximate_annual_tourists'].median(), inplace=True)

        # Standardize numerical features
        scaler = StandardScaler()
        clustering_df[features] = scaler.fit_transform(clustering_df[features])

        # Apply K-Means
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clustering_df['cost_cluster'] = kmeans.fit_predict(clustering_df[features])

        # Map clusters to categories
        cost_means = clustering_df.groupby('cost_cluster')['cost_numeric'].mean()
        sorted_cost_clusters = cost_means.sort_values().index

        cluster_cost_map = {
            sorted_cost_clusters[0]: 'Low',
            sorted_cost_clusters[1]: 'Medium',
            sorted_cost_clusters[2]: 'High'
        }

        model_info = {
            'kmeans': kmeans,
            'scaler': scaler,
            'label_encoder': le,
            'features': features,
            'cluster_cost_map': cluster_cost_map
        }

        return model_info

    except Exception as e:
        print(f"Error training cost tourism model: {str(e)}")
        return None

def recommend_by_cost_and_tourism(model_info, df, cost_level, tourism_level):
    """Recommend destinations based on cost of living and tourism levels."""
    if model_info is None:
        return pd.DataFrame(), "Model not available. Please try again later."

    try:
        # If both are "All", return all destinations
        if cost_level == "All" and tourism_level == "All":
            return df.copy(), None

        # Prepare data for filtering
        filtered_df = df.copy()

        # Define tourism thresholds (in millions)
        tourism_thresholds = {
            'Low': (0, 1),  # 0-1 million
            'Medium': (1, 5),  # 1-5 million
            'High': (5, float('inf'))  # 5+ million
        }

        # Apply cost level filter if specified
        if cost_level != "All":
            filtered_df = filtered_df[filtered_df['cost_of_living'] == cost_level]

        # Apply tourism level filter if specified
        if tourism_level != "All":
            min_tourists, max_tourists = tourism_thresholds[tourism_level]
            filtered_df = filtered_df[
                (filtered_df['approximate_annual_tourists'] >= min_tourists * 1000000) &
                (filtered_df['approximate_annual_tourists'] < max_tourists * 1000000)
            ]

        if filtered_df.empty:
            return pd.DataFrame(), "No destinations match your criteria. Try different selections."

        return filtered_df, None

    except Exception as e:
        return pd.DataFrame(), f"Error finding recommendations: {str(e)}"