import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from sklearn.neighbors import NearestNeighbors
import math

# Function to get geographical coordinates of a place using Geopy
def get_coordinates_from_place(place_name):
    """
    Get the latitude and longitude of a place name using Geopy's Nominatim service.
    
    Parameters:
    - place_name: String representing the location to geocode
    
    Returns:
    - Tuple of (latitude, longitude) or None if location not found
    """
    try:
        geolocator = Nominatim(user_agent="destination_recommender")
        location = geolocator.geocode(place_name)
        if location:
            return location.latitude, location.longitude
        else:
            return None
    except Exception as e:
        print(f"Error geocoding {place_name}: {str(e)}")
        return None

# Function to calculate the Haversine distance between two points
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on the Earth (specified in decimal degrees).
    
    Parameters:
    - lat1, lon1: Latitude and longitude of first point
    - lat2, lon2: Latitude and longitude of second point
    
    Returns:
    - Distance between points in kilometers
    """
    R = 6371.0  # Radius of Earth in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c  # Resulting distance in kilometers
    return distance

def train_geo_model(df):
    """
    Prepare the data for geographical filtering. This function doesn't actually 'train' a model
    in the traditional sense but prepares the necessary components for nearest neighbor calculations.
    
    Parameters:
    - df: DataFrame containing destination data with latitude and longitude columns
    
    Returns:
    - Dictionary containing the model-related information
    """
    try:
        # Ensure the dataframe has the necessary columns
        lat_col = 'Latitude' if 'Latitude' in df.columns else 'latitude' if 'latitude' in df.columns else None
        lon_col = 'Longitude' if 'Longitude' in df.columns else 'longitude' if 'longitude' in df.columns else None
        
        if lat_col is None or lon_col is None:
            return None
        
        # Prepare the coordinates data
        X = df[[lat_col, lon_col]].values
        
        # Create the NearestNeighbors model
        nbrs = NearestNeighbors(metric='haversine')
        nbrs.fit(np.radians(X))  # Convert to radians for Haversine calculation
        
        return {
            'nearest_neighbors': nbrs,
            'coordinates': X
        }
    except Exception as e:
        print(f"Error training geo model: {str(e)}")
        return None

def recommend_destinations_by_location(model_info, df, location_name, radius_km=None, max_results=10):
    """
    Recommend destinations near a specific location using NearestNeighbors algorithm.
    
    Parameters:
    - model_info: Dictionary containing the trained NearestNeighbors model
    - df: Original DataFrame containing destination data
    - location_name: Name of the location to search near
    - radius_km: Radius in kilometers to search (None for fixed number of results)
    - max_results: Maximum number of results to return
    
    Returns:
    - DataFrame with recommended destinations and distances, error message if any
    """
    if model_info is None:
        return pd.DataFrame(), "Geo model is not available. Please try again later."
    
    try:
        # Get coordinates of the specified location
        coords = get_coordinates_from_place(location_name)
        if coords is None:
            return pd.DataFrame(), f"Could not find coordinates for: {location_name}. Try a different location."
        
        user_lat, user_lon = coords
        
        # Determine which coordinate columns are used
        lat_col = 'Latitude' if 'Latitude' in df.columns else 'latitude' if 'latitude' in df.columns else None
        lon_col = 'Longitude' if 'Longitude' in df.columns else 'longitude' if 'longitude' in df.columns else None
        
        if lat_col is None or lon_col is None:
            return pd.DataFrame(), "Coordinate columns not found in dataset."
        
        # Get the NearestNeighbors model
        nbrs = model_info['nearest_neighbors']
        
        # Get appropriate number of neighbors
        if radius_km is None:
            # Fixed number of results
            n_neighbors = min(max_results, len(df))
            user_location = np.radians(np.array([[user_lat, user_lon]]))  # Convert to radians
            distances, indices = nbrs.kneighbors(user_location, n_neighbors=n_neighbors)
            
            # Convert distances from radians to kilometers
            distances = distances[0] * 6371.0
            indices = indices[0]
            
            # Only include destinations within the maximum range if specified
            if radius_km is not None:
                valid_indices = [i for i, d in zip(indices, distances) if d <= radius_km]
                valid_distances = [d for d in distances if d <= radius_km]
                indices = valid_indices
                distances = valid_distances
        else:
            # Use radius-based query
            user_location = np.radians(np.array([[user_lat, user_lon]]))  # Convert to radians
            radius_rad = radius_km / 6371.0  # Convert km to radians
            indices = nbrs.radius_neighbors(user_location, radius=radius_rad, return_distance=False)[0]
            
            # Calculate distances manually
            distances = []
            for idx in indices:
                lat2 = float(df.iloc[idx][lat_col])
                lon2 = float(df.iloc[idx][lon_col])
                distance = haversine(user_lat, user_lon, lat2, lon2)
                distances.append(distance)
            
            # Sort by distance
            sorted_indices = sorted(range(len(distances)), key=lambda k: distances[k])
            indices = [indices[i] for i in sorted_indices]
            distances = [distances[i] for i in sorted_indices]
            
            # Limit to max results
            indices = indices[:max_results]
            distances = distances[:max_results]
        
        if not indices:
            return pd.DataFrame(), f"No destinations found within {radius_km} km of {location_name}."
            
        # Create result dataframe
        result_df = df.iloc[indices].copy()
        result_df['distance_km'] = distances
        
        # Sort by distance
        result_df = result_df.sort_values('distance_km')
        
        return result_df, None
        
    except Exception as e:
        return pd.DataFrame(), f"Error finding nearby destinations: {str(e)}"