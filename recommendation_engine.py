import pandas as pd
import numpy as np
import streamlit as st

def filter_destinations(df, categories=None, countries=None, cost_ranges=None, best_times=None, min_rating=0):
    """
    Filter destinations based on user preferences.
    
    Parameters:
    - df: Dataframe containing destination data
    - categories: List of selected destination categories
    - countries: List of selected countries
    - cost_ranges: List of selected cost of living ranges
    - best_times: List of selected best times to visit
    - min_rating: Minimum rating value
    
    Returns:
    - Filtered dataframe
    """
    filtered_df = df.copy()
    
    # Apply filters if they are provided and not empty
    if categories and len(categories) > 0:
        filtered_df = filtered_df[filtered_df['category'].isin(categories)]
    
    if countries and len(countries) > 0:
        filtered_df = filtered_df[filtered_df['country'].isin(countries)]
    
    if cost_ranges and len(cost_ranges) > 0:
        filtered_df = filtered_df[filtered_df['cost_of_living'].isin(cost_ranges)]
    
    if best_times and len(best_times) > 0:
        filtered_df = filtered_df[filtered_df['best_time_to_visit'].isin(best_times)]
    
    # Apply rating filter if column exists
    if 'rating' in filtered_df.columns and min_rating > 0:
        filtered_df = filtered_df[filtered_df['rating'] >= min_rating]
    
    return filtered_df

def calculate_match_score(df, selected_categories=None, selected_countries=None, selected_cost=None, selected_time=None):
    """
    Calculate a match score for each destination based on how well it matches the user's preferences.
    
    Parameters:
    - df: Dataframe containing filtered destination data
    - selected_categories: List of selected destination categories
    - selected_countries: List of selected countries
    - selected_cost: List of selected cost of living ranges
    - selected_time: List of selected best times to visit
    
    Returns:
    - Dataframe with added match_score column
    """
    if df.empty:
        return df
    
    # Initialize match scores as zeros
    df['match_score'] = 0
    
    # Calculate scores based on matches with user preferences
    
    # Category match (highest weight)
    if selected_categories and len(selected_categories) < len(df['category'].unique()):
        category_weight = 40
        df['match_score'] += df['category'].apply(
            lambda x: category_weight if x in selected_categories else 0
        )
    
    # Country match
    if selected_countries and len(selected_countries) > 0:
        country_weight = 30
        df['match_score'] += df['country'].apply(
            lambda x: country_weight if x in selected_countries else 0
        )
    
    # Cost of living match
    if selected_cost and len(selected_cost) < len(df['cost_of_living'].unique()):
        cost_weight = 15
        df['match_score'] += df['cost_of_living'].apply(
            lambda x: cost_weight if x in selected_cost else 0
        )
    
    # Best time to visit match
    if selected_time and len(selected_time) > 0:
        time_weight = 15
        df['match_score'] += df['best_time_to_visit'].apply(
            lambda x: time_weight if x in selected_time else 0
        )
    
    # Add rating bonus if available
    if 'rating' in df.columns:
        # Normalize ratings to 0-10 scale and add to score
        max_rating = df['rating'].max()
        if max_rating > 0:  # Avoid division by zero
            df['match_score'] += (df['rating'] / max_rating) * 10
    
    # Normalize match score to 0-100 range
    max_score = df['match_score'].max()
    if max_score > 0:  # Avoid division by zero
        df['match_score'] = (df['match_score'] / max_score) * 100
    
    return df
