import streamlit as st
import folium
from folium.plugins import MarkerCluster
import pandas as pd

def display_destination_details(destination):
    """
    Display details for a single destination in a card-like format.
    
    Parameters:
    - destination: Series containing destination data
    """
    # Create a card-like display for each destination
    st.markdown(f"### {destination['name']}")
    
    # Display basic information - Removed ML Score display
    st.markdown(f"**Category:** {destination['category']}")
    st.markdown(f"**Country:** {destination['country']}")
    st.markdown(f"**Cost of Living:** {destination['cost_of_living']}")
    st.markdown(f"**Best Time to Visit:** {destination['best_time_to_visit']}")
    
    # Display distance if available (for Geo Filter results)
    if 'distance_km' in destination and pd.notna(destination['distance_km']):
        st.markdown(f"**Distance:** {destination['distance_km']:.1f} km")
    
    # Display rating if available
    if 'rating' in destination and pd.notna(destination['rating']):
        st.markdown(f"**Rating:** {destination['rating']}")
    
    # Display additional details if available
    if 'description' in destination and pd.notna(destination['description']):
        with st.expander("Description"):
            st.write(destination['description'])
    
    # Display activities if available
    if 'activities' in destination and pd.notna(destination['activities']):
        with st.expander("Popular Activities"):
            activities = destination['activities'].split(',') if isinstance(destination['activities'], str) else []
            for activity in activities:
                st.markdown(f"- {activity.strip()}")
    
    st.markdown("---")

def get_score_color(score):
    """
    Return a color based on the match score.
    
    Parameters:
    - score: Match score value (0-100)
    
    Returns:
    - Color code
    """
    if score >= 80:
        return "#4CAF50"  # Green
    elif score >= 60:
        return "#8BC34A"  # Light Green
    elif score >= 40:
        return "#FFC107"  # Amber
    elif score >= 20:
        return "#FF9800"  # Orange
    else:
        return "#F44336"  # Red

def create_map(df):
    """
    Create an interactive map with destination markers.
    
    Parameters:
    - df: Dataframe containing destination data with latitude and longitude
    
    Returns:
    - Folium map object
    """
    # Create a base map centered on the average coordinates
    avg_lat = df['Latitude'].mean() if 'Latitude' in df.columns else df['latitude'].mean() if 'latitude' in df.columns else 20.0
    avg_lon = df['Longitude'].mean() if 'Longitude' in df.columns else df['longitude'].mean() if 'longitude' in df.columns else 0.0
    
    # Default to world view if coordinates are missing or zero
    if pd.isna(avg_lat) or pd.isna(avg_lon) or (avg_lat == 0 and avg_lon == 0):
        avg_lat, avg_lon = 20.0, 0.0  # World center-ish
    
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=3)
    
    # Add a marker cluster for better visualization
    marker_cluster = MarkerCluster().add_to(m)
    
    # Add markers for each destination
    for idx, row in df.iterrows():
        # Handle different column names for coordinates
        lat_col = 'Latitude' if 'Latitude' in row.index else 'latitude' if 'latitude' in row.index else None
        lon_col = 'Longitude' if 'Longitude' in row.index else 'longitude' if 'longitude' in row.index else None
        
        # Skip if coordinate columns don't exist
        if lat_col is None or lon_col is None:
            continue
            
        # Skip if coordinates are missing or zero (likely placeholders)
        if pd.isna(row[lat_col]) or pd.isna(row[lon_col]) or (row[lat_col] == 0 and row[lon_col] == 0):
            continue
        
        # Create popup content
        popup_html = f"""
        <div style="width: 200px">
            <h4>{row['name']}</h4>
            <p><b>Category:</b> {row['category']}</p>
            <p><b>Country:</b> {row['country']}</p>
            <p><b>Cost:</b> {row['cost_of_living']}</p>
            <p><b>Best Time:</b> {row['best_time_to_visit']}</p>
        """
        
        # Add rating if available
        if 'rating' in row and pd.notna(row['rating']):
            popup_html += f"<p><b>Rating:</b> {row['rating']}</p>"
        
        # Add distance if available (for Geo Filter results)
        if 'distance_km' in row and pd.notna(row['distance_km']):
            popup_html += f"<p><b>Distance:</b> {row['distance_km']:.1f} km</p>"
        
        popup_html += "</div>"
        
        # Use a consistent blue color for all markers - Removed ML Score coloring
        color = 'blue'
        
        # Add marker to cluster
        folium.Marker(
            location=[row[lat_col], row[lon_col]],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=row['name'],
            icon=folium.Icon(color=color, icon='info-sign')
        ).add_to(marker_cluster)
    
    return m
