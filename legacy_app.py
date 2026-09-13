import os
import sys
import traceback
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt

# Set page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="Travel Destination Recommender",
    page_icon="✈️",
    layout="wide"
)

try:
    from data_processor import load_and_process_data
    from recommendation_engine import filter_destinations, calculate_match_score
    from ml_recommendation import train_recommendation_model, recommend_destinations_by_country_category
    from religion_safety_filter import train_safety_model, recommend_by_safety_and_religion
    from cost_time_filter import train_cost_time_model, recommend_by_cost_and_time
    from cost_tourism_filter import train_cost_tourism_model, recommend_by_cost_and_tourism
    from geo_filter import train_geo_model, recommend_destinations_by_location
    from religion_tourism_filter import train_religion_tourism_model, recommend_by_religion_and_tourism
    from country_time_tourists_filter import train_country_time_tourists_model, recommend_by_country_time_tourists
    from location_keyword_filter import train_location_keyword_model, recommend_by_location_and_keywords
    from all_features_filter import train_all_features_model, recommend_with_all_features
    from utils import display_destination_details, create_map
except Exception as e:
    st.error(f"Error importing modules: {str(e)}")
    st.error(traceback.format_exc())

# Application title and introduction
st.title("✈️ Travel Destination Recommender")
st.write("""
Find your perfect travel destination based on your preferences. 
Use the filters on the sidebar to narrow down destinations that match your interests.
""")

# Load data
try:
    df = load_and_process_data()
    if df is None or df.empty:
        st.error("No destination data available. Please check the data source.")
        st.stop()
except Exception as e:
    st.error(f"Error loading destination data: {str(e)}")
    st.error("Please ensure the destinations.xlsx file is available and properly formatted.")
    st.stop()

# Sidebar for filters
st.sidebar.title("Your Preferences")

# Extract unique values for filters
categories = sorted(df['category'].unique().tolist())
countries = sorted(df['country'].unique().tolist())
cost_ranges = sorted(df['cost_of_living'].unique().tolist())

# Standardize best time to visit options
best_times = ["All", "Spring", "Summer", "Fall", "Winter", "Any time"]
tourist_levels = sorted(df['approximate_annual_tourists'].unique().tolist())


# Create filters
with st.sidebar:
    # Creating a dropdown for filtering options
    filter_option = st.selectbox(
        "Filtering Options",
        ["Country & Category", "Religion & Safety", "Cost & Time", "Cost & Tourism", "Geo Filter", "Religion & Tourism", "Country-Time-Tourists", "Location & Keywords", "All Features"],
        index=0,
        key="filter_option"
    )

    # Initialize variables
    apply_ml_recommendation = False
    religion_safety_button = False
    cost_time_button = False
    cost_tourism_button = False
    geo_filter_button = False
    ml_country = ""
    ml_category = ""
    selected_religion = ""
    selected_safety = ""
    selected_cost = ""
    selected_time = ""
    selected_cost_level = ""
    selected_tourism_level = ""
    location_name = ""
    radius_km = None
    max_results = 10
    selected_country = ""
    selected_tourists = ""
    country_time_tourists_button = False


    # Train ML models in the background
    if 'ml_model' not in st.session_state:
        with st.spinner("Training recommendation model..."):
            st.session_state['ml_model'] = train_recommendation_model(df)

    # Train safety model in the background
    if 'safety_model' not in st.session_state:
        with st.spinner("Training safety model..."):
            safety_model, encoders, features = train_safety_model(df)
            st.session_state['safety_model'] = safety_model
            st.session_state['safety_encoders'] = encoders
            st.session_state['safety_features'] = features

    # Show appropriate filter options based on selection
    if filter_option == "Country & Category":
        st.write("Get personalized destination recommendations using machine learning.")

        # Single-select dropdowns for ML recommendation
        ml_country = st.selectbox(
            "Select Country",
            ["All"] + countries,
            index=0,
            key="ml_country"
        )

        ml_category = st.selectbox(
            "Select Category",
            ["All"] + categories,
            index=0,
            key="ml_category"
        )

        # Apply ML recommendation button - made more prominent
        apply_ml_recommendation = st.button("Get Recommendations", key="ml_button", use_container_width=True)

        # Information about the ML model
        with st.expander("About Country & Category Filter"):
            st.write("""
            This recommender uses machine learning to suggest destinations based on patterns in the data.
            It analyzes the relationships between countries, categories, and destinations to provide personalized recommendations.
            """)

    elif filter_option == "Religion & Safety":
        st.write("Find destinations by religion and safety preferences.")

        # Extract unique religions from the dataset
        religions = []
        if 'majority_religion' in df.columns:
            religions = sorted(df['majority_religion'].dropna().unique().tolist())

        # If no religions found, provide some common options
        if not religions:
            religions = [
                "Catholic", "Protestant", "Orthodox", "Christian", 
                "Muslim", "Hindu", "Buddhist", "Jewish",
                "Sikh", "Taoist", "Shinto"
            ]

        # Religion dropdown
        selected_religion = st.selectbox(
            "Select Religion",
            ["All"] + religions,
            index=0,
            key="religion_dropdown"
        )

        # Safety level dropdown
        selected_safety = st.selectbox(
            "Select Safety Level",
            ["All", "High Safety", "Low Safety"],
            index=0,
            key="safety_dropdown"
        )

        # Search button
        religion_safety_button = st.button(
            "Get Recommendations", 
            key="religion_safety_button",
            use_container_width=True
        )

        with st.expander("About Religion & Safety Filter"):
            st.write("""
            This filter helps you find destinations based on:

            1. Religious influence - locations where the selected religion has significant presence
            2. Safety level - destinations classified as having either high or low safety

            Select your preferences from the dropdowns and click 'Get Recommendations' to find matching destinations.
            """)

    elif filter_option == "Cost & Time":
        st.write("Find destinations based on budget and timing preferences.")

        # Train cost-time model in the background if not already trained
        if 'cost_time_model' not in st.session_state:
            with st.spinner("Training cost-time model..."):
                st.session_state['cost_time_model'] = train_cost_time_model(df)

        # Cost of Living dropdown
        selected_cost = st.selectbox(
            "Select Cost of Living",
            ["All"] + cost_ranges,
            index=0,
            key="cost_dropdown"
        )

        # Expanded Best Time to Visit options with individual seasons
        expanded_times = ["All"] + best_times

        # Add individual seasons that map to existing categories
        if "Spring/Fall" in best_times and "Spring" not in expanded_times:
            expanded_times.append("Spring")
            expanded_times.append("Fall")

        if "Winter/Summer" in best_times and "Winter" not in expanded_times:
            expanded_times.append("Winter")

        if "Summer" in best_times and expanded_times.count("Summer") < 1:
            expanded_times.append("Summer")

        # Sort the times but keep "All" at the beginning
        sorted_times = ["All"] + sorted([time for time in expanded_times if time != "All"])

        # Best Time to Visit dropdown
        selected_time = st.selectbox(
            "Select Best Time to Visit",
            sorted_times,
            index=0,
            key="time_dropdown"
        )

        # Search button
        cost_time_button = st.button(
            "Get Recommendations", 
            key="cost_time_button",
            use_container_width=True
        )

        with st.expander("About Cost & Time Filter"):
            st.write("""
            This filter helps you find destinations based on:

            1. Cost of Living - find destinations that match your budget
            2. Best Time to Visit - find places ideal to visit during your preferred season

            Select your preferences from the dropdowns and click 'Get Recommendations' to find matching destinations.
            """)

    elif filter_option == "Cost & Tourism":
        st.write("Find destinations based on cost of living and tourism popularity.")

        # Train cost-tourism model in the background if not already trained
        if 'cost_tourism_model' not in st.session_state:
            with st.spinner("Training cost-tourism model..."):
                st.session_state['cost_tourism_model'] = train_cost_tourism_model(df)

        # Cost Level dropdown
        selected_cost_level = st.selectbox(
            "Select Cost Level",
            ["All", "Low", "Medium", "High"],
            index=0,
            key="cost_level_dropdown"
        )

        # Tourism Level dropdown
        selected_tourism_level = st.selectbox(
            "Select Tourism Popularity",
            ["All", "Low", "Medium", "High"],
            index=0,
            key="tourism_level_dropdown"
        )

        # Search button
        cost_tourism_button = st.button(
            "Get Recommendations", 
            key="cost_tourism_button",
            use_container_width=True
        )

        with st.expander("About Cost & Tourism Filter"):
            st.write("""
            This filter helps you find destinations based on:

            1. Cost Level - find destinations that match your budget preference
            2. Tourism Popularity - find places based on tourist volume (low, medium, or high)

            The system uses machine learning clustering to group destinations by cost of living and annual tourist numbers.
            Select your preferences from the dropdowns and click 'Get Recommendations' to find matching destinations.
            """)

    elif filter_option == "Geo Filter":
        st.write("Find destinations near a specific location.")

        # Train geo model in the background if not already trained
        if 'geo_model' not in st.session_state:
            with st.spinner("Preparing geographical model..."):
                st.session_state['geo_model'] = train_geo_model(df)

        # Location input
        location_name = st.text_input(
            "Enter your location",
            "",
            key="location_input",
            help="Enter a city, region, or country name (e.g., Paris, Tokyo, New York)"
        )

        # Distance radius options
        radius_option = st.radio(
            "Search by:",
            ["Nearest destinations", "Custom radius"],
            index=0,
            key="radius_option"
        )

        # Radius input (shown only when custom radius is selected)
        radius_km = None
        max_results = 10

        if radius_option == "Custom radius":
            radius_km = st.slider(
                "Distance radius (km)",
                min_value=50,
                max_value=2000,
                value=500,
                step=50,
                key="radius_slider"
            )
        else:
            max_results = st.slider(
                "Number of nearest destinations",
                min_value=3,
                max_value=30,
                value=10,
                step=1,
                key="max_results_slider"
            )

        # Search button
        geo_filter_button = st.button(
            "Find Nearby Destinations", 
            key="geo_filter_button",
            use_container_width=True
        )

        with st.expander("About Geo Filter"):
            st.write("""
            This filter helps you find destinations based on location proximity.

            How to use:
            1. Enter your current location or a place of interest
            2. Choose whether to find the nearest destinations or set a custom search radius
            3. The system will find destinations that are closest to your specified location

            The distance is calculated using the Haversine formula, which accounts for the Earth's curvature to provide accurate distances.
            """)

    elif filter_option == "All Features":
        st.write("Get personalized recommendations using all features and cultural descriptions.")
        
        # Train all-features model in the background if not already trained
        if 'all_features_model' not in st.session_state:
            with st.spinner("Training comprehensive recommendation model..."):
                st.session_state['all_features_model'], error = train_all_features_model(df)
                if error:
                    st.error(error)
        
        # Create dropdowns for all features
        col1, col2 = st.columns(2)
        
        with col1:
            # Use preprocessed labels from the model
            model_labels = st.session_state['all_features_model']['unique_values']
            
            selected_country = st.selectbox(
                "Select Country",
                ["All"] + model_labels['countries'],
                key="all_features_country"
            )
            selected_category = st.selectbox(
                "Select Category",
                ["All"] + model_labels['categories'],
                key="all_features_category"
            )
            selected_religion = st.selectbox(
                "Select Religion",
                ["All"] + model_labels['religions'],
                key="all_features_religion"
            )
        
        with col2:
            selected_cost = st.selectbox(
                "Cost of Living",
                ["All"] + model_labels['costs'],
                key="all_features_cost"
            )
            selected_time = st.selectbox(
                "Best Time to Visit",
                ["All"] + model_labels['times'],
                key="all_features_time"
            )
            selected_safety = st.selectbox(
                "Safety Level",
                ["All"] + model_labels['safety'],
                key="all_features_safety"
            )
            
            keywords = st.text_input(
                "Enter Keywords of Interest",
                help="Enter keywords related to your interests (e.g., beaches, mountains, history)",
                key="all_features_keywords"
            )

        # Search button
        all_features_button = st.button(
            "Get Personalized Recommendations", 
            key="all_features_button",
            use_container_width=True
        )
        
        if all_features_button:
            if 'all_features_model' not in st.session_state:
                st.error("Recommendation model is not available. Please try again.")
            else:
                with st.spinner("Finding personalized recommendations..."):
                    # Prepare preferences
                    preferences = {
                        'country': selected_country if selected_country != "All" else None,
                        'category': selected_category if selected_category != "All" else None,
                        'majority_religion': selected_religion if selected_religion != "All" else None,
                        'cost_of_living': selected_cost if selected_cost != "All" else None,
                        'best_time_to_visit': selected_time if selected_time != "All" else None,
                        'safety': selected_safety if selected_safety != "All" else None,
                        'latitude': df['latitude'].mean(),
                        'longitude': df['longitude'].mean(),
                        'approximate_annual_tourists': df['approximate_annual_tourists'].mean()
                    }
                    
                    # Get recommendations
                    all_features_df, error = recommend_with_all_features(
                        st.session_state['all_features_model'],
                        df,
                        preferences,
                        keywords
                    )
                    
                    if error:
                        st.error(error)
                    elif all_features_df.empty:
                        st.warning("No destinations match your criteria. Try different selections.")
                    else:
                        # Store recommendations in session state
                        st.session_state['filtered_df'] = all_features_df
                        st.session_state['filter_mode'] = 'standard'
                        
                        st.subheader("Recommended Destinations")
                            
    elif filter_option == "Religion & Tourism":
        st.write("Find destinations based on religion and tourism popularity.")

        # Train religion-tourism model in the background if not already trained
        if 'religion_tourism_model' not in st.session_state:
            with st.spinner("Training religion-tourism model..."):
                st.session_state['religion_tourism_model'] = train_religion_tourism_model(df)

        # Get unique religions from the dataset
        religions = sorted(df['majority_religion'].unique().tolist())

        # Religion dropdown
        selected_religion = st.selectbox(
            "Select Religion",
            ["All"] + religions,
            index=0,
            key="religion_tourism_dropdown"
        )

        # Tourism Level dropdown
        selected_tourism_level = st.selectbox(
            "Select Tourism Level",
            ["All", "Low", "Medium", "High"],
            index=0,
            key="tourism_level_dropdown"
        )

        # Search button
        religion_tourism_button = st.button(
            "Get Recommendations",
            key="religion_tourism_button",
            use_container_width=True
        )

        with st.expander("About Religion & Tourism Filter"):
            st.write("""
            This filter helps you find destinations based on:

            1. Religion - find destinations with specific religious influences
            2. Tourism Level - find places based on tourist volume (low, medium, or high)

            The system uses machine learning clustering to group destinations by their religious significance and annual tourist numbers.
            Select your preferences from the dropdowns and click 'Get Recommendations' to find matching destinations.
            """)
    elif filter_option == "Location & Keywords":
        st.write("Find destinations based on location proximity and cultural keywords.")

        # Train location-keyword model in the background if not already trained
        if 'location_keyword_model' not in st.session_state:
            with st.spinner("Training location-keyword model..."):
                st.session_state['location_keyword_model'], error = train_location_keyword_model(df)
                if error:
                    st.error(error)

        # Location input
        location_name = st.text_input(
            "Enter your location",
            key="location_keyword_input",
            help="Enter a city, region, or country name (e.g., Paris, Tokyo, New York)"
        )

        # Keywords input
        keywords = st.text_input(
            "Enter keywords of interest",
            key="cultural_keywords_input",
            help="Enter keywords related to your interests (e.g., ancient temples, art museums, beaches)"
        )

        # Number of recommendations
        num_recommendations = st.slider(
            "Number of recommendations",
            min_value=3,
            max_value=10,
            value=5,
            key="num_recommendations_slider"
        )

        # Search button
        location_keyword_button = st.button(
            "Get Recommendations",
            key="location_keyword_button",
            use_container_width=True
        )

        with st.expander("About Location & Keywords Filter"):
            st.write("""
            This filter helps you find destinations based on:

            1. Proximity to your location
            2. Cultural and descriptive keywords matching your interests

            The system uses natural language processing to match your keywords with destination descriptions
            and calculates geographical distances to provide relevant recommendations near you.
            """)

    elif filter_option == "Country-Time-Tourists":
        st.write("Find destinations based on country, best time to visit, and tourist numbers.")

        # Country dropdown
        selected_country = st.selectbox(
            "Select Country",
            ["All"] + countries,
            index=0,
            key="country_dropdown"
        )

        # Best Time to Visit dropdown
        selected_time = st.selectbox(
            "Select Best Time to Visit",
            best_times,
            index=0,
            key="time_dropdown_country"
        )

        # Tourist Level dropdown
        selected_tourists = st.selectbox(
            "Select Tourist Level",
            ["All", "Low", "Medium", "High"],
            index=0,
            key="tourists_dropdown"
        )

        # Train model in the background if not already trained
        if 'country_time_tourists_model' not in st.session_state:
            with st.spinner("Training Country-Time-Tourists model..."):
                st.session_state['country_time_tourists_model'] = train_country_time_tourists_model(df)

        # Search button
        country_time_tourists_button = st.button(
            "Get Recommendations",
            key="country_time_tourists_button",
            use_container_width=True
        )

        with st.expander("About Country-Time-Tourists Filter"):
            st.write("""
            This filter helps you find destinations based on:

            1. Country - select a specific country or leave as "All"
            2. Best Time to Visit - choose a preferred season
            3. Tourist Level - specify your preference (low, medium, or high) tourist volume

            Select your preferences and click "Get Recommendations" for results.
            """)


    # Remove standard filters completely
    apply_filters = False
    min_rating = 0

# Main content area
# Initialize session state with ML filtered data if not present
if ('filtered_df' not in st.session_state and 'ml_filtered_df' not in st.session_state):
    # Set initial state to ML mode, but with no data yet
    st.session_state['filter_mode'] = 'ml'

# Handle ML-based recommendations
if apply_ml_recommendation:
    if ml_country == "All" and ml_category == "All":
        # If both are "All", just show all destinations
        ml_filtered_df = df.copy()

        # Store ML recommendations in session state
        st.session_state['ml_filtered_df'] = ml_filtered_df
        st.session_state['filter_mode'] = 'ml'

        # Show filter details as title
        st.subheader(f"All Destinations")

        # Clear standard filtered data if it exists
        if 'filtered_df' in st.session_state:
            del st.session_state['filtered_df']
    elif ml_country == "All" or ml_category == "All":
        st.warning("Please select a specific value for both Country and Category, or 'All' for both.")
    elif 'ml_model' in st.session_state and st.session_state['ml_model'] is not None:
        # Show loading indicator
        with st.spinner("Getting recommendations..."):
            # Use ML model to get recommendations
            ml_filtered_df = recommend_destinations_by_country_category(
                st.session_state['ml_model'],
                df,
                ml_country,
                ml_category
            )

            # Check if we got results
            if ml_filtered_df.empty:
                st.warning(f"No destinations found for {ml_category} in {ml_country}. Try a different combination.")
            else:
                # Store ML recommendations in session state
                st.session_state['ml_filtered_df'] = ml_filtered_df
                st.session_state['filter_mode'] = 'ml'

                # Show filter details as title
                st.subheader(f"ML Recommendations for {ml_category} in {ml_country}")

                # Clear standard filtered data if it exists
                if 'filtered_df' in st.session_state:
                    del st.session_state['filtered_df']
    else:
        st.error("ML recommendation model is not available. Please try again.")

# Handle Religion and Safety filter
if religion_safety_button:
    if selected_religion == "All" and selected_safety == "All":
        # If both are "All", show all destinations
        religion_safety_df = df.copy()
        error_message = None

        # Store recommendations in session state
        st.session_state['filtered_df'] = religion_safety_df
        st.session_state['filter_mode'] = 'standard'

        # Show filter details as title
        st.subheader(f"All Destinations")
    elif selected_religion == "All" or selected_safety == "All":
        # If only one is "All", prompt for both specific or both "All"
        st.warning("Please select a specific value for both Religion and Safety, or 'All' for both.")
        religion_safety_df = pd.DataFrame()
        error_message = "Incomplete filter selection"
    elif ('safety_model' in st.session_state and 
          'safety_encoders' in st.session_state and 
          'safety_features' in st.session_state):
        # Format the input for the filter function
        safety_level = "Low" if "Low" in selected_safety else "High"
        formatted_input = f"Find {safety_level.lower()} safety {selected_religion} destinations"

        # Show loading indicator
        with st.spinner("Searching for destinations matching your preferences..."):
            # Use the religion and safety filter
            religion_safety_df, error_message = recommend_by_safety_and_religion(
                formatted_input,
                df,
                st.session_state['safety_model'],
                st.session_state['safety_encoders'],
                st.session_state['safety_features']
            )

            # Check if we got results or an error
            if error_message:
                st.warning(error_message)
            elif religion_safety_df.empty:
                st.warning(f"No destinations match {selected_religion} religion with {selected_safety}. Try different criteria.")
            else:
                # Store recommendations in session state
                st.session_state['filtered_df'] = religion_safety_df
                st.session_state['filter_mode'] = 'standard'

                # Show filter details as title
                st.subheader(f"Destinations: {selected_religion} with {selected_safety}")

                # Clear ML filtered data if it exists
                if 'ml_filtered_df' in st.session_state:
                    del st.session_state['ml_filtered_df']
    else:
        st.error("Religion and safety model is not available. Please try again.")

# Handle Cost and Tourism filter
if cost_tourism_button:
    if 'cost_tourism_model' not in st.session_state:
        st.error("Cost & Tourism recommendation model is not available. Please try again.")
    else:
        # Show loading indicator
        with st.spinner("Finding destinations that match your cost and tourism preferences..."):
            # Use the cost and tourism filter
            cost_tourism_df, error_message = recommend_by_cost_and_tourism(
                st.session_state['cost_tourism_model'],
                df,
                selected_cost_level,
                selected_tourism_level
            )

            # Check if we got results or an error
            if error_message:
                st.warning(error_message)
            elif cost_tourism_df.empty:
                st.warning(f"No destinations match {selected_cost_level} cost level and {selected_tourism_level} tourism level. Try different criteria.")
            else:
                # Store recommendations in session state
                st.session_state['filtered_df'] = cost_tourism_df
                st.session_state['filter_mode'] = 'standard'

                # Show filter details as title
                if selected_cost_level == "All" and selected_tourism_level == "All":
                    st.subheader(f"All Destinations")
                elif selected_cost_level == "All":
                    st.subheader(f"Destinations with {selected_tourism_level} tourism popularity")
                elif selected_tourism_level == "All":
                    st.subheader(f"Destinations with {selected_cost_level} cost level")
                else:
                    st.subheader(f"Destinations: {selected_cost_level} cost with {selected_tourism_level} tourism popularity")

                # Clear ML filtered data if it exists
                if 'ml_filtered_df' in st.session_state:
                    del st.session_state['ml_filtered_df']

# Handle Cost and Time filter
if cost_time_button:
    if 'cost_time_model' not in st.session_state:
        st.error("Cost & Time recommendation model is not available. Please try again.")
    else:
        # Show loading indicator
        with st.spinner("Finding destinations that match your preferences..."):
            # Use the cost and time filter
            cost_time_df, error_message = recommend_by_cost_and_time(
                st.session_state['cost_time_model'],
                df,
                selected_cost,
                selected_time
            )

            # Check if we got results or an error
            if error_message:
                st.warning(error_message)
            elif cost_time_df.empty:
                st.warning(f"No destinations match {selected_cost} cost of living and {selected_time} best time to visit. Try different criteria.")
            else:
                # Store recommendations in session state
                st.session_state['filtered_df'] = cost_time_df
                st.session_state['filter_mode'] = 'standard'

                # Show filter details as title
                if selected_cost == "All" and selected_time == "All":
                    st.subheader(f"All Destinations")
                elif selected_cost == "All":
                    st.subheader(f"Destinations best to visit in {selected_time}")
                elif selected_time == "All":
                    st.subheader(f"Destinations with {selected_cost} cost of living")
                else:
                    st.subheader(f"Destinations: {selected_cost} cost with {selected_time} best time to visit")

                # Clear ML filtered data if it exists
                if 'ml_filtered_df' in st.session_state:
                    del st.session_state['ml_filtered_df']

# Handle Geo Filter
# Handle Religion and Tourism filter
if 'religion_tourism_button' in locals() and religion_tourism_button:
    if 'religion_tourism_model' not in st.session_state:
        st.error("Religion & Tourism recommendation model is not available. Please try again.")
    else:
        # Show loading indicator
        with st.spinner("Finding destinations that match your religion and tourism preferences..."):
            # Use the religion and tourism filter
            religion_tourism_df, error_message = recommend_by_religion_and_tourism(
                st.session_state['religion_tourism_model'],
                df,
                selected_religion,
                selected_tourism_level
            )

            # Check if we got results or an error
            if error_message:
                st.warning(error_message)
            elif religion_tourism_df.empty:
                st.warning(f"No destinations match {selected_religion} religion with {selected_tourism_level} tourism level. Try different criteria.")
            else:
                # Store recommendations in session state
                st.session_state['filtered_df'] = religion_tourism_df
                st.session_state['filter_mode'] = 'standard'

                # Show filter details as title
                if selected_religion == "All" and selected_tourism_level == "All":
                    st.subheader(f"All Destinations")
                elif selected_religion == "All":
                    st.subheader(f"Destinations with {selected_tourism_level} tourism level")
                elif selected_tourism_level == "All":
                    st.subheader(f"Destinations with {selected_religion} majority")
                else:
                    st.subheader(f"Destinations: {selected_religion} with {selected_tourism_level} tourism level")

                # Clear ML filtered data if it exists
                if 'ml_filtered_df' in st.session_state:
                    del st.session_state['ml_filtered_df']

if 'geo_filter_button' in locals() and geo_filter_button:
    if not location_name.strip():
        st.warning("Please enter a location to find nearby destinations.")
    elif 'geo_model' not in st.session_state or st.session_state['geo_model'] is None:
        st.error("Geographical recommendation model is not available. Please try again.")
    else:
        # Show loading indicator
        with st.spinner("Finding destinations near your location..."):
            # Use the geo filter
            geo_filtered_df, error_message = recommend_destinations_by_location(
                st.session_state['geo_model'],
                df,
                location_name,
                radius_km,
                max_results
            )

            # Check if we got results or an error
            if error_message:
                st.warning(error_message)
            elif geo_filtered_df.empty:
                if radius_km is not None:
                    st.warning(f"No destinations found within {radius_km} km of {location_name}. Try a larger radius or a different location.")
                else:
                    st.warning(f"No destinations found near {location_name}. Try a different location.")
            else:
                # Store recommendations in session state
                st.session_state['filtered_df'] = geo_filtered_df
                st.session_state['filter_mode'] = 'standard'

                # Show filter details as title
                if radius_km is not None:
                    st.subheader(f"Destinations within {radius_km} km of {location_name}")
                else:
                    st.subheader(f"Nearest {len(geo_filtered_df)} destinations to {location_name}")

                # Clear ML filtered data if it exists
                if 'ml_filtered_df' in st.session_state:
                    del st.session_state['ml_filtered_df']

# Handle Location & Keywords filter
if 'location_keyword_button' in locals() and location_keyword_button:
    if not location_name.strip():
        st.warning("Please enter a location to find nearby destinations.")
    elif not keywords.strip():
        st.warning("Please enter some keywords to help find matching destinations.")
    elif 'location_keyword_model' not in st.session_state:
        st.error("Location & Keywords recommendation model is not available. Please try again.")
    else:
        with st.spinner("Finding destinations that match your interests..."):
            filtered_df, error_message = recommend_by_location_and_keywords(
                st.session_state['location_keyword_model'],
                keywords,
                location_name,
                num_recommendations
            )

            if error_message:
                st.warning(error_message)
            elif filtered_df.empty:
                st.warning("No destinations found matching your criteria. Try different keywords or location.")
            else:
                st.session_state['filtered_df'] = filtered_df
                st.session_state['filter_mode'] = 'standard'

                st.subheader(f"Destinations near {location_name} matching '{keywords}'")

                # Clear ML filtered data if it exists
                if 'ml_filtered_df' in st.session_state:
                    del st.session_state['ml_filtered_df']

# Handle Country-Time-Tourists filter
if 'country_time_tourists_button' in locals() and country_time_tourists_button:
    if 'country_time_tourists_model' not in st.session_state:
        st.error("Country-Time-Tourists recommendation model is not available. Please try again.")
    else:
        with st.spinner("Finding destinations that match your preferences..."):
            country_time_tourists_df, error_message = recommend_by_country_time_tourists(
                st.session_state['country_time_tourists_model'],
                df,
                selected_country,
                selected_time,
                selected_tourists
            )

            if error_message:
                st.warning(error_message)
            elif country_time_tourists_df.empty:
                st.warning("No destinations match your criteria. Try different selections.")
            else:
                st.session_state['filtered_df'] = country_time_tourists_df
                st.session_state['filter_mode'] = 'standard'

                if selected_country == "All" and selected_time == "All" and selected_tourists == "All":
                    st.subheader("All Destinations")
                else:
                    filters = []
                    if selected_country != "All":
                        filters.append(f"{selected_country}")
                    if selected_time != "All":
                        filters.append(f"{selected_time}")
                    if selected_tourists != "All":
                        filters.append(f"{selected_tourists} tourism")
                    st.subheader(f"Destinations: {', '.join(filters)}")

                if 'ml_filtered_df' in st.session_state:
                    del st.session_state['ml_filtered_df']

# Display results based on filter mode
if 'filter_mode' in st.session_state:
    # Get appropriate dataframe based on filter mode
    if st.session_state['filter_mode'] == 'standard' and 'filtered_df' in st.session_state:
        filtered_df = st.session_state['filtered_df']
        is_ml_mode = False
    elif st.session_state['filter_mode'] == 'ml' and 'ml_filtered_df' in st.session_state:
        filtered_df = st.session_state['ml_filtered_df']
        is_ml_mode = True
    else:
        filtered_df = None
        is_ml_mode = False

    # Check if we have results
    if filtered_df is None or filtered_df.empty:
        st.warning("No destinations match your criteria. Try adjusting your filters.")
    else:
        if not is_ml_mode:
            st.subheader(f"Found {len(filtered_df)} Destinations Matching Your Preferences")

        # Sorting options
        sort_col1, sort_col2 = st.columns([1, 3])
        with sort_col1:
            # Determine sort options based on the data
            sort_options = ["Name", "Rating", "Cost of Living"]

            # Add Distance option if available in the DataFrame (for Geo Filter results)
            if 'filtered_df' in st.session_state and 'distance_km' in st.session_state['filtered_df'].columns:
                sort_options.append("Distance")
                default_index = 3  # Default to Distance for geo results
            else:
                default_index = 0  # Default to Name

            sort_by = st.selectbox(
                "Sort by",
                options=sort_options,
                index=default_index
            )

        with sort_col2:
            sort_order = st.radio(
                "Order",
                options=["Descending", "Ascending"],
                index=0,
                horizontal=True
            )

        # Apply sorting
        if sort_by == "Name":
            sort_column = "name"
            ascending = sort_order == "Ascending"
        elif sort_by == "Rating" and "rating" in filtered_df.columns:
            sort_column = "rating"
            ascending = sort_order == "Ascending"
        elif sort_by == "Cost of Living" and "cost_numeric" in filtered_df.columns:
            sort_column = "cost_numeric"
            ascending = sort_order == "Ascending"
        elif sort_by == "Distance" and "distance_km" in filtered_df.columns:
            sort_column = "distance_km"
            # For distance, we can respect the sort order but default to ascending
            ascending = True if sort_order == "Ascending" else False
        else:
            # Fallback to name if the selected column doesn't exist
            sort_column = "name"
            ascending = sort_order == "Ascending"

        sorted_df = filtered_df.sort_values(by=sort_column, ascending=ascending)

        # Display map and statistics side by side
        map_col, stats_col = st.columns([3, 2])

        with map_col:
            st.subheader("Destination Map")
            try:
                # Create and display map
                destination_map = create_map(sorted_df)
                folium_static(destination_map, width=800, height=500)
            except Exception as e:
                st.error(f"Error creating map: {str(e)}")

        with stats_col:
            st.subheader("Most Visited Destinations")
            try:
                # Make sure we have the tourists column in a numeric format
                if 'approximate_annual_tourists' in df.columns:
                    # Sort by most visited places and select the top 10
                    top_destinations = df.nlargest(10, "approximate_annual_tourists")

                    # Create figure and plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.barh(top_destinations["name"], top_destinations["approximate_annual_tourists"] / 1000000, color='skyblue')
                    ax.set_xlabel("Number of Tourists (in millions)")
                    ax.set_ylabel("Destination")
                    ax.set_title("Top 10 Most Visited Destinations")
                    ax.invert_yaxis()  # Invert y-axis to show highest at top
                    ax.grid(axis="x", linestyle="--", alpha=0.7)

                    # Add values on bars
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width:.1f}M', ha='left', va='center')

                    # Display the plot in Streamlit
                    st.pyplot(fig)
                else:
                    st.warning("Tourist data not available for visualization")
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")

        # Display destination results in a grid
        st.subheader("Recommended Destinations")

        # Create rows of 3 destinations each
        for i in range(0, len(sorted_df), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(sorted_df):
                    with cols[j]:
                        dest = sorted_df.iloc[i + j]
                        display_destination_details(dest)

        # Download results option
        csv = sorted_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Results as CSV",
            data=csv,
            file_name="recommended_destinations.csv",
            mime="text/csv",
        )

# Footer
st.markdown("---")
st.write("© 2023 Travel Destination Recommender | Data updated regularly")