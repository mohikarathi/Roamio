import pandas as pd
import numpy as np
import os
import streamlit as st

@st.cache_data
def load_and_process_data():
    """
    Load and process the destination data from Excel file.
    Returns a processed DataFrame or None if the file is not found.
    """
    try:
        # Try to load the Excel file from the attached_assets directory
        file_path = "attached_assets/destinations.xlsx"
        df = pd.read_excel(file_path)
        
        # Check if data is loaded
        if df.empty:
            st.warning("The destinations file is empty.")
            return None
            
        # Apply custom preprocessing as requested
        
        # Standardize Cost of Living values
        if 'Cost of Living' in df.columns:
            df['Cost of Living'].replace({
                "Extremely High": "High",
                "Medium-high": "Medium",
                "Varies": "Medium",
                "Free": "Low"
            }, inplace=True)
            
            # Fill missing Cost of Living values with 'Medium'
            df['Cost of Living'].fillna('Medium', inplace=True)
        
        # Standardize Safety values
        if 'Safety' in df.columns:
            df['Safety'].replace({
                "Generally safe, but watch out for pickpockets": "High",
                "Generally safe, but be aware of pickpockets": "High",
                "Generally safe, but watch for pickpockets": "High",
                "Generally safe, but be aware of crowds": "High",
                "Generally safe": "High",
                "Generally safe, but be aware of bears": "Low",
                "Generally safe, but be aware of ongoing conflict": "Low",
                "Generally safe, but be aware of potential risks": "Low",
                "Restricted access": "Low"
            }, inplace=True)
        
        # Combine Cultural Significance and Description into Cultural_Description
        if 'Cultural Significance' in df.columns and 'Description' in df.columns:
            df['Cultural_Description'] = df['Cultural Significance'].fillna('') + ' ' + df['Description'].fillna('')
            df = df.drop(columns=['Cultural Significance', 'Description'])
        
        # Standardize Best Time to Visit values
        if 'Best Time to Visit' in df.columns:
            df['Best Time to Visit'].replace({
                'Spring (April-May) or Fall (Sept-Oct)': 'Spring/Fall',
                'Summer (June-September)': 'Summer',
                'Winter (Dec-Mar) for skiing, Summer (Jun-Sept)': 'Winter/Summer',
                'Summer (Jun-Sept)': 'Summer',
                'Summer (June-August)': 'Summer',
                'Year-round': 'Any time',
                'Spring (May-June) or Fall (Sept-Oct)': 'Spring/Fall',
                'Winter (Dec-Mar) for Northern Lights, Summer (Jun-Aug) for hiking': 'Winter/Summer',
                'Summer (Jun-Aug)': 'Summer',
                'Spring (Apr-May) or Fall (Sep-Oct)': 'Spring/Fall',
                'Winter (Dec-Mar) for skiing, Summer (Jun-Sep) for hiking': 'Winter/Summer',
                'Spring (May-June) or Fall (Sep-Oct)': 'Spring/Fall',
                'Spring (April-May) or Fall (Sep-Oct)': 'Spring/Fall',
                'Winter (December-March) for skiing, Summer (June-August) for hiking': 'Winter/Summer'
            }, inplace=True)
        
        # Convert Approximate Annual Tourists to numeric values
        if 'Approximate Annual Tourists' in df.columns:
            df['Approximate Annual Tourists'].replace({
                '14 million': 14000000,
                '10 million': 10000000,
                '7 million': 7000000,
                '5 million': 5000000,
                '3 million': 3000000,
                '2 million': 2000000,
                '1.5 million': 1500000,
                '1 million': 1000000,
                '12.7 million': 12700000,
                '3.5 million': 3500000,
                '2.5 million': 2500000,
                '8 million': 8000000,
                '12 million': 12000000,
                '35-40 million': 40000000,
                '4 million': 4000000,
                '800,000': 800000,
                '10 million (region-wide)': 10000000,
                '12 million (region-wide)': 12000000,
                '7.5 million': 7500000,
                '500,000': 500000,
                '400,000': 400000,
                '200,000': 200000,
                '300,000': 300000,
                '100,000': 100000,
                '13.5 million': 13500000,
                '50,000': 50000,
                '12.5 million': 12500000,
                '2 million (tourists)': 2000000,
                '150,000': 150000,
                '350,000': 350000,
                '15 million': 15000000,
                '25 million': 25000000,
                '25,000': 25000,
                '20,000': 20000,
                '10,000': 10000
            }, inplace=True)
        
        # Drop specified columns if they exist
        columns_to_drop = ['Language', 'Currency', 'Region']
        existing_columns = [col for col in columns_to_drop if col in df.columns]
        if existing_columns:
            df = df.drop(columns=existing_columns)
        
        # Basic data cleaning
        # Convert column names to lowercase and remove spaces
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        # Ensure required columns exist
        required_columns = ['name', 'category', 'country', 'cost_of_living', 'best_time_to_visit', 'latitude', 'longitude']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            # If some columns are missing, try to infer them or create placeholders
            for col in missing_columns:
                if col == 'name' and 'destination' in df.columns:
                    df['name'] = df['destination']
                elif col == 'category' and 'type' in df.columns:
                    df['category'] = df['type']
                elif col == 'latitude' and 'lat' in df.columns:
                    df['latitude'] = df['lat']
                elif col == 'longitude' and 'long' in df.columns or 'lng' in df.columns:
                    df['longitude'] = df['long'] if 'long' in df.columns else df['lng']
                else:
                    # Create placeholder for missing columns
                    df[col] = "Unknown"
        
        # Fill missing values
        df = df.fillna({
            'name': 'Unknown',
            'category': 'Other',
            'country': 'Unknown',
            'cost_of_living': 'Medium',
            'best_time_to_visit': 'Any time'
        })
        
        # Convert latitude and longitude to numeric if they exist
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # Try to convert to float, replace errors with default values
            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
            
            # Fill missing coordinates with defaults (showing data is missing)
            df['latitude'] = df['latitude'].fillna(0)
            df['longitude'] = df['longitude'].fillna(0)
        
        # Add numeric representation of cost for sorting
        if 'cost_of_living' in df.columns:
            cost_map = {
                'very low': 1, 
                'low': 2, 
                'medium': 3, 
                'high': 4, 
                'very high': 5
            }
            
            # Convert to lowercase for consistent mapping
            df['cost_lower'] = df['cost_of_living'].str.lower()
            df['cost_numeric'] = df['cost_lower'].map(lambda x: next((v for k, v in cost_map.items() if k in x), 3))
            df.drop('cost_lower', axis=1, inplace=True)
        
        return df
    
    except FileNotFoundError:
        st.error(f"Destination data file not found. Please ensure 'destinations.xlsx' is in the attached_assets directory.")
        return None
    except Exception as e:
        st.error(f"Error processing destination data: {str(e)}")
        return None
