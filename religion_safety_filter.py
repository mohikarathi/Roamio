import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit as st

@st.cache_resource
def train_safety_model(df):
    """
    Train a machine learning model for safety classification.
    
    Parameters:
    - df: DataFrame containing the destination data
    
    Returns:
    - Trained model and label encoders
    """
    # Make a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Make sure column names are standardized
    df_copy.columns = [col.lower() for col in df_copy.columns]
    
    # Map columns to the ones expected in the code
    column_mapping = {
        'name': 'Destination',
        'country': 'Country',
        'category': 'Category',
        'cost_of_living': 'Cost of Living',
        'cultural_description': 'Cultural_Description',
        'safety': 'Safety',
        'majority_religion': 'Majority Religion'
    }
    
    # Rename columns based on available columns
    for new_col, old_col in column_mapping.items():
        if old_col in df_copy.columns and new_col not in df_copy.columns:
            df_copy.rename(columns={old_col: new_col}, inplace=True)
    
    # Add region information if it doesn't exist
    if 'region' not in df_copy.columns:
        # Create a simple region mapping based on country
        region_mapping = {
            # Europe
            'France': 'Europe', 'Spain': 'Europe', 'Italy': 'Europe', 'Germany': 'Europe',
            'United Kingdom': 'Europe', 'Greece': 'Europe', 'Portugal': 'Europe', 
            'Netherlands': 'Europe', 'Switzerland': 'Europe', 'Austria': 'Europe',
            'Czech Republic': 'Europe', 'Belgium': 'Europe', 'Ireland': 'Europe',
            'Poland': 'Europe', 'Hungary': 'Europe', 'Croatia': 'Europe',
            'Norway': 'Europe', 'Sweden': 'Europe', 'Denmark': 'Europe',
            'Finland': 'Europe', 'Iceland': 'Europe', 'Estonia': 'Europe',
            'Latvia': 'Europe', 'Lithuania': 'Europe', 'Slovakia': 'Europe',
            'Slovenia': 'Europe', 'Russia': 'Europe', 'Ukraine': 'Europe',
            'Belarus': 'Europe', 'Moldova': 'Europe', 'Romania': 'Europe',
            'Bulgaria': 'Europe', 'Serbia': 'Europe', 'Montenegro': 'Europe',
            'Bosnia and Herzegovina': 'Europe', 'Albania': 'Europe', 'North Macedonia': 'Europe',
            'Luxembourg': 'Europe', 'Malta': 'Europe', 'Cyprus': 'Europe',
            
            # Asia
            'Japan': 'Asia', 'China': 'Asia', 'Thailand': 'Asia', 'Vietnam': 'Asia',
            'Indonesia': 'Asia', 'Malaysia': 'Asia', 'Singapore': 'Asia', 'Philippines': 'Asia',
            'South Korea': 'Asia', 'India': 'Asia', 'Nepal': 'Asia', 'Cambodia': 'Asia',
            'Sri Lanka': 'Asia', 'Maldives': 'Asia', 'Laos': 'Asia', 'Myanmar': 'Asia',
            'Mongolia': 'Asia', 'Bhutan': 'Asia', 'Taiwan': 'Asia',
            
            # North America
            'United States': 'North America', 'Canada': 'North America', 'Mexico': 'North America',
            'Costa Rica': 'North America', 'Panama': 'North America', 'Jamaica': 'North America',
            'Cuba': 'North America', 'Dominican Republic': 'North America', 'Bahamas': 'North America',
            
            # South America
            'Brazil': 'South America', 'Argentina': 'South America', 'Peru': 'South America',
            'Colombia': 'South America', 'Chile': 'South America', 'Ecuador': 'South America',
            'Bolivia': 'South America', 'Venezuela': 'South America', 'Uruguay': 'South America',
            'Paraguay': 'South America', 'Guyana': 'South America', 'Suriname': 'South America',
            
            # Africa
            'South Africa': 'Africa', 'Egypt': 'Africa', 'Morocco': 'Africa', 'Kenya': 'Africa',
            'Tanzania': 'Africa', 'Ethiopia': 'Africa', 'Botswana': 'Africa', 'Namibia': 'Africa',
            'Zimbabwe': 'Africa', 'Uganda': 'Africa', 'Rwanda': 'Africa', 'Ghana': 'Africa',
            'Senegal': 'Africa', 'Tunisia': 'Africa', 'Algeria': 'Africa', 'Nigeria': 'Africa',
            
            # Oceania
            'Australia': 'Oceania', 'New Zealand': 'Oceania', 'Fiji': 'Oceania', 
            'Papua New Guinea': 'Oceania', 'Solomon Islands': 'Oceania'
        }
        
        # Apply the mapping or set to 'Unknown' if country is not in the mapping
        df_copy['region'] = df_copy['country'].apply(lambda x: region_mapping.get(x, 'Unknown'))
    
    # Ensure Cultural_Description exists
    if 'cultural_description' not in df_copy.columns:
        if 'description' in df_copy.columns:
            df_copy['cultural_description'] = df_copy['description']
        else:
            df_copy['cultural_description'] = 'No description available'
    
    # Make sure safety is properly formatted
    if 'safety' in df_copy.columns:
        # Standardize to High/Low based on existing values
        high_safety_patterns = ['generally safe', 'high']
        df_copy['safety'] = df_copy['safety'].apply(
            lambda x: 'High' if any(pattern in str(x).lower() for pattern in high_safety_patterns) 
                            and not any(risk in str(x).lower() for risk in ['bears', 'conflict', 'risks', 'restricted'])
                            else 'Low'
        )
    else:
        df_copy['safety'] = 'Unknown'  # Default safety value if not available
    
    # Define features for the safety model
    features = ["region", "country", "category", "cost_of_living", "cultural_description"]
    
    # Encode categorical features
    label_encoders = {}
    for feature in features:
        le = LabelEncoder()
        # Handle missing values
        df_copy[feature] = df_copy[feature].fillna('Unknown')
        df_copy[f"encoded_{feature}"] = le.fit_transform(df_copy[feature])
        label_encoders[feature] = le
        # Add 'Unknown' to classes if not already present
        if 'Unknown' not in le.classes_:
            label_encoders[feature].classes_ = np.append(le.classes_, "Unknown")
    
    # Convert safety to numerical labels
    df_copy["safety_label"] = df_copy["safety"].map({"High": 1, "Low": 0})
    
    # Define X (features) and y (target)
    X = df_copy[[f"encoded_{feature}" for feature in features]]
    y = df_copy["safety_label"]
    
    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, label_encoders, features

def recommend_by_safety_and_religion(user_input, df, model, label_encoders, features):
    """
    Recommend destinations based on safety preferences and religious influence.
    
    Parameters:
    - user_input: String containing user's query
    - df: DataFrame containing destination data
    - model: Trained safety classification model
    - label_encoders: Dictionary of label encoders for categorical features
    - features: List of feature names used in the model
    
    Returns:
    - Filtered DataFrame with destinations matching the criteria
    """
    # Make a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Standardize column names
    df_copy.columns = [col.lower() for col in df_copy.columns]
    
    # Ensure majority_religion exists
    if 'majority_religion' not in df_copy.columns:
        return pd.DataFrame(), "The dataset does not contain religion information."
    
    user_input = user_input.lower().strip()
    
    # Determine safety level from user input
    if re.search(r"\blow\b|\bdangerous\b|\bunsafe\b", user_input):
        desired_safety = 0  # Low safety
    elif re.search(r"\bsafe\b|\bhigh\b|\bsafest\b", user_input):
        desired_safety = 1  # High safety
    else:
        return pd.DataFrame(), "Please specify whether you want 'safe' (High) or 'low safety' places."
    
    # Clean and standardize religion data
    df_copy['majority_religion'] = df_copy['majority_religion'].astype(str).str.strip().str.lower()
    
    # Initialize detected religion
    detected_religion = None
    
    # List of common religion keywords and their variations
    religion_patterns = {
        'catholic': ['catholic', 'roman catholic', 'catholicism'],
        'protestant': ['protestant', 'protestantism', 'lutheran', 'baptist', 'methodist'],
        'orthodox': ['orthodox', 'russian orthodox', 'greek orthodox', 'serbian orthodox', 'ukrainian orthodox'],
        'christian': ['christian', 'christianity', 'anglican', 'presbyterian'],
        'muslim': ['muslim', 'islam', 'islamic'],
        'hindu': ['hindu', 'hinduism'],
        'buddhist': ['buddhist', 'buddhism'],
        'jewish': ['jewish', 'judaism'],
        'sikh': ['sikh', 'sikhism'],
        'taoist': ['taoist', 'taoism'],
        'shinto': ['shinto', 'shintoism']
    }
    
    # Detect religion from user input
    for religion_key, variations in religion_patterns.items():
        if any(re.search(rf"\b{re.escape(var)}\b", user_input, re.IGNORECASE) for var in variations):
            detected_religion = religion_key
            break
    
    # If no religion is detected, check against available religions in the dataset
    if not detected_religion:
        for religion in df_copy['majority_religion'].dropna().unique():
            religion_clean = religion.strip().lower()
            if re.search(rf"\b{re.escape(religion_clean)}\b", user_input, re.IGNORECASE):
                detected_religion = religion_clean
                break
    
    # If religion still not detected, return error
    if not detected_religion:
        available_religions = ", ".join(set(df_copy['majority_religion'].dropna().unique()))
        return pd.DataFrame(), f"Please specify a valid religion. Available options: {available_religions}"
    
    # Prepare data for prediction
    # Ensure all required columns exist
    for feature in features:
        if feature not in df_copy.columns:
            df_copy[feature] = 'Unknown'
    
    # Encode features
    for feature in features:
        df_copy[f"encoded_{feature}"] = df_copy[feature].apply(
            lambda x: label_encoders[feature].transform([x])[0] 
            if x in label_encoders[feature].classes_ 
            else label_encoders[feature].transform(["Unknown"])[0]
        )
    
    # Predict safety levels
    X_pred = df_copy[[f"encoded_{feature}" for feature in features]]
    df_copy["predicted_safety"] = model.predict(X_pred)
    
    # Filter destinations based on safety and religion
    # Check for religion in majority_religion using partial string matching
    religion_filter = df_copy['majority_religion'].str.contains(detected_religion, case=False, na=False)
    safety_filter = df_copy["predicted_safety"] == desired_safety
    
    filtered_df = df_copy[religion_filter & safety_filter]
    
    # If no results found, provide a helpful message
    if filtered_df.empty:
        safety_label = 'High' if desired_safety == 1 else 'Low'
        return pd.DataFrame(), f"Sorry, no places found with {detected_religion} influence and {safety_label} safety level."
    
    return filtered_df, None