import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

@st.cache_resource
def train_recommendation_model(df):
    """
    Train a machine learning model for destination recommendations based on Country and Category.
    
    Parameters:
    - df: DataFrame containing the preprocessed destination data
    
    Returns:
    - Trained pipeline and information about the model
    """
    try:
        # Make a copy to avoid modifying the original dataframe
        data = df.copy()
        
        # If 'name' column exists and 'destination' doesn't, create destination from name
        if 'destination' not in data.columns and 'name' in data.columns:
            data['destination'] = data['name']
        
        # Define the feature columns and target
        X = data[['country', 'category']]
        y = data['destination']
        
        # Create a preprocessing pipeline for country and category columns
        encoder = ColumnTransformer(
            transformers=[
                ('country', OneHotEncoder(handle_unknown='ignore'), ['country']),
                ('category', OneHotEncoder(handle_unknown='ignore'), ['category'])
            ],
            remainder='passthrough'  # Keeps other columns in the dataset
        )
        
        # Create a pipeline with preprocessing and Logistic Regression
        pipeline = Pipeline(steps=[
            ('preprocessor', encoder),
            ('classifier', LogisticRegression(max_iter=1000))
        ])
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Predict on test data
        y_pred = pipeline.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            'pipeline': pipeline,
            'accuracy': accuracy,
            'classes': pipeline.classes_,
            'features': ['country', 'category']
        }
    except Exception as e:
        st.error(f"Error training recommendation model: {str(e)}")
        return None

def recommend_destinations_by_country_category(model_info, df, country, category):
    """
    Recommend destinations based on country and category using the trained model.
    
    Parameters:
    - model_info: Dictionary containing the trained model and related information
    - df: Original DataFrame containing destination data
    - country: Selected country
    - category: Selected category
    
    Returns:
    - DataFrame with recommended destinations
    """
    try:
        if model_info is None:
            st.error("Recommendation model not available")
            return pd.DataFrame()  # Return empty DataFrame
        
        # STRICT FILTERING: Always filter by both country and category
        filtered_by_criteria = df.loc[(df['country'] == country) & (df['category'] == category)]
        
        # If no destinations match these exact criteria, return empty DataFrame with a message
        if filtered_by_criteria.empty:
            st.warning(f"No destinations found for category '{category}' in '{country}'.")
            return pd.DataFrame()  # Return empty DataFrame
        
        # If we have matches, apply ML model to get recommendations
        # Create input data for prediction
        input_data = pd.DataFrame([[country, category]], columns=['country', 'category'])
        
        # Get pipeline
        pipeline = model_info['pipeline']
        
        # Get predicted probabilities for all destinations
        probas = pipeline.predict_proba(input_data)[0]
        
        # Create a DataFrame with probabilities and corresponding destinations
        destination_probabilities = pd.DataFrame({
            'destination': pipeline.classes_,
            'probability': probas
        })
        
        # Sort by probability in descending order
        destination_probabilities = destination_probabilities.sort_values(by='probability', ascending=False)
        
        # Merge with filtered dataframe to get all details
        # This ensures we only include destinations matching BOTH country AND category
        result = pd.merge(
            destination_probabilities, 
            filtered_by_criteria,  # Only use the pre-filtered destinations
            left_on='destination', 
            right_on='name',
            how='inner'
        )
        
        # If no results after merging, return the filtered criteria directly
        if result.empty:
            return filtered_by_criteria.copy()
        
        return result
        
    except Exception as e:
        st.error(f"Error in recommendation: {str(e)}")
        # Log the error details for debugging
        print(f"Recommendation error details: {e}")
        # Fallback: return filtered dataframe - strictly match both country and category
        return df.loc[(df['country'] == country) & (df['category'] == category)].copy()