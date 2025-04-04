import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def train_cost_time_model(df):
    """
    Train a model to recommend destinations based on cost of living and best time to visit.
    
    Parameters:
    - df: DataFrame containing the preprocessed destination data
    
    Returns:
    - Dictionary containing the trained model and related information
    """
    try:
        # Drop any rows with missing values in these columns
        filtered_df = df.dropna(subset=['cost_of_living', 'best_time_to_visit', 'name'])
        
        # Define the feature columns and target
        X = filtered_df[['cost_of_living', 'best_time_to_visit']]
        y = filtered_df['name']

        # Create a preprocessing pipeline for cost and time columns
        encoder = ColumnTransformer(
            transformers=[
                ('cost_of_living', OneHotEncoder(handle_unknown='ignore'), ['cost_of_living']),
                ('best_time_to_visit', OneHotEncoder(handle_unknown='ignore'), ['best_time_to_visit'])
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
        
        # Store model information
        model_info = {
            'pipeline': pipeline,
            'feature_names': X.columns.tolist(),
            'classes': pipeline.classes_.tolist(),
            'cost_values': sorted(filtered_df['cost_of_living'].unique().tolist()),
            'time_values': sorted(filtered_df['best_time_to_visit'].unique().tolist())
        }
        
        return model_info
        
    except Exception as e:
        print(f"Error training cost-time model: {str(e)}")
        return None


def recommend_by_cost_and_time(model_info, df, cost_of_living, best_time_to_visit):
    """
    Recommend destinations based on cost of living and best time to visit preferences.
    
    Parameters:
    - model_info: Dictionary containing the trained model and related information
    - df: Original DataFrame containing destination data
    - cost_of_living: Selected cost of living (str)
    - best_time_to_visit: Selected best time to visit (str)
    
    Returns:
    - DataFrame with recommended destinations
    """
    if model_info is None:
        return pd.DataFrame(), "Model not available. Please try again later."
    
    try:
        # If both are "All", return all destinations
        if cost_of_living == "All" and best_time_to_visit == "All":
            return df.copy(), None
        
        # Handle individual seasons by mapping them to their combined categories
        mapped_time = best_time_to_visit
        if best_time_to_visit == "Spring" or best_time_to_visit == "Fall":
            mapped_time = "Spring/Fall"
        elif best_time_to_visit == "Winter":
            mapped_time = "Winter/Summer"
        # For Summer, we need to check both Summer directly and Winter/Summer
        # This is handled in the filtering below
            
        pipeline = model_info['pipeline']
        
        # Create input data
        if cost_of_living == "All":
            # If cost is "All", filter only by time
            if best_time_to_visit == "Summer":
                # Special case for Summer - needs to check both Summer and Winter/Summer
                filtered_destinations = df[(df['best_time_to_visit'] == "Summer") | 
                                          (df['best_time_to_visit'] == "Winter/Summer")].copy()
            elif mapped_time != best_time_to_visit:
                # Using mapped season (e.g., Spring/Fall for Spring)
                filtered_destinations = df[df['best_time_to_visit'] == mapped_time].copy()
            else:
                # Using exact match
                filtered_destinations = df[df['best_time_to_visit'] == best_time_to_visit].copy()
            message = f"Destinations with best time to visit in {best_time_to_visit}"
        elif best_time_to_visit == "All":
            # If time is "All", filter only by cost
            filtered_destinations = df[df['cost_of_living'] == cost_of_living].copy()
            message = f"Destinations with {cost_of_living} cost of living"
        else:
            # Both specific values, use the model to rank
            # For prediction, use the mapped time value if it's different
            prediction_time = mapped_time if mapped_time != best_time_to_visit else best_time_to_visit
            
            input_data = pd.DataFrame([[cost_of_living, prediction_time]], 
                                     columns=['cost_of_living', 'best_time_to_visit'])
            
            # Initial filter for exact matches with mapped time if needed
            if best_time_to_visit == "Summer":
                # Special case for Summer - needs to check both Summer and Winter/Summer
                exact_matches = df[(df['cost_of_living'] == cost_of_living) & 
                                 ((df['best_time_to_visit'] == "Summer") | 
                                  (df['best_time_to_visit'] == "Winter/Summer"))].copy()
            elif mapped_time != best_time_to_visit:
                # For individual seasons that map to combined categories
                exact_matches = df[(df['cost_of_living'] == cost_of_living) & 
                                  (df['best_time_to_visit'] == mapped_time)].copy()
            else:
                # For exact matches
                exact_matches = df[(df['cost_of_living'] == cost_of_living) & 
                                  (df['best_time_to_visit'] == best_time_to_visit)].copy()
            
            # If we have exact matches, return them
            if not exact_matches.empty:
                filtered_destinations = exact_matches
                message = None
            else:
                # Get predicted probabilities for all destinations
                try:
                    probas = pipeline.predict_proba(input_data)[0]
                    
                    # Create a DataFrame with probabilities and corresponding destinations
                    destination_probabilities = pd.DataFrame({
                        'name': pipeline.classes_,
                        'probability': probas
                    })
                    
                    # Sort by probability in descending order
                    destination_probabilities = destination_probabilities.sort_values(by='probability', ascending=False)
                    
                    # Get the top destinations
                    top_destinations = destination_probabilities.head(10)['name'].tolist()
                    
                    # Get the corresponding rows from the original dataframe
                    filtered_destinations = df[df['name'].isin(top_destinations)].copy()
                    
                    # Add a match score based on the probability (scale to 0-100)
                    name_to_prob = dict(zip(destination_probabilities['name'], destination_probabilities['probability'] * 100))
                    filtered_destinations['match_score'] = filtered_destinations['name'].map(name_to_prob)
                    
                    message = None
                    
                except Exception as e:
                    # Fallback to simple filtering for similar values
                    filtered_destinations = pd.DataFrame()
                    message = f"No exact matches found for {cost_of_living} cost and {best_time_to_visit} timing. Try different criteria."
        
        # Return the filtered destinations
        if filtered_destinations.empty:
            message = f"No destinations match your criteria. Try different selections."
            
        return filtered_destinations, message
        
    except Exception as e:
        return pd.DataFrame(), f"Error finding recommendations: {str(e)}"