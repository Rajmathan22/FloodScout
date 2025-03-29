import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster, HeatMap
from sklearn.model_selection import train_test_split
from models import FloodRiskModel
import tempfile
import base64
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

def train_models(data):
    """
    Train machine learning models for flood prediction
    
    Args:
        data (dict): Dictionary containing preprocessed data
        
    Returns:
        dict: Model results including predictions and performance metrics
    """
    try:
        # Extract features and target
        X = data['X']
        y = data['y']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize model
        model = FloodRiskModel()
        
        # Preprocess data
        X_train_scaled, y_train = model.preprocess_data(X_train, y_train)
        X_test_scaled, y_test = model.preprocess_data(X_test, y_test)
        
        # Train Random Forest
        logger.info("Training Random Forest model")
        model.train_random_forest(X_train_scaled, y_train)
        
        # Train MLP
        logger.info("Training MLP model")
        model.train_mlp(X_train_scaled, y_train)
        
        # Train improved models if data size allows
        if len(X) > 50:  # Only attempt with reasonably sized datasets
            logger.info("Training improved models")
            try:
                model.train_improved_models(X, y)
                logger.info("Successfully trained advanced models")
            except Exception as e:
                logger.warning(f"Unable to train improved models: {str(e)}")
        
        # Evaluate models
        logger.info("Evaluating models")
        evaluation = model.evaluate_models(X_test_scaled, y_test)
        
        # Make predictions on the entire dataset
        logger.info("Generating predictions")
        
        # Get standard predictions
        predictions = model.predict(X)
        
        # Prepare the results dictionary with base models
        results = {
            'rf_probabilities': predictions['rf_probabilities'].tolist(),
            'rf_predictions': predictions['rf_predictions'].tolist(),
            'mlp_probabilities': predictions['mlp_probabilities'].tolist(),
            'mlp_predictions': predictions['mlp_predictions'].tolist(),
            'ensemble_probabilities': predictions['ensemble_probabilities'].tolist(),
            'ensemble_predictions': predictions['ensemble_predictions'].tolist(),
            'rf_accuracy': evaluation['rf_accuracy'],
            'mlp_accuracy': evaluation['mlp_accuracy'],
            'ensemble_accuracy': evaluation['ensemble_accuracy'],
            'rf_report': evaluation['rf_report'],
            'mlp_report': evaluation['mlp_report'],
            'ensemble_report': evaluation['ensemble_report'],
            'feature_importances': evaluation['feature_importances']
        }
        
        # Add advanced model results if available
        if 'improved_rf_accuracy' in evaluation:
            results.update({
                'improved_rf_accuracy': evaluation['improved_rf_accuracy'],
                'combined_mlp_accuracy': evaluation['combined_mlp_accuracy'],
                'improved_rf_report': evaluation['improved_rf_report'],
                'combined_mlp_report': evaluation['combined_mlp_report']
            })
            
            # Add advanced predictions if available
            if 'improved_rf_probabilities' in predictions:
                results.update({
                    'improved_rf_probabilities': predictions['improved_rf_probabilities'].tolist(),
                    'improved_rf_predictions': predictions['improved_rf_predictions'].tolist(),
                    'combined_mlp_probabilities': predictions['combined_mlp_probabilities'].tolist(),
                    'combined_mlp_predictions': predictions['combined_mlp_predictions'].tolist()
                })
            
            # Add advanced ensemble if available
            if 'advanced_ensemble_accuracy' in evaluation:
                results.update({
                    'advanced_ensemble_accuracy': evaluation['advanced_ensemble_accuracy'],
                    'advanced_ensemble_report': evaluation['advanced_ensemble_report']
                })
                
                if 'advanced_ensemble_probabilities' in predictions:
                    results.update({
                        'advanced_ensemble_probabilities': predictions['advanced_ensemble_probabilities'].tolist(),
                        'advanced_ensemble_predictions': predictions['advanced_ensemble_predictions'].tolist()
                    })
        
        return results
    except Exception as e:
        logger.error(f"Error in train_models: {str(e)}")
        raise

def generate_visualizations(model_results, data):
    """
    Generate visualizations for model results
    
    Args:
        model_results (dict): Dictionary of model results
        data (dict): Dictionary containing preprocessed data
        
    Returns:
        dict: Paths to generated visualizations
    """
    try:
        # Create directory for visualizations if it doesn't exist
        static_dir = os.path.join('static', 'visualizations')
        os.makedirs(static_dir, exist_ok=True)
        
        # Model performance comparison chart
        # Check if we have advanced models
        has_advanced_models = 'improved_rf_accuracy' in model_results
        has_advanced_ensemble = 'advanced_ensemble_accuracy' in model_results
        
        if has_advanced_models:
            # Include all available models in the comparison
            models = ['Random Forest', 'MLP', 'Ensemble']
            accuracies = [
                model_results['rf_accuracy'] * 100,
                model_results['mlp_accuracy'] * 100,
                model_results['ensemble_accuracy'] * 100
            ]
            
            # Add improved models
            models.extend(['Improved RF', 'Combined MLP'])
            accuracies.extend([
                model_results['improved_rf_accuracy'] * 100,
                model_results['combined_mlp_accuracy'] * 100
            ])
            
            # Add advanced ensemble if available
            if has_advanced_ensemble:
                models.append('Advanced Ensemble')
                accuracies.append(model_results['advanced_ensemble_accuracy'] * 100)
            
            # Use different colors for each model
            colors = ['green', 'blue', 'purple', 'orange', 'red', 'brown']
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(models, accuracies, color=colors[:len(models)])
            plt.xticks(rotation=45, ha='right')
        else:
            # Simple comparison with just the base models
            models = ['Random Forest', 'MLP', 'Ensemble']
            accuracies = [
                model_results['rf_accuracy'] * 100,
                model_results['mlp_accuracy'] * 100,
                model_results['ensemble_accuracy'] * 100
            ]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(models, accuracies, color=['green', 'blue', 'purple'])
        
        # Add accuracy values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.2f}%', ha='center', va='bottom')
        
        plt.xlabel('Models')
        plt.ylabel('Accuracy (%)')
        plt.title('Model Performance Comparison')
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()  # Ensure everything fits
        
        # Save figure to memory
        performance_img = BytesIO()
        plt.savefig(performance_img, format='png')
        performance_img.seek(0)
        performance_img_b64 = base64.b64encode(performance_img.getvalue()).decode('utf-8')
        plt.close()
        
        # Feature importance chart
        feature_importances = pd.DataFrame(model_results['feature_importances'])
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importances.head(10))
        plt.title('Top 10 Important Features for Flood Prediction')
        plt.tight_layout()
        
        # Save figure to memory
        importance_img = BytesIO()
        plt.savefig(importance_img, format='png')
        importance_img.seek(0)
        importance_img_b64 = base64.b64encode(importance_img.getvalue()).decode('utf-8')
        plt.close()
        
        # Model prediction distribution
        # Use advanced ensemble probabilities if available, otherwise use the regular ensemble
        if has_advanced_ensemble and 'advanced_ensemble_probabilities' in model_results:
            ensemble_probs = np.array(model_results['advanced_ensemble_probabilities'])
            ensemble_title = 'Advanced Ensemble Probability Distribution'
        else:
            ensemble_probs = np.array(model_results['ensemble_probabilities'])
            ensemble_title = 'Ensemble Probability Distribution'
        
        plt.figure(figsize=(10, 6))
        sns.histplot(ensemble_probs, bins=20, kde=True)
        plt.axvline(x=0.5, color='red', linestyle='--')
        plt.xlabel('Probability')
        plt.ylabel('Count')
        plt.title(ensemble_title)
        
        # Save figure to memory
        dist_img = BytesIO()
        plt.savefig(dist_img, format='png')
        dist_img.seek(0)
        dist_img_b64 = base64.b64encode(dist_img.getvalue()).decode('utf-8')
        plt.close()
        
        # Model comparison distribution if we have multiple models
        if has_advanced_models:
            # Compare probability distributions of different models
            plt.figure(figsize=(12, 6))
            
            # Plot histograms for all available model probabilities
            sns.histplot(model_results['rf_probabilities'], bins=20, kde=True, alpha=0.5, label='Random Forest')
            sns.histplot(model_results['mlp_probabilities'], bins=20, kde=True, alpha=0.5, label='MLP')
            
            if 'improved_rf_probabilities' in model_results:
                sns.histplot(model_results['improved_rf_probabilities'], bins=20, kde=True, alpha=0.5, label='Improved RF')
            
            if 'combined_mlp_probabilities' in model_results:
                sns.histplot(model_results['combined_mlp_probabilities'], bins=20, kde=True, alpha=0.5, label='Combined MLP')
            
            # Decision boundary
            plt.axvline(x=0.5, color='red', linestyle='--', label='Decision Boundary')
            
            plt.xlabel('Probability')
            plt.ylabel('Count')
            plt.title('Probability Distribution Comparison Across Models')
            plt.legend()
            plt.tight_layout()
            
            # Save figure to memory
            model_comparison_img = BytesIO()
            plt.savefig(model_comparison_img, format='png')
            model_comparison_img.seek(0)
            model_comparison_img_b64 = base64.b64encode(model_comparison_img.getvalue()).decode('utf-8')
            plt.close()
            
            # Return both basic charts and the model comparison chart
            return {
                'performance_chart': f"data:image/png;base64,{performance_img_b64}",
                'feature_importance': f"data:image/png;base64,{importance_img_b64}",
                'probability_distribution': f"data:image/png;base64,{dist_img_b64}",
                'model_comparison': f"data:image/png;base64,{model_comparison_img_b64}"
            }
        
        # Return basic charts if we don't have advanced models
        return {
            'performance_chart': f"data:image/png;base64,{performance_img_b64}",
            'feature_importance': f"data:image/png;base64,{importance_img_b64}",
            'probability_distribution': f"data:image/png;base64,{dist_img_b64}"
        }
    
    except Exception as e:
        logger.error(f"Error in generate_visualizations: {str(e)}")
        # Return basic visualizations in case of error
        try:
            # Create a simple fallback performance chart
            plt.figure(figsize=(8, 6))
            plt.bar(['RF', 'MLP', 'Ensemble'], 
                   [model_results.get('rf_accuracy', 0.7) * 100,
                    model_results.get('mlp_accuracy', 0.7) * 100,
                    model_results.get('ensemble_accuracy', 0.75) * 100])
            plt.title('Model Accuracy Comparison')
            plt.ylabel('Accuracy (%)')
            plt.ylim(0, 100)
            
            fallback_img = BytesIO()
            plt.savefig(fallback_img, format='png')
            fallback_img.seek(0)
            fallback_img_b64 = base64.b64encode(fallback_img.getvalue()).decode('utf-8')
            plt.close()
            
            return {
                'performance_chart': f"data:image/png;base64,{fallback_img_b64}",
                'error_message': str(e)
            }
        except:
            logger.error("Failed to create fallback visualization")
            return {
                'error_message': f"Visualization error: {str(e)}"
            }

def generate_map(predictions, data, model_results=None):
    """
    Generate an interactive map with flood risk predictions
    
    Args:
        predictions (list): Ensemble model predictions
        data (dict): Dictionary containing preprocessed data with lat/lon
        model_results (dict, optional): Full model results for additional visualizations
        
    Returns:
        str: Path to the generated map HTML file
    """
    try:
        # Check if latitude and longitude data is available
        if 'latitude' not in data or 'longitude' not in data:
            logger.warning("Latitude or longitude data not available for map visualization")
            return None
        
        # Ensure all data types are correct
        latitudes = data.get('latitude', [])
        longitudes = data.get('longitude', [])
        
        # Ensure these are lists of numeric values
        if isinstance(latitudes, (int, float)):
            latitudes = [latitudes]
        if isinstance(longitudes, (int, float)):
            longitudes = [longitudes]
            
        # Convert any string values to float
        latitudes = [float(lat) if isinstance(lat, (int, float, str)) else 0.0 for lat in latitudes]
        longitudes = [float(lon) if isinstance(lon, (int, float, str)) else 0.0 for lon in longitudes]
        
        # Check if we have advanced ensemble predictions
        has_advanced_predictions = False
        advanced_predictions = None
        advanced_probabilities = None
        
        if model_results is not None:
            if 'advanced_ensemble_predictions' in model_results:
                has_advanced_predictions = True
                advanced_predictions = model_results['advanced_ensemble_predictions']
                advanced_probabilities = model_results['advanced_ensemble_probabilities']
                logger.info("Using advanced ensemble predictions for map visualization")
        
        # Convert predictions to a list of integers if it's not already
        if isinstance(predictions, list):
            # Ensure all elements are integers
            preds = [int(p) if isinstance(p, (int, float)) else 0 for p in predictions]
        else:
            # If predictions is not a list, create a default list
            preds = [0] * len(latitudes)
            
        # Get ensemble probabilities
        if 'ensemble_probabilities' in data:
            probabilities = data['ensemble_probabilities']
            # Ensure they're all floats
            prob_list = [float(p) if isinstance(p, (int, float)) else 0.5 for p in probabilities]
        else:
            prob_list = [0.5] * len(latitudes)
        
        # Create a DataFrame with predictions and coordinates
        df = pd.DataFrame({
            'Latitude': latitudes,
            'Longitude': longitudes,
            'Prediction': preds,
            'Probability': prob_list
        })
        
        # Add advanced predictions if available
        if has_advanced_predictions:
            df['Advanced_Prediction'] = [int(p) if isinstance(p, (int, float)) else 0 
                                        for p in advanced_predictions]
            df['Advanced_Probability'] = [float(p) if isinstance(p, (int, float)) else 0.5 
                                         for p in advanced_probabilities]
        
        # Filter for hotspots (predicted flood risk)
        if has_advanced_predictions:
            # Use advanced predictions for hotspots if available
            hotspots = df[df['Advanced_Prediction'] == 1]
            prob_column = 'Advanced_Probability'
        else:
            hotspots = df[df['Prediction'] == 1]
            prob_column = 'Probability'
        
        # Calculate map center (with error handling)
        try:
            map_center = [df['Latitude'].mean(), df['Longitude'].mean()]
        except:
            # Default to a central position if calculation fails
            map_center = [20.0, 77.0]  # Center of India
        
        # Create map
        my_map = folium.Map(location=map_center, zoom_start=5, 
                          tiles='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', 
                          attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors')
        
        # Add OpenStreetMap as an alternative
        folium.TileLayer('OpenStreetMap').add_to(my_map)
        
        # Add marker cluster for hotspots
        hotspot_cluster = MarkerCluster(name="Flood Risk Hotspots").add_to(my_map)
        
        # Add markers for hotspots (with error handling)
        for _, row in hotspots.iterrows():
            try:
                # Create a more informative popup
                popup_text = f"""
                <div style="font-family: Arial; min-width: 180px;">
                    <h4 style="margin-bottom: 5px;">Flood Risk Area</h4>
                    <b>Location:</b> {float(row['Latitude']):.4f}, {float(row['Longitude']):.4f}<br>
                    <b>Risk Probability:</b> {float(row[prob_column]):.2f}<br>
                """
                
                # Add additional model probabilities if available
                if has_advanced_predictions:
                    popup_text += f"""
                    <hr style="margin: 5px 0;">
                    <b>Standard Ensemble:</b> {float(row['Probability']):.2f}<br>
                    <b>Advanced Ensemble:</b> {float(row['Advanced_Probability']):.2f}<br>
                    """
                
                popup_text += "</div>"
                
                # Choose a color based on probability
                probability = float(row[prob_column])
                if probability > 0.8:
                    color = 'red'  # High risk
                elif probability > 0.6:
                    color = 'orange'  # Medium-high risk
                else:
                    color = 'blue'  # Medium risk
                
                folium.Marker(
                    location=[float(row['Latitude']), float(row['Longitude'])],
                    popup=folium.Popup(popup_text, max_width=300),
                    icon=folium.Icon(color=color, icon='tint', prefix='fa')
                ).add_to(hotspot_cluster)
            except Exception as marker_error:
                logger.warning(f"Error adding marker: {str(marker_error)}")
                continue
        
        # Add heatmap (with error handling)
        try:
            # Use advanced probabilities for the heatmap if available
            heat_data = []
            for _, row in df.iterrows():
                try:
                    lat = float(row['Latitude'])
                    lon = float(row['Longitude'])
                    prob = float(row[prob_column])
                    heat_data.append([lat, lon, prob])
                except:
                    continue
            
            if heat_data:
                HeatMap(
                    data=heat_data,
                    radius=15,
                    max_zoom=13,
                    gradient={0.4: 'blue', 0.65: 'yellow', 0.9: 'red'},
                    name='Risk Heatmap'
                ).add_to(my_map)
        except Exception as heat_error:
            logger.warning(f"Error creating heatmap: {str(heat_error)}")
        
        # Add layer control
        folium.LayerControl().add_to(my_map)
        
        # Add a legend
        legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; 
                    padding: 10px; border: 2px solid grey; border-radius: 5px;">
            <p><b>Flood Risk Legend</b></p>
            <p><i class="fa fa-tint" style="color:red"></i> High Risk (>80%)</p>
            <p><i class="fa fa-tint" style="color:orange"></i> Medium-High Risk (60-80%)</p>
            <p><i class="fa fa-tint" style="color:blue"></i> Medium Risk (<60%)</p>
        </div>
        '''
        my_map.get_root().html.add_child(folium.Element(legend_html))
        
        # Add a title
        title_html = '''
        <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%); z-index: 1000; 
                   background-color: rgba(255, 255, 255, 0.8); padding: 10px; border-radius: 5px;">
            <h3 style="margin: 0;">Flood Risk Prediction Map</h3>
        </div>
        '''
        my_map.get_root().html.add_child(folium.Element(title_html))
        
        # Save the map to a temporary file
        static_dir = os.path.join('static', 'maps')
        os.makedirs(static_dir, exist_ok=True)
        map_path = os.path.join(static_dir, 'flood_risk_map.html')
        my_map.save(map_path)
        
        return map_path.replace('static/', '')
    
    except Exception as e:
        logger.error(f"Error in generate_map: {str(e)}")
        # Create a simple fallback map
        try:
            static_dir = os.path.join('static', 'maps')
            os.makedirs(static_dir, exist_ok=True)
            map_path = os.path.join(static_dir, 'flood_risk_map.html')
            
            # Create a basic map centered on a default location
            simple_map = folium.Map(location=[20.0, 77.0], zoom_start=4)
            folium.LayerControl().add_to(simple_map)
            
            # Add a marker with error message
            folium.Marker(
                location=[20.0, 77.0],
                popup="Error generating detailed map - showing default view",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(simple_map)
            
            simple_map.save(map_path)
            return map_path.replace('static/', '')
        except:
            return None
