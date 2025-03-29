import os
import logging
import uuid
import json
import pickle
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import pandas as pd
import numpy as np
from utils.data_processing import preprocess_data, validate_data
from utils.model_utils import train_models, generate_visualizations, generate_map

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "fallback_secret_key_for_development")

# Configure upload folder
UPLOAD_FOLDER = 'temp_uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    """Route for landing page with file upload form"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start processing"""
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(request.url)
    
    if file and file.filename.endswith(('.xls', '.xlsx', '.csv')):
        # Generate unique ID for this upload session
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        
        # Make sure upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save the file temporarily
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{file.filename}")
        file.save(file_path)
        
        try:
            # Read the file based on extension
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            # Validate the data
            validation_result = validate_data(df)
            if not validation_result['valid']:
                flash(validation_result['message'], 'danger')
                return redirect(url_for('index'))
            
            # Store the processed data in the session (only store the file path)
            session['data_file'] = file_path
            
            # For debugging
            logger.debug(f"File saved at: {file_path}")
            logger.debug(f"File exists: {os.path.exists(file_path)}")
            
            return redirect(url_for('process_data'))
            
        except Exception as e:
            logger.error(f"Error processing uploaded file: {str(e)}")
            flash(f"Error processing file: {str(e)}", 'danger')
            return redirect(url_for('index'))
    else:
        flash('Allowed file types are .xls, .xlsx, and .csv', 'warning')
        return redirect(request.url)

@app.route('/process', methods=['GET'])
def process_data():
    """Process the uploaded data and train models"""
    if 'data_file' not in session:
        flash('No data found. Please upload a file first.', 'warning')
        return redirect(url_for('index'))
    
    file_path = session['data_file']
    session_id = session['session_id']
    
    try:
        # Read the stored file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Preprocess the data
        processed_data = preprocess_data(df)
        
        # Train models and get results
        model_results = train_models(processed_data)
        
        # Generate visualizations
        visualization_paths = generate_visualizations(model_results, processed_data)
        
        # Generate map using the best ensemble predictions available
        # Pass the full model_results to have access to advanced ensemble if available
        map_path = generate_map(model_results['ensemble_predictions'], processed_data, model_results)
        
        # Create results directory for this session
        results_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        os.makedirs(results_dir, exist_ok=True)
        
        # Store only essential data in session to avoid large cookie
        summary_data = {
            'rf_accuracy': model_results['rf_accuracy'],
            'mlp_accuracy': model_results['mlp_accuracy'],
            'ensemble_accuracy': model_results['ensemble_accuracy'],
            'feature_importances': model_results['feature_importances'][:5]  # Only top 5 features
        }
        
        # Save the full results as a pickle file
        results_file = os.path.join(results_dir, 'model_results.pkl')
        with open(results_file, 'wb') as f:
            pickle.dump(model_results, f)
        
        # Instead of storing paths directly in the session, store them on disk
        # Create a JSON file with the visualization paths
        paths_file = os.path.join(results_dir, 'visualization_paths.json')
        with open(paths_file, 'w') as f:
            json.dump({
                'visualization_paths': visualization_paths,
                'map_path': map_path
            }, f)
            
        # Store only the reference to the paths file in the session
        session['paths_file'] = paths_file
        session['results_file'] = results_file
        session['model_results_summary'] = summary_data
        
        # Redirect to dashboard
        return redirect(url_for('dashboard'))
        
    except Exception as e:
        logger.error(f"Error during data processing: {str(e)}")
        flash(f"Error during data processing: {str(e)}", 'danger')
        return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    """Display the dashboard with model results and visualizations"""
    if 'results_file' not in session:
        flash('No results found. Please upload and process a file first.', 'warning')
        return redirect(url_for('index'))
    
    try:
        # Load results from disk
        results_file = session['results_file']
        
        if os.path.exists(results_file):
            # Load the full model results from disk
            with open(results_file, 'rb') as f:
                model_results = pickle.load(f)
        else:
            # Fallback to summary if full results not available
            model_results = session.get('model_results_summary', {})
            flash('Detailed results not found. Showing summary data only.', 'warning')
            
        # Load visualization paths from the saved JSON file
        paths_file = session.get('paths_file')
        if paths_file and os.path.exists(paths_file):
            with open(paths_file, 'r') as f:
                paths_data = json.load(f)
                visualization_paths = paths_data.get('visualization_paths', {})
                map_path = paths_data.get('map_path')
        else:
            visualization_paths = {}
            map_path = None
            flash('Visualization data not found.', 'warning')
        
        # Read the file again to get sample data for display
        file_path = session['data_file']
        if file_path.endswith('.csv'):
            df_sample = pd.read_csv(file_path).head(10).to_dict('records')
        else:
            df_sample = pd.read_excel(file_path).head(10).to_dict('records')
        
        return render_template(
            'dashboard.html',
            model_results=model_results,
            visualization_paths=visualization_paths,
            map_path=map_path,
            data_sample=df_sample
        )
        
    except Exception as e:
        logger.error(f"Error displaying dashboard: {str(e)}")
        flash(f"Error displaying dashboard: {str(e)}", 'danger')
        return redirect(url_for('index'))

@app.route('/download_results', methods=['GET'])
def download_results():
    """Generate and serve a downloadable results file"""
    if 'results_file' not in session or 'data_file' not in session:
        flash('No results found. Please upload and process a file first.', 'warning')
        return redirect(url_for('index'))
    
    try:
        # Get the processed data
        file_path = session['data_file']
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Load model results from disk
        results_file = session['results_file']
        if os.path.exists(results_file):
            with open(results_file, 'rb') as f:
                model_results = pickle.load(f)
        else:
            return jsonify({
                'success': False,
                'error': 'Model results file not found'
            })
        
        # Add base model prediction columns to the dataframe
        df['RF_Probability'] = model_results['rf_probabilities']
        df['MLP_Probability'] = model_results['mlp_probabilities']
        df['Ensemble_Probability'] = model_results['ensemble_probabilities']
        df['RF_Prediction'] = model_results['rf_predictions']
        df['MLP_Prediction'] = model_results['mlp_predictions']
        df['Ensemble_Prediction'] = model_results['ensemble_predictions']
        
        # Add advanced model predictions if available
        if 'improved_rf_probabilities' in model_results:
            df['Improved_RF_Probability'] = model_results['improved_rf_probabilities']
            df['Improved_RF_Prediction'] = model_results['improved_rf_predictions']
            
        if 'combined_mlp_probabilities' in model_results:
            df['Combined_MLP_Probability'] = model_results['combined_mlp_probabilities']
            df['Combined_MLP_Prediction'] = model_results['combined_mlp_predictions']
            
        if 'advanced_ensemble_probabilities' in model_results:
            df['Advanced_Ensemble_Probability'] = model_results['advanced_ensemble_probabilities']
            df['Advanced_Ensemble_Prediction'] = model_results['advanced_ensemble_predictions']
        
        # Create downloads directory if it doesn't exist
        downloads_dir = os.path.join('static', 'downloads')
        os.makedirs(downloads_dir, exist_ok=True)
        
        # Generate a CSV file with results
        result_filename = f"{session['session_id']}_results.csv"
        result_file_path = os.path.join(downloads_dir, result_filename)
        df.to_csv(result_file_path, index=False)
        
        # Serve the file
        return jsonify({
            'success': True,
            'download_url': url_for('static', filename=f"downloads/{result_filename}")
        })
        
    except Exception as e:
        logger.error(f"Error generating downloadable results: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/progress', methods=['GET'])
def get_progress():
    """Mock API for progress tracking (would be implemented with a real progress tracker in production)"""
    return jsonify({
        'progress': 100,
        'status': 'complete'
    })

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('index.html', error="Page not found"), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    logger.error(f"Server error: {str(e)}")
    return render_template('index.html', error="Server error occurred"), 500

# File cleanup function to be called after data is no longer needed
@app.route('/cleanup', methods=['POST'])
def cleanup_files():
    """Clean up temporary files associated with the current session"""
    if 'session_id' not in session:
        return jsonify({'success': False, 'error': 'No session found'})
    
    session_id = session['session_id']
    
    try:
        # Clean up individual uploaded file
        if 'data_file' in session:
            file_path = session['data_file']
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Removed file: {file_path}")
        
        # Clean up results directory for this session
        results_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        if os.path.exists(results_dir):
            for filename in os.listdir(results_dir):
                file_path = os.path.join(results_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logger.debug(f"Removed file: {file_path}")
            os.rmdir(results_dir)
            logger.debug(f"Removed directory: {results_dir}")
        
        # Remove download files
        downloads_dir = os.path.join('static', 'downloads')
        if os.path.exists(downloads_dir):
            result_filename = f"{session_id}_results.csv"
            result_file_path = os.path.join(downloads_dir, result_filename)
            if os.path.exists(result_file_path):
                os.remove(result_file_path)
                logger.debug(f"Removed file: {result_file_path}")
        
        # Clear session data
        session.pop('data_file', None)
        session.pop('results_file', None)
        session.pop('paths_file', None)
        session.pop('model_results_summary', None)
        
        return jsonify({'success': True, 'message': 'Cleanup completed successfully'})
        
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})
