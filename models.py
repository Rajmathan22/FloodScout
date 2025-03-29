from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
import time
import logging

logger = logging.getLogger(__name__)

class FloodRiskModel:
    """
    Class to handle flood risk prediction using multiple ML models
    """
    def __init__(self):
        self.rf_model = None
        self.mlp_model = None
        self.improved_rf_model = None
        self.combined_mlp_model = None
        self.scaler = StandardScaler()
        self.features = None
        self.target = None
        self.top_features = None
        
    def preprocess_data(self, X, y=None):
        """
        Preprocess input data for model training or prediction
        
        Args:
            X (DataFrame): Input features
            y (Series, optional): Target variable
            
        Returns:
            Processed features and target (if provided)
        """
        # Save feature names
        self.features = X.columns.tolist()
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X) if y is not None else self.scaler.transform(X)
        
        if y is not None:
            self.target = y
            return X_scaled, y
        return X_scaled
    
    def train_random_forest(self, X_train, y_train):
        """
        Train Random Forest model with simplified parameters to avoid timeout
        
        Args:
            X_train: Scaled training features
            y_train: Training target
        """
        # Use a simpler model with fewer parameters to avoid timeout
        rf_param_grid = {
            'n_estimators': [100],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5]
        }
        
        # Set a timeout to prevent hanging
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            rf_param_grid,
            cv=2,  # Reduce cross-validation folds
            scoring='accuracy',
            verbose=1,
            n_jobs=2  # Limit parallel jobs
        )
        
        try:
            logger.info("Starting RandomForest grid search...")
            start_time = time.time()
            rf_grid.fit(X_train, y_train)
            logger.info(f"RandomForest grid search completed in {time.time() - start_time:.2f} seconds")
            self.rf_model = rf_grid.best_estimator_
            logger.info(f"Best RandomForest parameters: {rf_grid.best_params_}")
        except Exception as e:
            logger.error(f"GridSearch timeout or error: {str(e)}. Using default model.")
            # If timeout or error, use a simple model with default parameters
            self.rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.rf_model.fit(X_train, y_train)
            
        return self.rf_model
    
    def train_mlp(self, X_train, y_train):
        """
        Train MLP model with simplified parameters to avoid timeout
        
        Args:
            X_train: Scaled training features
            y_train: Training target
        """
        # Improved parameter grid based on research
        mlp_param_grid = {
            'hidden_layer_sizes': [(100,), (100, 50)],
            'activation': ['relu'],
            'max_iter': [2000],
            'early_stopping': [True],
            'learning_rate': ['adaptive'],
            'alpha': [0.0001, 0.001]
        }
        
        mlp_grid = GridSearchCV(
            MLPClassifier(random_state=42),
            mlp_param_grid,
            cv=2,  # Reduce cross-validation folds
            scoring='accuracy',
            verbose=1,
            n_jobs=2  # Limit parallel jobs
        )
        
        try:
            logger.info("Starting MLP grid search...")
            start_time = time.time()
            mlp_grid.fit(X_train, y_train)
            logger.info(f"MLP grid search completed in {time.time() - start_time:.2f} seconds")
            self.mlp_model = mlp_grid.best_estimator_
            logger.info(f"Best MLP parameters: {mlp_grid.best_params_}")
        except Exception as e:
            logger.error(f"GridSearch timeout or error: {str(e)}. Using default model.")
            # If timeout or error, use a simple model with default parameters
            self.mlp_model = MLPClassifier(
                hidden_layer_sizes=(100,),
                activation='relu',
                max_iter=2000,
                early_stopping=True,
                learning_rate='adaptive',
                random_state=42
            )
            self.mlp_model.fit(X_train, y_train)
            
        return self.mlp_model
    
    def train_improved_models(self, X, y):
        """
        Train improved models using feature selection
        
        Args:
            X: Original unscaled features DataFrame
            y: Target variable
        """
        # Train initial models if not already trained
        if self.rf_model is None:
            X_scaled = self.scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            self.train_random_forest(X_train, y_train)
            self.train_mlp(X_train, y_train)
        
        # Get feature importances
        feature_importances = pd.DataFrame({
            'Feature': self.features,
            'Importance': self.rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Select top features
        self.top_features = feature_importances.head(15)['Feature'].tolist()
        logger.info(f"Selected top {len(self.top_features)} features for improved model")
        
        # Get data with top features only
        X_top = X[self.top_features]
        
        # Split data
        X_train_top, X_test_top, y_train_top, y_test_top = train_test_split(
            X_top, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler_top = StandardScaler()
        X_train_top_scaled = scaler_top.fit_transform(X_train_top)
        
        # Train improved RF model
        logger.info("Training improved RF model with top features...")
        self.improved_rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced',
            random_state=42
        )
        
        self.improved_rf_model.fit(X_train_top_scaled, y_train_top)
        
        # Create combined features model
        if self.rf_model is not None:
            logger.info("Training MLP with RF predictions as features...")
            # Create a combined feature set that includes RF predictions
            X_scaled = self.scaler.transform(X)
            rf_probas = self.rf_model.predict_proba(X_scaled)[:, 1]
            
            # Add RF probabilities as a feature
            X_combined = X.copy()
            X_combined['RF_Probability'] = rf_probas
            
            # Split the combined data
            X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(
                X_combined, y, test_size=0.2, random_state=42
            )
            
            # Scale the features
            self.combined_scaler = StandardScaler()
            X_train_combined_scaled = self.combined_scaler.fit_transform(X_train_combined)
            
            # Train the combined MLP model
            self.combined_mlp_model = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                max_iter=2000,
                early_stopping=True,
                learning_rate='adaptive',
                random_state=42
            )
            
            self.combined_mlp_model.fit(X_train_combined_scaled, y_train_combined)
        
        return {
            'improved_rf_model': self.improved_rf_model,
            'combined_mlp_model': self.combined_mlp_model
        }
    
    def create_ensemble(self, X):
        """
        Create ensemble predictions by averaging RF and MLP probabilities
        
        Args:
            X: Scaled features
            
        Returns:
            Ensemble probabilities and predictions
        """
        if self.rf_model is None or self.mlp_model is None:
            raise ValueError("Models must be trained before creating ensemble predictions")
        
        rf_proba = self.rf_model.predict_proba(X)[:, 1]
        mlp_proba = self.mlp_model.predict_proba(X)[:, 1]
        
        ensemble_proba = (rf_proba + mlp_proba) / 2
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        
        return ensemble_proba, ensemble_pred
    
    def create_advanced_ensemble(self, X_orig):
        """
        Create advanced ensemble predictions using all trained models
        
        Args:
            X_orig: Original unscaled features
            
        Returns:
            Advanced ensemble probabilities and predictions
        """
        # Ensure all models are trained
        if (self.rf_model is None or self.mlp_model is None or 
            self.improved_rf_model is None or self.combined_mlp_model is None):
            logger.warning("Some models are not trained for advanced ensemble")
            # Fall back to basic ensemble
            X_scaled = self.scaler.transform(X_orig)
            return self.create_ensemble(X_scaled)
        
        # Get probabilities from all models
        X_scaled = self.scaler.transform(X_orig)
        rf_proba = self.rf_model.predict_proba(X_scaled)[:, 1]
        mlp_proba = self.mlp_model.predict_proba(X_scaled)[:, 1]
        
        # Get improved RF probabilities
        X_top = X_orig[self.top_features]
        X_top_scaled = StandardScaler().fit_transform(X_top)
        improved_rf_proba = self.improved_rf_model.predict_proba(X_top_scaled)[:, 1]
        
        # Get combined MLP probabilities
        X_combined = X_orig.copy()
        X_combined['RF_Probability'] = rf_proba
        X_combined_scaled = self.combined_scaler.transform(X_combined)
        combined_mlp_proba = self.combined_mlp_model.predict_proba(X_combined_scaled)[:, 1]
        
        # Create ensemble probabilities (average of all models)
        advanced_ensemble_proba = (rf_proba + mlp_proba + improved_rf_proba + combined_mlp_proba) / 4
        advanced_ensemble_pred = (advanced_ensemble_proba > 0.5).astype(int)
        
        return advanced_ensemble_proba, advanced_ensemble_pred
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all models on test data
        
        Args:
            X_test: Scaled test features
            y_test: Test target
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.rf_model is None or self.mlp_model is None:
            raise ValueError("Models must be trained before evaluation")
        
        # Get predictions
        rf_pred = self.rf_model.predict(X_test)
        mlp_pred = self.mlp_model.predict(X_test)
        ensemble_proba, ensemble_pred = self.create_ensemble(X_test)
        
        # Calculate accuracy scores
        rf_accuracy = accuracy_score(y_test, rf_pred)
        mlp_accuracy = accuracy_score(y_test, mlp_pred)
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        
        # Generate classification reports
        rf_report = classification_report(y_test, rf_pred, output_dict=True)
        mlp_report = classification_report(y_test, mlp_pred, output_dict=True)
        ensemble_report = classification_report(y_test, ensemble_pred, output_dict=True)
        
        # Get feature importances from RF model
        feature_importances = pd.DataFrame({
            'Feature': self.features,
            'Importance': self.rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        results = {
            'rf_accuracy': rf_accuracy,
            'mlp_accuracy': mlp_accuracy,
            'ensemble_accuracy': ensemble_accuracy,
            'rf_report': rf_report,
            'mlp_report': mlp_report,
            'ensemble_report': ensemble_report,
            'feature_importances': feature_importances.to_dict('records')
        }
        
        # Add advanced model evaluations if available
        if self.improved_rf_model is not None and self.combined_mlp_model is not None:
            # Need to recalculate test sets for these models
            X_test_orig = pd.DataFrame(self.scaler.inverse_transform(X_test), columns=self.features)
            
            # Evaluate improved RF
            X_top_test = X_test_orig[self.top_features]
            X_top_test_scaled = StandardScaler().fit_transform(X_top_test)
            improved_rf_pred = self.improved_rf_model.predict(X_top_test_scaled)
            improved_rf_accuracy = accuracy_score(y_test, improved_rf_pred)
            improved_rf_report = classification_report(y_test, improved_rf_pred, output_dict=True)
            
            # Evaluate combined MLP
            X_combined_test = X_test_orig.copy()
            X_combined_test['RF_Probability'] = self.rf_model.predict_proba(X_test)[:, 1]
            X_combined_test_scaled = self.combined_scaler.transform(X_combined_test)
            combined_mlp_pred = self.combined_mlp_model.predict(X_combined_test_scaled)
            combined_mlp_accuracy = accuracy_score(y_test, combined_mlp_pred)
            combined_mlp_report = classification_report(y_test, combined_mlp_pred, output_dict=True)
            
            # Add to results
            results.update({
                'improved_rf_accuracy': improved_rf_accuracy,
                'combined_mlp_accuracy': combined_mlp_accuracy,
                'improved_rf_report': improved_rf_report,
                'combined_mlp_report': combined_mlp_report
            })
            
            # Calculate advanced ensemble metrics if possible
            try:
                advanced_ensemble_proba, advanced_ensemble_pred = self.create_advanced_ensemble(X_test_orig)
                advanced_ensemble_accuracy = accuracy_score(y_test, advanced_ensemble_pred)
                advanced_ensemble_report = classification_report(y_test, advanced_ensemble_pred, output_dict=True)
                
                results.update({
                    'advanced_ensemble_accuracy': advanced_ensemble_accuracy,
                    'advanced_ensemble_report': advanced_ensemble_report
                })
            except Exception as e:
                logger.error(f"Could not calculate advanced ensemble metrics: {str(e)}")
        
        return results
    
    def predict(self, X):
        """
        Generate predictions for new data
        
        Args:
            X: Input features (unscaled)
            
        Returns:
            Dictionary of predictions from all models
        """
        if self.rf_model is None or self.mlp_model is None:
            raise ValueError("Models must be trained before prediction")
        
        # Scale the features
        X_scaled = self.scaler.transform(X)
        
        # Get RF predictions
        rf_proba = self.rf_model.predict_proba(X_scaled)[:, 1]
        rf_pred = self.rf_model.predict(X_scaled)
        
        # Get MLP predictions
        mlp_proba = self.mlp_model.predict_proba(X_scaled)[:, 1]
        mlp_pred = self.mlp_model.predict(X_scaled)
        
        # Create ensemble predictions
        ensemble_proba, ensemble_pred = self.create_ensemble(X_scaled)
        
        results = {
            'rf_probabilities': rf_proba,
            'rf_predictions': rf_pred,
            'mlp_probabilities': mlp_proba,
            'mlp_predictions': mlp_pred,
            'ensemble_probabilities': ensemble_proba,
            'ensemble_predictions': ensemble_pred
        }
        
        # Add advanced model predictions if available
        if self.improved_rf_model is not None and self.combined_mlp_model is not None:
            try:
                # Get improved RF predictions
                X_top = X[self.top_features]
                X_top_scaled = StandardScaler().fit_transform(X_top)
                improved_rf_proba = self.improved_rf_model.predict_proba(X_top_scaled)[:, 1]
                improved_rf_pred = self.improved_rf_model.predict(X_top_scaled)
                
                # Get combined MLP predictions
                X_combined = X.copy()
                X_combined['RF_Probability'] = rf_proba
                X_combined_scaled = self.combined_scaler.transform(X_combined)
                combined_mlp_proba = self.combined_mlp_model.predict_proba(X_combined_scaled)[:, 1]
                combined_mlp_pred = self.combined_mlp_model.predict(X_combined_scaled)
                
                # Add to results
                results.update({
                    'improved_rf_probabilities': improved_rf_proba,
                    'improved_rf_predictions': improved_rf_pred,
                    'combined_mlp_probabilities': combined_mlp_proba,
                    'combined_mlp_predictions': combined_mlp_pred
                })
                
                # Add advanced ensemble
                advanced_ensemble_proba, advanced_ensemble_pred = self.create_advanced_ensemble(X)
                results.update({
                    'advanced_ensemble_probabilities': advanced_ensemble_proba,
                    'advanced_ensemble_predictions': advanced_ensemble_pred
                })
            except Exception as e:
                logger.error(f"Error generating advanced model predictions: {str(e)}")
        
        return results

# Import necessary modules for the train_test_split which is used in new methods
from sklearn.model_selection import train_test_split
