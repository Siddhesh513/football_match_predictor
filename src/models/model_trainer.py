import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import logging

from ..data.preprocessor import DataPreprocessor
from ..data.feature_engineering import FeatureEngineer
from ..analysis.eda import ExploratoryDataAnalyzer
from ..analysis.cross_validation import CrossValidator
from .predictor import FootballPredictor

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and manage football prediction models"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.eda_analyzer = ExploratoryDataAnalyzer()
        self.cross_validator = CrossValidator(
            cv_folds=self.config.get('cv_folds', 5),
            random_state=self.config.get('random_state', 42)
        )
        self.predictor = FootballPredictor()
        self.training_results = {}

    def load_and_prepare_data(self, file_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load and prepare data for training

        Args:
            file_path: Path to data file

        Returns:
            Tuple of (features, target)
        """
        logger.info("Loading and preparing data")

        # Load data
        df = self.preprocessor.load_data(file_path)

        # Handle missing values
        df = self.preprocessor.handle_missing_values(df, strategy='zero')

        # Engineer features
        df = self.feature_engineer.engineer_features(df)

        # Encode categorical features
        categorical_features = self.config.get(
            'categorical_features', ['HTR', 'Referee'])
        df = self.preprocessor.encode_categorical_features(
            df, categorical_features)

        # Split features and target
        target_column = self.config.get('target_column', 'FTR')
        X, y = self.preprocessor.split_features_target(df, target_column)

        # Encode target
        y_encoded = self.preprocessor.encode_target(y)

        return X, y_encoded

    def train(self, file_path: str, save_path: Optional[str] = None) -> Dict:
        """
        Train model with complete pipeline

        Args:
            file_path: Path to training data
            save_path: Optional path to save trained model

        Returns:
            Dictionary with training results
        """
        logger.info("Starting training pipeline")

        # Load and prepare data
        X, y = self.load_and_prepare_data(file_path)

        # Perform EDA
        eda_results = self.eda_analyzer.perform_eda(X, y)
        X_selected = eda_results['X_selected']

        # Scale features
        X_scaled = self.preprocessor.scale_features(X_selected)

        # Cross-validation
        cv_results = self.cross_validator.validate_all_models(
            X_selected, y, X_scaled)

        # Get best model
        best_model_name, best_model, best_results = self.cross_validator.get_best_model()

        # Train best model on full dataset
        if best_model_name in ['SVM', 'Neural Network', 'Logistic Regression']:
            best_model.fit(X_scaled, y)
            use_scaling = True
        else:
            best_model.fit(X_selected, y)
            use_scaling = False

        # Update predictor
        self.predictor.model = best_model
        self.predictor.model_name = best_model_name
        self.predictor.feature_names = X_selected.columns.tolist()
        self.predictor.scaler = self.preprocessor.scaler
        self.predictor.label_encoder = self.preprocessor.label_encoder
        self.predictor.feature_engineer = self.feature_engineer
        self.predictor.use_scaling = use_scaling
        self.predictor.model_metadata = {
            'accuracy': best_results['mean_accuracy'],
            'std': best_results['std_accuracy'],
            'training_samples': len(X),
            'n_features': len(X_selected.columns)
        }

        # Save model if path provided
        if save_path:
            self.predictor.save_model(save_path)

        # Prepare results
        self.training_results = {
            'best_model': best_model_name,
            'accuracy': best_results['mean_accuracy'],
            'cv_results': cv_results,
            'selected_features': eda_results['selected_features'],
            'feature_importance': eda_results['sorted_features'][:10],
            'n_samples': len(X),
            'n_features': len(X_selected.columns)
        }

        logger.info(f"Training complete. Best model: {best_model_name} "
                    f"with accuracy {best_results['mean_accuracy']:.4f}")

        return self.training_results

    def get_training_summary(self) -> str:
        """
        Get summary of training results

        Returns:
            String with training summary
        """
        if not self.training_results:
            return "No training has been performed yet."

        summary = []
        summary.append("=" * 60)
        summary.append("TRAINING SUMMARY")
        summary.append("=" * 60)
        summary.append(f"Best Model: {self.training_results['best_model']}")
        summary.append(f"Accuracy: {self.training_results['accuracy']:.1%}")
        summary.append(
            f"Training Samples: {self.training_results['n_samples']}")
        summary.append(
            f"Selected Features: {self.training_results['n_features']}")
        summary.append("\nTop 10 Features:")

        for i, (feature, score) in enumerate(self.training_results['feature_importance'], 1):
            summary.append(f"{i:2d}. {feature:25s} - Score: {score:.4f}")

        summary.append("\nModel Comparison:")
        for model_name, results in self.training_results['cv_results'].items():
            summary.append(
                f"  {model_name:20s}: {results['mean_accuracy']:.4f}")

        return "\n".join(summary)
