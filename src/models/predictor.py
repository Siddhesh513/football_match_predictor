import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FootballPredictor:
    """Main predictor class for football match predictions"""

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_name = None
        self.feature_names = None
        self.scaler = None
        self.label_encoder = None
        self.feature_engineer = None
        self.use_scaling = False
        self.model_metadata = {}

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """
        Load trained model from disk

        Args:
            model_path: Path to saved model file
        """
        logger.info(f"Loading model from {model_path}")

        try:
            model_data = joblib.load(model_path)

            self.model = model_data['model']
            self.model_name = model_data['model_name']
            self.feature_names = model_data['feature_names']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.feature_engineer = model_data['feature_engineer']
            self.use_scaling = model_data['use_scaling']
            self.model_metadata = model_data.get('metadata', {})

            logger.info(f"Model loaded successfully: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def save_model(self, model_path: str):
        """
        Save trained model to disk

        Args:
            model_path: Path to save model file
        """
        logger.info(f"Saving model to {model_path}")

        # Create directory if it doesn't exist
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_engineer': self.feature_engineer,
            'use_scaling': self.use_scaling,
            'metadata': self.model_metadata
        }

        joblib.dump(model_data, model_path)
        logger.info("Model saved successfully")

    def create_feature_vector(self, home_team: str, away_team: str,
                              additional_features: Optional[Dict] = None) -> pd.DataFrame:
        """
        Create feature vector for prediction

        Args:
            home_team: Home team name
            away_team: Away team name
            additional_features: Optional additional features

        Returns:
            DataFrame with feature vector
        """
        logger.info(f"Creating feature vector for {home_team} vs {away_team}")

        # Initialize features with team information
        features = {
            'HomeTeamStrength': 0.5,
            'AwayTeamStrength': 0.5,
            'StrengthDifference': 0.0
        }

        # Add team strengths if available
        if self.feature_engineer and hasattr(self.feature_engineer, 'team_stats'):
            home_strength = self.feature_engineer.team_stats.get(
                home_team, 0.5)
            away_strength = self.feature_engineer.team_stats.get(
                away_team, 0.5)

            features['HomeTeamStrength'] = home_strength
            features['AwayTeamStrength'] = away_strength
            features['StrengthDifference'] = home_strength - away_strength

        # Add default statistical features
        default_features = {
            'HS': 10, 'AS': 10, 'HST': 4, 'AST': 4,
            'HF': 12, 'AF': 12, 'HC': 5, 'AC': 5,
            'HY': 2, 'AY': 2, 'HR': 0, 'AR': 0,
            'HTHG': 0, 'HTAG': 0,
            'ShotAccuracyHome': 0.4,
            'ShotAccuracyAway': 0.4,
            'ShotAccuracyDiff': 0.0
        }

        features.update(default_features)

        # Add any additional features provided
        if additional_features:
            features.update(additional_features)

        # Create DataFrame
        feature_df = pd.DataFrame([features])

        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in feature_df.columns:
                feature_df[feature] = 0

        # Select and order features
        feature_df = feature_df[self.feature_names]

        return feature_df

    def predict(self, home_team: str, away_team: str,
                additional_features: Optional[Dict] = None) -> Dict:
        """
        Make prediction for a match

        Args:
            home_team: Home team name
            away_team: Away team name
            additional_features: Optional additional features

        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")

        logger.info(f"Predicting: {home_team} vs {away_team}")

        # Create feature vector
        feature_df = self.create_feature_vector(
            home_team, away_team, additional_features)

        # Scale if necessary
        if self.use_scaling:
            feature_array = self.scaler.transform(feature_df)
        else:
            feature_array = feature_df.values

        # Make prediction
        prediction = self.model.predict(feature_array)[0]
        probabilities = self.model.predict_proba(feature_array)[0]

        # Get class labels
        classes = self.label_encoder.classes_

        # Format results
        result = {
            'home_team': home_team,
            'away_team': away_team,
            'model_used': self.model_name,
            'predicted_outcome': classes[prediction],
            'probabilities': {},
            'confidence': float(np.max(probabilities))
        }

        # Map probabilities to outcomes
        for i, class_name in enumerate(classes):
            if class_name == 'H':
                result['probabilities']['home_win'] = float(probabilities[i])
            elif class_name == 'A':
                result['probabilities']['away_win'] = float(probabilities[i])
            elif class_name == 'D':
                result['probabilities']['draw'] = float(probabilities[i])

        logger.info(f"Prediction: {result['predicted_outcome']} "
                    f"(confidence: {result['confidence']:.1%})")

        return result

    def predict_batch(self, matches: list) -> list:
        """
        Make predictions for multiple matches

        Args:
            matches: List of tuples (home_team, away_team) or dicts with match info

        Returns:
            List of prediction results
        """
        logger.info(f"Making batch predictions for {len(matches)} matches")

        results = []
        for match in matches:
            if isinstance(match, tuple):
                home_team, away_team = match
                result = self.predict(home_team, away_team)
            elif isinstance(match, dict):
                result = self.predict(
                    match['home_team'],
                    match['away_team'],
                    match.get('features')
                )
            else:
                logger.warning(f"Invalid match format: {match}")
                continue

            results.append(result)

        return results
