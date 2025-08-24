import pytest
import tempfile
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.models.predictor import FootballPredictor
from src.data.feature_engineering import FeatureEngineer


class TestFootballPredictor:

    @pytest.fixture
    def predictor(self):
        """Create a predictor with mock model"""
        predictor = FootballPredictor()

        # Setup mock model
        predictor.model = RandomForestClassifier(
            n_estimators=10, random_state=42)
        predictor.model_name = "Test Model"
        predictor.feature_names = ['HomeTeamStrength',
                                   'AwayTeamStrength', 'StrengthDifference']
        predictor.scaler = StandardScaler()
        predictor.label_encoder = LabelEncoder()
        predictor.feature_engineer = FeatureEngineer()
        predictor.use_scaling = False

        # Fit mock model with dummy data
        X_dummy = np.random.rand(100, 3)
        y_dummy = np.random.choice([0, 1, 2], 100)
        predictor.model.fit(X_dummy, y_dummy)
        predictor.label_encoder.fit(['A', 'D', 'H'])
        predictor.scaler.fit(X_dummy)

        return predictor

    def test_save_and_load_model(self, predictor):
        """Test saving and loading model"""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            # Save model
            predictor.save_model(tmp.name)

            # Load model
            new_predictor = FootballPredictor()
            new_predictor.load_model(tmp.name)

            assert new_predictor.model_name == predictor.model_name
            assert new_predictor.feature_names == predictor.feature_names
            assert new_predictor.use_scaling == predictor.use_scaling

    def test_create_feature_vector(self, predictor):
        """Test feature vector creation"""
        feature_df = predictor.create_feature_vector('Arsenal', 'Liverpool')

        assert len(feature_df) == 1
        assert list(feature_df.columns) == predictor.feature_names
        assert 'HomeTeamStrength' in feature_df.columns

    def test_predict(self, predictor):
        """Test prediction"""
        result = predictor.predict('Arsenal', 'Liverpool')

        assert 'predicted_outcome' in result
        assert 'probabilities' in result
        assert 'confidence' in result
        assert result['predicted_outcome'] in ['H', 'D', 'A']
        assert 0 <= result['confidence'] <= 1

    def test_predict_without_model(self):
        """Test prediction without loaded model"""
        predictor = FootballPredictor()

        with pytest.raises(ValueError):
            predictor.predict('Arsenal', 'Liverpool')

    def test_predict_batch(self, predictor):
        """Test batch predictions"""
        matches = [
            ('Arsenal', 'Liverpool'),
            ('Chelsea', 'Manchester United'),
            {'home_team': 'Tottenham', 'away_team': 'Leicester'}
        ]

        results = predictor.predict_batch(matches)

        assert len(results) == 3
        assert all('predicted_outcome' in r for r in results)
