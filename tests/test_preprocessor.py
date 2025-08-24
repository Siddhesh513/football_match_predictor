import pytest
import pandas as pd
import numpy as np
from src.data.preprocessor import DataPreprocessor


class TestDataPreprocessor:

    @pytest.fixture
    def preprocessor(self):
        return DataPreprocessor()

    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing"""
        data = {
            'HomeTeam': ['Arsenal', 'Liverpool', 'Chelsea'],
            'AwayTeam': ['Liverpool', 'Chelsea', 'Arsenal'],
            'FTR': ['H', 'D', 'A'],
            'FTHG': [2, 1, 0],
            'FTAG': [1, 1, 2],
            'HS': [15, 10, 8],
            'AS': [10, 12, 20],
            'HTR': ['H', 'D', 'A']
        }
        return pd.DataFrame(data)

    def test_load_data_file_not_found(self, preprocessor):
        """Test loading non-existent file"""
        with pytest.raises(FileNotFoundError):
            preprocessor.load_data('non_existent_file.csv')

    def test_handle_missing_values_zero_strategy(self, preprocessor, sample_data):
        """Test handling missing values with zero strategy"""
        # Add some NaN values
        sample_data.loc[0, 'HS'] = np.nan

        result = preprocessor.handle_missing_values(
            sample_data, strategy='zero')

        assert result.loc[0, 'HS'] == 0
        assert not result.isnull().any().any()

    def test_encode_categorical_features(self, preprocessor, sample_data):
        """Test categorical feature encoding"""
        categorical_features = ['HTR']
        result = preprocessor.encode_categorical_features(
            sample_data, categorical_features)

        # Check that original column is removed
        assert 'HTR' not in result.columns

        # Check that dummy columns are created
        assert 'HTR_D' in result.columns or 'HTR_H' in result.columns

    def test_encode_decode_target(self, preprocessor):
        """Test target encoding and decoding"""
        y = pd.Series(['H', 'A', 'D', 'H', 'A'])

        y_encoded = preprocessor.encode_target(y)
        assert len(np.unique(y_encoded)) == 3

        y_decoded = preprocessor.decode_target(y_encoded)
        assert list(y_decoded) == list(y)

    def test_split_features_target(self, preprocessor, sample_data):
        """Test splitting features and target"""
        X, y = preprocessor.split_features_target(sample_data, 'FTR')

        assert 'FTR' not in X.columns
        assert 'FTHG' not in X.columns  # Should be removed (data leakage)
        assert 'FTAG' not in X.columns  # Should be removed (data leakage)
        assert len(y) == len(sample_data)
