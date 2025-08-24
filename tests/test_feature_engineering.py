import pytest
import pandas as pd
import numpy as np
from src.data.feature_engineering import FeatureEngineer


class TestFeatureEngineer:

    @pytest.fixture
    def engineer(self):
        return FeatureEngineer()

    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing"""
        data = {
            'HomeTeam': ['Arsenal', 'Liverpool', 'Chelsea', 'Arsenal'],
            'AwayTeam': ['Liverpool', 'Chelsea', 'Arsenal', 'Chelsea'],
            'FTR': ['H', 'D', 'A', 'H'],
            'FTHG': [2, 1, 0, 3],
            'FTAG': [1, 1, 2, 1],
            'HS': [15, 10, 8, 20],
            'AS': [10, 12, 20, 8],
            'HST': [6, 4, 3, 10],
            'AST': [4, 5, 8, 3]
        }
        return pd.DataFrame(data)

    def test_calculate_team_strength(self, engineer, sample_data):
        """Test team strength calculation"""
        team_stats = engineer.calculate_team_strength(sample_data)

        assert 'Arsenal' in team_stats
        assert 'Liverpool' in team_stats
        assert 'Chelsea' in team_stats

        # Check that strengths are within valid range
        for team, strength in team_stats.items():
            assert 0.1 <= strength <= 0.9

    def test_add_team_strength_features(self, engineer, sample_data):
        """Test adding team strength features"""
        result = engineer.add_team_strength_features(sample_data)

        assert 'HomeTeamStrength' in result.columns
        assert 'AwayTeamStrength' in result.columns
        assert 'StrengthDifference' in result.columns

        # Check that difference is calculated correctly
        assert all(result['StrengthDifference'] ==
                   result['HomeTeamStrength'] - result['AwayTeamStrength'])

    def test_add_shot_accuracy_features(self, engineer, sample_data):
        """Test adding shot accuracy features"""
        result = engineer.add_shot_accuracy_features(sample_data)

        assert 'ShotAccuracyHome' in result.columns
        assert 'ShotAccuracyAway' in result.columns
        assert 'ShotAccuracyDiff' in result.columns

        # Check that values are reasonable (between 0 and 1)
        assert all(0 <= result['ShotAccuracyHome']) and all(
            result['ShotAccuracyHome'] <= 1)
        assert all(0 <= result['ShotAccuracyAway']) and all(
            result['ShotAccuracyAway'] <= 1)
