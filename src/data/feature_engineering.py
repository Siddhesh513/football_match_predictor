import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handle feature engineering operations"""

    def __init__(self):
        self.team_stats = {}

    def calculate_team_strength(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate team strength based on historical performance

        Args:
            df: DataFrame with match data

        Returns:
            Dictionary mapping team names to strength scores
        """
        logger.info("Calculating team strengths")
        team_stats = {}

        # Get unique teams
        teams = set(list(df['HomeTeam'].unique()) +
                    list(df['AwayTeam'].unique()))

        for team in teams:
            home_games = df[df['HomeTeam'] == team]
            away_games = df[df['AwayTeam'] == team]

            # Calculate statistics
            goals_scored = home_games['FTHG'].sum() + away_games['FTAG'].sum()
            goals_conceded = home_games['FTAG'].sum(
            ) + away_games['FTHG'].sum()

            home_wins = len(home_games[home_games['FTR'] == 'H'])
            away_wins = len(away_games[away_games['FTR'] == 'A'])
            total_games = len(home_games) + len(away_games)

            if total_games > 0:
                win_rate = (home_wins + away_wins) / total_games
                goal_diff = goals_scored - goals_conceded

                # Calculate strength score (normalized between 0.1 and 0.9)
                strength = 0.5 + (win_rate - 0.33) * 0.3 + \
                    (goal_diff / max(total_games, 1)) * 0.1
                team_stats[team] = max(0.1, min(0.9, strength))
            else:
                team_stats[team] = 0.5  # Default strength

        self.team_stats = team_stats
        logger.info(f"Calculated strength for {len(team_stats)} teams")
        return team_stats

    def add_team_strength_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add team strength features to DataFrame

        Args:
            df: DataFrame with match data

        Returns:
            DataFrame with added team strength features
        """
        logger.info("Adding team strength features")

        if not self.team_stats:
            self.calculate_team_strength(df)

        df['HomeTeamStrength'] = df['HomeTeam'].map(self.team_stats)
        df['AwayTeamStrength'] = df['AwayTeam'].map(self.team_stats)
        df['StrengthDifference'] = df['HomeTeamStrength'] - df['AwayTeamStrength']

        return df

    def add_shot_accuracy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add shot accuracy features

        Args:
            df: DataFrame with match data

        Returns:
            DataFrame with added shot accuracy features
        """
        logger.info("Adding shot accuracy features")

        # Avoid division by zero
        df['ShotAccuracyHome'] = df['HST'] / (df['HS'] + 1e-6)
        df['ShotAccuracyAway'] = df['AST'] / (df['AS'] + 1e-6)
        df['ShotAccuracyDiff'] = df['ShotAccuracyHome'] - df['ShotAccuracyAway']

        return df

    def add_form_features(self, df: pd.DataFrame, n_games: int = 5) -> pd.DataFrame:
        """
        Add recent form features for teams

        Args:
            df: DataFrame with match data (should be sorted by date)
            n_games: Number of recent games to consider

        Returns:
            DataFrame with added form features
        """
        logger.info(f"Adding form features (last {n_games} games)")

        # This would require date information and proper sorting
        # For now, returning df as is
        # In production, you would calculate rolling averages of recent performance

        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between existing features

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with added interaction features
        """
        logger.info("Creating interaction features")

        # Example interaction features
        if 'HS' in df.columns and 'AS' in df.columns:
            df['TotalShots'] = df['HS'] + df['AS']
            df['ShotsDifference'] = df['HS'] - df['AS']

        if 'HF' in df.columns and 'AF' in df.columns:
            df['TotalFouls'] = df['HF'] + df['AF']
            df['FoulsDifference'] = df['HF'] - df['AF']

        if 'HC' in df.columns and 'AC' in df.columns:
            df['TotalCorners'] = df['HC'] + df['AC']
            df['CornersDifference'] = df['HC'] - df['AC']

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps

        Args:
            df: DataFrame with raw features

        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering pipeline")

        # Apply all feature engineering steps
        df = self.add_team_strength_features(df)
        df = self.add_shot_accuracy_features(df)
        df = self.add_form_features(df)
        df = self.create_interaction_features(df)

        logger.info("Feature engineering complete")
        return df
