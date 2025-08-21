import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)


class ExploratoryDataAnalyzer:
    """Perform Exploratory Data Analysis"""

    def __init__(self):
        self.feature_scores = {}
        self.selected_features = []

    def calculate_feature_importance(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """
        Calculate feature importance using correlation and statistical tests

        Args:
            X: Features DataFrame
            y: Target array

        Returns:
            Dictionary of feature importance scores
        """
        logger.info("Calculating feature importance")

        feature_scores = {}

        # Convert target to numeric for correlation
        y_numeric = np.where(y == 0, -1, np.where(y == 1, 0, 1))

        # Method 1: Correlation with target
        correlations = {}
        for col in X.columns:
            if X[col].var() > 0:
                corr = np.corrcoef(X[col], y_numeric)[0, 1]
                correlations[col] = abs(corr) if not np.isnan(corr) else 0
            else:
                correlations[col] = 0

        # Method 2: F-score
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X, y)
        f_scores = dict(zip(X.columns, selector.scores_))

        # Combine scores
        max_f_score = max(f_scores.values()) if f_scores else 1
        for col in X.columns:
            feature_scores[col] = (
                correlations.get(col, 0) * 0.6 +
                (f_scores.get(col, 0) / max_f_score) * 0.4
            )

        self.feature_scores = feature_scores
        return feature_scores

    def select_features(self, X: pd.DataFrame,
                        feature_scores: Dict[str, float],
                        min_features: int = 8,
                        remove_bottom: int = 5) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select top features based on importance scores

        Args:
            X: Features DataFrame
            feature_scores: Dictionary of feature importance scores
            min_features: Minimum number of features to keep
            remove_bottom: Number of least important features to remove

        Returns:
            Tuple of (selected features DataFrame, list of selected feature names)
        """
        logger.info("Selecting features")

        # Sort features by importance
        sorted_features = sorted(
            feature_scores.items(), key=lambda x: x[1], reverse=True)

        # Select features
        n_features = max(min_features, len(sorted_features) - remove_bottom)
        selected_features = [f[0] for f in sorted_features[:n_features]]

        logger.info(
            f"Selected {len(selected_features)} features out of {len(sorted_features)}")

        self.selected_features = selected_features
        return X[selected_features], selected_features

    def get_feature_statistics(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get statistical summary of features

        Args:
            X: Features DataFrame

        Returns:
            DataFrame with feature statistics
        """
        logger.info("Calculating feature statistics")

        stats = pd.DataFrame({
            'mean': X.mean(),
            'std': X.std(),
            'min': X.min(),
            'max': X.max(),
            'median': X.median(),
            'skew': X.skew(),
            'kurtosis': X.kurtosis()
        })

        return stats

    def analyze_target_distribution(self, y: np.ndarray,
                                    class_names: List[str]) -> Dict[str, float]:
        """
        Analyze target variable distribution

        Args:
            y: Target array
            class_names: List of class names

        Returns:
            Dictionary with class distribution
        """
        logger.info("Analyzing target distribution")

        unique, counts = np.unique(y, return_counts=True)
        distribution = {}

        for idx, count in zip(unique, counts):
            if idx < len(class_names):
                distribution[class_names[idx]] = count / len(y)

        return distribution

    def perform_eda(self, X: pd.DataFrame, y: np.ndarray) -> Dict:
        """
        Perform complete EDA pipeline

        Args:
            X: Features DataFrame
            y: Target array

        Returns:
            Dictionary with EDA results
        """
        logger.info("Starting EDA pipeline")

        # Calculate feature importance
        feature_scores = self.calculate_feature_importance(X, y)

        # Select features
        X_selected, selected_features = self.select_features(X, feature_scores)

        # Get statistics
        feature_stats = self.get_feature_statistics(X_selected)

        # Sort features by importance
        sorted_features = sorted(
            feature_scores.items(), key=lambda x: x[1], reverse=True)

        results = {
            'feature_scores': feature_scores,
            'selected_features': selected_features,
            'sorted_features': sorted_features,
            'feature_statistics': feature_stats,
            'X_selected': X_selected
        }

        logger.info("EDA complete")
        return results
