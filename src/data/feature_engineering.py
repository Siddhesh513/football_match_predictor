import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handle data loading and preprocessing operations"""

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = None
        self.original_data = None

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load CSV data

        Args:
            file_path: Path to CSV file

        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading data from {file_path}")

        try:
            df = pd.read_csv(file_path)
            logger.info(f"Dataset shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            self.original_data = df.copy()
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'zero') -> pd.DataFrame:
        """
        Handle missing values in dataset

        Args:
            df: DataFrame with potentially missing values
            strategy: Strategy for handling missing values ('zero', 'mean', 'median', 'drop')

        Returns:
            DataFrame with handled missing values
        """
        logger.info(f"Handling missing values with strategy: {strategy}")

        if strategy == 'zero':
            return df.fillna(0)
        elif strategy == 'mean':
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(
                df[numeric_columns].mean())
            return df
        elif strategy == 'median':
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(
                df[numeric_columns].median())
            return df
        elif strategy == 'drop':
            return df.dropna()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def encode_categorical_features(self, df: pd.DataFrame,
                                    categorical_features: list) -> pd.DataFrame:
        """
        One-hot encode categorical features

        Args:
            df: DataFrame with categorical features
            categorical_features: List of categorical feature names

        Returns:
            DataFrame with encoded features
        """
        logger.info(f"Encoding categorical features: {categorical_features}")

        for feature in categorical_features:
            if feature in df.columns:
                dummies = pd.get_dummies(
                    df[feature], prefix=feature, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(feature, axis=1)
                logger.info(
                    f"Encoded {feature} into {len(dummies.columns)} columns")

        return df

    def encode_target(self, y: pd.Series) -> np.ndarray:
        """
        Encode target variable

        Args:
            y: Target variable series

        Returns:
            Encoded target array
        """
        logger.info("Encoding target variable")
        y_encoded = self.label_encoder.fit_transform(y)
        logger.info(f"Target classes: {self.label_encoder.classes_}")
        return y_encoded

    def decode_target(self, y_encoded: np.ndarray) -> np.ndarray:
        """
        Decode target variable back to original labels

        Args:
            y_encoded: Encoded target array

        Returns:
            Decoded target array
        """
        return self.label_encoder.inverse_transform(y_encoded)

    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Scale features using StandardScaler

        Args:
            X: Features DataFrame
            fit: Whether to fit the scaler (True for training, False for prediction)

        Returns:
            Scaled features array
        """
        logger.info(f"Scaling features (fit={fit})")

        if fit:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)

    def split_features_target(self, df: pd.DataFrame,
                              target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split dataframe into features and target

        Args:
            df: Complete DataFrame
            target_column: Name of target column

        Returns:
            Tuple of (features, target)
        """
        logger.info(
            f"Splitting features and target (target column: {target_column})")

        # Remove outcome-related columns that would cause data leakage
        columns_to_drop = [target_column, 'FTHG', 'FTAG']

        # Select only numeric columns for features
        numeric_df = df.select_dtypes(include=[np.number])

        # Drop target and leakage columns
        X = numeric_df.drop(columns_to_drop, axis=1, errors='ignore')
        y = df[target_column]

        self.feature_names = X.columns.tolist()
        logger.info(f"Features shape: {X.shape}")

        return X, y
