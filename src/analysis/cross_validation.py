import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class CrossValidator:
    """Perform cross-validation for model selection"""

    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.models = self._initialize_models()
        self.results = {}
        self.best_model = None
        self.best_model_name = None

    def _initialize_models(self) -> Dict[str, Any]:
        """
        Initialize machine learning models

        Returns:
            Dictionary of model instances
        """
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=self.random_state
            ),
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ),
            'SVM': SVC(
                probability=True,
                random_state=self.random_state
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                random_state=self.random_state,
                max_iter=1000
            )
        }

        return models

    def add_model(self, name: str, model: Any):
        """
        Add a custom model to the validator

        Args:
            name: Model name
            model: Model instance
        """
        self.models[name] = model
        logger.info(f"Added model: {name}")

    def validate_model(self, model: Any, X: np.ndarray, y: np.ndarray,
                       model_name: str) -> Dict[str, Any]:
        """
        Validate a single model using cross-validation

        Args:
            model: Model instance
            X: Features array
            y: Target array
            model_name: Name of the model

        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating {model_name}")

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                             random_state=self.random_state)

        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

        results = {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'scores': scores,
            'min_accuracy': scores.min(),
            'max_accuracy': scores.max()
        }

        logger.info(f"{model_name}: {results['mean_accuracy']:.4f} "
                    f"(+/- {results['std_accuracy'] * 2:.4f})")

        return results

    def validate_all_models(self, X: np.ndarray, y: np.ndarray,
                            X_scaled: np.ndarray = None) -> Dict[str, Dict]:
        """
        Validate all models

        Args:
            X: Features array
            y: Target array
            X_scaled: Scaled features array (for models that need scaling)

        Returns:
            Dictionary with all validation results
        """
        logger.info("Starting cross-validation for all models")

        results = {}

        for name, model in self.models.items():
            # Use scaled data for certain models
            if X_scaled is not None and name in ['SVM', 'Neural Network', 'Logistic Regression']:
                results[name] = self.validate_model(model, X_scaled, y, name)
            else:
                results[name] = self.validate_model(model, X, y, name)

        self.results = results

        # Find best model
        self.best_model_name = max(results.keys(),
                                   key=lambda x: results[x]['mean_accuracy'])

        logger.info(f"Best model: {self.best_model_name} with accuracy "
                    f"{results[self.best_model_name]['mean_accuracy']:.4f}")

        return results

    def get_best_model(self) -> Tuple[str, Any, Dict]:
        """
        Get the best performing model

        Returns:
            Tuple of (model name, model instance, results)
        """
        if self.best_model_name is None:
            raise ValueError("No validation has been performed yet")

        return (
            self.best_model_name,
            self.models[self.best_model_name],
            self.results[self.best_model_name]
        )

    def get_model_comparison(self) -> pd.DataFrame:
        """
        Get comparison of all models

        Returns:
            DataFrame with model comparison
        """
        if not self.results:
            raise ValueError("No validation has been performed yet")

        comparison_data = []
        for name, result in self.results.items():
            comparison_data.append({
                'Model': name,
                'Mean Accuracy': result['mean_accuracy'],
                'Std Accuracy': result['std_accuracy'],
                'Min Accuracy': result['min_accuracy'],
                'Max Accuracy': result['max_accuracy']
            })

        return pd.DataFrame(comparison_data).sort_values('Mean Accuracy', ascending=False)
