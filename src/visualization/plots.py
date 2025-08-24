# src/visualization/plots.py - FIXED VERSION
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class Visualizer:
    """Handle all visualization tasks"""

    def __init__(self, style: str = 'default', figsize: Tuple[int, int] = (15, 10)):
        self.figsize = figsize

        # Fix for matplotlib style issue
        available_styles = plt.style.available

        # Try to use seaborn-v0_8 or seaborn-darkgrid if available
        if 'seaborn-v0_8' in available_styles:
            plt.style.use('seaborn-v0_8')
        elif 'seaborn-darkgrid' in available_styles:
            plt.style.use('seaborn-darkgrid')
        elif 'ggplot' in available_styles:
            plt.style.use('ggplot')
        else:
            # Use default style if nothing else is available
            plt.style.use('default')

        # Set seaborn defaults if seaborn is installed
        try:
            sns.set_theme()
            sns.set_palette("husl")
        except:
            # If seaborn is not installed or has issues, continue without it
            pass

    def plot_feature_importance(self, feature_scores: List[Tuple[str, float]],
                                top_n: int = 15, save_path: Optional[str] = None):
        """
        Plot feature importance scores

        Args:
            feature_scores: List of tuples (feature_name, score)
            top_n: Number of top features to display
            save_path: Optional path to save figure
        """
        logger.info(f"Plotting top {top_n} feature importance")

        plt.figure(figsize=(10, 8))

        # Get top features
        top_features = feature_scores[:min(top_n, len(feature_scores))]
        features, scores = zip(*top_features)

        # Create horizontal bar plot
        y_pos = np.arange(len(features))
        plt.barh(y_pos, scores, color='skyblue', edgecolor='navy', alpha=0.7)

        plt.yticks(y_pos, features)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title('Feature Importance Analysis',
                  fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (feature, score) in enumerate(top_features):
            plt.text(score, i, f' {score:.3f}', va='center')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")

        plt.show()
        plt.close()

    def plot_correlation_matrix(self, df: pd.DataFrame, features: Optional[List[str]] = None,
                                save_path: Optional[str] = None):
        """
        Plot correlation matrix heatmap

        Args:
            df: DataFrame with features
            features: Optional list of features to include
            save_path: Optional path to save figure
        """
        logger.info("Plotting correlation matrix")

        if features:
            df_subset = df[features]
        else:
            df_subset = df

        plt.figure(figsize=(12, 10))

        corr_matrix = df_subset.corr()

        # Check if seaborn is available for heatmap
        try:
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                        cmap='coolwarm', center=0, square=True,
                        linewidths=0.5, cbar_kws={"shrink": 0.8})
        except:
            # Fallback to matplotlib if seaborn is not available
            plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
            plt.colorbar()
            plt.xticks(range(len(corr_matrix.columns)),
                       corr_matrix.columns, rotation=45, ha='right')
            plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)

        plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")

        plt.show()
        plt.close()

    def plot_model_comparison(self, cv_results: Dict[str, Dict],
                              save_path: Optional[str] = None):
        """
        Plot model comparison results

        Args:
            cv_results: Dictionary with cross-validation results
            save_path: Optional path to save figure
        """
        logger.info("Plotting model comparison")

        fig, axes = plt.subplots(1, 2, figsize=self.figsize)

        # Extract data
        model_names = list(cv_results.keys())
        accuracies = [cv_results[name]['mean_accuracy']
                      for name in model_names]
        stds = [cv_results[name]['std_accuracy'] for name in model_names]

        # Find best model
        best_idx = accuracies.index(max(accuracies))
        colors = ['gold' if i ==
                  best_idx else 'skyblue' for i in range(len(model_names))]

        # Plot 1: Bar chart with error bars
        ax1 = axes[0]
        bars = ax1.bar(model_names, accuracies, yerr=stds, capsize=5,
                       color=colors, edgecolor='navy', alpha=0.7)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Model Performance Comparison',
                      fontsize=14, fontweight='bold')
        ax1.set_ylim([0, 1])
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                     f'{acc:.3f}', ha='center', va='bottom')

        # Rotate x labels
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot 2: Box plot of CV scores
        ax2 = axes[1]
        cv_scores = [cv_results[name]['scores'] for name in model_names]
        bp = ax2.boxplot(cv_scores, labels=model_names, patch_artist=True)

        # Color the best model
        for i, patch in enumerate(bp['boxes']):
            if i == best_idx:
                patch.set_facecolor('gold')
            else:
                patch.set_facecolor('lightblue')

        ax2.set_ylabel('Cross-Validation Accuracy', fontsize=12)
        ax2.set_title('Cross-Validation Score Distribution',
                      fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")

        plt.show()
        plt.close()

    def plot_target_distribution(self, y: np.ndarray, class_names: List[str],
                                 save_path: Optional[str] = None):
        """
        Plot target variable distribution

        Args:
            y: Target array
            class_names: List of class names
            save_path: Optional path to save figure
        """
        logger.info("Plotting target distribution")

        plt.figure(figsize=(8, 8))

        unique, counts = np.unique(y, return_counts=True)
        labels = [class_names[i] if i < len(
            class_names) else str(i) for i in unique]

        colors = plt.cm.Set3(np.linspace(0, 1, len(unique)))
        plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors,
                startangle=90, textprops={'fontsize': 12})

        plt.title('Target Distribution (Match Outcomes)',
                  fontsize=14, fontweight='bold')
        plt.axis('equal')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")

        plt.show()
        plt.close()

    def plot_prediction_confidence(self, predictions: List[Dict],
                                   save_path: Optional[str] = None):
        """
        Plot prediction confidence distribution

        Args:
            predictions: List of prediction results
            save_path: Optional path to save figure
        """
        logger.info("Plotting prediction confidence")

        confidences = [pred['confidence'] for pred in predictions]
        outcomes = [pred['predicted_outcome'] for pred in predictions]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Confidence histogram
        ax1 = axes[0]
        ax1.hist(confidences, bins=20, color='skyblue',
                 edgecolor='navy', alpha=0.7)
        ax1.set_xlabel('Confidence', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Prediction Confidence Distribution',
                      fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        # Add statistics
        mean_conf = np.mean(confidences)
        ax1.axvline(mean_conf, color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {mean_conf:.3f}')
        ax1.legend()

        # Plot 2: Confidence by outcome
        ax2 = axes[1]
        outcome_types = list(set(outcomes))
        confidence_by_outcome = {outcome: [] for outcome in outcome_types}

        for pred in predictions:
            confidence_by_outcome[pred['predicted_outcome']].append(
                pred['confidence'])

        bp = ax2.boxplot([confidence_by_outcome[outcome] for outcome in outcome_types],
                         labels=outcome_types, patch_artist=True)

        for patch in bp['boxes']:
            patch.set_facecolor('lightgreen')

        ax2.set_xlabel('Predicted Outcome', fontsize=12)
        ax2.set_ylabel('Confidence', fontsize=12)
        ax2.set_title('Confidence by Predicted Outcome',
                      fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")

        plt.show()
        plt.close()

    def create_eda_report(self, eda_results: Dict, save_dir: Optional[str] = None):
        """
        Create comprehensive EDA report with multiple plots

        Args:
            eda_results: Dictionary with EDA results
            save_dir: Optional directory to save figures
        """
        logger.info("Creating EDA report")

        if save_dir:
            from pathlib import Path
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Create multi-panel figure
        fig = plt.figure(figsize=(20, 12))

        # Plot 1: Feature importance
        ax1 = plt.subplot(2, 3, 1)
        top_features = eda_results['sorted_features'][:10]
        features, scores = zip(*top_features)
        ax1.barh(range(len(features)), scores,
                 color='skyblue', edgecolor='navy', alpha=0.7)
        ax1.set_yticks(range(len(features)))
        ax1.set_yticklabels(features)
        ax1.set_xlabel('Importance Score')
        ax1.set_title('Top 10 Features', fontweight='bold')
        ax1.invert_yaxis()

        # Plot 2: Feature statistics
        ax2 = plt.subplot(2, 3, 2)
        stats_df = eda_results['feature_statistics']
        if len(stats_df) > 0:
            try:
                sns.heatmap(stats_df.iloc[:10, :3], annot=True, fmt='.2f',
                            cmap='YlOrRd', ax=ax2, cbar_kws={'shrink': 0.8})
            except:
                # Fallback if seaborn has issues
                ax2.imshow(stats_df.iloc[:10, :3].values,
                           cmap='YlOrRd', aspect='auto')
                ax2.set_xticks(range(3))
                ax2.set_xticklabels(stats_df.columns[:3])
                ax2.set_yticks(range(min(10, len(stats_df))))
                ax2.set_yticklabels(stats_df.index[:10])
            ax2.set_title('Feature Statistics (Top 10)', fontweight='bold')

        # Plot 3: Feature distribution sample
        ax3 = plt.subplot(2, 3, 3)
        X_selected = eda_results['X_selected']
        if len(X_selected.columns) > 0:
            sample_feature = X_selected.columns[0]
            ax3.hist(X_selected[sample_feature], bins=30, color='lightgreen',
                     edgecolor='darkgreen', alpha=0.7)
            ax3.set_xlabel(sample_feature)
            ax3.set_ylabel('Frequency')
            ax3.set_title(f'Distribution: {sample_feature}', fontweight='bold')

        plt.suptitle('Exploratory Data Analysis Report',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_dir:
            save_path = Path(save_dir) / 'eda_report.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"EDA report saved to {save_path}")

        plt.show()
        plt.close()


# src/visualization/__init__.py

__all__ = ['Visualizer']
