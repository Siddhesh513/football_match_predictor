# scripts/train_model.py - UPDATED VERSION
"""
Training script for Football Match Predictor
"""
from src.models.predictor import FootballPredictor
import json
from src.visualization.plots import Visualizer
from src.utils.logger import get_logger
from src.utils.config import Config
from src.models.model_trainer import ModelTrainer
import sys
import os
import argparse
from pathlib import Path

# Fix import issue - Add parent directory to Python path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Now imports will work


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Train Football Match Predictor')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to training data CSV file')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output', type=str, default='data/models/model.pkl',
                        help='Path to save trained model')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization plots')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')

    args = parser.parse_args()

    # Setup logger
    logger = get_logger('train_model', log_file='logs/training.log',
                        level=args.log_level)

    try:
        # Load configuration
        logger.info("Loading configuration")
        config = Config(args.config)

        # Initialize trainer
        logger.info("Initializing model trainer")
        trainer = ModelTrainer(config.model_config)

        # Train model
        logger.info(f"Starting training with data from {args.data}")
        results = trainer.train(args.data, save_path=args.output)

        # Print summary
        print("\n" + "="*60)
        print(trainer.get_training_summary())
        print("="*60)

        # Generate visualizations if requested
        if args.visualize:
            logger.info("Generating visualizations")
            visualizer = Visualizer()

            # Create output directory for plots
            plot_dir = Path('outputs/plots')
            plot_dir.mkdir(parents=True, exist_ok=True)

            # Plot model comparison
            visualizer.plot_model_comparison(
                results['cv_results'],
                save_path=plot_dir / 'model_comparison.png'
            )

            # Plot feature importance
            visualizer.plot_feature_importance(
                results['feature_importance'],
                save_path=plot_dir / 'feature_importance.png'
            )

            logger.info(f"Plots saved to {plot_dir}")

        logger.info(f"Model saved to {args.output}")
        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()

# ============================================
# scripts/predict.py - UPDATED VERSION
# ============================================
"""
Prediction script for Football Match Predictor
"""

# Fix import issue - Add parent directory to Python path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))


def interactive_mode(predictor, logger):
    """Run interactive prediction mode"""
    logger.info("Starting interactive prediction mode")

    print("\n" + "="*60)
    print("FOOTBALL MATCH PREDICTOR - INTERACTIVE MODE")
    print("="*60)
    print("\nEnter 'quit' to exit")

    while True:
        print("\n" + "-"*40)
        home_team = input("Enter HOME team: ").strip()

        if home_team.lower() == 'quit':
            break

        away_team = input("Enter AWAY team: ").strip()

        if away_team.lower() == 'quit':
            break

        try:
            # Make prediction
            result = predictor.predict(home_team, away_team)

            # Display results
            print("\n" + "="*40)
            print("PREDICTION RESULTS")
            print("="*40)
            print(f"Match: {result['home_team']} vs {result['away_team']}")
            print(f"Model: {result['model_used']}")
            print(f"Predicted Outcome: {result['predicted_outcome']}")
            print(f"Confidence: {result['confidence']:.1%}")
            print("\nProbabilities:")
            print(
                f"  Home Win: {result['probabilities'].get('home_win', 0):.1%}")
            print(f"  Draw: {result['probabilities'].get('draw', 0):.1%}")
            print(
                f"  Away Win: {result['probabilities'].get('away_win', 0):.1%}")

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            print(f"Error: {e}")


def batch_mode(predictor, batch_file, output_file, logger):
    """Run batch prediction mode"""
    logger.info(f"Starting batch prediction mode with {batch_file}")

    # Load batch file
    with open(batch_file, 'r') as f:
        if batch_file.endswith('.json'):
            matches = json.load(f)
        else:
            # Assume CSV format: home_team,away_team
            matches = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split(',')
                    if len(parts) >= 2:
                        matches.append({
                            'home_team': parts[0].strip(),
                            'away_team': parts[1].strip()
                        })

    logger.info(f"Loaded {len(matches)} matches for prediction")

    # Make predictions
    results = []
    for match in matches:
        try:
            result = predictor.predict(match['home_team'], match['away_team'])
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to predict {match}: {e}")
            results.append({
                'home_team': match['home_team'],
                'away_team': match['away_team'],
                'error': str(e)
            })

    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")

    # Display summary
    print("\n" + "="*60)
    print("BATCH PREDICTION SUMMARY")
    print("="*60)
    print(f"Total matches: {len(matches)}")
    print(
        f"Successful predictions: {sum(1 for r in results if 'error' not in r)}")
    print(f"Failed predictions: {sum(1 for r in results if 'error' in r)}")

    if not output_file:
        print("\nResults:")
        for result in results:
            if 'error' not in result:
                print(f"{result['home_team']:15} vs {result['away_team']:15} => "
                      f"{result['predicted_outcome']} ({result['confidence']:.1%})")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Make predictions with Football Match Predictor')
    parser.add_argument('--model', type=str, default='data/models/model.pkl',
                        help='Path to trained model')
    parser.add_argument('--home', type=str, help='Home team name')
    parser.add_argument('--away', type=str, help='Away team name')
    parser.add_argument('--batch', type=str,
                        help='Path to batch prediction file')
    parser.add_argument('--output', type=str, help='Path to save predictions')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')

    args = parser.parse_args()

    # Setup logger
    logger = get_logger('predict', log_file='logs/predictions.log',
                        level=args.log_level)

    try:
        # Load model
        logger.info(f"Loading model from {args.model}")
        predictor = FootballPredictor(args.model)

        # Determine mode
        if args.interactive:
            interactive_mode(predictor, logger)
        elif args.batch:
            batch_mode(predictor, args.batch, args.output, logger)
        elif args.home and args.away:
            # Single prediction
            result = predictor.predict(args.home, args.away)

            # Display result
            print("\n" + "="*40)
            print("PREDICTION RESULT")
            print("="*40)
            print(f"Match: {result['home_team']} vs {result['away_team']}")
            print(f"Predicted Outcome: {result['predicted_outcome']}")
            print(f"Confidence: {result['confidence']:.1%}")
            print("\nProbabilities:")
            for outcome, prob in result['probabilities'].items():
                print(f"  {outcome}: {prob:.1%}")

            # Save if requested
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                logger.info(f"Result saved to {args.output}")
        else:
            print("Please specify teams (--home and --away), "
                  "batch file (--batch), or interactive mode (--interactive)")
            parser.print_help()

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


if __name__ == "__main__":
    main()
