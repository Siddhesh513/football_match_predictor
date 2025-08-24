"""
Web Interface for Football Match Predictor
Run: python web_app.py
Then open: http://localhost:5000
"""

from src.utils.logger import get_logger
from src.models.predictor import FootballPredictor
from flask import Flask, render_template, request, jsonify, send_from_directory
import sys
from pathlib import Path
import os
import json
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Setup logger
logger = get_logger('web_app', log_file='logs/web_app.log')

# Global predictor instance
predictor = None
MODEL_PATH = 'data/models/model.pkl'

# Available teams (you can expand this list based on your data)
TEAMS = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham",
    "Leicester", "Liverpool", "Luton", "Manchester City", "Manchester United",
    "Newcastle", "Norwich", "Nottingham Forest", "Sheffield United", "Tottenham",
    "West Ham", "Wolves", "Southampton", "Leeds", "Watford"
]


def load_model():
    """Load the prediction model"""
    global predictor
    try:
        predictor = FootballPredictor(MODEL_PATH)
        logger.info(f"Model loaded successfully: {predictor.model_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', teams=sorted(TEAMS))


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        data = request.json
        home_team = data.get('home_team')
        away_team = data.get('away_team')

        if not home_team or not away_team:
            return jsonify({'error': 'Please select both teams'}), 400

        if home_team == away_team:
            return jsonify({'error': 'Teams must be different'}), 400

        # Make prediction
        result = predictor.predict(home_team, away_team)

        # Add timestamp
        result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Format probabilities for display
        result['formatted_probabilities'] = {
            'home': f"{result['probabilities'].get('home_win', 0):.1%}",
            'draw': f"{result['probabilities'].get('draw', 0):.1%}",
            'away': f"{result['probabilities'].get('away_win', 0):.1%}"
        }

        # Add interpretation
        confidence = result['confidence']
        if confidence < 0.45:
            result['interpretation'] = "Very uncertain match - could go any way!"
        elif confidence < 0.55:
            result['interpretation'] = "Fairly competitive match with slight edge"
        elif confidence < 0.65:
            result['interpretation'] = "Clear favorite but upset is possible"
        else:
            result['interpretation'] = "Strong favorite expected to win"

        logger.info(
            f"Prediction: {home_team} vs {away_team} => {result['predicted_outcome']}")

        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Handle batch predictions"""
    try:
        data = request.json
        matches = data.get('matches', [])

        results = []
        for match in matches:
            try:
                result = predictor.predict(match['home'], match['away'])
                result['match_id'] = f"{match['home']}_vs_{match['away']}"
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting {match}: {e}")
                results.append({
                    'match_id': f"{match['home']}_vs_{match['away']}",
                    'error': str(e)
                })

        return jsonify({'predictions': results})

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/model_info')
def model_info():
    """Get model information"""
    if predictor:
        info = {
            'model_name': predictor.model_name,
            'num_features': len(predictor.feature_names),
            'features': predictor.feature_names[:10],  # First 10 features
            'model_loaded': True
        }

        if hasattr(predictor, 'model_metadata') and predictor.model_metadata:
            info['accuracy'] = f"{predictor.model_metadata.get('accuracy', 0):.1%}"
            info['training_samples'] = predictor.model_metadata.get(
                'training_samples', 'N/A')

        return jsonify(info)
    else:
        return jsonify({'model_loaded': False, 'error': 'Model not loaded'}), 503


@app.route('/api/teams')
def get_teams():
    """Get list of available teams"""
    return jsonify({'teams': sorted(TEAMS)})


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None,
        'timestamp': datetime.now().isoformat()
    })

# Error handlers


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("FOOTBALL MATCH PREDICTOR - WEB INTERFACE")
    print("=" * 60)

    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)

    # Load model
    if load_model():
        print(f"✅ Model loaded: {predictor.model_name}")
        print(f"📊 Features: {len(predictor.feature_names)}")
    else:
        print("❌ Failed to load model")
        print(f"   Please ensure model exists at: {MODEL_PATH}")
        sys.exit(1)

    print("\n🚀 Starting web server...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("-" * 60)

    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
