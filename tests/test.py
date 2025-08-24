# test.py
from src.data.feature_engineering import FeatureEngineer
from src.data.preprocessor import DataPreprocessor
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))


print("✅ All imports working!")
