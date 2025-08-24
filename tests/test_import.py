# test_imports.py - Place this in your project root directory
"""
Quick script to test if imports are working correctly
"""

import sys
import os
from pathlib import Path

print("Python version:", sys.version)
print("Current directory:", os.getcwd())
print("Python path:", sys.path[:3])  # Show first 3 paths

# Method 1: Add to path manually
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

print("\nAfter adding current directory to path:")
print("Python path:", sys.path[:3])

# Now try imports
try:
    from src.models.model_trainer import ModelTrainer
    print("\n✅ Successfully imported ModelTrainer")
except ImportError as e:
    print(f"\n❌ Failed to import ModelTrainer: {e}")

try:
    from src.models.predictor import FootballPredictor
    print("✅ Successfully imported FootballPredictor")
except ImportError as e:
    print(f"❌ Failed to import FootballPredictor: {e}")

try:
    from src.data.preprocessor import DataPreprocessor
    print("✅ Successfully imported DataPreprocessor")
except ImportError as e:
    print(f"❌ Failed to import DataPreprocessor: {e}")

# Check if src directory exists
src_path = Path("src")
if src_path.exists():
    print(f"\n✅ 'src' directory found at: {src_path.resolve()}")
    print("Contents of src:")
    for item in src_path.iterdir():
        print(f"  - {item.name}")
else:
    print(f"\n❌ 'src' directory not found in {Path.cwd()}")

# Check for __init__.py files
init_files = [
    "src/__init__.py",
    "src/models/__init__.py",
    "src/data/__init__.py",
    "src/analysis/__init__.py",
    "src/visualization/__init__.py",
    "src/utils/__init__.py"
]

print("\nChecking for __init__.py files:")
for init_file in init_files:
    if Path(init_file).exists():
        print(f"  ✅ {init_file} exists")
    else:
        print(f"  ❌ {init_file} missing")

print("\n" + "="*50)
print("RECOMMENDED FIXES:")
print("="*50)
print("""
1. Make sure you're in the project root directory:
   cd /path/to/football-match-predictor

2. Install the package in development mode:
   pip install -e .

3. Or run scripts from the project root:
   python scripts/train_model.py --data data/raw/foot_data.csv

4. Create missing __init__.py files if needed:
   touch src/__init__.py
   touch src/models/__init__.py
   touch src/data/__init__.py
   touch src/analysis/__init__.py
   touch src/visualization/__init__.py
   touch src/utils/__init__.py
""")
