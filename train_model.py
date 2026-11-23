"""Optional: create iris.pkl using the classic Iris dataset (requires scikit-learn)."""
from pathlib import Path
import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(iris.data, iris.target)
output = Path(__file__).resolve().parent / 'iris.pkl'
with output.open('wb') as f:
    pickle.dump(model, f)
print(f'Trained example model saved to {output}')
