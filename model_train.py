import joblib
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
git add model.pkl
git commit -m "Add pre-trained model"
git push
def train_model():
    print("Loading MNIST dataset... (this may take a minute)")
    # fetch_openml returns the dataset as a bunch object
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='liac-arff')
    
    # Scale pixels to [0, 1]
    X = X / 255.0
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training MLP Classifier...")
    # Using a simple Multi-layer Perceptron
    mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=20, alpha=1e-4,
                        solver='sgd', verbose=10, random_state=1,
                        learning_rate_init=.1)
    
    mlp.fit(X_train, y_train)
    
    # Evaluate
    y_pred = mlp.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Training Complete. Accuracy: {acc * 100:.2f}%")
    
    # Save the model
    joblib.dump(mlp, 'model.pkl')
    print("Model saved as model.pkl")

if __name__ == "__main__":
    train_model()
