import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model():
    df = pd.read_csv('iris.csv')
    print(f"Training with dataset shape: {df.shape}")
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'num_samples': len(df),
        'num_features': len(X.columns),
        'num_classes': len(y.unique())
    }
    
    with open('metrics.csv', 'w') as f:
        f.write('metric,value\n')
        for key, value in metrics.items():
            f.write(f'{key},{value}\n')
    
    print(f"Model trained successfully!")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Dataset size: {len(df)} samples")
    print(f"Features: {len(X.columns)}")
    
    return accuracy

if __name__ == "__main__":
    train_model()
