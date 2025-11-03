# Cell: Import KNN and related libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import numpy as np
import joblib
import os


# Cell: Full-Featured KNN Model Setup
class KNNAudioClassifier:
    def __init__(self, n_neighbors=15, weights='uniform', algorithm='auto', random_state=42):
        """
        Full-featured KNN classifier for audio classification (optimized to reduce overfitting)
        
        Args:
            n_neighbors: Number of neighbors to use (default: 15 - higher to reduce overfitting)
            weights: Weight function ('uniform' or 'distance') - uniform reduces overfitting
            algorithm: Algorithm to compute nearest neighbors ('auto', 'ball_tree', 'kd_tree', 'brute')
            random_state: Random state for reproducibility
        """
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            n_jobs=-1,  # Use all available cores for maximum performance
            metric='minkowski',  # Standard metric
            p=2  # Euclidean distance
        )
        self.is_fitted = False
        self.random_state = random_state
        
        # Store parameters for tuning
        self.n_neighbors = n_neighbors
        self.weights = weights
        
    def compile(self, **kwargs):
        """
        Compile method (for consistency with other models)
        """
        print("="*70)
        print("🔧 KNN MODEL CONFIGURATION (Optimized for Generalization)")
        print("="*70)
        print(f"👥 Number of Neighbors: {self.model.n_neighbors}")
        print(f"⚖️  Weights: {self.model.weights}")
        print(f"📐 Algorithm: {self.model.algorithm}")
        print(f"🔢 Metric: {self.model.metric} (p={self.model.p})")
        print(f"⚡ Parallel Jobs: {self.model.n_jobs} (using all cores)")
        print("\n💡 Tips to reduce overfitting:")
        print("   • Use n_neighbors=15-25 (higher = less overfitting)")
        print("   • Use weights='uniform' instead of 'distance'")
        print("   • Consider feature scaling/normalization")
        print("="*70)
        
    def fit(self, train_data, epochs=None, validation_data=None, normalize=True):
        """
        Full-featured fit method using all available data with normalization
        
        Args:
            train_data: Training dataset
            epochs: Not used for KNN (kept for API consistency)
            validation_data: Optional validation dataset
            normalize: Whether to normalize features (default: True, recommended for KNN)
        """
        print("\n" + "="*70)
        print("🚀 TRAINING KNN CLASSIFIER")
        print("="*70)
        print("📦 Extracting training data...")
        
        # Extract features and labels from all batches
        X_train = []
        y_train = []
        
        batch_count = 0
        for batch_x, batch_y in train_data:
            batch_x_flat = batch_x.numpy().reshape(batch_x.shape[0], -1)
            X_train.extend(batch_x_flat)
            y_train.extend(batch_y.numpy())
            batch_count += 1
            
            if batch_count % 10 == 0:
                print(f"   Processed {batch_count} batches, total samples: {len(X_train)}")
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        print(f"\n📊 Dataset Information:")
        print(f"   • Training samples: {X_train.shape[0]}")
        print(f"   • Features per sample: {X_train.shape[1]}")
        print(f"   • Memory usage: ~{X_train.nbytes / 1024**2:.1f} MB")
        
        # Normalize features to prevent distance bias (IMPORTANT for KNN!)
        if normalize:
            print(f"\n⚖️  Normalizing features (StandardScaler)...")
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            print(f"   ✅ Features normalized (mean=0, std=1)")
            self.normalized = True
        else:
            self.scaler = None
            self.normalized = False
            print(f"\n⚠️  WARNING: Training without normalization may cause overfitting!")
        
        # Fit the model with all data
        print(f"\n🔄 Training KNN with {self.n_neighbors} neighbors...")
        print("="*70)
        
        import sys
        from datetime import datetime
        start_time = datetime.now()
        
        sys.stdout.write("   Training in progress... ")
        sys.stdout.flush()
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"✅ Done in {elapsed:.2f} seconds!")
        print("="*70)
        
        # Calculate training metrics on subset to detect overfitting
        print("\n📊 Calculating training metrics (on subset to save time)...")
        
        # Use subset for training metrics to save time
        subset_size = min(1000, len(X_train))
        indices = np.random.choice(len(X_train), subset_size, replace=False)
        X_train_subset = X_train[indices]
        y_train_subset = y_train[indices]
        
        train_pred = self.model.predict(X_train_subset)
        train_accuracy = accuracy_score(y_train_subset, train_pred)
        train_precision = precision_score(y_train_subset, train_pred, zero_division=0)
        train_recall = recall_score(y_train_subset, train_pred, zero_division=0)
        
        # Validation metrics
        val_accuracy, val_precision, val_recall = None, None, None
        if validation_data is not None:
            print("📊 Evaluating on validation data...")
            X_val = []
            y_val = []
            
            for batch_x, batch_y in validation_data:
                batch_x_flat = batch_x.numpy().reshape(batch_x.shape[0], -1)
                X_val.extend(batch_x_flat)
                y_val.extend(batch_y.numpy())
            
            X_val = np.array(X_val)
            y_val = np.array(y_val)
            
            # Apply same normalization to validation data
            if self.normalized and self.scaler is not None:
                X_val = self.scaler.transform(X_val)
            
            val_pred = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)
            val_precision = precision_score(y_val, val_pred, zero_division=0)
            val_recall = recall_score(y_val, val_pred, zero_division=0)
            
            # Check for overfitting
            accuracy_gap = train_accuracy - val_accuracy
            if accuracy_gap > 0.10:  # More than 10% gap
                print(f"\n⚠️  WARNING: Potential overfitting detected!")
                print(f"   Training accuracy: {train_accuracy:.4f}")
                print(f"   Validation accuracy: {val_accuracy:.4f}")
                print(f"   Gap: {accuracy_gap:.4f} ({accuracy_gap*100:.1f}%)")
                print(f"\n💡 Suggestions to reduce overfitting:")
                print(f"   • Increase n_neighbors (current: {self.n_neighbors})")
                print(f"   • Try n_neighbors=20-30 for better generalization")
                print(f"   • Ensure normalization is enabled (current: {self.normalized})")
        
        # Create history dictionary
        history = {
            'binary_accuracy': [train_accuracy],
            'precision': [train_precision],
            'recall': [train_recall],
            'val_binary_accuracy': [val_accuracy] if val_accuracy else [None],
            'val_precision': [val_precision] if val_precision else [None],
            'val_recall': [val_recall] if val_recall else [None]
        }
        
        # Print results
        print(f"\n" + "="*70)
        print("📈 TRAINING RESULTS")
        print("="*70)
        print(f"Training Metrics (on {subset_size} sample subset):")
        print(f"   • Accuracy:  {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"   • Precision: {train_precision:.4f}")
        print(f"   • Recall:    {train_recall:.4f}")
        
        if validation_data is not None and val_accuracy is not None:
            print(f"\nValidation Metrics (full validation set):")
            print(f"   • Accuracy:  {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
            print(f"   • Precision: {val_precision:.4f}")
            print(f"   • Recall:    {val_recall:.4f}")
            
            # Show generalization performance
            if accuracy_gap <= 0.05:
                print(f"\n✅ Good generalization! (gap: {accuracy_gap*100:.1f}%)")
            elif accuracy_gap <= 0.10:
                print(f"\n⚠️  Moderate overfitting (gap: {accuracy_gap*100:.1f}%)")
            else:
                print(f"\n❌ High overfitting! (gap: {accuracy_gap*100:.1f}%)")
        
        print("="*70)
        
        return type('History', (), {'history': history})()
    
    def predict(self, X_test):
        """
        Full-featured prediction method with normalization
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions!")
        
        # If X_test is from a dataset batch, flatten it
        if len(X_test.shape) > 2:
            X_test = X_test.reshape(X_test.shape[0], -1)
        
        # Apply normalization if it was used during training
        if self.normalized and self.scaler is not None:
            X_test = self.scaler.transform(X_test)
        
        # Get prediction probabilities
        pred_proba = self.model.predict_proba(X_test)
        # Return probability of positive class (class 1)
        return pred_proba[:, 1].reshape(-1, 1)
    
    def summary(self):
        """
        Print model summary with overfitting prevention info
        """
        print("="*70)
        print("📊 KNN MODEL SUMMARY")
        print("="*70)
        print(f"Model Type: K-Nearest Neighbors Classifier")
        print(f"\n🔧 Configuration:")
        print(f"   • Number of Neighbors: {self.model.n_neighbors}")
        print(f"   • Weights: {self.model.weights}")
        print(f"   • Algorithm: {self.model.algorithm}")
        print(f"   • Metric: {self.model.metric} (p={self.model.p})")
        print(f"   • Parallel Jobs: {self.model.n_jobs}")
        
        if self.is_fitted:
            print(f"\n📈 Training Info:")
            print(f"   • Training Samples: {self.model._fit_X.shape[0]:,}")
            print(f"   • Number of Features: {self.model.n_features_in_}")
            print(f"   • Feature Normalization: {'✅ Enabled' if self.normalized else '❌ Disabled'}")
            
            # Calculate neighbors to samples ratio
            ratio = self.model.n_neighbors / self.model._fit_X.shape[0]
            if ratio < 0.001:
                status = "⚠️  Very Low (High overfitting risk)"
            elif ratio < 0.01:
                status = "✅ Good"
            else:
                status = "✅ Excellent (Low overfitting risk)"
            
            print(f"   • Neighbors/Samples Ratio: {ratio:.4f} ({status})")
            
            print(f"\n💡 Overfitting Prevention:")
            print(f"   • Higher n_neighbors = Less overfitting")
            print(f"   • Recommended n_neighbors: 15-30")
            print(f"   • Feature normalization: CRITICAL for KNN")
        else:
            print(f"\n⚠️  Model not yet trained")
        
        print("="*70)
    
    def save_model(self, filename=None):
        """
        Save the trained KNN model to disk
        
        Args:
            filename: Name of the file to save (default: knn_model.joblib)
            
        Returns:
            filepath: Full path to the saved model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving!")
        
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"knn_model_trained_{timestamp}.joblib"
        
        # Ensure .joblib extension
        if not filename.endswith('.joblib'):
            filename += '.joblib'
        
        # Get current directory
        current_dir = os.getcwd()
        filepath = os.path.join(current_dir, filename)
        
        # Save model
        joblib.dump(self, filepath)
        print(f"✓ KNN model saved to: {filepath}")
        
        return filepath


def load_knn_model(filepath):
    """
    Load a trained KNN model from disk
    
    Args:
        filepath: Path to the saved model file
        
    Returns:
        model: Loaded KNNAudioClassifier instance
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    model = joblib.load(filepath)
    
    if not isinstance(model, KNNAudioClassifier):
        raise ValueError("Loaded object is not a KNNAudioClassifier instance!")
    
    print(f"✓ KNN model loaded from: {filepath}")
    print(f"  - Training samples: {model.model._fit_X.shape[0]}")
    print(f"  - Number of features: {model.model.n_features_in_}")
    print(f"  - Number of neighbors: {model.model.n_neighbors}")
    
    return model