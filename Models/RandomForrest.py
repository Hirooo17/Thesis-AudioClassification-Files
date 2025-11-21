
# Cell: Modified Random Forest - Feature Selection Only (No PCA)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import pickle
from datetime import datetime
import os
import numpy as np

class RandomForestFullFeatures():
    """Random Forest that keeps all 16,384 features - NO dimensionality reduction"""
    
    def __init__(self, n_estimators=100, max_depth=None, random_state=42, **kwargs):
        # Initialize the Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )
        self.fitted_preprocessing = False
        self.is_fitted = False
        self.n_features_in_ = None  # Track expected feature count
    
    def _extract_from_dataset(self, dataset, dataset_name="data"):
        """
        Helper method to properly extract data from TensorFlow dataset without cache warnings
        
        Args:
            dataset: TensorFlow dataset
            dataset_name: Name for logging
            
        Returns:
            X, y: Numpy arrays
        """
        X = []
        y = []
        
        print(f"   Extracting {dataset_name}...")
        
        # Use unbatch() to avoid cache warnings
        try:
            # Check if dataset is already unbatched or batched
            unbatched = dataset.unbatch() if hasattr(dataset, 'unbatch') else dataset
            
            batch_count = 0
            for features, labels in unbatched:
                # Convert to numpy and flatten
                if hasattr(features, 'numpy'):
                    features = features.numpy()
                if hasattr(labels, 'numpy'):
                    labels = labels.numpy()
                
                # Flatten if needed
                if len(features.shape) > 1:
                    features = features.flatten()
                
                X.append(features)
                y.append(labels)
                
                batch_count += 1
                if batch_count % 500 == 0:
                    print(f"      Processed {batch_count} samples...")
        
        except Exception as e:
            print(f"   Warning during extraction: {e}")
            if len(X) == 0:
                raise
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"   ✅ Extracted {len(X)} samples, shape: {X.shape}")
        return X, y
        
    def compile(self, **kwargs):
        """Enhanced compile method"""
        print(f"✅ Using FULL feature set: 16,384 features (128x128)")
        print(f"✅ NO dimensionality reduction applied")
        
    def fit(self, train_data, epochs=None, validation_data=None):
        """Fit method WITHOUT dimensionality reduction"""
        print("\n" + "="*70)
        print("🚀 TRAINING RANDOM FOREST")
        print("="*70)
        print("📦 Extracting training data with FULL features (16,384)...")
        
        # Use helper method to extract data properly
        X_train, y_train = self._extract_from_dataset(train_data, "training data")
        
        # Store expected feature count for validation during prediction
        self.n_features_in_ = X_train.shape[1]
        
        print(f"\n📊 Dataset Information:")
        print(f"   • Training samples: {X_train.shape[0]}")
        print(f"   • Features per sample: {X_train.shape[1]:,}")
        print(f"   • Memory usage: ~{X_train.nbytes / 1024**2:.1f} MB")
        print(f"   ✅ Using ALL features (no reduction)")
        
        self.fitted_preprocessing = True
        
        # Train the Random Forest on FULL feature set
        print(f"\n🌳 Training Random Forest...")
        print(f"   • Trees: {self.model.n_estimators}")
        print(f"   • Max depth: {self.model.max_depth if self.model.max_depth else 'Unlimited'}")
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
        
        # Calculate training metrics on ALL samples for accurate overfitting detection
        print("\n📊 Calculating training metrics on ALL training samples...")
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        train_precision = precision_score(y_train, train_pred)
        train_recall = recall_score(y_train, train_pred)
        
        # Process validation data if provided
        val_accuracy, val_precision, val_recall = None, None, None
        if validation_data is not None:
            print("\n� Extracting validation data...")
            X_val, y_val = self._extract_from_dataset(validation_data, "validation data")
            
            print("   Evaluating on validation set...")
            val_pred = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)
            val_precision = precision_score(y_val, val_pred)
            val_recall = recall_score(y_val, val_pred)
        
        # Create history dictionary
        history = {
            'binary_accuracy': [train_accuracy],
            'precision': [train_precision],
            'recall': [train_recall],
            'val_binary_accuracy': [val_accuracy] if val_accuracy else [None],
            'val_precision': [val_precision] if val_precision else [None],
            'val_recall': [val_recall] if val_recall else [None]
        }
        
        # Print results with better formatting
        print(f"\n{'='*70}")
        print("📊 TRAINING RESULTS")
        print(f"{'='*70}")
        print(f"   📈 Training Metrics (on ALL {len(y_train)} samples):")
        print(f"      • Accuracy:  {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"      • Precision: {train_precision:.4f}")
        print(f"      • Recall:    {train_recall:.4f}")
        
        if validation_data is not None:
            print(f"\n   � Validation Metrics:")
            print(f"      • Accuracy:  {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
            print(f"      • Precision: {val_precision:.4f}")
            print(f"      • Recall:    {val_recall:.4f}")
            
            # Overfitting detection
            accuracy_gap = (train_accuracy - val_accuracy) * 100
            if accuracy_gap > 10:
                print(f"\n   ⚠️  High overfitting detected!")
                print(f"      Gap: {accuracy_gap:.2f}% (train - val)")
                print(f"      💡 Consider: Increase min_samples_split or max_depth")
            elif accuracy_gap > 5:
                print(f"\n   ⚡ Moderate overfitting: {accuracy_gap:.2f}% gap")
            else:
                print(f"\n   ✅ Good generalization: {accuracy_gap:.2f}% gap")
        
        print(f"{'='*70}\n")
        
        return type('History', (), {'history': history})()
    
    def _prepare_features(self, X_test):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions!")
        if hasattr(X_test, 'numpy'):
            X_test = X_test.numpy()
        if len(X_test.shape) > 2:
            X_test = X_test.reshape(X_test.shape[0], -1)
        if X_test.shape[1] != self.n_features_in_:
            raise ValueError(
                f"❌ FEATURE MISMATCH ERROR!\n"
                f"   Model was trained on {self.n_features_in_} features\n"
                f"   But prediction data has {X_test.shape[1]} features\n\n"
                f"💡 SOLUTION:\n"
                f"   • If trained on SPECTROGRAM data (~16k features):\n"
                f"     → Use spectrogram dataset for prediction\n"
                f"   • If trained on MFCC data (40 features):\n"
                f"     → Use MFCC dataset (mfcc_*) for prediction\n\n"
                f"   🔍 Training feature type: "
                f"{'SPECTROGRAM' if self.n_features_in_ > 1000 else 'MFCC'}\n"
                f"   🔍 Prediction feature type: "
                f"{'SPECTROGRAM' if X_test.shape[1] > 1000 else 'MFCC'}\n"
            )
        return X_test

    def predict_proba(self, X_test):
        """Return class probabilities after validating feature dimensions."""
        X_test = self._prepare_features(X_test)
        pred_proba = self.model.predict_proba(X_test)
        if pred_proba.ndim == 1:
            pred_proba = pred_proba.reshape(-1, 1)
        if pred_proba.shape[1] == 1:
            real_col = pred_proba[:, 0]
            fake_col = 1 - real_col
            pred_proba = np.stack([fake_col, real_col], axis=1)
        return pred_proba

    def predict(self, X_test):
        """Return class labels derived from probability estimates."""
        pred_proba = self.predict_proba(X_test)
        real_probs = pred_proba[:, 1]
        return (real_probs >= 0.5).astype(int)

    @property
    def feature_importances_(self):
        return getattr(self.model, 'feature_importances_', None)
    
    def summary(self):
        """Summary showing full features"""
        print("Random Forest with FULL Features Summary:")
        print("="*60)
        print(f"Model Type: Random Forest (No Dimensionality Reduction)")
        print(f"Feature Count: 16,384 (128x128 mel-spectrogram)")
        print(f"Number of Estimators: {self.model.n_estimators}")
        print(f"Max Depth: {self.model.max_depth}")
        
        if self.is_fitted:
            print(f"Actual Feature Count: {self.model.n_features_in_:,}")
            print(f"Number of Classes: {self.model.n_classes_}")
        print("="*60)
    
    # === ADD THESE METHODS FOR SAVING/LOADING ===
    
    def save_model(self, filepath=None):
        """Save the trained model to disk (auto-deletes old Random Forest models)"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving!")
        
        # Delete old Random Forest models first
        print("\n🗑️  Cleaning up old Random Forest models...")
        deleted_count = 0
        for file in os.listdir('.'):
            if file.endswith(('.joblib', '.pkl')) and ('rf' in file.lower() or 'random' in file.lower() or 'forest' in file.lower()):
                try:
                    os.remove(file)
                    print(f"   ✅ Deleted: {file}")
                    deleted_count += 1
                except Exception as e:
                    print(f"   ⚠️  Could not delete {file}: {e}")
        
        if deleted_count > 0:
            print(f"   🧹 Cleaned up {deleted_count} old model(s)")
        else:
            print(f"   ℹ️  No old models found")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"rf_model_full_features_{timestamp}.joblib"
        
        # Save using joblib (better for large numpy arrays)
        print(f"\n💾 Saving new model...")
        joblib.dump(self, filepath)
        print(f"✅ Model saved to: {filepath}")
        print(f"💾 File size: {os.path.getsize(filepath) / 1024**2:.2f} MB")
        return filepath
    
    def save_model_pickle(self, filepath=None):
        """Alternative: Save using pickle"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving!")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"rf_model_full_features_{timestamp}.pkl"
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"✅ Model saved to: {filepath}")
        print(f"💾 File size: {os.path.getsize(filepath) / 1024**2:.2f} MB")
        return filepath

# Helper function to load models
def load_rf_model(filepath):
    """Load a saved Random Forest model"""
    try:
        # Try joblib first
        model = joblib.load(filepath)
        print(f"✅ Model loaded from: {filepath}")
        print(f"📊 Model type: {type(model).__name__}")
        print(f"🔧 Model fitted: {model.is_fitted}")
        return model
    except Exception as e:
        print(f"❌ Error loading with joblib: {e}")
        # Fallback to pickle
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            print(f"✅ Model loaded from: {filepath}")
            print(f"📊 Model type: {type(model).__name__}")
            print(f"🔧 Model fitted: {model.is_fitted}")
            return model
        except Exception as e2:
            print(f"❌ Error loading with pickle: {e2}")
            raise ValueError(f"Failed to load model from {filepath}")