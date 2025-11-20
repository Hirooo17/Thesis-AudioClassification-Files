# Cell: Import XGBoost and related libraries
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import joblib
import os


# Cell: Full-Featured XGBoost Model Setup
class XGBoostAudioClassifier:
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42):
        """
        Full-featured XGBoost classifier for audio classification
        
        Args:
            n_estimators: Number of boosting rounds (default: 100)
            max_depth: Maximum depth of trees (default: 6)
            learning_rate: Learning rate (default: 0.1)
            random_state: Random state for reproducibility
        """
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=1,  # CHANGED: Use single thread to reduce memory
            eval_metric='logloss',
            tree_method='hist',  # Memory-efficient histogram method
            subsample=0.8,  # Use subset of training data for each tree
            colsample_bytree=0.8,  # Use subset of features for each tree
            max_bin=128,  # ADDED: Reduce histogram bins to save memory (default=256)
            grow_policy='depthwise',  # ADDED: Memory-efficient growth policy
            verbosity=2  # Show detailed training progress (2 = info level)
        )
        self.is_fitted = False
        self.n_estimators = n_estimators  # Store for progress tracking
        self.n_features_in_ = None  # Track expected feature count
        
    def compile(self, **kwargs):
        """
        Compile method (for consistency with other models)
        """
        print("Full-Featured XGBoost model 'compiled' successfully!")
        print(f"Model parameters: n_estimators={self.model.n_estimators}, max_depth={self.model.max_depth}")
        print(f"Learning rate: {self.model.learning_rate}, n_jobs={self.model.n_jobs}")
        print(f"Tree method: {self.model.tree_method}, Max bins: {self.model.max_bin}")
        print(f"Subsample: {self.model.subsample}, Growth policy: {self.model.grow_policy}")
        print("⚠️  Memory-optimized settings enabled for large datasets")
        
    def fit(self, train_data, epochs=None, validation_data=None):
        """
        Full-featured fit method using all available data
        """
        print("Preparing training data...")
        
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
                print(f"Processed {batch_count} batches, total samples: {len(X_train)}")
        
        X_train = np.array(X_train)
        y_train = np.array(y_train, dtype=np.int32)
        
        # Store expected feature count for validation during prediction
        self.n_features_in_ = X_train.shape[1]
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Memory usage: ~{X_train.nbytes / 1024**2:.1f} MB")
        print(f"Training with {len(X_train)} samples...")
        
        # Prepare validation data for early stopping if provided
        eval_set = None
        if validation_data is not None:
            print("Preparing validation data for early stopping...")
            X_val = []
            y_val = []
            
            for batch_x, batch_y in validation_data:
                batch_x_flat = batch_x.numpy().reshape(batch_x.shape[0], -1)
                X_val.extend(batch_x_flat)
                y_val.extend(batch_y.numpy())
            
            X_val = np.array(X_val)
            y_val = np.array(y_val, dtype=np.int32)
            
            print(f"Validation data shape: {X_val.shape}")
            eval_set = [(X_val, y_val)]
        
        # Fit the model with all data and progress tracking
        print("\n" + "="*70)
        print("🚀 TRAINING XGBOOST CLASSIFIER")
        print("="*70)
        print(f"📊 Training samples: {len(X_train)}")
        print(f"🎯 Features per sample: {X_train.shape[1]}")
        print(f"🌳 Number of trees: {self.n_estimators}")
        print(f"📈 Max depth: {self.model.max_depth}")
        print(f"⚡ Learning rate: {self.model.learning_rate}")
        print("="*70)
        
        from datetime import datetime
        start_time = datetime.now()
        
        if eval_set:
            print("🔍 Training with validation set for early stopping...")
            print("=" * 70)
            
            # Use set_params for early_stopping_rounds in newer XGBoost versions
            self.model.set_params(early_stopping_rounds=10, verbosity=1)
            
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=True  # Show training progress
            )
            
            print(f"\n✅ Training completed with early stopping!")
        else:
            print("📊 Training without validation set...")
            print("=" * 70)
            
            # Set verbosity to show progress
            self.model.set_params(verbosity=1)
            
            self.model.fit(
                X_train, y_train,
                verbose=True  # Show training progress
            )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n✅ Training completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        print("=" * 70)
        
        self.is_fitted = True
        print("\n" + "="*70)
        print("✅ XGBOOST TRAINING SUCCESSFUL!")
        print("="*70)
        
        # Calculate training metrics on ALL samples for accurate overfitting detection
        print(f"\n📊 Calculating training metrics on ALL {len(X_train)} training samples...")
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        train_precision = precision_score(y_train, train_pred, zero_division=0)
        train_recall = recall_score(y_train, train_pred, zero_division=0)
        
        # Calculate validation metrics if data exists
        val_accuracy, val_precision, val_recall = None, None, None
        if eval_set is not None:
            val_pred = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)
            val_precision = precision_score(y_val, val_pred, zero_division=0)
            val_recall = recall_score(y_val, val_pred, zero_division=0)
        
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
        
        if val_accuracy is not None:
            print(f"\n   🔍 Validation Metrics:")
            print(f"      • Accuracy:  {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
            print(f"      • Precision: {val_precision:.4f}")
            print(f"      • Recall:    {val_recall:.4f}")
            
            # Overfitting detection
            accuracy_gap = (train_accuracy - val_accuracy) * 100
            if accuracy_gap > 10:
                print(f"\n   ⚠️  High overfitting detected!")
                print(f"      Gap: {accuracy_gap:.2f}% (train - val)")
                print(f"      💡 Consider: Reduce max_depth, increase min_child_weight, or add regularization")
            elif accuracy_gap > 5:
                print(f"\n   ⚡ Moderate overfitting: {accuracy_gap:.2f}% gap")
            else:
                print(f"\n   ✅ Good generalization: {accuracy_gap:.2f}% gap")
        
        print(f"{'='*70}\n")
        
        return type('History', (), {'history': history})()
    
    def predict(self, X_test):
        """
        Full-featured prediction method with feature validation
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions!")
        
        # If X_test is from a dataset batch, flatten it
        if len(X_test.shape) > 2:
            X_test = X_test.reshape(X_test.shape[0], -1)
        
        # CRITICAL: Validate feature count matches training data
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
        
        # Get prediction probabilities
        pred_proba = self.model.predict_proba(X_test)
        
        # Check if we have both classes
        if pred_proba.shape[1] == 1:
            print(f"WARNING: Model only learned one class! Pred_proba shape: {pred_proba.shape}")
            return pred_proba.reshape(-1, 1)
        else:
            # Normal case - return probability of positive class (class 1)
            return pred_proba[:, 1].reshape(-1, 1)
    
    def summary(self):
        """
        Print model summary
        """
        print("Full-Featured XGBoost Model Summary:")
        print("="*50)
        print(f"Model Type: XGBoost Classifier")
        print(f"Number of Estimators: {self.model.n_estimators}")
        print(f"Max Depth: {self.model.max_depth}")
        print(f"Learning Rate: {self.model.learning_rate}")
        print(f"Tree Method: {self.model.tree_method}")
        print(f"Number of Jobs: {self.model.n_jobs} (parallel processing)")
        print(f"Subsample: {self.model.subsample}")
        print(f"Column Sample by Tree: {self.model.colsample_bytree}")
        if self.is_fitted:
            print(f"Number of Features: {self.model.n_features_in_}")
            print(f"Number of Classes: {self.model.n_classes_}")
        print("="*50)
    
    def save_model(self, filename=None):
        """
        Save the trained XGBoost model to disk (auto-deletes old XGBoost models)
        
        Args:
            filename: Name of the file to save (default: xgboost_model.joblib)
            
        Returns:
            filepath: Full path to the saved model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving!")
        
        # Delete old XGBoost models first
        print("\n🗑️  Cleaning up old XGBoost models...")
        deleted_count = 0
        for file in os.listdir('.'):
            if file.endswith(('.joblib', '.pkl')) and ('xgb' in file.lower() or 'xgboost' in file.lower() or 'boost' in file.lower()):
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
        
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"xgboost_model_trained_{timestamp}.joblib"
        
        # Ensure .joblib extension
        if not filename.endswith('.joblib'):
            filename += '.joblib'
        
        # Get current directory
        current_dir = os.getcwd()
        filepath = os.path.join(current_dir, filename)
        
        # Save model
        print(f"\n💾 Saving new model...")
        joblib.dump(self, filepath)
        print(f"✅ XGBoost model saved to: {filepath}")
        
        return filepath


def load_xgboost_model(filepath):
    """
    Load a trained XGBoost model from disk
    
    Args:
        filepath: Path to the saved model file
        
    Returns:
        model: Loaded XGBoostAudioClassifier instance
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    model = joblib.load(filepath)
    
    if not isinstance(model, XGBoostAudioClassifier):
        raise ValueError("Loaded object is not a XGBoostAudioClassifier instance!")
    
    print(f"✓ XGBoost model loaded from: {filepath}")
    print(f"  - Number of features: {model.model.n_features_in_}")
    print(f"  - Number of classes: {model.model.n_classes_}")
    print(f"  - Number of estimators: {model.model.n_estimators}")
    
    return model