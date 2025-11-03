# Cell: Import SVM and related libraries
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import numpy as np
import joblib
import os

# Cell: Full-Featured SVM Model Setup
class SVMAudioClassifier:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42, max_iter=-1):
        """
        Full-featured SVM classifier for audio classification
        
        Args:
            kernel: SVM kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            C: Regularization parameter (default: 1.0)
            gamma: Kernel coefficient ('scale', 'auto', or float)
            probability: Enable probability estimates (required for predict_proba)
            random_state: Random state for reproducibility
            max_iter: Maximum iterations (-1 for no limit, or specify number for faster training)
        """
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=probability,
            random_state=random_state,
            cache_size=1000,  # Increase cache for faster training
            max_iter=max_iter,  # Control iterations
            verbose=True  # Show progress
        )
        self.is_fitted = False
        self.kernel = kernel
        self.C = C
        
    def compile(self, **kwargs):
        """Compile method for consistency with other models"""
        print("="*70)
        print("🔧 SVM MODEL CONFIGURATION")
        print("="*70)
        print(f"🎯 Kernel: {self.model.kernel}")
        print(f"📊 Regularization (C): {self.model.C}")
        print(f"🔢 Gamma: {self.model.gamma}")
        print(f"💾 Cache Size: {self.model.cache_size} MB")
        print(f"🔄 Max Iterations: {self.model.max_iter}")
        print(f"📈 Probability Estimates: {self.model.probability}")
        print("\n💡 Tips for faster training:")
        print("   • Use kernel='linear' for large datasets")
        print("   • Normalize features (critical for SVM!)")
        print("   • Set max_iter=1000-5000 for time limit")
        print("="*70)
        
    def fit(self, train_data, epochs=None, validation_data=None, normalize=True):
        """Full-featured fit method with progress tracking and normalization"""
        print("\n" + "="*70)
        print("🚀 TRAINING SVM CLASSIFIER")
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
        
        # Normalize features (CRITICAL for SVM performance!)
        if normalize:
            print(f"\n⚖️  Normalizing features (StandardScaler)...")
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            print(f"   ✅ Features normalized (mean=0, std=1)")
            print(f"   💡 This significantly speeds up SVM training!")
            self.normalized = True
        else:
            self.scaler = None
            self.normalized = False
            print(f"\n⚠️  WARNING: Training without normalization - will be VERY SLOW!")
        
        # Fit the model with progress tracking
        print(f"\n🔄 Training SVM with {self.kernel} kernel...")
        print(f"   C={self.C}, gamma={self.model.gamma}")
        print("="*70)
        
        import sys
        from datetime import datetime
        start_time = datetime.now()
        
        # Use verbose mode to show iterations
        print("⏳ Training in progress (this may take a few minutes)...")
        print("   [SVM will display convergence progress below]")
        print("-"*70)
        
        try:
            self.model.fit(X_train, y_train)
            elapsed = (datetime.now() - start_time).total_seconds()
            
            print("-"*70)
            print(f"\n✅ Training completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
            
        except Exception as e:
            print(f"\n❌ Training interrupted: {e}")
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"   Time elapsed: {elapsed:.2f} seconds")
            raise
        
        self.is_fitted = True
        print("="*70)
        
        # Calculate training metrics on subset to save time
        print("\n📊 Calculating training metrics (on subset to save time)...")
        
        subset_size = min(1000, len(X_train))
        indices = np.random.choice(len(X_train), subset_size, replace=False)
        X_train_subset = X_train[indices]
        y_train_subset = y_train[indices]
        
        train_pred = self.model.predict(X_train_subset)
        train_accuracy = accuracy_score(y_train_subset, train_pred)
        train_precision = precision_score(y_train_subset, train_pred, zero_division=0)
        train_recall = recall_score(y_train_subset, train_pred, zero_division=0)
        
        # Process validation data if provided
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
                print(f"   • Decrease C (current: {self.C}) - try C=0.1 or C=0.01")
                print(f"   • Try kernel='linear' for simpler model")
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
        
        # Show support vector info
        if hasattr(self.model, 'support_'):
            n_sv = len(self.model.support_)
            sv_ratio = n_sv / len(X_train)
            print(f"\n📊 Support Vectors:")
            print(f"   • Number: {n_sv:,}")
            print(f"   • Ratio: {sv_ratio:.2%} of training data")
            
            if sv_ratio > 0.5:
                print(f"   ⚠️  High support vector ratio - model may be complex")
            elif sv_ratio > 0.3:
                print(f"   ⚠️  Moderate support vector ratio")
            else:
                print(f"   ✅ Good support vector ratio")
        
        print("="*70)
        
        return type('History', (), {'history': history})()
    
    def predict(self, X_test):
        """Full-featured prediction method with normalization"""
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
        """Print model summary with overfitting info"""
        print("="*70)
        print("📊 SVM MODEL SUMMARY")
        print("="*70)
        print(f"Model Type: Support Vector Machine")
        print(f"\n🔧 Configuration:")
        print(f"   • Kernel: {self.model.kernel}")
        print(f"   • Regularization (C): {self.model.C}")
        print(f"   • Gamma: {self.model.gamma}")
        print(f"   • Cache Size: {self.model.cache_size} MB")
        print(f"   • Max Iterations: {self.model.max_iter}")
        
        if self.is_fitted:
            print(f"\n📈 Training Info:")
            n_sv = len(self.model.support_)
            print(f"   • Number of Support Vectors: {n_sv:,}")
            print(f"   • Number of Features: {self.model.n_features_in_}")
            print(f"   • Feature Normalization: {'✅ Enabled' if self.normalized else '❌ Disabled'}")
            
            # Analyze support vector ratio
            if hasattr(self.model, '_n_support'):
                total_samples = sum(self.model._n_support)
                sv_ratio = n_sv / total_samples if total_samples > 0 else 0
                print(f"   • Support Vector Ratio: {sv_ratio:.2%}")
                
                if sv_ratio > 0.5:
                    status = "⚠️  Very High (possible overfitting)"
                elif sv_ratio > 0.3:
                    status = "⚠️  High"
                elif sv_ratio > 0.1:
                    status = "✅ Good"
                else:
                    status = "✅ Excellent (well-generalized)"
                
                print(f"   • Model Complexity: {status}")
            
            print(f"\n💡 Overfitting Prevention:")
            print(f"   • Lower C value = More regularization = Less overfitting")
            print(f"   • Recommended C range: 0.01 - 10.0")
            print(f"   • Linear kernel = Simpler = Less overfitting")
            print(f"   • Feature normalization: CRITICAL for SVM!")
        else:
            print(f"\n⚠️  Model not yet trained")
        
        print("="*70)
    
    def save_model(self, filename=None):
        """
        Save the trained SVM model to disk
        
        Args:
            filename: Name of the file to save (default: svm_model.joblib)
            
        Returns:
            filepath: Full path to the saved model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving!")
        
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"svm_model_trained_{timestamp}.joblib"
        
        # Ensure .joblib extension
        if not filename.endswith('.joblib'):
            filename += '.joblib'
        
        # Get current directory
        current_dir = os.getcwd()
        filepath = os.path.join(current_dir, filename)
        
        # Save model
        joblib.dump(self, filepath)
        print(f"✓ SVM model saved to: {filepath}")
        
        return filepath


def load_svm_model(filepath):
    """
    Load a trained SVM model from disk
    
    Args:
        filepath: Path to the saved model file
        
    Returns:
        model: Loaded SVMAudioClassifier instance
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    model = joblib.load(filepath)
    
    if not isinstance(model, SVMAudioClassifier):
        raise ValueError("Loaded object is not a SVMAudioClassifier instance!")
    
    print(f"✓ SVM model loaded from: {filepath}")
    print(f"  - Number of support vectors: {len(model.model.support_)}")
    print(f"  - Number of features: {model.model.n_features_in_}")
    print(f"  - Kernel: {model.model.kernel}")
    
    return model