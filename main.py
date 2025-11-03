import datetime
import tensorflow as tf
import numpy as np  
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from datetime import datetime
import joblib
import pandas
from gui import launch_advanced_simulation_gui
from Models.RandomForrest import RandomForestFullFeatures, load_rf_model
from Models.KNN import KNNAudioClassifier
from Models.SVM import SVMAudioClassifier
from Models.XgBoost import XGBoostAudioClassifier
# from Models.CNN import CNNAudioClassifier  # Uncomment when CNN is ready
def save_augmented_dataset(train_dataset, val_dataset, test_dataset, dataset_name="augmented", save_dir='saved_datasets'):
    """
    Save augmented datasets to disk for later use
    
    Args:
        train_dataset: Training tf.data.Dataset
        val_dataset: Validation tf.data.Dataset  
        test_dataset: Test tf.data.Dataset
        dataset_name: Name identifier for the dataset (e.g., "augmented_5000")
        save_dir: Directory to save datasets
    
    Returns:
        str: Path to saved file
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n💾 Saving {dataset_name} datasets to disk...")
    print(f"📁 Save directory: {os.path.abspath(save_dir)}")
    
    # Extract data from datasets
    print("📦 Extracting training data...")
    train_X, train_y = [], []
    for batch_x, batch_y in train_dataset.unbatch():
        train_X.append(batch_x.numpy())
        train_y.append(batch_y.numpy())
    
    print("📦 Extracting validation data...")
    val_X, val_y = [], []
    for batch_x, batch_y in val_dataset.unbatch():
        val_X.append(batch_x.numpy())
        val_y.append(batch_y.numpy())
    
    print("📦 Extracting test data...")
    test_X, test_y = [], []
    for batch_x, batch_y in test_dataset.unbatch():
        test_X.append(batch_x.numpy())
        test_y.append(batch_y.numpy())
    
    # Convert to numpy arrays
    train_X = np.array(train_X)
    train_y = np.array(train_y)
    val_X = np.array(val_X)
    val_y = np.array(val_y)
    test_X = np.array(test_X)
    test_y = np.array(test_y)
    
    # Create dataset dictionary
    dataset_dict = {
        'train_X': train_X,
        'train_y': train_y,
        'val_X': val_X,
        'val_y': val_y,
        'test_X': test_X,
        'test_y': test_y,
        'dataset_name': dataset_name,
        'created_date': datetime.now().isoformat(),
        'total_samples': len(train_X) + len(val_X) + len(test_X),
        'split_info': {
            'train': len(train_X),
            'val': len(val_X),
            'test': len(test_X)
        }
    }
    
    # Save to disk
    filename = os.path.join(save_dir, f'{dataset_name}_dataset.pkl')
    
    print(f"💾 Saving to: {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(dataset_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Get file size
    file_size_mb = os.path.getsize(filename) / (1024 * 1024)
    
    print(f"✅ Successfully saved {dataset_name} datasets!")
    print(f"📊 Dataset Info:")
    print(f"   • Training samples: {len(train_X)}")
    print(f"   • Validation samples: {len(val_X)}")
    print(f"   • Test samples: {len(test_X)}")
    print(f"   • Total samples: {len(train_X) + len(val_X) + len(test_X)}")
    print(f"   • File size: {file_size_mb:.2f} MB")
    print(f"   • Location: {os.path.abspath(filename)}")
    
    return filename

def load_augmented_dataset(dataset_name="augmented", save_dir='saved_datasets', batch_size=32):
    """
    Load previously saved augmented datasets from disk
    
    Args:
        dataset_name: Name identifier for the dataset
        save_dir: Directory where datasets are saved
        batch_size: Batch size for tf.data.Dataset
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, dataset_info)
    """
    filename = os.path.join(save_dir, f'{dataset_name}_dataset.pkl')
    
    if not os.path.exists(filename):
        print(f"❌ Dataset file not found: {filename}")
        print(f"📁 Looking in: {os.path.abspath(save_dir)}")
        return None, None, None, None
    
    print(f"📂 Loading {dataset_name} dataset from disk...")
    print(f"📁 Loading from: {os.path.abspath(filename)}")
    
    # Load dataset dictionary
    with open(filename, 'rb') as f:
        dataset_dict = pickle.load(f)
    
    # Extract data
    train_X = dataset_dict['train_X']
    train_y = dataset_dict['train_y']
    val_X = dataset_dict['val_X']
    val_y = dataset_dict['val_y']
    test_X = dataset_dict['test_X']
    test_y = dataset_dict['test_y']
    
    # Create tf.data.Datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y))
    train_dataset = train_dataset.cache().shuffle(buffer_size=500).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((val_X, val_y))
    val_dataset = val_dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    test_dataset = test_dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Prepare info dictionary
    dataset_info = {
        'dataset_name': dataset_dict['dataset_name'],
        'created_date': dataset_dict['created_date'],
        'total_samples': dataset_dict['total_samples'],
        'split_info': dataset_dict['split_info'],
        'file_size_mb': os.path.getsize(filename) / (1024 * 1024)
    }
    
    print(f"✅ Successfully loaded {dataset_name} datasets!")
    print(f"📊 Dataset Info:")
    print(f"   • Training samples: {dataset_dict['split_info']['train']}")
    print(f"   • Validation samples: {dataset_dict['split_info']['val']}")
    print(f"   • Test samples: {dataset_dict['split_info']['test']}")
    print(f"   • Total samples: {dataset_dict['total_samples']}")
    print(f"   • Created: {dataset_dict['created_date'][:10]}")
    print(f"   • File size: {dataset_info['file_size_mb']:.2f} MB")
    
    return train_dataset, val_dataset, test_dataset, dataset_info

def quick_load_augmented_dataset(dataset_name="augmented_5000"):
    """Quick helper to load augmented dataset"""
    print(f" Quick Loading {dataset_name} Dataset...")
    
    train, val, test, info = load_augmented_dataset(dataset_name)
    
    if train is not None:
        # Create global variables with standard names
        globals()['train_aug_safe'] = train
        globals()['val_aug_safe'] = val
        globals()['test_aug_safe'] = test
        
        print(f"\n✅ Dataset loaded and assigned to variables:")
        print(f"   • train_aug_safe")
        print(f"   • val_aug_safe")
        print(f"   • test_aug_safe")
        
        return train, val, test, info
    else:
        print(f" Dataset '{dataset_name}' not found. Create and save it first.")
        return None, None, None, None
    


def load_snr_augmented_dataset(snr_level, save_dir='saved_datasets', batch_size=32):
    """
    Load previously saved augmented datasets from disk
    
    Args:
        snr_level: SNR level to load (5, 10, 15, or 20)
        save_dir: Directory where datasets are saved
        batch_size: Batch size for tf.data.Dataset
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, dataset_info)
    """
    filename = os.path.join(save_dir, f'snr_{snr_level}db_augmented_dataset.pkl')
    
    if not os.path.exists(filename):
        print(f"❌ Dataset file not found: {filename}")
        print(f"📁 Looking in: {os.path.abspath(save_dir)}")
        return None, None, None, None
    
    print(f"📂 Loading SNR {snr_level} dB dataset from disk...")
    print(f"📁 Loading from: {os.path.abspath(filename)}")
    
    # Load dataset dictionary
    with open(filename, 'rb') as f:
        dataset_dict = pickle.load(f)
    
    # Extract data
    train_X = dataset_dict['train_X']
    train_y = dataset_dict['train_y']
    val_X = dataset_dict['val_X']
    val_y = dataset_dict['val_y']
    test_X = dataset_dict['test_X']
    test_y = dataset_dict['test_y']
    
    # Create tf.data.Datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y))
    train_dataset = train_dataset.cache().shuffle(buffer_size=500).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((val_X, val_y))
    val_dataset = val_dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    test_dataset = test_dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Prepare info dictionary
    dataset_info = {
        'snr_level': dataset_dict['snr_level'],
        'created_date': dataset_dict['created_date'],
        'total_samples': dataset_dict['total_samples'],
        'split_info': dataset_dict['split_info'],
        'file_size_mb': os.path.getsize(filename) / (1024 * 1024)
    }
    
    print(f"✅ Successfully loaded SNR {snr_level} dB datasets!")
    print(f"📊 Dataset Info:")
    print(f"   • Training samples: {dataset_dict['split_info']['train']}")
    print(f"   • Validation samples: {dataset_dict['split_info']['val']}")
    print(f"   • Test samples: {dataset_dict['split_info']['test']}")
    print(f"   • Total samples: {dataset_dict['total_samples']}")
    print(f"   • Created: {dataset_dict['created_date'][:10]}")
    print(f"   • File size: {dataset_info['file_size_mb']:.2f} MB")
    
    return train_dataset, val_dataset, test_dataset, dataset_info

def quick_load_snr_datasets(snr_levels=[5, 10, 15, 20], save_dir='saved_datasets'):
    """
    Quick helper to load all SNR datasets at once
    
    Args:
        snr_levels: List of SNR levels to load
        save_dir: Directory where datasets are saved
    """
    print("🚀 Quick Loading SNR Augmented Datasets...")
    print("="*70)
    
    loaded_datasets = {}
    
    for snr in snr_levels:
        print(f"\n📂 Loading SNR {snr} dB...")
        train, val, test, info = load_snr_augmented_dataset(snr, save_dir)
        
        if train is not None:
            # Create global variables
            globals()[f'train_aug_snr_{["five", "ten", "fifteen", "twenty"][snr_levels.index(snr)]}'] = train
            globals()[f'val_aug_snr_{["five", "ten", "fifteen", "twenty"][snr_levels.index(snr)]}'] = val
            globals()[f'test_aug_snr_{["five", "ten", "fifteen", "twenty"][snr_levels.index(snr)]}'] = test
            
            loaded_datasets[snr] = {'train': train, 'val': val, 'test': test, 'info': info}
            print(f"   ✅ SNR {snr} dB loaded successfully!")
        else:
            print(f"   ❌ SNR {snr} dB not found on disk")
    
    print("\n" + "="*70)
    print(f"📊 LOADING SUMMARY:")
    print(f"   ✅ Loaded {len(loaded_datasets)}/{len(snr_levels)} SNR datasets")
    
    if loaded_datasets:
        print(f"\n✅ Available dataset variables:")
        for snr in loaded_datasets.keys():
            snr_name = ["five", "ten", "fifteen", "twenty"][snr_levels.index(snr)]
            print(f"   • train_aug_snr_{snr_name}, val_aug_snr_{snr_name}, test_aug_snr_{snr_name}")
    
    return loaded_datasets

def list_saved_models():
    """List all saved model files"""
    model_files = [f for f in os.listdir('.') if f.startswith('rf_model') and (f.endswith('.joblib') or f.endswith('.pkl'))]
    print("📁 Saved Models:")
    for i, model_file in enumerate(model_files, 1):
        size = os.path.getsize(model_file) / 1024**2
        print(f"  {i}. {model_file} ({size:.2f} MB)")
    return model_files

def load_model(filepath):
    """
    Load a saved model from file and return it as a usable object
    
    Args:
        filepath: Path to the saved model file (.joblib or .pkl)
    
    Returns:
        Loaded model object ready to use
    
    Example:
        rf_model_full = load_model('rf_model_full_trained.joblib')
        predictions = rf_model_full.predict(X_test)
    """
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        print(f"📁 Available models:")
        list_saved_models()
        return None
    
    try:
        # Try loading with joblib first (recommended for sklearn models)
        print(f"📂 Loading model from: {filepath}")
        model = joblib.load(filepath)
        
        print(f"✅ Model loaded successfully!")
        print(f"📊 Model type: {type(model).__name__}")
        
        # Check if model is fitted
        if hasattr(model, 'is_fitted'):
            print(f"🔧 Model fitted: {model.is_fitted}")
        
        # Display model summary if available
        if hasattr(model, 'summary'):
            print("\n📋 Model Summary:")
            model.summary()
        
        print(f"\n✅ Model ready to use! You can now call:")
        print(f"   • model.predict(X_test)")
        print(f"   • model.summary()")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading with joblib: {e}")
        
        # Fallback to pickle
        try:
            print("🔄 Trying pickle method...")
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            
            print(f"✅ Model loaded successfully with pickle!")
            print(f"📊 Model type: {type(model).__name__}")
            
            if hasattr(model, 'is_fitted'):
                print(f"🔧 Model fitted: {model.is_fitted}")
            
            return model
            
        except Exception as e2:
            print(f"❌ Error loading with pickle: {e2}")
            return None

def delete_model(filepath):
    """Delete a saved model file"""
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"🗑️ Deleted: {filepath}")
    else:
        print(f"❌ File not found: {filepath}")

def model_info(filepath):
    """Get info about a saved model without loading it fully"""
    try:
        model = load_model(filepath)
        if model and hasattr(model, 'summary'):
            model.summary()
        return model
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# Quick load helper function
def quick_load(filename=None):
    """
    Quick load function - automatically finds and loads rf_model_full_trained.joblib
    
    Args:
        filename: Optional specific filename to load (default: 'rf_model_full_trained.joblib')
    
    Returns:
        Loaded model
    
    Example:
        rf_model_full = quick_load()  # Loads default trained model
        rf_model_full = quick_load('my_custom_model.joblib')  # Loads specific model
    """
    if filename is None:
        filename = 'rf_model_full_trained.joblib'
    
    return load_model(filename)

# Usage examples
print("\n🔧 Available commands:")
print("="*70)
print("📂 Load Model:")
print("   rf_model_full = load_model('rf_model_full_trained.joblib')")
print("   rf_model_full = quick_load()  # Automatically loads default model")
print()
print("📋 Get Model Info:")
print("   model_info('rf_model_full_trained.joblib')")
print()
print("📁 List All Models:")
print("   list_saved_models()")
print()
print("🗑️ Delete Model:")
print("   delete_model('old_model.joblib')")
print("="*70)

# List current models
print("\n📁 Currently Available Models:")
saved_models = list_saved_models()

# Example: Auto-load if default model exists
if 'rf_model_full_trained.joblib' in saved_models:
    print("\n💡 Quick Tip: Load your trained model with:")
    print("   rf_model_full = quick_load()")


# This is the most important part!
if __name__ == "__main__":

    # Load all available models
    print("\n🤖 Loading Models...")
    print("="*70)
    
    # Try to load saved Random Forest model
    list_saved_models()
    rf_model_full = load_model('rf_model_full_trained.joblib')
    
    # Create instances of other models (untrained, ready to train with full potential)
    print("\n🔧 Creating full-featured model instances...")
    knn_model = KNNAudioClassifier(n_neighbors=5, weights='distance', algorithm='auto')
    svm_model = SVMAudioClassifier(kernel='rbf', C=1.0, gamma='scale')
    xgboost_model = XGBoostAudioClassifier(n_estimators=100, max_depth=6, learning_rate=0.1)
    
    print("✅ KNN model instance created (full-featured, n_neighbors=5)")
    print("✅ SVM model instance created (full-featured, kernel=rbf)")
    print("✅ XGBoost model instance created (full-featured, n_estimators=100)")
    
    print("\n🚀 Quick Loading Augmented 5000 Dataset...")

    # This automatically loads and assigns to train_aug_safe, val_aug_safe, test_aug_safe
    train_aug_safe, val_aug_safe, test_aug_safe, info = quick_load_augmented_dataset("augmented_5000")

    # Load ALL SNR datasets
    print("\n🚀 Loading ALL SNR datasets...")
    print("="*70)
    
    snr_datasets = {}
    
    # Load SNR 5, 10, 15, 20 dB datasets
    snr_levels = [5, 10, 15, 20]
    for snr in snr_levels:
        print(f"\n📂 Loading SNR {snr} dB dataset...")
        train, val, test, info_snr = load_snr_augmented_dataset(snr_level=snr)
        
        if train is not None:
            snr_datasets[f'snr_{snr}'] = {
                'train': train,
                'val': val,
                'test': test,
                'info': info_snr
            }
            print(f"   ✅ SNR {snr} dB loaded successfully!")
        else:
            print(f"   ⚠️ SNR {snr} dB not found, skipping...")

    if train_aug_safe is not None:
        print("\n✅ Base dataset loaded successfully!")
        print("🎯 Ready to train models!")
    else:
        print("\n❌ Base dataset not found. Create and save it first.")

    # Create a comprehensive dictionary of ALL datasets
    all_datasets = {
        'safe_5000': {
            'train': train_aug_safe, 
            'val': val_aug_safe, 
            'test': test_aug_safe,
            'info': info
        }
    }
    
    # Add all loaded SNR datasets
    all_datasets.update(snr_datasets)
    
    print("\n📊 DATASET LOADING SUMMARY:")
    print("="*70)
    print(f"✅ Total datasets loaded: {len(all_datasets)}")
    for ds_name in all_datasets.keys():
        print(f"   • {ds_name}")
    print("="*70)
    
    # Create a dictionary of all available models
    all_models = {
        'RandomForest': rf_model_full,
        'KNN': knn_model,
        'SVM': svm_model,
        'XGBoost': xgboost_model
    }
    
    print("\n🤖 MODEL SUMMARY:")
    print("="*70)
    print(f"✅ Total models available: {len(all_models)}")
    for model_name, model_obj in all_models.items():
        trained_status = "✅ Trained" if hasattr(model_obj, 'is_fitted') and model_obj.is_fitted else "⚠️ Not trained"
        print(f"   • {model_name}: {trained_status}")
    print("="*70)
    
    # Pass all models and datasets to the GUI
    launch_advanced_simulation_gui(all_models, all_datasets)