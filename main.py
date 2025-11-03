import datetime
import tensorflow as tf
import librosa
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

# ============================================================================
# MFCC FUNCTIONS
# ============================================================================

def load_wav_16k_mono(filename):
    """Load and resample audio to 16kHz mono using librosa"""
    # Decode filename from tensor if needed
    if isinstance(filename, tf.Tensor):
        filename = filename.numpy().decode('utf-8')
    
    # Load audio with librosa (automatically resamples to target sr)
    wav, sample_rate = librosa.load(filename, sr=16000, mono=True)
    
    # Convert to TensorFlow tensor
    wav = tf.convert_to_tensor(wav, dtype=tf.float32)
    
    return wav


def preprocess_mfcc(file_path, label, n_mfcc=40):
    """
    Extract MFCC features from audio file - optimized for tree-based models
    
    Args:
        file_path: Path to audio file
        label: Class label (0 or 1)
        n_mfcc: Number of MFCC coefficients to extract (default: 40)
    
    Returns:
        mfcc_features: Flattened MFCC feature vector
        label: Class label
    """
    TARGET_LENGTH = 4800  # Same as spectrogram version
    
    # Load audio
    wav = load_wav_16k_mono(file_path)
    
    # Truncate/pad to target length
    if tf.shape(wav)[0] > TARGET_LENGTH:
        wav = wav[:TARGET_LENGTH]
    else:
        padding = TARGET_LENGTH - tf.shape(wav)[0]
        wav = tf.concat([wav, tf.zeros([padding], dtype=tf.float32)], 0)
    
    # Compute STFT
    stft = tf.signal.stft(wav, frame_length=320, frame_step=32, fft_length=512)
    spectrogram = tf.abs(stft)
    
    # Convert to mel-scale (40 bins for MFCC)
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mfcc,
        num_spectrogram_bins=tf.shape(spectrogram)[-1],
        sample_rate=16000,
        lower_edge_hertz=0.0,
        upper_edge_hertz=8000.0
    )
    mel_spec = tf.tensordot(spectrogram, mel_matrix, 1)
    mel_spec = tf.math.log(mel_spec + 1e-6)
    
    # Compute MFCC (Discrete Cosine Transform of mel spectrogram)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(mel_spec)[..., :n_mfcc]
    
    # Average across time dimension to get fixed-length feature vector
    mfcc_mean = tf.reduce_mean(mfccs, axis=0)
    
    # Normalize features
    mfcc_min = tf.reduce_min(mfcc_mean)
    mfcc_max = tf.reduce_max(mfcc_mean)
    mfcc_normalized = (mfcc_mean - mfcc_min) / (mfcc_max - mfcc_min + 1e-8)
    
    return mfcc_normalized, label


def create_mfcc_datasets(train_path, val_path, test_path, 
                         samples_per_class=2500, n_mfcc=40,
                         seed=42, batch_size=32):
    """
    Create MFCC feature datasets from audio files
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data
        test_path: Path to test data
        samples_per_class: Number of samples per class (default: 2500)
        n_mfcc: Number of MFCC coefficients (default: 40)
        seed: Random seed for reproducibility
        batch_size: Batch size for dataset
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    print(f"\n🎵 Creating MFCC Datasets with {n_mfcc} coefficients...")
    print(f"📊 Target: {samples_per_class} samples per class")
    
    def create_single_dataset(data_path, dataset_name, target_samples_per_class):
        print(f"\n📂 Processing {dataset_name}...")
        
        # Define paths
        positive_data = os.path.join(data_path, 'real')
        negative_data = os.path.join(data_path, 'fake')
        
        # List files
        pos_files = [os.path.join(positive_data, f) for f in os.listdir(positive_data) if f.endswith('.wav')]
        neg_files = [os.path.join(negative_data, f) for f in os.listdir(negative_data) if f.endswith('.wav')]
        
        print(f"   Available - Real: {len(pos_files)}, Fake: {len(neg_files)}")
        
        # Sample if needed
        if len(pos_files) > target_samples_per_class:
            np.random.seed(seed)
            pos_files = np.random.choice(pos_files, target_samples_per_class, replace=False).tolist()
        if len(neg_files) > target_samples_per_class:
            np.random.seed(seed)
            neg_files = np.random.choice(neg_files, target_samples_per_class, replace=False).tolist()
        
        print(f"   Using - Real: {len(pos_files)}, Fake: {len(neg_files)}")
        
        # Create datasets
        pos_ds = tf.data.Dataset.from_tensor_slices(pos_files)
        neg_ds = tf.data.Dataset.from_tensor_slices(neg_files)
        
        # Add labels
        pos_labeled = tf.data.Dataset.zip((pos_ds, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos_files)))))
        neg_labeled = tf.data.Dataset.zip((neg_ds, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg_files)))))
        
        # Combine and shuffle
        combined = pos_labeled.concatenate(neg_labeled)
        combined = combined.shuffle(buffer_size=len(pos_files) + len(neg_files), seed=seed)
        
        # Apply MFCC preprocessing
        combined = combined.map(lambda x, y: preprocess_mfcc(x, y, n_mfcc=n_mfcc), 
                               num_parallel_calls=tf.data.AUTOTUNE)
        
        return combined
    
    # Create datasets
    train_ds = create_single_dataset(train_path, "Training", samples_per_class)
    val_ds = create_single_dataset(val_path, "Validation", samples_per_class)
    test_ds = create_single_dataset(test_path, "Test", samples_per_class)
    
    # Cache and batch
    train_ds = train_ds.cache().shuffle(buffer_size=500).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    print(f"\n✅ MFCC Datasets created successfully!")
    
    return train_ds, val_ds, test_ds


def save_mfcc_dataset(train_dataset, val_dataset, test_dataset, 
                      dataset_name="mfcc", save_dir='mfcc_datasets'):
    """
    Save MFCC datasets to disk
    
    Args:
        train_dataset: Training tf.data.Dataset
        val_dataset: Validation tf.data.Dataset
        test_dataset: Test tf.data.Dataset
        dataset_name: Name identifier for the dataset
        save_dir: Directory to save datasets (default: 'mfcc_datasets')
    
    Returns:
        str: Path to saved file
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n💾 Saving {dataset_name} MFCC datasets to disk...")
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
        'feature_type': 'mfcc',
        'n_features': train_X.shape[1],
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
    
    print(f"✅ Successfully saved {dataset_name} MFCC datasets!")
    print(f"📊 Dataset Info:")
    print(f"   • Feature type: MFCC")
    print(f"   • Features per sample: {train_X.shape[1]}")
    print(f"   • Training samples: {len(train_X)}")
    print(f"   • Validation samples: {len(val_X)}")
    print(f"   • Test samples: {len(test_X)}")
    print(f"   • Total samples: {len(train_X) + len(val_X) + len(test_X)}")
    print(f"   • File size: {file_size_mb:.2f} MB")
    print(f"   • Location: {os.path.abspath(filename)}")
    
    return filename


def load_mfcc_dataset(dataset_name="mfcc", save_dir='mfcc_datasets', batch_size=32):
    """
    Load previously saved MFCC datasets from disk
    
    Args:
        dataset_name: Name identifier for the dataset
        save_dir: Directory where datasets are saved (default: 'mfcc_datasets')
        batch_size: Batch size for tf.data.Dataset
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, dataset_info)
    """
    filename = os.path.join(save_dir, f'{dataset_name}_dataset.pkl')
    
    if not os.path.exists(filename):
        print(f"❌ Dataset file not found: {filename}")
        print(f"📁 Looking in: {os.path.abspath(save_dir)}")
        return None, None, None, None
    
    print(f"📂 Loading {dataset_name} MFCC dataset from disk...")
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
        'feature_type': dataset_dict.get('feature_type', 'mfcc'),
        'n_features': dataset_dict.get('n_features', train_X.shape[1]),
        'created_date': dataset_dict['created_date'],
        'total_samples': dataset_dict['total_samples'],
        'split_info': dataset_dict['split_info'],
        'file_size_mb': os.path.getsize(filename) / (1024 * 1024)
    }
    
    print(f"✅ Successfully loaded {dataset_name} MFCC datasets!")
    print(f"📊 Dataset Info:")
    print(f"   • Feature type: {dataset_info['feature_type']}")
    print(f"   • Features per sample: {dataset_info['n_features']}")
    print(f"   • Training samples: {dataset_dict['split_info']['train']}")
    print(f"   • Validation samples: {dataset_dict['split_info']['val']}")
    print(f"   • Test samples: {dataset_dict['split_info']['test']}")
    print(f"   • Total samples: {dataset_dict['total_samples']}")
    print(f"   • Created: {dataset_dict['created_date'][:10]}")
    print(f"   • File size: {dataset_info['file_size_mb']:.2f} MB")
    
    return train_dataset, val_dataset, test_dataset, dataset_info


def mfcc_dataset_to_numpy(dataset):
    """
    Convert MFCC tf.data.Dataset to numpy arrays
    
    Args:
        dataset: tf.data.Dataset with MFCC features
    
    Returns:
        X: numpy array of features (n_samples, n_features)
        y: numpy array of labels (n_samples,)
    """
    X, y = [], []
    for batch_x, batch_y in dataset.unbatch():
        X.append(batch_x.numpy())
        y.append(batch_y.numpy())
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y


def quick_load_mfcc_datasets(dataset_names=None, save_dir='mfcc_datasets'):
    """
    Quick helper to load multiple MFCC datasets at once
    
    Args:
        dataset_names: List of dataset names to load (default: all standard datasets)
        save_dir: Directory where datasets are saved
    
    Returns:
        Dictionary of loaded datasets
    """
    if dataset_names is None:
        dataset_names = [
            'train_test_val',
            'train_test_val_aug',
            'snr_5db',
            'snr_10db',
            'snr_15db',
            'snr_20db'
        ]
    
    print("🚀 Quick Loading MFCC Datasets...")
    print("="*70)
    
    loaded_datasets = {}
    
    for dataset_name in dataset_names:
        print(f"\n📂 Loading {dataset_name}...")
        train, val, test, info = load_mfcc_dataset(dataset_name, save_dir)
        
        if train is not None:
            loaded_datasets[dataset_name] = {
                'train': train,
                'val': val,
                'test': test,
                'info': info
            }
            print(f"   ✅ {dataset_name} loaded successfully!")
        else:
            print(f"   ❌ {dataset_name} not found on disk")
    
    print("\n" + "="*70)
    print(f"📊 LOADING SUMMARY:")
    print(f"   ✅ Loaded {len(loaded_datasets)}/{len(dataset_names)} MFCC datasets")
    
    if loaded_datasets:
        print(f"\n✅ Available MFCC datasets:")
        for ds_name in loaded_datasets.keys():
            print(f"   • {ds_name}")
    
    return loaded_datasets

# ============================================================================
# END MFCC FUNCTIONS
# ============================================================================

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

def list_saved_models(verbose=True):
    """
    List all saved model files (.joblib and .pkl)
    
    Args:
        verbose: If True, print model list. If False, return silently.
    
    Returns:
        Dictionary with model type as key and list of files as value
    """
    # Get all .joblib and .pkl files in current directory
    all_files = [f for f in os.listdir('.') if f.endswith('.joblib') or f.endswith('.pkl')]
    
    # Categorize by model type
    model_files = {
        'RandomForest': [],
        'KNN': [],
        'SVM': [],
        'XGBoost': [],
        'CNN': [],
        'Other': []
    }
    
    for file in all_files:
        file_lower = file.lower()
        if 'rf' in file_lower or 'random' in file_lower or 'forest' in file_lower:
            model_files['RandomForest'].append(file)
        elif 'knn' in file_lower or 'k_nearest' in file_lower or 'neighbor' in file_lower:
            model_files['KNN'].append(file)
        elif 'svm' in file_lower or 'support' in file_lower or 'vector' in file_lower:
            model_files['SVM'].append(file)
        elif 'xgb' in file_lower or 'xgboost' in file_lower or 'boost' in file_lower:
            model_files['XGBoost'].append(file)
        elif 'cnn' in file_lower or 'conv' in file_lower or 'neural' in file_lower:
            model_files['CNN'].append(file)
        else:
            model_files['Other'].append(file)
    
    if verbose:
        print("📁 Saved Model Files:")
        total_count = 0
        for model_type, files in model_files.items():
            if files:
                print(f"\n  🔹 {model_type} Models ({len(files)}):")
                for file in files:
                    size = os.path.getsize(file) / 1024**2
                    print(f"     • {file} ({size:.2f} MB)")
                    total_count += 1
        
        if total_count == 0:
            print("  ⚠️ No saved model files found")
        else:
            print(f"\n  ✅ Total: {total_count} model file(s)")
    
    return model_files

def load_model(filepath, verbose=True):
    """
    Load a saved model from file and return it as a usable object
    
    Args:
        filepath: Path to the saved model file (.joblib or .pkl)
        verbose: If True, print detailed loading information
    
    Returns:
        Loaded model object ready to use
    
    Example:
        rf_model_full = load_model('rf_model_full_trained.joblib')
        predictions = rf_model_full.predict(X_test)
    """
    if not os.path.exists(filepath):
        if verbose:
            print(f"❌ File not found: {filepath}")
            print(f"📁 Available models:")
            list_saved_models()
        return None
    
    try:
        # Try loading with joblib first (recommended for sklearn models)
        if verbose:
            print(f"📂 Loading model from: {filepath}")
        
        model = joblib.load(filepath)
        
        if verbose:
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
        if verbose:
            print(f"❌ Error loading with joblib: {e}")
        
        # Fallback to pickle
        try:
            if verbose:
                print("🔄 Trying pickle method...")
            
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            
            if verbose:
                print(f"✅ Model loaded successfully with pickle!")
                print(f"📊 Model type: {type(model).__name__}")
                
                if hasattr(model, 'is_fitted'):
                    print(f"🔧 Model fitted: {model.is_fitted}")
            
            return model
            
        except Exception as e2:
            if verbose:
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

def auto_load_all_models():
    """
    Automatically detect and load all saved model files
    
    Returns:
        Dictionary with model instances, organized by type
    """
    print("\n🔍 Auto-detecting saved models...")
    print("="*70)
    
    model_files = list_saved_models(verbose=False)
    loaded_models = {}
    
    # Load Random Forest models
    if model_files['RandomForest']:
        print(f"\n📦 Loading Random Forest models...")
        for file in model_files['RandomForest']:
            try:
                model = load_model(file, verbose=False)
                if model:
                    # Use filename without extension as key
                    model_key = os.path.splitext(file)[0]
                    loaded_models[f'rf_{model_key}'] = model
                    print(f"   ✅ Loaded: {file}")
            except Exception as e:
                print(f"   ❌ Failed to load {file}: {e}")
    
    # Load KNN models
    if model_files['KNN']:
        print(f"\n📦 Loading KNN models...")
        for file in model_files['KNN']:
            try:
                model = load_model(file, verbose=False)
                if model:
                    model_key = os.path.splitext(file)[0]
                    loaded_models[f'knn_{model_key}'] = model
                    print(f"   ✅ Loaded: {file}")
            except Exception as e:
                print(f"   ❌ Failed to load {file}: {e}")
    
    # Load SVM models
    if model_files['SVM']:
        print(f"\n📦 Loading SVM models...")
        for file in model_files['SVM']:
            try:
                model = load_model(file, verbose=False)
                if model:
                    model_key = os.path.splitext(file)[0]
                    loaded_models[f'svm_{model_key}'] = model
                    print(f"   ✅ Loaded: {file}")
            except Exception as e:
                print(f"   ❌ Failed to load {file}: {e}")
    
    # Load XGBoost models
    if model_files['XGBoost']:
        print(f"\n📦 Loading XGBoost models...")
        for file in model_files['XGBoost']:
            try:
                model = load_model(file, verbose=False)
                if model:
                    model_key = os.path.splitext(file)[0]
                    loaded_models[f'xgb_{model_key}'] = model
                    print(f"   ✅ Loaded: {file}")
            except Exception as e:
                print(f"   ❌ Failed to load {file}: {e}")
    
    # Load CNN models
    if model_files['CNN']:
        print(f"\n� Loading CNN models...")
        for file in model_files['CNN']:
            try:
                model = load_model(file, verbose=False)
                if model:
                    model_key = os.path.splitext(file)[0]
                    loaded_models[f'cnn_{model_key}'] = model
                    print(f"   ✅ Loaded: {file}")
            except Exception as e:
                print(f"   ❌ Failed to load {file}: {e}")
    
    # Load other models
    if model_files['Other']:
        print(f"\n📦 Loading other models...")
        for file in model_files['Other']:
            try:
                model = load_model(file, verbose=False)
                if model:
                    model_key = os.path.splitext(file)[0]
                    loaded_models[f'other_{model_key}'] = model
                    print(f"   ✅ Loaded: {file}")
            except Exception as e:
                print(f"   ❌ Failed to load {file}: {e}")
    
    print("\n" + "="*70)
    print(f"📊 AUTO-LOAD SUMMARY:")
    print(f"   ✅ Successfully loaded: {len(loaded_models)} model(s)")
    
    if loaded_models:
        print(f"\n✅ Loaded models:")
        for model_name in loaded_models.keys():
            print(f"   • {model_name}")
    else:
        print(f"   ⚠️ No models were loaded")
    
    print("="*70)
    
    return loaded_models

# Example: Auto-load if default model exists
saved_models = list_saved_models(verbose=False)
if saved_models['RandomForest'] or saved_models['KNN'] or saved_models['SVM'] or saved_models['XGBoost']:
    print("\n💡 Quick Tip: Auto-load all models with:")
    print("   loaded_models = auto_load_all_models()")


# This is the most important part!
if __name__ == "__main__":

    # Auto-detect and load all saved models
    print("\n🤖 Loading Models...")
    print("="*70)
    
    # List all available model files
    list_saved_models(verbose=True)
    
    # Auto-load all saved models
    loaded_models_dict = auto_load_all_models()
    
    # Extract specific models if they exist
    rf_model_full = None
    knn_model = None
    svm_model = None
    xgboost_model = None
    
    # Try to get the first model of each type from loaded models
    for key, model in loaded_models_dict.items():
        if key.startswith('rf_') and rf_model_full is None:
            rf_model_full = model
            print(f"\n✅ Using Random Forest: {key}")
        elif key.startswith('knn_') and knn_model is None:
            knn_model = model
            print(f"✅ Using KNN: {key}")
        elif key.startswith('svm_') and svm_model is None:
            svm_model = model
            print(f"✅ Using SVM: {key}")
        elif key.startswith('xgb_') and xgboost_model is None:
            xgboost_model = model
            print(f"✅ Using XGBoost: {key}")
    
    # Create new instances for models that weren't found
    print("\n🔧 Creating model instances for models not found on disk...")
    if rf_model_full is None:
        rf_model_full = RandomForestFullFeatures(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42)
        print("✅ Random Forest instance created (untrained)")
    
    if knn_model is None:
        knn_model = KNNAudioClassifier(n_neighbors=15, weights='uniform', algorithm='auto')
        print("✅ KNN instance created (untrained, n_neighbors=15)")
    
    if svm_model is None:
        svm_model = SVMAudioClassifier(kernel='rbf', C=1.0, gamma='scale', max_iter=-1)
        print("✅ SVM instance created (untrained, kernel=rbf)")
    
    if xgboost_model is None:
        xgboost_model = XGBoostAudioClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, verbosity=2)
        print("✅ XGBoost instance created (untrained, n_estimators=100)")
    
    print("\n🚀 Quick Loading Augmented 5000 Dataset...")

    # This automatically loads and assigns to train_aug_safe, val_aug_safe, test_aug_safe
    train_aug_safe, val_aug_safe, test_aug_safe, info = quick_load_augmented_dataset("augmented_5000")

    # Load ALL MFCC datasets
    print("\n🚀 Loading ALL MFCC datasets...")
    print("="*70)
    
    mfcc_datasets = quick_load_mfcc_datasets()

    # Load ALL SNR datasets (spectrogram-based)
    print("\n🚀 Loading ALL SNR datasets (spectrogram-based)...")
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
    
    # Add all loaded SNR datasets (spectrogram-based)
    all_datasets.update(snr_datasets)
    
    # Add all loaded MFCC datasets
    if mfcc_datasets:
        print("\n📊 Adding MFCC datasets to dataset dictionary...")
        for mfcc_name, mfcc_data in mfcc_datasets.items():
            # Add MFCC prefix to distinguish from spectrogram datasets
            dataset_key = f"mfcc_{mfcc_name}"
            all_datasets[dataset_key] = mfcc_data
            print(f"   • Added: {dataset_key}")
    
    print("\n📊 DATASET LOADING SUMMARY:")
    print("="*70)
    print(f"✅ Total datasets loaded: {len(all_datasets)}")
    
    # Categorize datasets
    spectrogram_datasets = [k for k in all_datasets.keys() if not k.startswith('mfcc_')]
    mfcc_ds_list = [k for k in all_datasets.keys() if k.startswith('mfcc_')]
    
    if spectrogram_datasets:
        print(f"\n📈 Spectrogram-based datasets ({len(spectrogram_datasets)}):")
        for ds_name in spectrogram_datasets:
            print(f"   • {ds_name}")
    
    if mfcc_ds_list:
        print(f"\n🎵 MFCC-based datasets ({len(mfcc_ds_list)}):")
        for ds_name in mfcc_ds_list:
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
