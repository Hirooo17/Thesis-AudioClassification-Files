# SNR-Augmented MFCC Dataset Implementation

## Overview
This document explains the SNR (Signal-to-Noise Ratio) augmented MFCC dataset generation implementation for robust audio classification model training.

## What is SNR Augmentation?

**SNR (Signal-to-Noise Ratio)** measures the level of desired signal relative to background noise:
- **Higher SNR** (e.g., 20 dB) = Less noise, clearer audio
- **Lower SNR** (e.g., 5 dB) = More noise, challenging conditions

By training on datasets with different SNR levels, models become more robust to real-world noisy conditions.

---

## Implementation Details

### Key Components

1. **Noise Augmentation Function** (`preprocess_mfcc_with_noise`)
   - Loads audio file and converts to MFCC features
   - Optionally adds noise at specific SNR level
   - Maintains consistent feature dimensions (40 MFCC coefficients)

2. **SNR Dataset Creator** (`create_mfcc_snr_datasets`)
   - Generates datasets with controlled noise levels
   - Creates clean + noisy versions of audio samples
   - Ensures balanced class distribution (2500 real + 2500 fake)

3. **Noise Sample Loader** (`load_noise_samples`)
   - Loads background noise audio from `Noise_Kaggle/` directory
   - Supports up to 100 different noise samples
   - Randomly selects noise for augmentation

### SNR Levels Generated

| SNR Level | Noise Amount | Use Case |
|-----------|--------------|----------|
| 5 dB | Very High | Extreme noise conditions |
| 10 dB | High | Noisy environments |
| 15 dB | Moderate | Typical background noise |
| 20 dB | Low | Clean conditions with slight noise |

---

## How to Use

### Step 1: Prepare Noise Files

1. Collect background noise audio files (.wav format)
2. Place them in the `Noise_Kaggle/` directory
3. Recommended: 50-100 diverse noise samples (traffic, crowd, white noise, etc.)

**Example directory structure:**
```
Noise_Kaggle/
├── traffic_noise_1.wav
├── crowd_noise_1.wav
├── white_noise_1.wav
├── ambient_noise_1.wav
└── ...
```

### Step 2: Load Noise Samples

Run the noise loading cell:
```python
# This will load up to 100 noise samples
noise_samples = load_noise_samples(noise_dir='Noise_Kaggle', max_samples=100)
```

**Output:**
```
📂 Loading noise samples from: Noise_Kaggle
   Found 100 noise files
   Loaded 20/100 noise samples...
   Loaded 40/100 noise samples...
   ...
✅ Successfully loaded 100 noise samples
```

### Step 3: Generate SNR Datasets

Run the SNR generation cell:
```python
# This generates all 4 SNR levels (5, 10, 15, 20 dB)
# Estimated time: 10-30 minutes depending on system
```

**What it does:**
1. Creates MFCC features from original audio
2. Adds noise at specific SNR levels
3. Generates 5000 samples per dataset (2500 per class)
4. Saves to `mfcc_datasets/` as pickle files

**Output files:**
- `snr_5db_dataset.pkl` (~2 MB)
- `snr_10db_dataset.pkl` (~2 MB)
- `snr_15db_dataset.pkl` (~2 MB)
- `snr_20db_dataset.pkl` (~2 MB)

### Step 4: Load and Use

```python
# Load SNR dataset
train, val, test, info = load_mfcc_dataset("snr_10db", save_dir='mfcc_datasets')

# Convert to numpy for sklearn
X_train, y_train = mfcc_dataset_to_numpy(train)
X_test, y_test = mfcc_dataset_to_numpy(test)

# Train model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf.fit(X_train, y_train)

# Evaluate
accuracy = rf.score(X_test, y_test)
print(f"SNR 10dB Accuracy: {accuracy:.4f}")
```

---

## Benefits of SNR-Augmented Training

### 1. **Improved Robustness**
Models trained on noisy data perform better in real-world conditions where audio quality varies.

### 2. **Better Generalization**
Exposure to different noise levels prevents overfitting to clean audio.

### 3. **Realistic Testing**
Evaluate model performance under various noise conditions to understand limitations.

### 4. **Deployment Readiness**
Models tested on low SNR datasets are more reliable for production deployment.

---

## Technical Details

### Noise Addition Process

1. **Load original audio** (clean speech sample)
2. **Select random noise** from noise samples
3. **Match lengths** (repeat or truncate noise to match audio length)
4. **Calculate signal power**: $P_{signal} = \frac{1}{N}\sum_{i=1}^{N} x_i^2$
5. **Calculate noise power**: $P_{noise} = \frac{1}{N}\sum_{i=1}^{N} n_i^2$
6. **Compute scaling factor**:
   $$\alpha = \sqrt{\frac{P_{signal}}{P_{noise} \cdot 10^{SNR/10}}}$$
7. **Mix audio**: $y = x + \alpha \cdot n$
8. **Normalize** to prevent clipping

### MFCC Feature Extraction

After noise addition:
1. Compute STFT (Short-Time Fourier Transform)
2. Convert to Mel-scale (40 bins)
3. Apply logarithm
4. Compute DCT (Discrete Cosine Transform) → 40 MFCC coefficients
5. Average across time → Fixed 40-feature vector
6. Normalize to [0, 1] range

---

## Performance Expectations

### Training Time
- **Per SNR level**: ~5-10 minutes (5000 samples)
- **All 4 levels**: ~20-40 minutes total
- Much faster than spectrogram-based approaches

### File Sizes
- **Each SNR dataset**: ~2 MB (vs ~800 MB for spectrograms)
- **Total storage**: ~8 MB for all 4 SNR levels
- **400× reduction** compared to spectrogram datasets

### Model Accuracy Trends
- **SNR 20 dB**: Highest accuracy (~90-95%)
- **SNR 15 dB**: Good accuracy (~85-90%)
- **SNR 10 dB**: Moderate accuracy (~75-85%)
- **SNR 5 dB**: Challenging (~65-75%)

*Actual results depend on model architecture and training parameters*

---

## Troubleshooting

### Issue: "No noise samples loaded"
**Cause**: Noise directory not found or empty  
**Solution**:
1. Check that `Noise_Kaggle/` directory exists
2. Ensure it contains .wav files
3. Update `noise_dir` parameter if using different location

### Issue: "Datasets created without noise augmentation"
**Cause**: `noise_samples` list is empty  
**Solution**: Run the noise loading cell successfully before generation

### Issue: "Generation taking too long"
**Cause**: Large number of samples or slow disk I/O  
**Solution**:
- Reduce `samples_per_class` parameter (e.g., to 1000)
- Close other applications to free system resources
- Use SSD instead of HDD for faster processing

### Issue: "NaN or Inf values in features"
**Cause**: Invalid audio samples or extreme noise levels  
**Solution**:
- Validation checks automatically skip invalid samples
- Check audio file integrity
- Ensure noise samples are clean recordings

---

## Comparison with Spectrogram Approach

| Aspect | Spectrograms | MFCC (SNR-Augmented) |
|--------|--------------|----------------------|
| **Features per sample** | 16,384 (128×128) | 40 |
| **File size per dataset** | ~800 MB | ~2 MB |
| **Training time (RF)** | 30-60 min | 2-5 min |
| **Noise robustness** | Moderate | Excellent |
| **Memory usage** | High | Low |
| **Best for** | CNNs, Deep Learning | Tree-based, Traditional ML |

---

## Advanced Usage

### Custom SNR Levels
```python
# Generate dataset with custom SNR level (e.g., 12 dB)
train, val, test = create_mfcc_snr_datasets(
    train_path='data/for-2sec/for-2seconds/training',
    val_path='data/for-2sec/for-2seconds/validation',
    test_path='data/for-2sec/for-2seconds/testing',
    noise_samples=noise_samples,
    snr_level=12,  # Custom SNR
    samples_per_class=2500,
    n_mfcc=40
)

# Save with custom name
save_mfcc_dataset(train, val, test, 
                  dataset_name="snr_12db_custom",
                  save_dir='mfcc_datasets')
```

### Cross-SNR Training/Testing
```python
# Train on high SNR, test on low SNR (robustness check)
train_20, _, _, _ = load_mfcc_dataset("snr_20db")
_, _, test_5, _ = load_mfcc_dataset("snr_5db")

X_train, y_train = mfcc_dataset_to_numpy(train_20)
X_test, y_test = mfcc_dataset_to_numpy(test_5)

rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf.fit(X_train, y_train)

accuracy = rf.score(X_test, y_test)
print(f"Trained on 20dB, tested on 5dB: {accuracy:.4f}")
# Lower accuracy expected - tests model robustness!
```

### Mixed SNR Training
```python
# Combine multiple SNR levels for ultra-robust training
X_train_combined, y_train_combined = [], []

for snr in [5, 10, 15, 20]:
    train, _, _, _ = load_mfcc_dataset(f"snr_{snr}db")
    X, y = mfcc_dataset_to_numpy(train)
    X_train_combined.extend(X)
    y_train_combined.extend(y)

X_train_combined = np.array(X_train_combined)
y_train_combined = np.array(y_train_combined)

# Train on combined dataset
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
rf.fit(X_train_combined, y_train_combined)

print(f"Trained on {len(X_train_combined)} samples from all SNR levels!")
```

---

## Best Practices

1. **Start with clean datasets** - Generate base MFCC datasets first
2. **Use diverse noise** - More variety = better robustness
3. **Test across SNR levels** - Evaluate on all levels to understand limits
4. **Consider deployment environment** - Match training SNR to expected conditions
5. **Monitor validation performance** - Avoid overfitting to specific noise patterns
6. **Document noise sources** - Track which noise files were used for reproducibility

---

## Research Applications

### 1. Robustness Analysis
- Compare model performance across SNR levels
- Identify failure modes at low SNR
- Quantify degradation curves

### 2. Deployment Planning
- Determine minimum SNR for reliable operation
- Set confidence thresholds based on noise level
- Plan preprocessing requirements

### 3. Model Comparison
- Benchmark different algorithms under noise
- Compare CNN vs tree-based approaches
- Evaluate feature extraction methods

### 4. Data Augmentation Studies
- Test impact of noise augmentation on accuracy
- Compare with other augmentation techniques
- Optimize training data composition

---

## Citation & References

**MFCC (Mel-Frequency Cepstral Coefficients)**
- Davis, S., & Mermelstein, P. (1980). "Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences." IEEE Transactions on Acoustics, Speech, and Signal Processing.

**SNR in Audio Processing**
- Martin, R. (2001). "Noise power spectral density estimation based on optimal smoothing and minimum statistics." IEEE Transactions on Speech and Audio Processing.

---

**Last Updated**: November 2025  
**Version**: 1.0  
**Implemented for**: Audio Classification Thesis Project
