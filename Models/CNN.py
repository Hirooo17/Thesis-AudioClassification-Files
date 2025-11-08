"""
CNN Model for Audio Classification
Binary classification: Real vs Fake Audio
Uses mel-spectrogram features (128x128x1)
"""

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np


class CNNAudioClassifier:
    """CNN Model for Audio Classification matching notebook architecture"""
    
    def __init__(self, input_shape=(128, 128, 1)):
        """
        Initialize CNN model with 128x128 mel-spectrogram input
        
        Args:
            input_shape: Shape of input spectrograms (default: 128x128x1)
        """
        self.input_shape = input_shape
        self.model = None
        self.history = None
        self.is_compiled = False
        self.is_fitted = False
        
    def build_model(self, conv_filters=[16, 32, 64], dense_units=[128, 64], 
                   conv_dropout=[0.0, 0.0, 0.0], dense_dropout=[0.3, 0.2],
                   use_l2=False, l2_strength=0.001):
        """
        Build the CNN architecture with customizable parameters
        
        Args:
            conv_filters: List of filter counts for conv layers [layer1, layer2, layer3]
            dense_units: List of units for dense layers [layer1, layer2] (0 to skip)
            conv_dropout: Dropout after each conv block [block1, block2, block3]
            dense_dropout: Dropout for dense layers [layer1, layer2]
            use_l2: Whether to use L2 regularization
            l2_strength: L2 regularization strength
        """
        from tensorflow.keras.regularizers import l2
        from tensorflow.keras.initializers import HeNormal, GlorotUniform
        
        # Regularizer (only if enabled and strength is reasonable)
        regularizer = l2(l2_strength) if use_l2 and l2_strength > 0 else None
        
        # He initialization for ReLU layers (better than default)
        he_init = HeNormal(seed=42)
        glorot_init = GlorotUniform(seed=42)
        
        model = Sequential()
        
        print(f"\n🏗️ Building CNN with custom architecture:")
        print(f"   Conv filters: {conv_filters}")
        print(f"   Dense units: {dense_units}")
        print(f"   Conv dropout: {conv_dropout}")
        print(f"   Dense dropout: {dense_dropout}")
        print(f"   L2 regularization: {use_l2} (strength: {l2_strength})")
        print(f"   Using He initialization for better convergence")
        
        # Layer 1: First Convolution Block
        model.add(Conv2D(conv_filters[0], (3, 3), activation='relu', 
                        input_shape=self.input_shape,
                        kernel_initializer=he_init,
                        kernel_regularizer=regularizer,
                        padding='same',
                        name='conv1'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), name='pool1'))
        if conv_dropout[0] > 0:
            model.add(Dropout(conv_dropout[0], name='dropout_conv1'))
        
        # Layer 2: Second Convolution Block
        model.add(Conv2D(conv_filters[1], (3, 3), activation='relu',
                        kernel_initializer=he_init,
                        kernel_regularizer=regularizer,
                        padding='same',
                        name='conv2'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))
        if conv_dropout[1] > 0:
            model.add(Dropout(conv_dropout[1], name='dropout_conv2'))
        
        # Layer 3: Third Convolution Block
        model.add(Conv2D(conv_filters[2], (3, 3), activation='relu',
                        kernel_initializer=he_init,
                        kernel_regularizer=regularizer,
                        padding='same',
                        name='conv3'))
        model.add(BatchNormalization())
        if conv_dropout[2] > 0:
            model.add(Dropout(conv_dropout[2], name='dropout_conv3'))
        
        # Flatten for dense layers
        model.add(Flatten())
        
        # Dense layers (dynamic - can be removed by setting to 0)
        if dense_units[0] > 0:
            model.add(Dense(dense_units[0], activation='relu',
                           kernel_initializer=he_init,
                           kernel_regularizer=regularizer,
                           name='dense1'))
            if dense_dropout[0] > 0:
                model.add(Dropout(dense_dropout[0], name='dropout_dense1'))
        
        if len(dense_units) > 1 and dense_units[1] > 0:
            model.add(Dense(dense_units[1], activation='relu',
                           kernel_initializer=he_init,
                           kernel_regularizer=regularizer,
                           name='dense2'))
            if dense_dropout[1] > 0:
                model.add(Dropout(dense_dropout[1], name='dropout_dense2'))
        
        # Output layer with Glorot initialization (better for sigmoid)
        model.add(Dense(1, activation='sigmoid', 
                       kernel_initializer=glorot_init,
                       bias_initializer=tf.keras.initializers.Constant(0.0),
                       name='output'))
        
        self.model = model
        print(f"✅ Model built with {model.count_params():,} parameters")
        return model
    
    def compile(self, optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy', 'precision', 'recall']):
        """
        Compile the model
        
        Args:
            optimizer: Optimizer to use (default: adam) - can be string or optimizer instance
            loss: Loss function (default: binary_crossentropy)
            metrics: List of metrics to track (now includes precision and recall)
        """
        if self.model is None:
            raise ValueError("Model must be built before compilation! Call build_model() first.")
        
        # If optimizer is a learning rate (float), create Adam with that LR
        if isinstance(optimizer, float):
            from tensorflow.keras.optimizers import Adam
            optimizer = Adam(learning_rate=optimizer, clipnorm=1.0)
            print(f"   Using Adam optimizer with LR={optimizer.learning_rate.numpy():.6f} and gradient clipping")
        elif isinstance(optimizer, str) and optimizer.lower() == 'adam':
            from tensorflow.keras.optimizers import Adam
            # Use a safer default learning rate with gradient clipping
            optimizer = Adam(learning_rate=0.0005, clipnorm=1.0)
            print(f"   Using Adam optimizer with LR=0.0005 and gradient clipping (safer default)")
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        self.is_compiled = True
        print("✅ CNN Model compiled successfully!")
        print(f"   Input shape: {self.input_shape}")
        print(f"   Output: Binary classification (Real vs Fake)")
        print(f"   Loss: {loss}")
        print(f"   Metrics: {', '.join([str(m) if isinstance(m, str) else m.name for m in metrics])}")
    
    def fit(self, train_data, epochs=10, validation_data=None, callbacks=None, verbose=1):
        """
        Train the model
        
        Args:
            train_data: Training dataset
            epochs: Number of epochs to train
            validation_data: Validation dataset
            callbacks: List of callbacks
            verbose: Verbosity level
        
        Returns:
            Training history
        """
        if not self.is_compiled:
            raise ValueError("Model must be compiled before training! Call model.compile() first.")
        
        # Default callbacks
        if callbacks is None:
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
        
        print(f"\n🏋️ Training CNN Model for {epochs} epochs...")
        print(f"   Training samples: {sum(1 for _ in train_data.unbatch())}")
        if validation_data:
            print(f"   Validation samples: {sum(1 for _ in validation_data.unbatch())}")
        
        # Train the model
        self.history = self.model.fit(
            train_data,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_fitted = True
        
        # Print training summary
        print(f"\n✅ Training completed!")
        if validation_data:
            final_val_acc = self.history.history['val_binary_accuracy'][-1]
            final_val_loss = self.history.history['val_loss'][-1]
            print(f"   Final Validation Accuracy: {final_val_acc:.4f}")
            print(f"   Final Validation Loss: {final_val_loss:.4f}")
        
        return self.history
    
    def predict(self, X_test, batch_size=32):
        """
        Make predictions on test data
        
        Args:
            X_test: Test data (numpy array or tf.data.Dataset)
            batch_size: Batch size for prediction
        
        Returns:
            Predictions (probabilities)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions!")
        
        # Handle different input types
        if isinstance(X_test, tf.data.Dataset):
            predictions = self.model.predict(X_test, verbose=0)
        else:
            predictions = self.model.predict(X_test, batch_size=batch_size, verbose=0)
        
        return predictions
    
    def evaluate(self, test_data, verbose=1):
        """
        Evaluate the model on test data
        
        Args:
            test_data: Test dataset
            verbose: Verbosity level
        
        Returns:
            Test loss and metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation!")
        
        results = self.model.evaluate(test_data, verbose=verbose)
        
        print(f"\n📊 Test Results:")
        print(f"   Test Loss: {results[0]:.4f}")
        print(f"   Test Accuracy: {results[1]:.4f}")
        
        return results
    
    def summary(self):
        """Print model architecture summary"""
        if self.model is None:
            self.build_model()
        
        print("\n" + "="*70)
        print("CNN MODEL ARCHITECTURE")
        print("="*70)
        self.model.summary()
        print("="*70)
    
    def save_model(self, filepath):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving!")
        
        self.model.save(filepath)
        print(f"✅ Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """
        Load a saved model
        
        Args:
            filepath: Path to the saved model
        """
        self.model = tf.keras.models.load_model(filepath)
        self.is_compiled = True
        self.is_fitted = True
        print(f"✅ Model loaded from: {filepath}")


# Helper function to create and compile a CNN model
def create_cnn_model(input_shape=(128, 128, 1), optimizer='adam'):
    """
    Quick function to create and compile a CNN model
    
    Args:
        input_shape: Input shape for spectrograms
        optimizer: Optimizer to use
    
    Returns:
        Compiled CNN model
    """
    model = CNNAudioClassifier(input_shape=input_shape)
    model.compile(optimizer=optimizer)
    return model
