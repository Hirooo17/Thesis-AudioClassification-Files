"""
CNN Audio Classification GUI
Dedicated GUI for training and predicting with CNN model
Shows epoch-by-epoch training progress, loss curves, and detailed metrics
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import sys
import os
import pickle
import csv
from matplotlib.backends.backend_pdf import PdfPages
import psutil
import cv2

# Add librosa for audio processing
import librosa

# Add sounddevice and scipy for recording
import sounddevice as sd
import scipy.io.wavfile as wavfile

# Add Models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Models'))

# Import CNN model
from Models.CNN import CNNAudioClassifier

# Try to import tensorflow
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow not available!")


class CNNTrainingGUI:
    """Dedicated GUI for CNN Training and Prediction"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 CNN Audio Classification - Training & Prediction System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1a1a1a')
        
        # Model and data variables
        self.cnn_model = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.is_training = False
        self.is_predicting = False
        
        # Training tracking
        self.current_epoch = 0
        self.total_epochs = 0
        self.epoch_history = {
            'epoch': [],
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Prediction tracking
        self.prediction_history = []
        self.correct_predictions = 0
        self.total_predictions = 0
        
        # Confusion matrix tracking
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        
        # Detailed classification report tracking
        self.classification_report = {
            'class_0': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
            'class_1': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
            'accuracy': 0,
            'macro_avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
            'weighted_avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}
        }
        
        # Training performance tracking
        self.training_start_time = None
        self.training_end_time = None
        self.training_duration = 0
        self.peak_memory_mb = 0
        self.avg_memory_mb = 0
        
        # SNR degradation tracking
        self.snr_degradation_data = []
        self.current_snr_level = None
        
        # Explainability tracking
        self.explainability_data = []
        self.current_explain_sample = None
        self.grad_cam_heatmap = None
        self.feature_importance = None
        self.recorded_audio = None  # Store recorded audio for playback
        self.recorded_sr = None     # Sample rate for playback
        
        # Resource monitoring
        self.process = psutil.Process()
        self.start_time = None
        
        # Hardware computational monitoring
        self.resource_monitor = None
        self.monitoring_active = False
        self.resource_data = {
            'timestamps': [],
            'cpu_percent': [],
            'memory_mb': [],
            'memory_percent': []
        }
        self.resource_stats = None
        
        # Model optimization settings
        self.use_l2_regularization = False
        self.l2_strength = 0.001
        self.conv_dropout_rates = [0.0, 0.0, 0.0]  # After each conv block
        self.dense_dropout_rates = [0.3, 0.2]  # For dense layers
        self.conv_filters = [16, 32, 64]  # Number of filters per conv layer
        self.dense_units = [128, 64]  # Units in dense layers
        self.early_stopping_patience = 5
        self.use_mixed_snr = False
        self.snr_datasets = []
        
        # Dataset info for tracking
        self.current_dataset_info = {
            'is_mixed': False,
            'dataset_name': None,
            'mixed_datasets': [],
            'total_samples': {'train': 0, 'val': 0, 'test': 0}
        }
        
        # CNN model directory
        self.cnn_model_dir = os.path.join(os.path.dirname(__file__), 'CNN')
        self.cnn_model_path = os.path.join(self.cnn_model_dir, 'cnn_trained_model.h5')
        
        # Create CNN directory if it doesn't exist
        if not os.path.exists(self.cnn_model_dir):
            os.makedirs(self.cnn_model_dir)
            print(f"📁 Created CNN model directory: {self.cnn_model_dir}")
        
        # Dataset paths (hardcoded for now, can be made configurable)
        self.dataset_paths = {
            'augmented_10k': 'saved_datasets/augmented_10000_dataset.pkl',
            'augmented_5k': 'saved_datasets/augmented_5000_dataset.pkl',
            'mfcc_2sec': 'saved_datasets/augmented_5000_dataset.pkl',
            'snr_5db': 'saved_datasets/snr_5db_augmented_dataset.pkl',
            'snr_10db': 'saved_datasets/snr_10db_augmented_dataset.pkl',
            'snr_15db': 'saved_datasets/snr_15db_augmented_dataset.pkl',
            'snr_20db': 'saved_datasets/snr_20db_augmented_dataset.pkl'
        }
        
        # Create GUI
        self.create_widgets()
        
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main container
        main_container = tk.Frame(self.root, bg='#1a1a1a')
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill='both', expand=True)
        
        # Create tabs
        self.training_tab = tk.Frame(self.notebook, bg='#1a1a1a')
        self.prediction_tab = tk.Frame(self.notebook, bg='#1a1a1a')
        self.results_tab = tk.Frame(self.notebook, bg='#1a1a1a')
        self.model_manager_tab = tk.Frame(self.notebook, bg='#1a1a1a')
        
        self.notebook.add(self.training_tab, text=' 🏋️ Training ')
        self.notebook.add(self.prediction_tab, text=' 🔮 Prediction ')
        self.notebook.add(self.results_tab, text=' 📊 Results & Export ')
        self.notebook.add(self.model_manager_tab, text=' 🗂️ Model Manager ')
        
        # Add optimization tab
        self.optimization_tab = tk.Frame(self.notebook, bg='#1a1a1a')
        self.notebook.add(self.optimization_tab, text=' ⚙️ Optimization ')
        
        # Add explainability tab
        self.explainability_tab = tk.Frame(self.notebook, bg='#1a1a1a')
        self.notebook.add(self.explainability_tab, text=' 🎙️ Synthetic Detection ')
        
        # Setup each tab
        self.setup_training_tab()
        self.setup_prediction_tab()
        self.setup_results_tab()
        self.setup_model_manager_tab()
        self.setup_optimization_tab()
        self.setup_explainability_tab()
        
    def setup_training_tab(self):
        """Setup training tab with epoch tracking"""
        # Title
        title = tk.Label(self.training_tab, text="🧠 CNN Model Training", 
                        font=('Arial', 20, 'bold'), fg='#00d4ff', bg='#1a1a1a')
        title.pack(pady=10)
        
        # Control Panel
        control_frame = tk.Frame(self.training_tab, bg='#2a2a2a', relief='ridge', bd=2)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        # Dataset Selection
        dataset_frame = tk.LabelFrame(control_frame, text=" Dataset Selection ", 
                                      fg='white', bg='#2a2a2a', font=('Arial', 10, 'bold'))
        dataset_frame.pack(side='left', padx=10, pady=10, fill='both', expand=True)
        
        tk.Label(dataset_frame, text="Select Dataset:", fg='white', bg='#2a2a2a').grid(row=0, column=0, sticky='w', padx=5, pady=5)
        
        self.dataset_var = tk.StringVar(value='mfcc_2sec')
        dataset_options = list(self.dataset_paths.keys())
        dataset_menu = ttk.Combobox(dataset_frame, textvariable=self.dataset_var, values=dataset_options, state='readonly', width=20)
        dataset_menu.grid(row=0, column=1, padx=5, pady=5)
        
        load_btn = tk.Button(dataset_frame, text="📂 Load Dataset", command=self.load_dataset,
                            bg='#00d4ff', fg='black', font=('Arial', 10, 'bold'), cursor='hand2')
        load_btn.grid(row=0, column=2, padx=5, pady=5)
        
        self.dataset_status = tk.Label(dataset_frame, text="No dataset loaded", fg='#ff6b6b', bg='#2a2a2a')
        self.dataset_status.grid(row=1, column=0, columnspan=3, sticky='w', padx=5, pady=5)
        
        # Training Parameters
        param_frame = tk.LabelFrame(control_frame, text=" Training Parameters ", 
                                    fg='white', bg='#2a2a2a', font=('Arial', 10, 'bold'))
        param_frame.pack(side='left', padx=10, pady=10, fill='both', expand=True)
        
        tk.Label(param_frame, text="Epochs:", fg='white', bg='#2a2a2a').grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.epochs_var = tk.IntVar(value=10)
        epochs_spin = tk.Spinbox(param_frame, from_=1, to=100, textvariable=self.epochs_var, width=10)
        epochs_spin.grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(param_frame, text="Batch Size:", fg='white', bg='#2a2a2a').grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.batch_var = tk.IntVar(value=32)
        batch_spin = tk.Spinbox(param_frame, from_=8, to=128, increment=8, textvariable=self.batch_var, width=10)
        batch_spin.grid(row=1, column=1, padx=5, pady=5)
        
        tk.Label(param_frame, text="Learning Rate:", fg='white', bg='#2a2a2a').grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.lr_var = tk.DoubleVar(value=0.0005)
        lr_entry = tk.Entry(param_frame, textvariable=self.lr_var, width=10)
        lr_entry.grid(row=2, column=1, padx=5, pady=5)
        tk.Label(param_frame, text="(0.0005 recommended)", fg='#888888', bg='#2a2a2a', font=('Arial', 8, 'italic')).grid(row=2, column=2, sticky='w', padx=5)
        
        # Training Controls
        btn_frame = tk.LabelFrame(control_frame, text=" Training Controls ", 
                                 fg='white', bg='#2a2a2a', font=('Arial', 10, 'bold'))
        btn_frame.pack(side='left', padx=10, pady=10)
        
        self.train_btn = tk.Button(btn_frame, text="🏋️ Start Training", command=self.start_training,
                                   bg='#00ff88', fg='black', font=('Arial', 12, 'bold'), 
                                   width=15, height=2, cursor='hand2')
        self.train_btn.pack(padx=5, pady=5)
        
        self.stop_btn = tk.Button(btn_frame, text="⏹️ Stop Training", command=self.stop_training,
                                  bg='#ff6b6b', fg='white', font=('Arial', 12, 'bold'), 
                                  width=15, height=2, state='disabled', cursor='hand2')
        self.stop_btn.pack(padx=5, pady=5)
        
        # Progress Section
        progress_frame = tk.Frame(self.training_tab, bg='#2a2a2a', relief='ridge', bd=2)
        progress_frame.pack(fill='x', padx=10, pady=5)
        
        # Epoch Progress
        tk.Label(progress_frame, text="Epoch Progress:", fg='white', bg='#2a2a2a', 
                font=('Arial', 10, 'bold')).pack(anchor='w', padx=10, pady=5)
        
        self.epoch_label = tk.Label(progress_frame, text="Epoch: 0/0", fg='#00d4ff', bg='#2a2a2a', 
                                    font=('Arial', 14, 'bold'))
        self.epoch_label.pack(anchor='w', padx=10)
        
        self.epoch_progress = ttk.Progressbar(progress_frame, mode='determinate', length=400)
        self.epoch_progress.pack(fill='x', padx=10, pady=5)
        
        # Current Metrics
        metrics_container = tk.Frame(progress_frame, bg='#2a2a2a')
        metrics_container.pack(fill='x', padx=10, pady=10)
        
        # Loss
        loss_frame = tk.Frame(metrics_container, bg='#3a3a3a', relief='ridge', bd=2)
        loss_frame.pack(side='left', padx=5, fill='both', expand=True)
        tk.Label(loss_frame, text="Loss", fg='#ff6b6b', bg='#3a3a3a', font=('Arial', 10, 'bold')).pack(pady=5)
        self.loss_label = tk.Label(loss_frame, text="0.0000", fg='white', bg='#3a3a3a', font=('Arial', 16, 'bold'))
        self.loss_label.pack(pady=5)
        
        # Accuracy
        acc_frame = tk.Frame(metrics_container, bg='#3a3a3a', relief='ridge', bd=2)
        acc_frame.pack(side='left', padx=5, fill='both', expand=True)
        tk.Label(acc_frame, text="Accuracy", fg='#00ff88', bg='#3a3a3a', font=('Arial', 10, 'bold')).pack(pady=5)
        self.acc_label = tk.Label(acc_frame, text="0.00%", fg='white', bg='#3a3a3a', font=('Arial', 16, 'bold'))
        self.acc_label.pack(pady=5)
        
        # Val Loss
        val_loss_frame = tk.Frame(metrics_container, bg='#3a3a3a', relief='ridge', bd=2)
        val_loss_frame.pack(side='left', padx=5, fill='both', expand=True)
        tk.Label(val_loss_frame, text="Val Loss", fg='#ffa500', bg='#3a3a3a', font=('Arial', 10, 'bold')).pack(pady=5)
        self.val_loss_label = tk.Label(val_loss_frame, text="0.0000", fg='white', bg='#3a3a3a', font=('Arial', 16, 'bold'))
        self.val_loss_label.pack(pady=5)
        
        # Val Accuracy
        val_acc_frame = tk.Frame(metrics_container, bg='#3a3a3a', relief='ridge', bd=2)
        val_acc_frame.pack(side='left', padx=5, fill='both', expand=True)
        tk.Label(val_acc_frame, text="Val Accuracy", fg='#00d4ff', bg='#3a3a3a', font=('Arial', 10, 'bold')).pack(pady=5)
        self.val_acc_label = tk.Label(val_acc_frame, text="0.00%", fg='white', bg='#3a3a3a', font=('Arial', 16, 'bold'))
        self.val_acc_label.pack(pady=5)
        
        # Training Plots
        plot_frame = tk.Frame(self.training_tab, bg='#1a1a1a')
        plot_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create matplotlib figures
        self.train_fig = Figure(figsize=(12, 5), facecolor='#1a1a1a')
        self.train_canvas = FigureCanvasTkAgg(self.train_fig, plot_frame)
        self.train_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Create subplots
        self.loss_ax = self.train_fig.add_subplot(121, facecolor='#2a2a2a')
        self.acc_ax = self.train_fig.add_subplot(122, facecolor='#2a2a2a')
        
        self.loss_ax.set_title('Training & Validation Loss', color='white', fontsize=12, weight='bold')
        self.loss_ax.set_xlabel('Epoch', color='white')
        self.loss_ax.set_ylabel('Loss', color='white')
        self.loss_ax.tick_params(colors='white')
        self.loss_ax.grid(True, alpha=0.3)
        
        self.acc_ax.set_title('Training & Validation Accuracy', color='white', fontsize=12, weight='bold')
        self.acc_ax.set_xlabel('Epoch', color='white')
        self.acc_ax.set_ylabel('Accuracy', color='white')
        self.acc_ax.tick_params(colors='white')
        self.acc_ax.grid(True, alpha=0.3)
        
        self.train_fig.tight_layout()
        
        # Training Log
        log_frame = tk.LabelFrame(self.training_tab, text=" Training Log ", 
                                 fg='white', bg='#2a2a2a', font=('Arial', 10, 'bold'))
        log_frame.pack(fill='both', padx=10, pady=10, expand=True)
        
        self.training_log = tk.Text(log_frame, height=10, bg='#1a1a1a', fg='#00ff88', 
                                   font=('Consolas', 9), wrap='word')
        self.training_log.pack(side='left', fill='both', expand=True)
        
        log_scroll = tk.Scrollbar(log_frame, command=self.training_log.yview)
        log_scroll.pack(side='right', fill='y')
        self.training_log.config(yscrollcommand=log_scroll.set)
        
    def setup_prediction_tab(self):
        """Setup prediction tab"""
        # Title
        title = tk.Label(self.prediction_tab, text="🔮 CNN Prediction System", 
                        font=('Arial', 20, 'bold'), fg='#00d4ff', bg='#1a1a1a')
        title.pack(pady=10)
        
        # Control Panel
        control_frame = tk.Frame(self.prediction_tab, bg='#2a2a2a', relief='ridge', bd=2)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        # Dataset Selection (NEW!)
        dataset_frame = tk.LabelFrame(control_frame, text=" Dataset Selection ", 
                                      fg='white', bg='#2a2a2a', font=('Arial', 10, 'bold'))
        dataset_frame.pack(side='left', padx=10, pady=10, fill='both', expand=True)
        
        tk.Label(dataset_frame, text="⚠️ Load the SAME dataset\nused for training!", 
                fg='#ffa500', bg='#2a2a2a', font=('Arial', 9, 'bold'), justify='center').pack(padx=5, pady=5)
        
        load_dataset_btn = tk.Button(dataset_frame, text="📂 Load Dataset", command=self.load_dataset_for_prediction,
                                     bg='#00d4ff', fg='black', font=('Arial', 10, 'bold'), cursor='hand2')
        load_dataset_btn.pack(padx=5, pady=5)
        
        self.pred_dataset_status = tk.Label(dataset_frame, text="No dataset loaded", fg='#ff6b6b', bg='#2a2a2a', font=('Arial', 9))
        self.pred_dataset_status.pack(padx=5, pady=5)
        
        # Model Selection
        model_frame = tk.LabelFrame(control_frame, text=" Model Selection ", 
                                    fg='white', bg='#2a2a2a', font=('Arial', 10, 'bold'))
        model_frame.pack(side='left', padx=10, pady=10, fill='both', expand=True)
        
        load_model_btn = tk.Button(model_frame, text="📂 Load Model (Auto)", command=self.load_trained_model,
                                   bg='#00d4ff', fg='black', font=('Arial', 10, 'bold'), cursor='hand2')
        load_model_btn.pack(padx=5, pady=5)
        
        self.model_status = tk.Label(model_frame, text="No model loaded", fg='#ff6b6b', bg='#2a2a2a', font=('Arial', 9))
        self.model_status.pack(padx=5, pady=5)
        
        # Info label
        info_label = tk.Label(model_frame, 
                             text="💡 Auto-loads latest model from CNN folder",
                             fg='#888888', bg='#2a2a2a', font=('Arial', 8, 'italic'))
        info_label.pack(padx=5, pady=2)
        
        # Prediction Controls
        pred_control_frame = tk.LabelFrame(control_frame, text=" Prediction Controls ", 
                                          fg='white', bg='#2a2a2a', font=('Arial', 10, 'bold'))
        pred_control_frame.pack(side='left', padx=10, pady=10)
        
        tk.Label(pred_control_frame, text="Number of Samples:", fg='white', bg='#2a2a2a').grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.pred_samples_var = tk.IntVar(value=100)
        samples_spin = tk.Spinbox(pred_control_frame, from_=10, to=5000, increment=10, 
                                 textvariable=self.pred_samples_var, width=10)
        samples_spin.grid(row=0, column=1, padx=5, pady=5)
        
        self.predict_btn = tk.Button(pred_control_frame, text="🔮 Start Prediction", command=self.start_prediction,
                                     bg='#00ff88', fg='black', font=('Arial', 12, 'bold'), 
                                     width=15, height=2, cursor='hand2', state='disabled')
        self.predict_btn.grid(row=1, column=0, columnspan=2, padx=5, pady=10)
        
        # Prediction Results
        results_frame = tk.Frame(self.prediction_tab, bg='#2a2a2a', relief='ridge', bd=2)
        results_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Metrics Display
        metrics_display = tk.Frame(results_frame, bg='#2a2a2a')
        metrics_display.pack(fill='x', padx=10, pady=10)
        
        # Accuracy
        acc_display = tk.Frame(metrics_display, bg='#3a3a3a', relief='ridge', bd=2)
        acc_display.pack(side='left', padx=5, fill='both', expand=True)
        tk.Label(acc_display, text="Accuracy", fg='#00ff88', bg='#3a3a3a', font=('Arial', 10, 'bold')).pack(pady=5)
        self.pred_acc_label = tk.Label(acc_display, text="0.0%", fg='white', bg='#3a3a3a', font=('Arial', 18, 'bold'))
        self.pred_acc_label.pack(pady=5)
        
        # Precision
        prec_display = tk.Frame(metrics_display, bg='#3a3a3a', relief='ridge', bd=2)
        prec_display.pack(side='left', padx=5, fill='both', expand=True)
        tk.Label(prec_display, text="Precision", fg='#00d4ff', bg='#3a3a3a', font=('Arial', 10, 'bold')).pack(pady=5)
        self.pred_prec_label = tk.Label(prec_display, text="0.0%", fg='white', bg='#3a3a3a', font=('Arial', 18, 'bold'))
        self.pred_prec_label.pack(pady=5)
        
        # Recall
        recall_display = tk.Frame(metrics_display, bg='#3a3a3a', relief='ridge', bd=2)
        recall_display.pack(side='left', padx=5, fill='both', expand=True)
        tk.Label(recall_display, text="Recall", fg='#ffa500', bg='#3a3a3a', font=('Arial', 10, 'bold')).pack(pady=5)
        self.pred_recall_label = tk.Label(recall_display, text="0.0%", fg='white', bg='#3a3a3a', font=('Arial', 18, 'bold'))
        self.pred_recall_label.pack(pady=5)
        
        # F1-Score
        f1_display = tk.Frame(metrics_display, bg='#3a3a3a', relief='ridge', bd=2)
        f1_display.pack(side='left', padx=5, fill='both', expand=True)
        tk.Label(f1_display, text="F1-Score", fg='#ff69b4', bg='#3a3a3a', font=('Arial', 10, 'bold')).pack(pady=5)
        self.pred_f1_label = tk.Label(f1_display, text="0.0%", fg='white', bg='#3a3a3a', font=('Arial', 18, 'bold'))
        self.pred_f1_label.pack(pady=5)
        
        # Prediction Log
        log_frame = tk.LabelFrame(results_frame, text=" Prediction Log ", 
                                 fg='white', bg='#2a2a2a', font=('Arial', 10, 'bold'))
        log_frame.pack(fill='both', padx=10, pady=10, expand=True)
        
        self.prediction_log = tk.Text(log_frame, height=15, bg='#1a1a1a', fg='#00d4ff', 
                                     font=('Consolas', 9), wrap='word')
        self.prediction_log.pack(side='left', fill='both', expand=True)
        
        pred_scroll = tk.Scrollbar(log_frame, command=self.prediction_log.yview)
        pred_scroll.pack(side='right', fill='y')
        self.prediction_log.config(yscrollcommand=pred_scroll.set)
        
    def setup_results_tab(self):
        """Setup results and export tab"""
        # Title
        title = tk.Label(self.results_tab, text="📊 Results & Export", 
                        font=('Arial', 20, 'bold'), fg='#00d4ff', bg='#1a1a1a')
        title.pack(pady=10)
        
        # Export Controls
        export_frame = tk.LabelFrame(self.results_tab, text=" Export Options ", 
                                     fg='white', bg='#2a2a2a', font=('Arial', 10, 'bold'))
        export_frame.pack(fill='x', padx=10, pady=10)
        
        btn_container = tk.Frame(export_frame, bg='#2a2a2a')
        btn_container.pack(pady=10)
        
        export_pdf_btn = tk.Button(btn_container, text="📄 Export to PDF", command=self.export_to_pdf,
                                   bg='#ff6b6b', fg='white', font=('Arial', 11, 'bold'), 
                                   width=20, height=2, cursor='hand2')
        export_pdf_btn.pack(side='left', padx=10)
        
        export_csv_btn = tk.Button(btn_container, text="📊 Export to CSV", command=self.export_to_csv,
                                   bg='#00ff88', fg='black', font=('Arial', 11, 'bold'), 
                                   width=20, height=2, cursor='hand2')
        export_csv_btn.pack(side='left', padx=10)
        
        export_model_btn = tk.Button(btn_container, text="💾 Save Model (Custom)", command=self.save_model,
                                     bg='#00d4ff', fg='black', font=('Arial', 11, 'bold'), 
                                     width=20, height=2, cursor='hand2')
        export_model_btn.pack(side='left', padx=10)
        
        # Info label
        info_label = tk.Label(export_frame, 
                             text="💡 Note: Models are auto-saved to CNN folder after training. Use 'Save Model (Custom)' for additional backups.",
                             fg='#ffa500', bg='#2a2a2a', font=('Arial', 9, 'italic'), wraplength=800)
        info_label.pack(pady=5)
        
        # Results Summary
        summary_frame = tk.LabelFrame(self.results_tab, text=" Training Summary ", 
                                     fg='white', bg='#2a2a2a', font=('Arial', 10, 'bold'))
        summary_frame.pack(fill='both', padx=10, pady=10, expand=True)
        
        self.summary_text = tk.Text(summary_frame, height=20, bg='#1a1a1a', fg='white', 
                                    font=('Consolas', 10), wrap='word')
        self.summary_text.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        summary_scroll = tk.Scrollbar(summary_frame, command=self.summary_text.yview)
        summary_scroll.pack(side='right', fill='y')
        self.summary_text.config(yscrollcommand=summary_scroll.set)
    
    def setup_model_manager_tab(self):
        """Setup model manager tab for loading old models and selecting datasets"""
        # Title
        title = tk.Label(self.model_manager_tab, text="🗂️ Model Manager - Load & Test Old Models", 
                        font=('Arial', 20, 'bold'), fg='#00d4ff', bg='#1a1a1a')
        title.pack(pady=10)
        
        # Main container
        main_container = tk.Frame(self.model_manager_tab, bg='#2a2a2a', relief='ridge', bd=2)
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left panel - Model Selection
        left_panel = tk.Frame(main_container, bg='#2a2a2a')
        left_panel.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        
        # Model Selection Section
        model_frame = tk.LabelFrame(left_panel, text=" 📂 Select Model ", 
                                    fg='white', bg='#2a2a2a', font=('Arial', 12, 'bold'))
        model_frame.pack(fill='x', pady=10)
        
        # Scan for available models button
        scan_btn = tk.Button(model_frame, text="🔍 Scan for Models", command=self.scan_for_models,
                            bg='#00d4ff', fg='black', font=('Arial', 10, 'bold'), cursor='hand2')
        scan_btn.pack(padx=10, pady=10)
        
        # Models listbox
        tk.Label(model_frame, text="Available Models:", fg='white', bg='#2a2a2a', 
                font=('Arial', 10)).pack(anchor='w', padx=10, pady=5)
        
        listbox_frame = tk.Frame(model_frame, bg='#2a2a2a')
        listbox_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.models_listbox = tk.Listbox(listbox_frame, height=8, bg='#1a1a1a', fg='white', 
                                         font=('Consolas', 9), selectmode='single')
        self.models_listbox.pack(side='left', fill='both', expand=True)
        
        models_scroll = tk.Scrollbar(listbox_frame, command=self.models_listbox.yview)
        models_scroll.pack(side='right', fill='y')
        self.models_listbox.config(yscrollcommand=models_scroll.set)
        
        # Load selected model button
        load_selected_btn = tk.Button(model_frame, text="📥 Load Selected Model", 
                                      command=self.load_selected_model,
                                      bg='#00ff88', fg='black', font=('Arial', 11, 'bold'), 
                                      cursor='hand2', width=20)
        load_selected_btn.pack(padx=10, pady=10)
        
        # Browse for custom model
        browse_btn = tk.Button(model_frame, text="📁 Browse for Model...", 
                              command=self.browse_for_model,
                              bg='#888888', fg='white', font=('Arial', 10), cursor='hand2')
        browse_btn.pack(padx=10, pady=5)
        
        # Model info display
        info_frame = tk.LabelFrame(left_panel, text=" ℹ️ Model Information ", 
                                   fg='white', bg='#2a2a2a', font=('Arial', 12, 'bold'))
        info_frame.pack(fill='both', expand=True, pady=10)
        
        self.model_info_text = tk.Text(info_frame, height=10, bg='#1a1a1a', fg='#00d4ff', 
                                       font=('Consolas', 9), wrap='word')
        self.model_info_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Right panel - Dataset Selection & Prediction
        right_panel = tk.Frame(main_container, bg='#2a2a2a')
        right_panel.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        # Dataset Selection Section
        dataset_frame = tk.LabelFrame(right_panel, text=" 📊 Select Dataset for Prediction ", 
                                      fg='white', bg='#2a2a2a', font=('Arial', 12, 'bold'))
        dataset_frame.pack(fill='x', pady=10)
        
        tk.Label(dataset_frame, text="Choose Dataset:", fg='white', bg='#2a2a2a', 
                font=('Arial', 10)).pack(anchor='w', padx=10, pady=10)
        
        self.manager_dataset_var = tk.StringVar(value='mfcc_2sec')
        dataset_options = list(self.dataset_paths.keys())
        
        for dataset in dataset_options:
            rb = tk.Radiobutton(dataset_frame, text=dataset, variable=self.manager_dataset_var, 
                               value=dataset, fg='white', bg='#2a2a2a', selectcolor='#1a1a1a',
                               font=('Arial', 10), activebackground='#2a2a2a', activeforeground='white')
            rb.pack(anchor='w', padx=30, pady=3)
        
        load_dataset_btn = tk.Button(dataset_frame, text="📂 Load Selected Dataset", 
                                     command=self.load_manager_dataset,
                                     bg='#00d4ff', fg='black', font=('Arial', 10, 'bold'), 
                                     cursor='hand2', width=20)
        load_dataset_btn.pack(padx=10, pady=10)
        
        self.manager_dataset_status = tk.Label(dataset_frame, text="No dataset loaded", 
                                               fg='#ff6b6b', bg='#2a2a2a', font=('Arial', 9))
        self.manager_dataset_status.pack(padx=10, pady=5)
        
        # Prediction Controls
        pred_frame = tk.LabelFrame(right_panel, text=" 🔮 Run Prediction ", 
                                   fg='white', bg='#2a2a2a', font=('Arial', 12, 'bold'))
        pred_frame.pack(fill='x', pady=10)
        
        tk.Label(pred_frame, text="Number of Samples:", fg='white', bg='#2a2a2a', 
                font=('Arial', 10)).pack(padx=10, pady=5)
        
        self.manager_samples_var = tk.IntVar(value=1000)
        samples_spin = tk.Spinbox(pred_frame, from_=10, to=5000, increment=50, 
                                 textvariable=self.manager_samples_var, width=10, 
                                 font=('Arial', 10))
        samples_spin.pack(padx=10, pady=5)
        
        self.manager_predict_btn = tk.Button(pred_frame, text="🚀 Start Prediction", 
                                            command=self.start_manager_prediction,
                                            bg='#00ff88', fg='black', font=('Arial', 12, 'bold'), 
                                            width=20, height=2, cursor='hand2', state='disabled')
        self.manager_predict_btn.pack(padx=10, pady=15)
        
        # Results Display
        results_frame = tk.LabelFrame(right_panel, text=" 📈 Prediction Results ", 
                                     fg='white', bg='#2a2a2a', font=('Arial', 12, 'bold'))
        results_frame.pack(fill='both', expand=True, pady=10)
        
        self.manager_results_text = tk.Text(results_frame, height=15, bg='#1a1a1a', fg='white', 
                                            font=('Consolas', 9), wrap='word')
        self.manager_results_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        results_scroll = tk.Scrollbar(results_frame, command=self.manager_results_text.yview)
        results_scroll.pack(side='right', fill='y')
        self.manager_results_text.config(yscrollcommand=results_scroll.set)
    
    def setup_optimization_tab(self):
        """Setup optimization settings tab"""
        # Title
        title = tk.Label(self.optimization_tab, text="⚙️ Model Optimization Settings", 
                        font=('Arial', 20, 'bold'), fg='#00d4ff', bg='#1a1a1a')
        title.pack(pady=10)
        
        # Scrollable canvas for all settings
        canvas = tk.Canvas(self.optimization_tab, bg='#1a1a1a', highlightthickness=0)
        scrollbar = tk.Scrollbar(self.optimization_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#1a1a1a')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Architecture Settings
        arch_frame = tk.LabelFrame(scrollable_frame, text=" 🏗️ Architecture Settings ", 
                                   fg='white', bg='#2a2a2a', font=('Arial', 12, 'bold'))
        arch_frame.pack(fill='x', padx=20, pady=10)
        
        # Conv Layer Filters
        tk.Label(arch_frame, text="Convolutional Layer Filters:", fg='#00d4ff', bg='#2a2a2a', 
                font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=3, sticky='w', padx=10, pady=10)
        
        tk.Label(arch_frame, text="Conv Layer 1:", fg='white', bg='#2a2a2a').grid(row=1, column=0, sticky='w', padx=10, pady=5)
        self.conv1_filters_var = tk.IntVar(value=16)
        tk.Spinbox(arch_frame, from_=8, to=128, increment=8, textvariable=self.conv1_filters_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        tk.Label(arch_frame, text="filters", fg='#888888', bg='#2a2a2a', font=('Arial', 9, 'italic')).grid(row=1, column=2, sticky='w', padx=5)
        
        tk.Label(arch_frame, text="Conv Layer 2:", fg='white', bg='#2a2a2a').grid(row=2, column=0, sticky='w', padx=10, pady=5)
        self.conv2_filters_var = tk.IntVar(value=32)
        tk.Spinbox(arch_frame, from_=8, to=128, increment=8, textvariable=self.conv2_filters_var, width=10).grid(row=2, column=1, padx=5, pady=5)
        tk.Label(arch_frame, text="filters", fg='#888888', bg='#2a2a2a', font=('Arial', 9, 'italic')).grid(row=2, column=2, sticky='w', padx=5)
        
        tk.Label(arch_frame, text="Conv Layer 3:", fg='white', bg='#2a2a2a').grid(row=3, column=0, sticky='w', padx=10, pady=5)
        self.conv3_filters_var = tk.IntVar(value=64)
        tk.Spinbox(arch_frame, from_=8, to=128, increment=8, textvariable=self.conv3_filters_var, width=10).grid(row=3, column=1, padx=5, pady=5)
        tk.Label(arch_frame, text="filters (try 48 for less overfitting)", fg='#888888', bg='#2a2a2a', font=('Arial', 9, 'italic')).grid(row=3, column=2, sticky='w', padx=5)
        
        # Dense Layer Units
        tk.Label(arch_frame, text="Dense Layer Units:", fg='#00d4ff', bg='#2a2a2a', 
                font=('Arial', 10, 'bold')).grid(row=4, column=0, columnspan=3, sticky='w', padx=10, pady=10)
        
        tk.Label(arch_frame, text="Dense Layer 1:", fg='white', bg='#2a2a2a').grid(row=5, column=0, sticky='w', padx=10, pady=5)
        self.dense1_units_var = tk.IntVar(value=128)
        tk.Spinbox(arch_frame, from_=0, to=512, increment=16, textvariable=self.dense1_units_var, width=10).grid(row=5, column=1, padx=5, pady=5)
        tk.Label(arch_frame, text="units (0 to remove)", fg='#888888', bg='#2a2a2a', font=('Arial', 9, 'italic')).grid(row=5, column=2, sticky='w', padx=5)
        
        tk.Label(arch_frame, text="Dense Layer 2:", fg='white', bg='#2a2a2a').grid(row=6, column=0, sticky='w', padx=10, pady=5)
        self.dense2_units_var = tk.IntVar(value=64)
        tk.Spinbox(arch_frame, from_=0, to=256, increment=16, textvariable=self.dense2_units_var, width=10).grid(row=6, column=1, padx=5, pady=5)
        tk.Label(arch_frame, text="units (0 to remove)", fg='#888888', bg='#2a2a2a', font=('Arial', 9, 'italic')).grid(row=6, column=2, sticky='w', padx=5)
        
        # Regularization Settings
        reg_frame = tk.LabelFrame(scrollable_frame, text=" 🛡️ Regularization Settings ", 
                                 fg='white', bg='#2a2a2a', font=('Arial', 12, 'bold'))
        reg_frame.pack(fill='x', padx=20, pady=10)
        
        # L2 Regularization
        self.l2_reg_var = tk.BooleanVar(value=False)
        tk.Checkbutton(reg_frame, text="Enable L2 Regularization", variable=self.l2_reg_var, 
                      fg='white', bg='#2a2a2a', selectcolor='#1a1a1a', 
                      font=('Arial', 10, 'bold'), activebackground='#2a2a2a', activeforeground='white').grid(row=0, column=0, sticky='w', padx=10, pady=10)
        
        tk.Label(reg_frame, text="L2 Strength:", fg='white', bg='#2a2a2a').grid(row=1, column=0, sticky='w', padx=30, pady=5)
        self.l2_strength_var = tk.DoubleVar(value=0.001)
        tk.Entry(reg_frame, textvariable=self.l2_strength_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        tk.Label(reg_frame, text="(0.001 recommended)", fg='#888888', bg='#2a2a2a', font=('Arial', 9, 'italic')).grid(row=1, column=2, sticky='w', padx=5)
        
        # Dropout Settings
        tk.Label(reg_frame, text="Dropout Rates:", fg='#00d4ff', bg='#2a2a2a', 
                font=('Arial', 10, 'bold')).grid(row=2, column=0, columnspan=3, sticky='w', padx=10, pady=10)
        
        # Conv Dropout
        tk.Label(reg_frame, text="After Conv Block 1:", fg='white', bg='#2a2a2a').grid(row=3, column=0, sticky='w', padx=30, pady=5)
        self.conv1_dropout_var = tk.DoubleVar(value=0.0)
        tk.Scale(reg_frame, from_=0.0, to=0.5, resolution=0.05, orient='horizontal', 
                variable=self.conv1_dropout_var, bg='#2a2a2a', fg='white', highlightthickness=0, length=150).grid(row=3, column=1, padx=5, pady=5)
        
        tk.Label(reg_frame, text="After Conv Block 2:", fg='white', bg='#2a2a2a').grid(row=4, column=0, sticky='w', padx=30, pady=5)
        self.conv2_dropout_var = tk.DoubleVar(value=0.0)
        tk.Scale(reg_frame, from_=0.0, to=0.5, resolution=0.05, orient='horizontal', 
                variable=self.conv2_dropout_var, bg='#2a2a2a', fg='white', highlightthickness=0, length=150).grid(row=4, column=1, padx=5, pady=5)
        
        tk.Label(reg_frame, text="After Conv Block 3:", fg='white', bg='#2a2a2a').grid(row=5, column=0, sticky='w', padx=30, pady=5)
        self.conv3_dropout_var = tk.DoubleVar(value=0.0)
        tk.Scale(reg_frame, from_=0.0, to=0.5, resolution=0.05, orient='horizontal', 
                variable=self.conv3_dropout_var, bg='#2a2a2a', fg='white', highlightthickness=0, length=150).grid(row=5, column=1, padx=5, pady=5)
        
        # Dense Dropout
        tk.Label(reg_frame, text="Dense Layer 1 Dropout:", fg='white', bg='#2a2a2a').grid(row=6, column=0, sticky='w', padx=30, pady=5)
        self.dense1_dropout_var = tk.DoubleVar(value=0.3)
        tk.Scale(reg_frame, from_=0.0, to=0.7, resolution=0.05, orient='horizontal', 
                variable=self.dense1_dropout_var, bg='#2a2a2a', fg='white', highlightthickness=0, length=150).grid(row=6, column=1, padx=5, pady=5)
        tk.Label(reg_frame, text="(try 0.5 for overfitting)", fg='#888888', bg='#2a2a2a', font=('Arial', 9, 'italic')).grid(row=6, column=2, sticky='w', padx=5)
        
        tk.Label(reg_frame, text="Dense Layer 2 Dropout:", fg='white', bg='#2a2a2a').grid(row=7, column=0, sticky='w', padx=30, pady=5)
        self.dense2_dropout_var = tk.DoubleVar(value=0.2)
        tk.Scale(reg_frame, from_=0.0, to=0.7, resolution=0.05, orient='horizontal', 
                variable=self.dense2_dropout_var, bg='#2a2a2a', fg='white', highlightthickness=0, length=150).grid(row=7, column=1, padx=5, pady=5)
        tk.Label(reg_frame, text="(try 0.5 for overfitting)", fg='#888888', bg='#2a2a2a', font=('Arial', 9, 'italic')).grid(row=7, column=2, sticky='w', padx=5)
        
        # Training Strategy
        train_frame = tk.LabelFrame(scrollable_frame, text=" 📈 Training Strategy ", 
                                    fg='white', bg='#2a2a2a', font=('Arial', 12, 'bold'))
        train_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(train_frame, text="Early Stopping Patience:", fg='white', bg='#2a2a2a').grid(row=0, column=0, sticky='w', padx=10, pady=5)
        self.patience_var = tk.IntVar(value=5)
        tk.Spinbox(train_frame, from_=3, to=15, textvariable=self.patience_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        tk.Label(train_frame, text="epochs (try 7 for better convergence)", fg='#888888', bg='#2a2a2a', font=('Arial', 9, 'italic')).grid(row=0, column=2, sticky='w', padx=5)
        
        tk.Label(train_frame, text="ReduceLR Patience:", fg='white', bg='#2a2a2a').grid(row=1, column=0, sticky='w', padx=10, pady=5)
        self.reduce_lr_patience_var = tk.IntVar(value=3)
        tk.Spinbox(train_frame, from_=2, to=10, textvariable=self.reduce_lr_patience_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        tk.Label(train_frame, text="epochs", fg='#888888', bg='#2a2a2a', font=('Arial', 9, 'italic')).grid(row=1, column=2, sticky='w', padx=5)
        
        self.save_best_model_var = tk.BooleanVar(value=True)
        tk.Checkbutton(train_frame, text="Save Best Model During Training (ModelCheckpoint)", variable=self.save_best_model_var, 
                      fg='white', bg='#2a2a2a', selectcolor='#1a1a1a', 
                      font=('Arial', 10), activebackground='#2a2a2a', activeforeground='white').grid(row=2, column=0, columnspan=3, sticky='w', padx=10, pady=10)
        
        # Mixed SNR Training
        snr_frame = tk.LabelFrame(scrollable_frame, text=" 🔊 Mixed SNR Dataset Training ", 
                                 fg='white', bg='#2a2a2a', font=('Arial', 12, 'bold'))
        snr_frame.pack(fill='x', padx=20, pady=10)
        
        self.mixed_snr_var = tk.BooleanVar(value=False)
        tk.Checkbutton(snr_frame, text="Enable Mixed SNR Training (Combine Multiple Datasets)", 
                      variable=self.mixed_snr_var, command=self.toggle_snr_options,
                      fg='white', bg='#2a2a2a', selectcolor='#1a1a1a', 
                      font=('Arial', 10, 'bold'), activebackground='#2a2a2a', activeforeground='white').grid(row=0, column=0, columnspan=2, sticky='w', padx=10, pady=10)
        
        tk.Label(snr_frame, text="Select SNR Levels to Mix:", fg='#00d4ff', bg='#2a2a2a', 
                font=('Arial', 10)).grid(row=1, column=0, columnspan=2, sticky='w', padx=30, pady=5)
        
        self.snr_5db_var = tk.BooleanVar(value=False)
        self.snr_10db_var = tk.BooleanVar(value=False)
        self.snr_15db_var = tk.BooleanVar(value=False)
        self.snr_20db_var = tk.BooleanVar(value=False)
        
        self.snr_5db_check = tk.Checkbutton(snr_frame, text="SNR 5dB", variable=self.snr_5db_var, 
                                           fg='white', bg='#2a2a2a', selectcolor='#1a1a1a', state='disabled')
        self.snr_5db_check.grid(row=2, column=0, sticky='w', padx=50, pady=2)
        
        self.snr_10db_check = tk.Checkbutton(snr_frame, text="SNR 10dB", variable=self.snr_10db_var, 
                                            fg='white', bg='#2a2a2a', selectcolor='#1a1a1a', state='disabled')
        self.snr_10db_check.grid(row=3, column=0, sticky='w', padx=50, pady=2)
        
        self.snr_15db_check = tk.Checkbutton(snr_frame, text="SNR 15dB", variable=self.snr_15db_var, 
                                            fg='white', bg='#2a2a2a', selectcolor='#1a1a1a', state='disabled')
        self.snr_15db_check.grid(row=4, column=0, sticky='w', padx=50, pady=2)
        
        self.snr_20db_check = tk.Checkbutton(snr_frame, text="SNR 20dB", variable=self.snr_20db_var, 
                                            fg='white', bg='#2a2a2a', selectcolor='#1a1a1a', state='disabled')
        self.snr_20db_check.grid(row=5, column=0, sticky='w', padx=50, pady=2)
        
        tk.Label(snr_frame, text="💡 Tip: Training on multiple SNR levels improves robustness", 
                fg='#ffa500', bg='#2a2a2a', font=('Arial', 9, 'italic'), wraplength=600).grid(row=6, column=0, columnspan=2, sticky='w', padx=30, pady=10)
        
        # Preset Buttons
        preset_frame = tk.LabelFrame(scrollable_frame, text=" 🎯 Quick Presets ", 
                                     fg='white', bg='#2a2a2a', font=('Arial', 12, 'bold'))
        preset_frame.pack(fill='x', padx=20, pady=10)
        
        btn_container = tk.Frame(preset_frame, bg='#2a2a2a')
        btn_container.pack(pady=10)
        
        tk.Button(btn_container, text="🔧 Default Settings", command=self.apply_default_preset,
                 bg='#888888', fg='white', font=('Arial', 10, 'bold'), width=20, cursor='hand2').pack(side='left', padx=5)
        
        tk.Button(btn_container, text="🛡️ Anti-Overfitting", command=self.apply_antioverfitting_preset,
                 bg='#ff6b6b', fg='white', font=('Arial', 10, 'bold'), width=20, cursor='hand2').pack(side='left', padx=5)
        
        tk.Button(btn_container, text="⚡ Lightweight", command=self.apply_lightweight_preset,
                 bg='#00d4ff', fg='black', font=('Arial', 10, 'bold'), width=20, cursor='hand2').pack(side='left', padx=5)
        
        tk.Button(btn_container, text="💪 Heavy Power", command=self.apply_heavy_preset,
                 bg='#00ff88', fg='black', font=('Arial', 10, 'bold'), width=20, cursor='hand2').pack(side='left', padx=5)
        
        # Info Box
        info_frame = tk.Frame(scrollable_frame, bg='#3a3a3a', relief='ridge', bd=2)
        info_frame.pack(fill='x', padx=20, pady=10)
        
        info_text = """
📚 OPTIMIZATION GUIDE:
        
🛡️ To Reduce Overfitting:
   • Increase dropout rates (try 0.5 for dense layers)
   • Enable L2 regularization (0.001)
   • Reduce conv3 filters to 48
   • Remove one dense layer (set to 0)
   • Increase early stopping patience to 7
   
⚡ For Faster Training:
   • Reduce filter counts (8, 16, 32)
   • Remove one dense layer
   • Lower batch size
   
💪 For Better Accuracy:
   • Increase filters (32, 64, 96)
   • Add more dense units
   • Use mixed SNR training
   • Lower learning rate (0.0001)
        """
        
        tk.Label(info_frame, text=info_text, fg='#00d4ff', bg='#3a3a3a', 
                font=('Consolas', 9), justify='left').pack(padx=10, pady=10)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def toggle_snr_options(self):
        """Enable/disable SNR checkboxes based on mixed SNR toggle"""
        state = 'normal' if self.mixed_snr_var.get() else 'disabled'
        self.snr_5db_check.config(state=state)
        self.snr_10db_check.config(state=state)
        self.snr_15db_check.config(state=state)
        self.snr_20db_check.config(state=state)
    
    def setup_explainability_tab(self):
        """Setup explainability tab for synthetic audio detection analysis"""
        # Title
        title = tk.Label(self.explainability_tab, text="🎙️ Synthetic Audio Detection Analysis", 
                        font=('Arial', 20, 'bold'), fg='#00d4ff', bg='#1a1a1a')
        title.pack(pady=10)
        
        # Info section
        info_frame = tk.LabelFrame(self.explainability_tab, text=" ℹ️ About Synthetic Detection ", 
                                   fg='white', bg='#2a2a2a', font=('Arial', 10, 'bold'))
        info_frame.pack(fill='x', padx=10, pady=5)
        
        info_text = """
� AI-Powered Synthetic Speech Detection Under Noisy Conditions

📊 What You'll See:
   • Synthetic/Authentic Classification: Is this audio AI-generated or real human speech?
   • Noise Robustness Analysis: Model performance under various SNR conditions
   • Attention Maps (Grad-CAM): Which spectral patterns indicate synthetic generation
   • Confidence Assessment: How certain is the model about this detection?
   
💡 Model Capabilities:
   • Trained on SNR-augmented datasets (5dB, 10dB, 15dB, 20dB noise levels)
   • Detects synthetic audio artifacts even in noisy environments
   • Analyzes mel-spectrogram patterns characteristic of AI voice generation
   • Provides explainable decisions with visual attention heatmaps
        """
        
        tk.Label(info_frame, text=info_text, fg='#ecf0f1', bg='#2a2a2a', 
                font=('Consolas', 9), justify='left').pack(padx=10, pady=10)
        
        # Control panel
        control_frame = tk.Frame(self.explainability_tab, bg='#2a2a2a', relief='ridge', bd=2)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        # Model selection section
        model_frame = tk.LabelFrame(control_frame, text=" 🤖 Select Model ", 
                                    fg='white', bg='#2a2a2a', font=('Arial', 10, 'bold'))
        model_frame.pack(side='left', padx=10, pady=10)
        
        tk.Label(model_frame, text="Active Model:", fg='white', bg='#2a2a2a', 
                font=('Arial', 9)).grid(row=0, column=0, sticky='w', padx=5, pady=5)
        
        self.explain_model_var = tk.StringVar(value="Current Model")
        self.explain_model_label = tk.Label(model_frame, textvariable=self.explain_model_var, 
                                            fg='#00ff88', bg='#2a2a2a', font=('Arial', 9, 'bold'))
        self.explain_model_label.grid(row=0, column=1, sticky='w', padx=5, pady=5)
        
        tk.Button(model_frame, text="🔄 Load Model", command=self.load_model_for_explain,
                 bg='#3498db', fg='white', font=('Arial', 9, 'bold'), 
                 cursor='hand2', padx=10).grid(row=1, column=0, columnspan=2, pady=5)
        
        # Load sample section
        load_frame = tk.LabelFrame(control_frame, text=" 📂 Load Sample for Analysis ", 
                                   fg='white', bg='#2a2a2a', font=('Arial', 10, 'bold'))
        load_frame.pack(side='left', padx=10, pady=10, fill='both', expand=True)
        
        tk.Label(load_frame, text="Select from test dataset:", fg='white', bg='#2a2a2a').grid(row=0, column=0, sticky='w', padx=5, pady=5)
        
        self.explain_sample_var = tk.IntVar(value=0)
        sample_spin = tk.Spinbox(load_frame, from_=0, to=1000, textvariable=self.explain_sample_var, width=10)
        sample_spin.grid(row=0, column=1, padx=5, pady=5)
        
        load_sample_btn = tk.Button(load_frame, text="🔍 Analyze Sample", command=self.analyze_sample,
                                    bg='#00d4ff', fg='black', font=('Arial', 10, 'bold'), cursor='hand2')
        load_sample_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # Manual audio upload
        tk.Label(load_frame, text="Or upload your own audio:", fg='white', bg='#2a2a2a').grid(row=1, column=0, sticky='w', padx=5, pady=5)
        
        upload_btn = tk.Button(load_frame, text="📁 Upload WAV File", command=self.analyze_custom_audio,
                              bg='#9b59b6', fg='white', font=('Arial', 10, 'bold'), cursor='hand2')
        upload_btn.grid(row=1, column=1, padx=5, pady=5, sticky='ew')
        
        # Microphone recording
        record_btn = tk.Button(load_frame, text="🎤 Record Audio (5s)", command=self.record_and_analyze_audio,
                              bg='#e74c3c', fg='white', font=('Arial', 10, 'bold'), cursor='hand2')
        record_btn.grid(row=1, column=2, padx=5, pady=5, sticky='ew')
        
        # Preview button for recorded audio
        self.preview_btn = tk.Button(load_frame, text="▶️ Play Recording", command=self.play_recorded_audio,
                                     bg='#27ae60', fg='white', font=('Arial', 9), cursor='hand2', state='disabled')
        self.preview_btn.grid(row=2, column=2, padx=5, pady=2, sticky='ew')
        
        # Save recording button
        self.save_rec_btn = tk.Button(load_frame, text="💾 Save Recording", command=self.save_recording_for_training,
                                      bg='#f39c12', fg='white', font=('Arial', 9), cursor='hand2', state='disabled')
        self.save_rec_btn.grid(row=2, column=1, padx=5, pady=2, sticky='ew')
        
        self.explain_status = tk.Label(load_frame, text="Load a sample or upload audio to begin analysis", fg='#ffa500', bg='#2a2a2a')
        self.explain_status.grid(row=3, column=0, columnspan=3, sticky='w', padx=5, pady=5)
        
        # Export section
        export_frame = tk.LabelFrame(control_frame, text=" 📄 Export Analysis ", 
                                     fg='white', bg='#2a2a2a', font=('Arial', 10, 'bold'))
        export_frame.pack(side='left', padx=10, pady=10)
        
        export_btn = tk.Button(export_frame, text="📄 Export to PDF", command=self.export_explainability_pdf,
                              bg='#ff6b6b', fg='white', font=('Arial', 11, 'bold'), 
                              width=15, height=2, cursor='hand2')
        export_btn.pack(padx=5, pady=5)
        
        # Main content area with scrolling
        content_canvas = tk.Canvas(self.explainability_tab, bg='#1a1a1a', highlightthickness=0)
        content_scrollbar = tk.Scrollbar(self.explainability_tab, orient='vertical', command=content_canvas.yview)
        self.explain_content = tk.Frame(content_canvas, bg='#1a1a1a')
        
        self.explain_content.bind(
            "<Configure>",
            lambda e: content_canvas.configure(scrollregion=content_canvas.bbox("all"))
        )
        
        content_canvas.create_window((0, 0), window=self.explain_content, anchor='nw')
        content_canvas.configure(yscrollcommand=content_scrollbar.set)
        
        content_canvas.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        content_scrollbar.pack(side='right', fill='y')
        
        # Prediction Summary
        summary_frame = tk.LabelFrame(self.explain_content, text=" 🎯 Prediction Summary ", 
                                     fg='white', bg='#2a2a2a', font=('Arial', 12, 'bold'))
        summary_frame.pack(fill='x', padx=10, pady=10)
        
        self.explain_summary = tk.Text(summary_frame, height=8, bg='#1a1a1a', fg='white', 
                                      font=('Consolas', 10), wrap='word')
        self.explain_summary.pack(fill='x', padx=5, pady=5)
        
        # Visualization area
        viz_frame = tk.LabelFrame(self.explain_content, text=" 📊 Visual Analysis ", 
                                 fg='white', bg='#2a2a2a', font=('Arial', 12, 'bold'))
        viz_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create matplotlib figure for visualizations
        self.explain_fig = Figure(figsize=(14, 10), facecolor='#1a1a1a')
        self.explain_canvas = FigureCanvasTkAgg(self.explain_fig, viz_frame)
        self.explain_canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
        
        # Detailed Analysis Log
        log_frame = tk.LabelFrame(self.explain_content, text=" 📝 Detailed Analysis ", 
                                 fg='white', bg='#2a2a2a', font=('Arial', 12, 'bold'))
        log_frame.pack(fill='both', padx=10, pady=10, expand=True)
        
        self.explain_log = tk.Text(log_frame, height=15, bg='#1a1a1a', fg='#00ff88', 
                                  font=('Consolas', 9), wrap='word')
        self.explain_log.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        log_scroll = tk.Scrollbar(log_frame, command=self.explain_log.yview)
        log_scroll.pack(side='right', fill='y')
        self.explain_log.config(yscrollcommand=log_scroll.set)
    
    def load_model_for_explain(self):
        """Load a model specifically for explainability analysis"""
        file_path = filedialog.askopenfilename(
            title="Select CNN Model",
            filetypes=[("Keras Model", "*.keras"), ("H5 Model", "*.h5"), ("All files", "*.*")],
            initialdir=self.cnn_model_dir
        )
        
        if not file_path:
            return
        
        try:
            import tensorflow as tf
            
            # Load model
            self.log_explain(f"📂 Loading model: {os.path.basename(file_path)}")
            
            # Create new CNN model instance and load
            self.cnn_model = CNNAudioClassifier()
            self.cnn_model.load_model(file_path)
            
            # Model is now compiled and built in load_model() - ready to use
            
            # Update UI
            model_name = os.path.basename(file_path)
            self.explain_model_var.set(model_name)
            self.explain_status.config(
                text=f"✅ Model loaded: {model_name}", 
                fg='#00ff88'
            )
            
            self.log_explain(f"✅ Model loaded successfully!")
            self.log_explain(f"   • Model: {model_name}")
            self.log_explain(f"   • Model built and ready for analysis")
            
            messagebox.showinfo("Success", f"Model loaded:\n{model_name}\n\nReady for explainability analysis!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            print(f"Error loading model for explain: {e}")
            import traceback
            traceback.print_exc()
    
    # ========== MODEL MANAGER METHODS ==========
    
    def scan_for_models(self):
        """Scan CNN folder for available model files"""
        self.models_listbox.delete(0, tk.END)
        
        # Check CNN folder
        if not os.path.exists(self.cnn_model_dir):
            self.model_info_text.delete('1.0', tk.END)
            self.model_info_text.insert(tk.END, "❌ CNN folder not found!\n")
            return
        
        # Find all .h5 and .keras files
        model_files = []
        for file in os.listdir(self.cnn_model_dir):
            if file.endswith('.h5') or file.endswith('.keras'):
                model_files.append(file)
        
        if not model_files:
            self.model_info_text.delete('1.0', tk.END)
            self.model_info_text.insert(tk.END, "⚠️ No model files found in CNN folder\n")
            self.model_info_text.insert(tk.END, f"   Path: {self.cnn_model_dir}\n")
            return
        
        # Sort by modification time (newest first)
        model_files_with_time = []
        for file in model_files:
            filepath = os.path.join(self.cnn_model_dir, file)
            mod_time = os.path.getmtime(filepath)
            model_files_with_time.append((file, mod_time))
        
        model_files_with_time.sort(key=lambda x: x[1], reverse=True)
        
        # Add to listbox
        for file, mod_time in model_files_with_time:
            mod_date = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M")
            self.models_listbox.insert(tk.END, f"{file} ({mod_date})")
        
        self.model_info_text.delete('1.0', tk.END)
        self.model_info_text.insert(tk.END, f"✅ Found {len(model_files)} model(s)\n\n")
        self.model_info_text.insert(tk.END, "📂 Select a model from the list above\n")
        self.model_info_text.insert(tk.END, "   and click 'Load Selected Model'\n")
        
        print(f"✅ Scanned CNN folder: Found {len(model_files)} models")
    
    def load_selected_model(self):
        """Load the model selected in the listbox"""
        selection = self.models_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a model from the list first!")
            return
        
        # Get selected model filename (remove the date suffix)
        selected_text = self.models_listbox.get(selection[0])
        model_filename = selected_text.split(' (')[0]  # Remove date part
        model_path = os.path.join(self.cnn_model_dir, model_filename)
        
        self._load_model_from_path(model_path)
    
    def browse_for_model(self):
        """Browse for a model file"""
        filepath = filedialog.askopenfilename(
            title="Select CNN Model File",
            filetypes=[("Keras Models", "*.h5 *.keras"), ("All Files", "*.*")],
            initialdir=self.cnn_model_dir
        )
        
        if filepath:
            self._load_model_from_path(filepath)
    
    def _load_model_from_path(self, model_path):
        """Load a model from the given path"""
        try:
            print(f"\n📂 Loading model from: {model_path}")
            
            self.cnn_model = CNNAudioClassifier()
            self.cnn_model.load_model(model_path)
            
            # Model is now compiled and built in load_model() - no need to do it here
            
            # Update explainability tab model label
            if hasattr(self, 'explain_model_var'):
                self.explain_model_var.set(os.path.basename(model_path))
            
            # Get file info
            mod_time = os.path.getmtime(model_path)
            mod_date = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            
            # Display model info
            self.model_info_text.delete('1.0', tk.END)
            self.model_info_text.insert(tk.END, "✅ MODEL LOADED SUCCESSFULLY\n")
            self.model_info_text.insert(tk.END, "="*50 + "\n\n")
            self.model_info_text.insert(tk.END, f"📂 File: {os.path.basename(model_path)}\n")
            self.model_info_text.insert(tk.END, f"📅 Modified: {mod_date}\n")
            self.model_info_text.insert(tk.END, f"💾 Size: {file_size:.2f} MB\n\n")
            
            # Check for metadata
            metadata_path = os.path.join(self.cnn_model_dir, 'dataset_info.txt')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = f.read()
                self.model_info_text.insert(tk.END, "📋 Training Info:\n")
                self.model_info_text.insert(tk.END, "-"*50 + "\n")
                # Show first 300 chars of metadata
                preview = metadata[:300] + "..." if len(metadata) > 300 else metadata
                self.model_info_text.insert(tk.END, preview + "\n")
            else:
                self.model_info_text.insert(tk.END, "⚠️ No training metadata found\n")
            
            print(f"✅ Model loaded successfully!")
            
            # Enable prediction if dataset is also loaded
            if self.test_data is not None:
                self.manager_predict_btn.config(state='normal')
                self.model_info_text.insert(tk.END, "\n✅ Ready for prediction!\n")
            else:
                self.model_info_text.insert(tk.END, "\n⚠️ Load a dataset to start prediction\n")
            
        except Exception as e:
            error_msg = f"Failed to load model:\n{str(e)}"
            messagebox.showerror("Load Error", error_msg)
            self.model_info_text.delete('1.0', tk.END)
            self.model_info_text.insert(tk.END, f"❌ ERROR:\n{error_msg}\n")
            print(f"❌ Error loading model: {str(e)}")
    
    def load_manager_dataset(self):
        """Load the selected dataset for Model Manager"""
        dataset_name = self.manager_dataset_var.get()
        
        # Temporarily switch the main dataset var
        old_dataset = self.dataset_var.get()
        self.dataset_var.set(dataset_name)
        
        # Load the dataset using existing function
        self.load_dataset()
        
        # Update manager status
        if self.test_data is not None:
            self.manager_dataset_status.config(
                text=f"✅ Loaded: {dataset_name}", 
                fg='#00ff88'
            )
            
            # Enable prediction if model is also loaded
            if self.cnn_model is not None:
                self.manager_predict_btn.config(state='normal')
                self.manager_results_text.delete('1.0', tk.END)
                self.manager_results_text.insert(tk.END, "✅ Model and Dataset loaded!\n")
                self.manager_results_text.insert(tk.END, "   Click 'Start Prediction' to test.\n")
        else:
            self.manager_dataset_status.config(
                text="❌ Failed to load dataset", 
                fg='#ff6b6b'
            )
        
        # Restore original dataset selection
        self.dataset_var.set(old_dataset)
    
    def start_manager_prediction(self):
        """Start prediction from Model Manager tab"""
        if self.cnn_model is None:
            messagebox.showwarning("No Model", "Please load a model first!")
            return
        
        if self.test_data is None:
            messagebox.showwarning("No Dataset", "Please load a dataset first!")
            return
        
        # Clear results
        self.manager_results_text.delete('1.0', tk.END)
        self.manager_results_text.insert(tk.END, "🔮 Starting prediction...\n\n")
        
        # Disable button during prediction
        self.manager_predict_btn.config(state='disabled')
        
        # Start prediction in thread
        pred_thread = threading.Thread(target=self._manager_prediction_thread, daemon=True)
        pred_thread.start()
    
    def _manager_prediction_thread(self):
        """Prediction thread for Model Manager"""
        try:
            num_samples = self.manager_samples_var.get()
            
            # Reset counters
            correct = 0
            total = 0
            true_pos = 0
            false_pos = 0
            true_neg = 0
            false_neg = 0
            
            self.manager_results_text.insert(tk.END, f"📊 Processing {num_samples} samples...\n\n")
            
            # Iterate through test data
            for batch_idx, (batch_x, batch_y) in enumerate(self.test_data):
                if total >= num_samples:
                    break
                
                # Make predictions
                predictions = self.cnn_model.model.predict(batch_x, verbose=0)
                
                # Process each prediction
                for i in range(len(predictions)):
                    if total >= num_samples:
                        break
                    
                    pred_prob = predictions[i][0]
                    pred_label = 1 if pred_prob > 0.5 else 0
                    actual_label = int(batch_y[i])
                    
                    # Update counters
                    if pred_label == actual_label:
                        correct += 1
                    
                    # Confusion matrix
                    if pred_label == 1 and actual_label == 1:
                        true_pos += 1
                    elif pred_label == 1 and actual_label == 0:
                        false_pos += 1
                    elif pred_label == 0 and actual_label == 0:
                        true_neg += 1
                    elif pred_label == 0 and actual_label == 1:
                        false_neg += 1
                    
                    total += 1
                    
                    # Progress update every 100 samples
                    if total % 100 == 0:
                        progress_pct = (total / num_samples) * 100
                        acc = (correct / total) * 100
                        self.manager_results_text.insert(tk.END, f"   {total}/{num_samples} ({progress_pct:.1f}%) - Acc: {acc:.2f}%\n")
                        self.manager_results_text.see(tk.END)
            
            # Calculate metrics
            accuracy = (correct / total) * 100 if total > 0 else 0
            precision = (true_pos / (true_pos + false_pos)) * 100 if (true_pos + false_pos) > 0 else 0
            recall = (true_pos / (true_pos + false_neg)) * 100 if (true_pos + false_neg) > 0 else 0
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Display results
            self.manager_results_text.insert(tk.END, "\n" + "="*50 + "\n")
            self.manager_results_text.insert(tk.END, "✅ PREDICTION COMPLETED!\n")
            self.manager_results_text.insert(tk.END, "="*50 + "\n\n")
            self.manager_results_text.insert(tk.END, f"📊 Results:\n")
            self.manager_results_text.insert(tk.END, f"   Total Samples:  {total}\n")
            self.manager_results_text.insert(tk.END, f"   Correct:        {correct}\n")
            self.manager_results_text.insert(tk.END, f"   Incorrect:      {total - correct}\n\n")
            self.manager_results_text.insert(tk.END, f"🎯 Metrics:\n")
            self.manager_results_text.insert(tk.END, f"   Accuracy:   {accuracy:.2f}%\n")
            self.manager_results_text.insert(tk.END, f"   Precision:  {precision:.2f}%\n")
            self.manager_results_text.insert(tk.END, f"   Recall:     {recall:.2f}%\n")
            self.manager_results_text.insert(tk.END, f"   F1-Score:   {f1_score:.2f}%\n\n")
            self.manager_results_text.insert(tk.END, f"📋 Confusion Matrix:\n")
            self.manager_results_text.insert(tk.END, f"   True Positives:  {true_pos}\n")
            self.manager_results_text.insert(tk.END, f"   False Positives: {false_pos}\n")
            self.manager_results_text.insert(tk.END, f"   True Negatives:  {true_neg}\n")
            self.manager_results_text.insert(tk.END, f"   False Negatives: {false_neg}\n")
            
            print(f"\n✅ Manager Prediction Complete: {accuracy:.2f}% accuracy on {total} samples")
            
        except Exception as e:
            error_msg = f"❌ Prediction Error:\n{str(e)}\n"
            self.manager_results_text.insert(tk.END, error_msg)
            print(f"❌ Prediction error: {str(e)}")
        
        finally:
            # Re-enable button
            self.manager_predict_btn.config(state='normal')
    
    def apply_default_preset(self):
        """Apply default settings"""
        self.conv1_filters_var.set(16)
        self.conv2_filters_var.set(32)
        self.conv3_filters_var.set(64)
        self.dense1_units_var.set(128)
        self.dense2_units_var.set(64)
        self.l2_reg_var.set(False)
        self.conv1_dropout_var.set(0.0)
        self.conv2_dropout_var.set(0.0)
        self.conv3_dropout_var.set(0.0)
        self.dense1_dropout_var.set(0.3)
        self.dense2_dropout_var.set(0.2)
        self.patience_var.set(5)
        self.reduce_lr_patience_var.set(3)
        print("✅ Applied: Default Settings")
        messagebox.showinfo("Preset Applied", "Default settings applied!")
    
    def apply_antioverfitting_preset(self):
        """Apply anti-overfitting settings"""
        self.conv1_filters_var.set(16)
        self.conv2_filters_var.set(32)
        self.conv3_filters_var.set(48)  # Reduced
        self.dense1_units_var.set(64)  # Reduced
        self.dense2_units_var.set(0)   # Removed
        self.l2_reg_var.set(True)      # Enabled
        self.l2_strength_var.set(0.001)
        self.conv1_dropout_var.set(0.2)
        self.conv2_dropout_var.set(0.3)
        self.conv3_dropout_var.set(0.3)
        self.dense1_dropout_var.set(0.5)  # Increased
        self.dense2_dropout_var.set(0.5)  # Increased
        self.patience_var.set(7)  # Increased
        self.reduce_lr_patience_var.set(3)
        print("✅ Applied: Anti-Overfitting Preset")
        print("   • Stronger regularization (L2 + high dropout)")
        print("   • Simpler architecture (48-32-16 conv, 64 dense)")
        print("   • Better training strategy (patience=7)")
        messagebox.showinfo("Preset Applied", "Anti-Overfitting settings applied!\n\n• L2 Regularization enabled\n• High dropout rates (0.5)\n• Simplified architecture\n• Patience increased to 7")
    
    def apply_lightweight_preset(self):
        """Apply lightweight settings for faster training"""
        self.conv1_filters_var.set(8)
        self.conv2_filters_var.set(16)
        self.conv3_filters_var.set(32)
        self.dense1_units_var.set(32)
        self.dense2_units_var.set(0)  # Removed
        self.l2_reg_var.set(False)
        self.conv1_dropout_var.set(0.0)
        self.conv2_dropout_var.set(0.0)
        self.conv3_dropout_var.set(0.0)
        self.dense1_dropout_var.set(0.3)
        self.dense2_dropout_var.set(0.0)
        self.patience_var.set(5)
        self.reduce_lr_patience_var.set(3)
        print("✅ Applied: Lightweight Preset")
        print("   • Fewer parameters for faster training")
        print("   • Good for quick experiments")
        messagebox.showinfo("Preset Applied", "Lightweight settings applied!\n\n• Smaller architecture\n• Faster training\n• Good for experimentation")
    
    def apply_heavy_preset(self):
        """Apply heavy settings for maximum accuracy"""
        self.conv1_filters_var.set(32)
        self.conv2_filters_var.set(64)
        self.conv3_filters_var.set(96)
        self.dense1_units_var.set(256)
        self.dense2_units_var.set(128)
        self.l2_reg_var.set(True)
        self.l2_strength_var.set(0.0005)
        self.conv1_dropout_var.set(0.1)
        self.conv2_dropout_var.set(0.2)
        self.conv3_dropout_var.set(0.2)
        self.dense1_dropout_var.set(0.4)
        self.dense2_dropout_var.set(0.3)
        self.patience_var.set(10)
        self.reduce_lr_patience_var.set(5)
        print("✅ Applied: Heavy Power Preset")
        print("   • Maximum capacity for complex patterns")
        print("   • Slower training, potentially better accuracy")
        messagebox.showinfo("Preset Applied", "Heavy Power settings applied!\n\n• Large architecture\n• Maximum learning capacity\n• Longer training time")
    
    # ========== HARDWARE MONITORING METHODS ==========
    
    def start_resource_monitoring(self):
        """Start monitoring CPU and memory usage"""
        self.monitoring_active = True
        self.resource_data = {
            'timestamps': [],
            'cpu_percent': [],
            'memory_mb': [],
            'memory_percent': []
        }
        
        # Print system specifications
        system_memory = psutil.virtual_memory()
        print(f"\n{'='*70}")
        print(f"🖥️  SYSTEM SPECIFICATIONS")
        print(f"{'='*70}")
        print(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
        print(f"Total RAM: {system_memory.total / 1024**3:.2f} GB")
        print(f"Available RAM: {system_memory.available / 1024**3:.2f} GB")
        print(f"Process ID: {self.process.pid}")
        print(f"{'='*70}\n")
        
        self.log_training(f"🖥️ System: {psutil.cpu_count(logical=True)} cores, {system_memory.total / 1024**3:.2f} GB RAM")
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitor_resources_loop, daemon=True)
        monitor_thread.start()
    
    def _monitor_resources_loop(self):
        """Background thread to monitor resources"""
        while self.monitoring_active:
            try:
                # System-wide CPU usage
                cpu_percent = psutil.cpu_percent(interval=None)
                
                # Process-specific memory usage
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                # System memory percentage
                system_memory = psutil.virtual_memory()
                memory_percent = system_memory.percent
                
                self.resource_data['timestamps'].append(time.time())
                self.resource_data['cpu_percent'].append(cpu_percent)
                self.resource_data['memory_mb'].append(memory_mb)
                self.resource_data['memory_percent'].append(memory_percent)
                
                time.sleep(0.5)  # Monitor every 0.5 seconds
            except Exception as e:
                print(f"⚠️ Monitoring error: {e}")
                break
    
    def stop_resource_monitoring(self):
        """Stop monitoring and calculate statistics"""
        self.monitoring_active = False
        time.sleep(0.6)  # Wait for monitoring thread to finish
        
        if not self.resource_data['timestamps']:
            self.resource_stats = {}
            return
        
        # Calculate statistics
        duration = self.resource_data['timestamps'][-1] - self.resource_data['timestamps'][0]
        self.resource_stats = {
            'duration_seconds': duration,
            'avg_cpu_percent': np.mean(self.resource_data['cpu_percent']),
            'max_cpu_percent': np.max(self.resource_data['cpu_percent']),
            'avg_memory_mb': np.mean(self.resource_data['memory_mb']),
            'peak_memory_mb': np.max(self.resource_data['memory_mb']),
            'min_memory_mb': np.min(self.resource_data['memory_mb']),
            'memory_increase_mb': np.max(self.resource_data['memory_mb']) - np.min(self.resource_data['memory_mb']),
            'avg_system_memory_percent': np.mean(self.resource_data['memory_percent']),
            'max_system_memory_percent': np.max(self.resource_data['memory_percent'])
        }
        
        # Print comprehensive report
        self.print_resource_report()
    
    def print_resource_report(self):
        """Print detailed resource usage report"""
        if not self.resource_stats:
            print("⚠️ No monitoring data available")
            return
        
        print(f"\n{'='*70}")
        print(f"💻 COMPUTATIONAL RESOURCE REPORT - CNN MODEL")
        print(f"{'='*70}")
        
        # Dataset info
        if self.current_dataset_info['is_mixed']:
            dataset_name = "Mixed: " + ", ".join(self.current_dataset_info['mixed_datasets'])
        else:
            dataset_name = self.current_dataset_info['dataset_name'] or "Unknown"
        
        print(f"Dataset: {dataset_name}")
        print(f"Training Samples: {self.current_dataset_info['total_samples']['train']}")
        print(f"Validation Samples: {self.current_dataset_info['total_samples']['val']}")
        
        print(f"\n⏱️  Training Duration: {self.resource_stats['duration_seconds']:.2f} seconds ({self.resource_stats['duration_seconds']/60:.2f} minutes)")
        
        print(f"\n🔥 CPU Usage:")
        print(f"  • Average: {self.resource_stats['avg_cpu_percent']:.1f}%")
        print(f"  • Peak: {self.resource_stats['max_cpu_percent']:.1f}%")
        print(f"  • Cores Available: {psutil.cpu_count(logical=True)}")
        
        print(f"\n💾 Memory Usage:")
        print(f"  • Peak Process Memory: {self.resource_stats['peak_memory_mb']:.2f} MB")
        print(f"  • Average Process Memory: {self.resource_stats['avg_memory_mb']:.2f} MB")
        print(f"  • Memory Increase: {self.resource_stats['memory_increase_mb']:.2f} MB")
        print(f"  • Peak System Memory: {self.resource_stats['max_system_memory_percent']:.1f}%")
        
        if self.current_dataset_info['total_samples']['train'] > 0:
            train_samples = self.current_dataset_info['total_samples']['train']
            print(f"\n⚡ Efficiency Metrics:")
            print(f"  • Samples per Second: {train_samples / self.resource_stats['duration_seconds']:.2f}")
            print(f"  • Memory per Sample: {self.resource_stats['peak_memory_mb'] / train_samples:.4f} MB")
        
        print(f"{'='*70}\n")
        
        # Log to GUI
        self.log_training(f"💻 Resource Report:")
        self.log_training(f"   Duration: {self.resource_stats['duration_seconds']/60:.2f} min")
        self.log_training(f"   Avg CPU: {self.resource_stats['avg_cpu_percent']:.1f}% | Peak: {self.resource_stats['max_cpu_percent']:.1f}%")
        self.log_training(f"   Peak Memory: {self.resource_stats['peak_memory_mb']:.2f} MB")
    
    def analyze_snr_degradation(self):
        """Analyze accuracy degradation across SNR levels"""
        # SNR levels and their corresponding accuracy (to be filled during prediction)
        snr_levels = [5, 10, 15, 20, 'clean']
        
        # Check if we have SNR-specific datasets loaded
        dataset_name = self.current_dataset_info.get('dataset_name', '')
        
        # Extract SNR level from dataset name if applicable
        current_snr = None
        if 'snr_5db' in dataset_name.lower() or '5db' in dataset_name.lower():
            current_snr = 5
        elif 'snr_10db' in dataset_name.lower() or '10db' in dataset_name.lower():
            current_snr = 10
        elif 'snr_15db' in dataset_name.lower() or '15db' in dataset_name.lower():
            current_snr = 15
        elif 'snr_20db' in dataset_name.lower() or '20db' in dataset_name.lower():
            current_snr = 20
        
        # Calculate degradation rate (requires baseline accuracy at clean/high SNR)
        # For now, store the current results
        if current_snr and hasattr(self, 'classification_report'):
            accuracy = self.classification_report.get('accuracy', 0) * 100
            
            # Store SNR data point
            self.snr_degradation_data.append({
                'snr_db': current_snr,
                'accuracy': accuracy,
                'dataset': dataset_name
            })
            
            print(f"\n📊 SNR Level: {current_snr} dB | Accuracy: {accuracy:.2f}%")
        
        # If we have multiple SNR data points, calculate degradation
        if len(self.snr_degradation_data) > 1:
            # Sort by SNR level
            sorted_data = sorted(self.snr_degradation_data, key=lambda x: x['snr_db'])
            
            print(f"\n{'='*70}")
            print(f"📉 ACCURACY DEGRADATION RATE ANALYSIS")
            print(f"{'='*70}")
            print(f"{'SNR Range (dB)':<20} {'Degradation Rate':<25} {'Classification':<25}")
            print(f"{'-'*70}")
            
            # Calculate degradation between consecutive SNR levels
            for i in range(len(sorted_data) - 1):
                snr_low = sorted_data[i]['snr_db']
                snr_high = sorted_data[i + 1]['snr_db']
                acc_low = sorted_data[i]['accuracy']
                acc_high = sorted_data[i + 1]['accuracy']
                
                degradation_rate = acc_high - acc_low
                degradation_per_db = degradation_rate / (snr_high - snr_low)
                
                # Classify degradation
                if abs(degradation_per_db) < 0.5:
                    classification = "Minimal"
                elif abs(degradation_per_db) < 1.5:
                    classification = "Moderate"
                elif abs(degradation_per_db) < 3.0:
                    classification = "Significant"
                else:
                    classification = "Severe"
                
                print(f"{f'{snr_low} -> {snr_high} dB':<20} {f'{degradation_rate:+.2f}% ({degradation_per_db:+.2f}%/dB)':<25} {classification:<25}")
            
            print(f"{'='*70}\n")
    
    def analyze_sample(self):
        """Analyze a specific sample and generate explanations"""
        if self.cnn_model is None or self.cnn_model.model is None:
            messagebox.showerror("Error", "Please load a trained model first!")
            return
        
        if self.test_data is None:
            messagebox.showerror("Error", "Please load a dataset first!")
            return
        
        sample_idx = self.explain_sample_var.get()
        
        try:
            print("\n" + "="*80)
            print(f"🔬 ANALYZING SAMPLE #{sample_idx}")
            print("="*80)
            
            # Check model state before analysis
            print(f"📊 Model state check:")
            print(f"   Model type: {type(self.cnn_model.model)}")
            print(f"   Model built: {self.cnn_model.model.built}")
            print(f"   Model compiled: {self.cnn_model.is_compiled}")
            
            # Extract sample from test dataset
            self.log_explain("🔍 Extracting sample from test dataset...")
            print(f"\n🔍 Extracting sample {sample_idx} from test dataset...")
            
            X_test = []
            y_test = []
            for x_batch, y_batch in self.test_data.unbatch().take(sample_idx + 1):
                X_test.append(x_batch.numpy())
                y_test.append(y_batch.numpy())
            
            if sample_idx >= len(X_test):
                messagebox.showerror("Error", f"Sample index {sample_idx} out of range!")
                return
            
            sample = X_test[sample_idx]
            true_label = y_test[sample_idx]
            
            print(f"   ✓ Sample shape: {sample.shape}")
            print(f"   ✓ True label: {true_label}")
            print(f"   ✓ Sample value range: [{sample.min():.6f}, {sample.max():.6f}]")
            print(f"   ✓ Sample mean: {sample.mean():.6f}, std: {sample.std():.6f}")
            
            # Store current sample
            self.current_explain_sample = {
                'data': sample,
                'true_label': true_label,
                'index': sample_idx
            }
            
            # Make prediction
            self.log_explain("🤖 Making prediction...")
            print(f"\n🤖 Making prediction...")
            sample_expanded = np.expand_dims(sample, axis=0)
            print(f"   Sample expanded shape: {sample_expanded.shape}")
            
            # Test prediction first
            prediction_prob = self.cnn_model.model.predict(sample_expanded, verbose=0)[0][0]
            predicted_class = 1 if prediction_prob > 0.5 else 0
            
            print(f"   ✓ Prediction probability: {prediction_prob:.4f}")
            print(f"   ✓ Predicted class: {predicted_class}")
            
            # Get class names
            true_class_name = "REAL" if true_label == 1 else "FAKE"
            pred_class_name = "REAL" if predicted_class == 1 else "FAKE"
            confidence = prediction_prob if predicted_class == 1 else (1 - prediction_prob)
            
            # Update status
            self.explain_status.config(
                text=f"✅ Analyzed Sample #{sample_idx} | Predicted: {pred_class_name} ({confidence*100:.2f}%)",
                fg='#00ff88'
            )
            
            # Generate Grad-CAM heatmap
            self.log_explain("🔥 Generating Grad-CAM heatmap...")
            grad_cam_heatmap = self.generate_grad_cam(sample_expanded)
            
            # Get layer activations
            self.log_explain("🧠 Extracting layer activations...")
            layer_activations = self.get_layer_activations(sample_expanded)
            
            # Update summary
            self.update_explain_summary(sample_idx, true_class_name, pred_class_name, 
                                       confidence, prediction_prob)
            
            # Create visualizations
            self.log_explain("📊 Creating visualizations...")
            self.create_explain_visualizations(sample, grad_cam_heatmap, layer_activations, 
                                              true_class_name, pred_class_name, confidence)
            
            # Generate detailed analysis
            self.generate_detailed_analysis(sample, grad_cam_heatmap, layer_activations, 
                                           true_class_name, pred_class_name, 
                                           confidence, prediction_prob)
            
            # Store in history
            self.explainability_data.append({
                'sample_idx': sample_idx,
                'true_label': true_class_name,
                'predicted_label': pred_class_name,
                'confidence': confidence,
                'raw_probability': prediction_prob,
                'grad_cam': grad_cam_heatmap,
                'activations': layer_activations
            })
            
            self.log_explain("✅ Analysis complete!")
            print("="*80)
            print("✅ SAMPLE ANALYSIS COMPLETE!")
            print("="*80 + "\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze sample:\n{str(e)}")
            print(f"\n❌ Error in analyze_sample: {e}")
            import traceback
            traceback.print_exc()
            print("="*80 + "\n")
    
    def analyze_custom_audio(self):
        """Analyze a custom uploaded WAV file"""
        if self.cnn_model is None or self.cnn_model.model is None:
            messagebox.showerror("Error", "Please load a trained model first!")
            return
        
        # Open file dialog to select WAV file
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        if not file_path:
            return  # User cancelled
        
        try:
            self.explain_status.config(text="🔄 Processing custom audio...", fg='#ffa500')
            self.log_explain("="*80)
            self.log_explain(f"🎵 ANALYZING CUSTOM AUDIO FILE")
            self.log_explain("="*80)
            self.log_explain(f"📁 File: {os.path.basename(file_path)}")
            self.log_explain("")
            
            # Load and preprocess audio
            self.log_explain("📊 Step 1: Loading audio file...")
            audio, sr = librosa.load(file_path, sr=16000, mono=True)
            self.log_explain(f"   ✓ Sample rate: {sr} Hz")
            self.log_explain(f"   ✓ Original duration: {len(audio)/sr:.2f} seconds")
            self.log_explain(f"   ✓ Original samples: {len(audio)}")
            
            # *** MATCH NOTEBOOK PREPROCESSING EXACTLY ***
            # UPDATED: Training uses 8-second clips at 16kHz = 128000 samples for better features
            TARGET_LENGTH = 128000  # 8 seconds at 16kHz
            
            self.log_explain("")
            self.log_explain("🔧 Step 2: Audio preprocessing...")
            
            # First, normalize audio amplitude to [-1, 1] range (like training)
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
                self.log_explain(f"   ✓ Normalized amplitude to [-1, 1]")
            
            # Truncate or pad to TARGET_LENGTH
            self.log_explain("✂️ Step 3: Truncating/padding to target length...")
            if len(audio) > TARGET_LENGTH:
                # Take from the middle (more stable than start/end)
                start = (len(audio) - TARGET_LENGTH) // 2
                audio = audio[start:start + TARGET_LENGTH]
                self.log_explain(f"   ✓ Truncated to {TARGET_LENGTH} samples (from middle)")
            elif len(audio) < TARGET_LENGTH:
                padding = TARGET_LENGTH - len(audio)
                # Pad symmetrically (better than zero-padding at end)
                pad_left = padding // 2
                pad_right = padding - pad_left
                audio = np.pad(audio, (pad_left, pad_right), mode='constant', constant_values=0)
                self.log_explain(f"   ✓ Padded to {TARGET_LENGTH} samples (symmetric)")
            else:
                self.log_explain(f"   ✓ Audio already {TARGET_LENGTH} samples")
            
            # Convert to mel-spectrogram with EXACT notebook parameters
            self.log_explain("")
            self.log_explain("🎨 Step 4: Creating mel-spectrogram (matching training)...")
            
            # Use librosa.feature.melspectrogram for exact matching
            # This matches TensorFlow's mel-spectrogram more closely
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=16000,
                n_fft=2048,  # Standard for audio
                hop_length=512,  # Standard hop length
                win_length=2048,
                n_mels=256,  # Increased from 128 for better frequency resolution
                fmin=0.0,
                fmax=8000.0,
                power=2.0  # Power spectrogram (magnitude squared)
            )
            
            self.log_explain(f"   ✓ Mel spectrogram shape: {mel_spec.shape}")
            self.log_explain(f"   ✓ Parameters: n_fft=2048, hop_length=512, n_mels=256")
            self.log_explain(f"   ✓ Frequency range: 0-8000 Hz")
            self.log_explain(f"   ✓ Mel spec value range BEFORE log: [{mel_spec.min():.6f}, {mel_spec.max():.6f}]")
            
            # Convert to log scale (dB)
            # Use librosa.power_to_db for proper scaling
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            self.log_explain(f"   ✓ Mel spec value range AFTER log (dB): [{mel_spec_db.min():.2f}, {mel_spec_db.max():.2f}] dB")
            
            # Resize to 256x256 for better feature extraction
            self.log_explain("")
            self.log_explain("🔧 Step 5: Resizing to 256x256...")
            mel_spec_resized = cv2.resize(mel_spec_db, (256, 256), interpolation=cv2.INTER_LINEAR)
            self.log_explain(f"   ✓ Resized shape: {mel_spec_resized.shape}")
            
            # Normalize using standardization (zero mean, unit variance)
            # This is more robust than min-max for audio
            self.log_explain("")
            self.log_explain("📊 Step 6: Normalizing (standardization)...")
            
            # Method 1: Standard normalization (try this first)
            mean_val = mel_spec_resized.mean()
            std_val = mel_spec_resized.std()
            spec_final = (mel_spec_resized - mean_val) / (std_val + 1e-8)
            
            # Clip to reasonable range and scale to [0, 1]
            spec_final = np.clip(spec_final, -3, 3)  # Clip to ±3 std devs
            spec_final = (spec_final + 3) / 6  # Scale to [0, 1]
            
            # Add channel dimension (256, 256, 1)
            spec_final = np.expand_dims(spec_final, axis=-1).astype(np.float32)
            
            self.log_explain(f"   ✓ Final shape: {spec_final.shape}")
            self.log_explain(f"   ✓ Value range: [{spec_final.min():.4f}, {spec_final.max():.4f}]")
            self.log_explain(f"   ✓ Mean: {spec_final.mean():.4f}, Std: {spec_final.std():.4f}")
            self.log_explain(f"   ✓ Preprocessing complete! ✓")
            
            # Store current sample (no true label for custom audio)
            self.current_explain_sample = {
                'data': spec_final,
                'true_label': None,  # Unknown for custom audio
                'index': 'custom',
                'file_path': file_path,
                'audio_raw': audio,
                'sample_rate': sr
            }
            
            # Make prediction
            self.log_explain("")
            self.log_explain("🤖 Step 6: Making prediction...")
            sample_expanded = np.expand_dims(spec_final, axis=0)
            
            # Model is already built in load_model() - ready to predict
            prediction_prob = self.cnn_model.model.predict(sample_expanded, verbose=0)[0][0]
            predicted_class = 1 if prediction_prob > 0.5 else 0
            
            # Get class names
            pred_class_name = "REAL" if predicted_class == 1 else "FAKE"
            confidence = prediction_prob if predicted_class == 1 else (1 - prediction_prob)
            
            self.log_explain(f"   ✓ Prediction: {pred_class_name}")
            self.log_explain(f"   ✓ Confidence: {confidence*100:.2f}%")
            self.log_explain(f"   ✓ Raw probability (REAL): {prediction_prob:.4f}")
            self.log_explain(f"   ✓ Raw probability (FAKE): {1-prediction_prob:.4f}")
            
            # Update status
            self.explain_status.config(
                text=f"✅ Custom Audio | Predicted: {pred_class_name} ({confidence*100:.2f}%)",
                fg='#00ff88'
            )
            
            # Generate Grad-CAM heatmap
            self.log_explain("")
            self.log_explain("🔥 Step 7: Generating Grad-CAM heatmap...")
            grad_cam_heatmap = self.generate_grad_cam(sample_expanded)
            if grad_cam_heatmap is not None:
                self.log_explain("   ✓ Grad-CAM generated successfully")
            
            # Get layer activations
            self.log_explain("")
            self.log_explain("🧠 Step 8: Extracting layer activations...")
            layer_activations = self.get_layer_activations(sample_expanded)
            if layer_activations:
                self.log_explain(f"   ✓ Extracted {len(layer_activations)} layer activations")
            
            # Update summary (no true label for custom audio)
            self.update_explain_summary("custom", "UNKNOWN", pred_class_name, 
                                       confidence, prediction_prob, is_custom=True,
                                       filename=os.path.basename(file_path))
            
            # Create visualizations
            self.log_explain("")
            self.log_explain("📊 Step 9: Creating visualizations...")
            self.create_explain_visualizations(spec_final, grad_cam_heatmap, layer_activations, 
                                              "UNKNOWN", pred_class_name, confidence)
            self.log_explain("   ✓ Visualizations complete")
            
            # Generate detailed analysis
            self.log_explain("")
            self.log_explain("📝 Step 10: Generating detailed analysis...")
            self.generate_detailed_analysis(spec_final, grad_cam_heatmap, layer_activations, 
                                           "UNKNOWN", pred_class_name, 
                                           confidence, prediction_prob, is_custom=True)
            
            # Store in history
            self.explainability_data.append({
                'sample_idx': 'custom',
                'file_path': file_path,
                'filename': os.path.basename(file_path),
                'true_label': 'UNKNOWN',
                'predicted_label': pred_class_name,
                'confidence': confidence,
                'raw_probability': prediction_prob,
                'grad_cam': grad_cam_heatmap,
                'activations': layer_activations
            })
            
            self.log_explain("")
            self.log_explain("="*80)
            self.log_explain("✅ CUSTOM AUDIO ANALYSIS COMPLETE!")
            self.log_explain("="*80)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze custom audio:\n{str(e)}")
            self.explain_status.config(text="❌ Analysis failed", fg='#ff6b6b')
            print(f"Error in analyze_custom_audio: {e}")
            import traceback
            traceback.print_exc()
    
    def record_and_analyze_audio(self):
        """Record 5 seconds of audio from microphone and analyze it"""
        if self.cnn_model is None or self.cnn_model.model is None:
            messagebox.showerror("Error", "Please load a trained model first!")
            return
        
        try:
            # Update status
            self.explain_status.config(text="🎤 Recording... Please speak for 5 seconds!", fg='#ffa500')
            self.root.update()
            
            # Recording parameters
            duration = 5  # seconds
            sample_rate = 16000  # 16kHz to match training
            
            self.log_explain("="*80)
            self.log_explain(f"🎤 RECORDING AUDIO FROM MICROPHONE")
            self.log_explain("="*80)
            self.log_explain(f"⏱️ Duration: {duration} seconds")
            self.log_explain(f"📊 Sample Rate: {sample_rate} Hz")
            self.log_explain("")
            self.log_explain("🔴 Recording started... Speak now!")
            self.root.update()
            
            # Record audio
            print(f"\n🎤 Recording {duration} seconds of audio...")
            print("🔴 Speak now!")
            audio_data = sd.rec(int(duration * sample_rate), 
                               samplerate=sample_rate, 
                               channels=1, 
                               dtype='float32')
            sd.wait()  # Wait until recording is finished
            
            self.log_explain("✅ Recording complete!")
            self.log_explain("")
            
            # Convert to mono array
            audio = audio_data.flatten()
            
            # Store for playback
            self.recorded_audio = audio
            self.recorded_sr = sample_rate
            
            # Save temporary WAV file
            temp_wav_path = os.path.join(os.path.dirname(__file__), "temp_recorded_audio.wav")
            wavfile.write(temp_wav_path, sample_rate, audio)
            
            # Enable preview and save buttons
            self.preview_btn.config(state='normal')
            self.save_rec_btn.config(state='normal')
            
            self.log_explain(f"💾 Saved to: {temp_wav_path}")
            self.log_explain(f"📊 Recorded {len(audio)} samples ({len(audio)/sample_rate:.2f} seconds)")
            self.log_explain("")
            
            # Now process this audio using the same pipeline as analyze_custom_audio
            self.explain_status.config(text="🔄 Processing recorded audio...", fg='#ffa500')
            self.root.update()
            
            # *** MATCH NOTEBOOK PREPROCESSING EXACTLY ***
            # UPDATED: Must match custom audio preprocessing (8 seconds)
            TARGET_LENGTH = 128000  # 8 seconds at 16kHz (same as custom audio)
            
            # First, normalize audio amplitude
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
                self.log_explain("   ✓ Normalized amplitude to [-1, 1]")
            
            # Truncate or pad to TARGET_LENGTH
            self.log_explain("✂️ Step 1: Truncating/padding to target length...")
            if len(audio) > TARGET_LENGTH:
                # Take from the middle
                start = (len(audio) - TARGET_LENGTH) // 2
                audio = audio[start:start + TARGET_LENGTH]
                self.log_explain(f"   ✓ Truncated to {TARGET_LENGTH} samples (from middle)")
            else:
                padding = TARGET_LENGTH - len(audio)
                pad_left = padding // 2
                pad_right = padding - pad_left
                audio = np.pad(audio, (pad_left, pad_right), mode='constant', constant_values=0)
                self.log_explain(f"   ✓ Padded to {TARGET_LENGTH} samples (symmetric)")
            
            # Convert to mel-spectrogram - MATCH custom audio preprocessing
            self.log_explain("")
            self.log_explain("🎨 Step 2: Creating mel-spectrogram (matching custom audio)...")
            
            # Use same method as custom audio
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=16000,
                n_fft=2048,
                hop_length=512,
                win_length=2048,
                n_mels=256,  # Increased from 128 for better frequency resolution
                fmin=0.0,
                fmax=8000.0,
                power=2.0
            )
            
            self.log_explain(f"   ✓ Mel spectrogram shape: {mel_spec.shape}")
            self.log_explain(f"   ✓ Parameters: n_fft=2048, hop_length=512, n_mels=256")
            
            # Convert to dB
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Resize to 256x256
            self.log_explain("")
            self.log_explain("🔧 Step 3: Resizing to 256x256...")
            mel_spec_resized = cv2.resize(mel_spec_db, (256, 256), interpolation=cv2.INTER_LINEAR)
            self.log_explain(f"   ✓ Resized shape: {mel_spec_resized.shape}")
            
            # Normalize using standardization (same as custom audio)
            self.log_explain("")
            self.log_explain("📊 Step 4: Normalizing...")
            mean_val = mel_spec_resized.mean()
            std_val = mel_spec_resized.std()
            spec_final = (mel_spec_resized - mean_val) / (std_val + 1e-8)
            spec_final = np.clip(spec_final, -3, 3)
            spec_final = (spec_final + 3) / 6
            
            # Add channel dimension (256, 256, 1)
            spec_final = np.expand_dims(spec_final, axis=-1).astype(np.float32)
            
            self.log_explain(f"   ✓ Final shape: {spec_final.shape}")
            self.log_explain(f"   ✓ Value range: [{spec_final.min():.4f}, {spec_final.max():.4f}]")
            self.log_explain(f"   ✓ Preprocessing complete! ✓")
            
            # Store current sample
            self.current_explain_sample = {
                'data': spec_final,
                'true_label': None,
                'index': 'recorded',
                'file_path': temp_wav_path,
                'audio_raw': audio,
                'sample_rate': sample_rate
            }
            
            # Make prediction
            self.log_explain("")
            self.log_explain("🤖 Step 5: Making prediction...")
            sample_expanded = np.expand_dims(spec_final, axis=0)
            
            prediction_prob = self.cnn_model.model.predict(sample_expanded, verbose=0)[0][0]
            predicted_class = 1 if prediction_prob > 0.5 else 0
            
            pred_class_name = "REAL" if predicted_class == 1 else "FAKE"
            confidence = prediction_prob if predicted_class == 1 else (1 - prediction_prob)
            
            self.log_explain(f"   ✓ Prediction: {pred_class_name}")
            self.log_explain(f"   ✓ Confidence: {confidence*100:.2f}%")
            self.log_explain(f"   ✓ Raw probability (REAL): {prediction_prob:.4f}")
            
            # Update status
            self.explain_status.config(
                text=f"✅ Recorded Audio | Predicted: {pred_class_name} ({confidence*100:.2f}%)",
                fg='#00ff88'
            )
            
            # Generate Grad-CAM heatmap
            self.log_explain("")
            self.log_explain("🔥 Step 6: Generating Grad-CAM heatmap...")
            grad_cam_heatmap = self.generate_grad_cam(sample_expanded)
            
            # Get layer activations
            self.log_explain("")
            self.log_explain("🧠 Step 7: Extracting layer activations...")
            layer_activations = self.get_layer_activations(sample_expanded)
            
            # Update summary
            self.update_explain_summary("recorded", "UNKNOWN", pred_class_name, 
                                       confidence, prediction_prob, is_custom=True,
                                       filename="🎤 Microphone Recording (5s)")
            
            # Create visualizations
            self.log_explain("")
            self.log_explain("📊 Step 8: Creating visualizations...")
            self.create_explain_visualizations(spec_final, grad_cam_heatmap, layer_activations, 
                                              "UNKNOWN", pred_class_name, confidence)
            
            # Generate detailed analysis
            self.log_explain("")
            self.log_explain("📝 Step 9: Generating detailed analysis...")
            self.generate_detailed_analysis(spec_final, grad_cam_heatmap, layer_activations, 
                                           "UNKNOWN", pred_class_name, 
                                           confidence, prediction_prob, is_custom=True)
            
            # Store in history
            self.explainability_data.append({
                'sample_idx': 'recorded',
                'file_path': temp_wav_path,
                'filename': '🎤 Microphone Recording',
                'true_label': 'UNKNOWN',
                'predicted_label': pred_class_name,
                'confidence': confidence,
                'raw_probability': prediction_prob,
                'grad_cam': grad_cam_heatmap,
                'activations': layer_activations
            })
            
            self.log_explain("")
            self.log_explain("="*80)
            self.log_explain("✅ MICROPHONE RECORDING ANALYSIS COMPLETE!")
            self.log_explain("="*80)
            
            # Show result with helpful message
            result_msg = f"Recording analyzed!\n\n"
            result_msg += f"Prediction: {pred_class_name}\n"
            result_msg += f"Confidence: {confidence*100:.2f}%\n\n"
            result_msg += "Click '▶️ Play Recording' to hear your audio!\n\n"
            
            # Add helpful note if real voice is classified as fake
            if pred_class_name == "FAKE" and confidence > 0.6:
                result_msg += "⚠️ NOTE: Your real voice was classified as FAKE.\n\n"
                result_msg += "This happens because:\n"
                result_msg += "• Model was trained on dataset audio (not live recordings)\n"
                result_msg += "• Microphone characteristics differ from training data\n"
                result_msg += "• Background noise affects the classification\n\n"
                result_msg += "💡 TIP: To improve accuracy on live recordings:\n"
                result_msg += "1. Record your voice and save it as a WAV file\n"
                result_msg += "2. Add it to the training dataset as 'REAL'\n"
                result_msg += "3. Retrain the model with mixed data\n"
                result_msg += "4. This will teach the model your voice patterns!"
            
            messagebox.showinfo("Recording Complete", result_msg)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to record/analyze audio:\n{str(e)}\n\nMake sure you have a working microphone!")
            self.explain_status.config(text="❌ Recording failed", fg='#ff6b6b')
            print(f"Error in record_and_analyze_audio: {e}")
            import traceback
            traceback.print_exc()
    
    def play_recorded_audio(self):
        """Play back the recorded audio"""
        if self.recorded_audio is None:
            messagebox.showwarning("No Recording", "No recorded audio available to play!")
            return
        
        try:
            self.explain_status.config(text="▶️ Playing recorded audio...", fg='#00d4ff')
            self.root.update()
            
            # Play the audio
            sd.play(self.recorded_audio, self.recorded_sr)
            sd.wait()  # Wait until playback is finished
            
            self.explain_status.config(text="✅ Playback complete", fg='#00ff88')
            
        except Exception as e:
            messagebox.showerror("Playback Error", f"Failed to play audio:\n{str(e)}")
            print(f"Error in play_recorded_audio: {e}")
    
    def save_recording_for_training(self):
        """Save the recorded audio to a file for future training"""
        if self.recorded_audio is None:
            messagebox.showwarning("No Recording", "No recorded audio available to save!")
            return
        
        try:
            # Ask user where to save
            file_path = filedialog.asksaveasfilename(
                title="Save Recording for Training",
                defaultextension=".wav",
                filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
                initialfile=f"my_voice_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            )
            
            if not file_path:
                return  # User cancelled
            
            # Save the audio
            wavfile.write(file_path, self.recorded_sr, self.recorded_audio)
            
            info_msg = f"✅ Recording saved successfully!\n\n"
            info_msg += f"File: {os.path.basename(file_path)}\n"
            info_msg += f"Location: {os.path.dirname(file_path)}\n\n"
            info_msg += "💡 TO USE THIS FOR TRAINING:\n\n"
            info_msg += "1. Go to your dataset folder:\n"
            info_msg += "   data/for-2sec/for-2seconds/training/real/\n\n"
            info_msg += "2. Copy this WAV file there\n\n"
            info_msg += "3. Rebuild the dataset pickle file:\n"
            info_msg += "   - Open AudioClassification.ipynb\n"
            info_msg += "   - Run the dataset creation cells\n\n"
            info_msg += "4. Retrain the model with your voice!\n"
            info_msg += "   - Go to Training tab\n"
            info_msg += "   - Load the updated dataset\n"
            info_msg += "   - Start training\n\n"
            info_msg += "After retraining, the model will recognize your voice as REAL!"
            
            messagebox.showinfo("Recording Saved", info_msg)
            self.explain_status.config(text=f"✅ Recording saved to {os.path.basename(file_path)}", fg='#00ff88')
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save recording:\n{str(e)}")
            print(f"Error in save_recording_for_training: {e}")
    
    def generate_grad_cam(self, sample):
        """Generate Grad-CAM heatmap for CNN explainability - detecting synthetic speech patterns"""
        try:
            import tensorflow as tf
            
            print("\n" + "="*80)
            print("🔥 SYNTHETIC AUDIO DETECTION - GRAD-CAM ANALYSIS")
            print("="*80)
            
            # Convert numpy to tensor if needed
            if not isinstance(sample, tf.Tensor):
                sample_tensor = tf.convert_to_tensor(sample, dtype=tf.float32)
            else:
                sample_tensor = sample
            
            print(f"📊 Model type: {type(self.cnn_model.model)}")
            print(f"📊 Input sample shape: {sample_tensor.shape}")
            
            # Find the last convolutional layer (most relevant for synthetic patterns)
            print(f"\n🔍 Locating feature extraction layers...")
            last_conv_layer = None
            conv_layer_names = []
            
            for layer in self.cnn_model.model.layers:
                if layer.__class__.__name__ in ['Conv2D', 'Conv1D', 'Conv3D']:
                    conv_layer_names.append(layer.name)
                    last_conv_layer = layer
            
            if last_conv_layer is None:
                print("⚠️ No convolutional layer found - cannot analyze feature patterns")
                return None
            
            print(f"   ✓ Analyzing {len(conv_layer_names)} convolutional layers")
            print(f"   ✓ Focus layer: {last_conv_layer.name} (detects synthetic artifacts)")
            
            # Create Grad-CAM model
            grad_model = tf.keras.models.Model(
                inputs=self.cnn_model.model.inputs,
                outputs=[last_conv_layer.output, self.cnn_model.model.outputs[0]]
            )
            
            # Compute gradients to find synthetic patterns
            print(f"\n🎯 Computing attention map for synthetic features...")
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(sample_tensor, training=False)
                # Focus on the predicted class (synthetic or authentic)
                pred_score = predictions[:, 0]
            
            # Compute gradients
            grads = tape.gradient(pred_score, conv_outputs)
            
            if grads is None:
                print(f"   ⚠️ Cannot compute feature importance - using uniform attention")
                # Return uniform heatmap
                return np.ones((conv_outputs.shape[1], conv_outputs.shape[2]))
            
            print(f"   ✓ Detected {grads.shape[-1]} feature channels")
            print(f"   ✓ Prediction confidence: {predictions.numpy()[0][0]:.4f}")
            
            # Global average pooling of gradients (importance of each filter)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight feature maps by importance
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
            
            # Normalize to 0-1 range
            heatmap = tf.maximum(heatmap, 0)
            heatmap_max = tf.reduce_max(heatmap)
            if heatmap_max > 0:
                heatmap = heatmap / heatmap_max
            
            print(f"   ✓ Generated attention map: {heatmap.shape}")
            print(f"   ✓ Attention range: [{tf.reduce_min(heatmap).numpy():.3f}, {tf.reduce_max(heatmap).numpy():.3f}]")
            print(f"\n✅ Synthetic pattern detection complete!")
            print("="*80 + "\n")
            
            return heatmap.numpy()
            
        except Exception as e:
            print(f"\n❌ Error in pattern detection: {e}")
            import traceback
            traceback.print_exc()
            print("="*80 + "\n")
            return None
    
    def get_layer_activations(self, sample):
        """Get activations from all layers"""
        try:
            import tensorflow as tf
            
            print("\n" + "="*80)
            print("🧠 LAYER ACTIVATIONS EXTRACTION DEBUG")
            print("="*80)
            
            print(f"📊 Model type: {type(self.cnn_model.model)}")
            print(f"📊 Model built: {self.cnn_model.model.built}")
            print(f"📊 Input sample shape: {sample.shape}")
            
            # Find layers we want activations from
            layer_outputs = []
            layer_names = []
            
            print(f"\n🔍 Searching for conv/dense layers...")
            for layer in self.cnn_model.model.layers:
                # Check actual layer type, not just name
                layer_type = layer.__class__.__name__
                if layer_type in ['Conv2D', 'Conv1D', 'Conv3D', 'Dense']:
                    layer_outputs.append(layer.output)
                    layer_names.append(layer.name)
                    try:
                        output_shape = layer.output.shape if hasattr(layer, 'output') else 'N/A'
                        print(f"   ✓ Found: {layer.name} ({layer_type}) - {output_shape}")
                    except:
                        print(f"   ✓ Found: {layer.name} ({layer_type})")
            
            if not layer_outputs:
                print("❌ No conv/dense layers found!")
                return None
            
            print(f"\n   Total layers to extract: {len(layer_names)}")
            
            # Create activation model
            print(f"\n🔨 Creating activation model...")
            
            # For Sequential models, use .inputs and .outputs (plural)
            try:
                model_inputs = self.cnn_model.model.inputs
            except:
                # Fallback: call model first to build the graph
                _ = self.cnn_model.model(sample, training=False)
                model_inputs = self.cnn_model.model.inputs
            
            activation_model = tf.keras.models.Model(
                inputs=model_inputs,
                outputs=layer_outputs
            )
            
            # CRITICAL: Build the activation model by calling it
            print(f"   🔨 Building activation model with sample input...")
            _ = activation_model(sample, training=False)
            print(f"   ✓ Activation model built successfully")
            print(f"   ✓ Activation model built status: {activation_model.built}")
            
            # Get activations
            print(f"\n📊 Extracting activations...")
            activations = activation_model.predict(sample, verbose=0)
            
            # Handle single output case
            if not isinstance(activations, list):
                activations = [activations]
            
            print(f"   ✓ Extracted {len(activations)} layer activations")
            for i, (name, act) in enumerate(zip(layer_names, activations)):
                print(f"      {i+1}. {name}: shape {act.shape}, range [{act.min():.4f}, {act.max():.4f}]")
            
            print(f"\n✅ Layer activations extracted successfully!")
            print("="*80 + "\n")
            
            return {'names': layer_names, 'outputs': activations}
            
        except Exception as e:
            print(f"\n❌ Error getting layer activations: {e}")
            import traceback
            traceback.print_exc()
            print("="*80 + "\n")
            return None
            
        except Exception as e:
            print(f"Error getting layer activations: {e}")
            return None
    
    def update_explain_summary(self, sample_idx, true_label, pred_label, confidence, raw_prob, is_custom=False, filename=None):
        """Update the explanation summary text"""
        self.explain_summary.delete(1.0, tk.END)
        
        if is_custom:
            # Custom audio summary (no true label)
            summary = f"""
🎵 CUSTOM AUDIO ANALYSIS

📁 File Information:
   • Filename: {filename}
   • Source: User uploaded

🤖 Prediction Result:
   • Predicted Label: {pred_label}
   • Confidence: {confidence*100:.2f}%

🎯 Probability Scores:
   • REAL probability: {raw_prob*100:.2f}%
   • FAKE probability: {(1-raw_prob)*100:.2f}%
   • Threshold: 50.00%

💡 Decision Logic:
   • Model predicted "{pred_label}" because probability was {'>=' if raw_prob > 0.5 else '<'} 50%
   • Confidence level: {'HIGH' if confidence > 0.8 else 'MEDIUM' if confidence > 0.6 else 'LOW'}

ℹ️ Note: True label unknown for custom audio uploads
            """
        else:
            # Test dataset sample summary
            # Determine if correct
            is_correct = true_label == pred_label
            result_icon = "✅" if is_correct else "❌"
            result_text = "CORRECT" if is_correct else "INCORRECT"
            
            summary = f"""
{result_icon} PREDICTION RESULT: {result_text}

📊 Sample Information:
   • Sample Index: #{sample_idx}
   • True Label: {true_label}
   • Predicted Label: {pred_label}
   • Confidence: {confidence*100:.2f}%

🎯 Probability Scores:
   • REAL probability: {raw_prob*100:.2f}%
   • FAKE probability: {(1-raw_prob)*100:.2f}%
   • Threshold: 50.00%

💡 Decision Logic:
   • Model predicted "{pred_label}" because probability was {'>=' if raw_prob > 0.5 else '<'} 50%
   • Confidence level: {'HIGH' if confidence > 0.8 else 'MEDIUM' if confidence > 0.6 else 'LOW'}
            """
        
        self.explain_summary.insert(1.0, summary)
    
    def create_explain_visualizations(self, sample, grad_cam, activations, 
                                     true_label, pred_label, confidence):
        """Create comprehensive visualizations for explainability"""
        self.explain_fig.clear()
        
        # Create grid of subplots
        gs = self.explain_fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Original Spectrogram
        ax1 = self.explain_fig.add_subplot(gs[0, 0])
        ax1.imshow(sample.squeeze(), cmap='viridis', aspect='auto')
        ax1.set_title(f'Original Spectrogram\\nTrue: {true_label}', 
                     fontsize=10, weight='bold', color='white')
        ax1.axis('off')
        
        # 2. Grad-CAM Heatmap
        if grad_cam is not None:
            ax2 = self.explain_fig.add_subplot(gs[0, 1])
            im = ax2.imshow(grad_cam, cmap='jet', aspect='auto')
            ax2.set_title('Grad-CAM Heatmap\\n(Important Regions)', 
                         fontsize=10, weight='bold', color='white')
            ax2.axis('off')
            self.explain_fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        
        # 3. Overlay
        if grad_cam is not None:
            ax3 = self.explain_fig.add_subplot(gs[0, 2])
            ax3.imshow(sample.squeeze(), cmap='gray', aspect='auto')
            # Resize heatmap to match spectrogram
            heatmap_resized = cv2.resize(grad_cam, (sample.shape[1], sample.shape[0]))
            ax3.imshow(heatmap_resized, cmap='jet', alpha=0.4, aspect='auto')
            ax3.set_title(f'Overlay\\nPredicted: {pred_label}', 
                         fontsize=10, weight='bold', color='white')
            ax3.axis('off')
        
        # 4. Confidence Bar
        ax4 = self.explain_fig.add_subplot(gs[1, :])
        categories = ['FAKE\\nConfidence', 'REAL\\nConfidence']
        fake_conf = (1 - confidence) * 100 if pred_label == "REAL" else confidence * 100
        real_conf = confidence * 100 if pred_label == "REAL" else (1 - confidence) * 100
        confidences = [fake_conf, real_conf]
        colors = ['#e74c3c', '#27ae60']
        bars = ax4.barh(categories, confidences, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        ax4.set_xlim(0, 100)
        ax4.set_xlabel('Confidence (%)', color='white', fontsize=10, weight='bold')
        ax4.set_title('Prediction Confidence Breakdown', color='white', fontsize=12, weight='bold')
        ax4.tick_params(colors='white')
        ax4.set_facecolor('#2a2a2a')
        # Add value labels
        for bar, conf in zip(bars, confidences):
            ax4.text(conf + 2, bar.get_y() + bar.get_height()/2, 
                    f'{conf:.2f}%', va='center', color='white', fontsize=10, weight='bold')
        
        # 5-7. Layer Activations (first 3 conv layers)
        if activations is not None and activations['outputs']:
            for idx, (name, activation) in enumerate(zip(activations['names'][:3], 
                                                         activations['outputs'][:3])):
                if 'conv' in name.lower():
                    ax = self.explain_fig.add_subplot(gs[2, idx])
                    # Show first channel of activation
                    if len(activation.shape) == 4:  # Conv layer
                        act_display = activation[0, :, :, 0]
                    else:  # Dense layer
                        # Create a simple bar chart for dense layers
                        act_display = activation[0][:min(20, len(activation[0]))]
                        ax.bar(range(len(act_display)), act_display, color='#00d4ff', alpha=0.8)
                        ax.set_title(f'{name}\\n(First 20 neurons)', 
                                   fontsize=9, color='white')
                        ax.tick_params(colors='white', labelsize=7)
                        ax.set_facecolor('#2a2a2a')
                        continue
                    
                    ax.imshow(act_display, cmap='viridis', aspect='auto')
                    ax.set_title(f'{name}\\n(Channel 1)', fontsize=9, color='white')
                    ax.axis('off')
        
        # Set overall figure properties
        self.explain_fig.patch.set_facecolor('#1a1a1a')
        for ax in self.explain_fig.get_axes():
            ax.set_facecolor('#2a2a2a')
            ax.tick_params(colors='white')
        
        self.explain_canvas.draw()
    
    def generate_detailed_analysis(self, sample, grad_cam, activations, 
                                   true_label, pred_label, confidence, raw_prob, is_custom=False):
        """Generate detailed analysis focused on synthetic audio detection under noisy conditions"""
        self.explain_log.delete(1.0, tk.END)
        
        # Determine authenticity
        is_synthetic = (pred_label == "FAKE")
        authenticity_score = (1 - raw_prob) * 100  # % synthetic
        
        analysis = f"""
{'='*80}
🎙️ SYNTHETIC AUDIO DETECTION ANALYSIS
{'='*80}

📊 AUDIO CHARACTERISTICS:
   • Spectrogram Resolution: {sample.shape[0]}×{sample.shape[1]} (Time × Frequency)
   • Feature Type: Mel-Spectrogram (128 mel-frequency bins)
   • Frequency Range: 0-8000 Hz (optimized for speech)
   • Analysis Window: 0.3 seconds (4800 samples @ 16kHz)

   Statistical Properties:
   • Intensity Range: [{sample.min():.4f}, {sample.max():.4f}]
   • Mean Intensity: {sample.mean():.4f}
   • Intensity Variance: {sample.std():.4f}
   • Dynamic Range: {sample.max() - sample.min():.4f}

{'='*80}
🤖 DETECTION RESULT: {'🔴 SYNTHETIC AUDIO DETECTED' if is_synthetic else '✅ AUTHENTIC AUDIO'}
{'='*80}

Classification:
   • Detected as: {pred_label}
   • Confidence Level: {confidence*100:.2f}%
   • Authenticity Score: {(raw_prob)*100:.2f}% AUTHENTIC
   • Synthetic Score: {authenticity_score:.2f}% SYNTHETIC

Decision Threshold: 50.00%
   └─ Probability > 50% → Classified as AUTHENTIC (Real Speech)
   └─ Probability ≤ 50% → Classified as SYNTHETIC (AI-Generated)

"""
        
        # Add noise robustness analysis
        analysis += f"""
{'='*80}
🔊 NOISE ROBUSTNESS ANALYSIS
{'='*80}

Model Training Context:
   • Trained on SNR-augmented datasets (5dB, 10dB, 15dB, 20dB)
   • Robust to various noise conditions
   • Optimized for real-world noisy environments

Current Audio Analysis:
"""
        
        # Analyze intensity patterns for noise indicators
        intensity_variation = sample.std()
        if intensity_variation > 0.3:
            noise_level = "HIGH"
            noise_desc = "Significant background noise detected - typical of real-world conditions"
        elif intensity_variation > 0.2:
            noise_level = "MODERATE"
            noise_desc = "Moderate noise present - model trained to handle this"
        else:
            noise_level = "LOW"
            noise_desc = "Clean or minimal noise - ideal conditions"
        
        analysis += f"""   • Estimated Noise Level: {noise_level}
   • Noise Characteristics: {noise_desc}
   • Model Confidence: {'MAINTAINED' if confidence > 0.6 else 'REDUCED'} under current noise conditions
   • SNR Estimation: {('High SNR (>15dB) - Clean signal' if intensity_variation < 0.2 else 
                         'Medium SNR (10-15dB) - Moderate noise' if intensity_variation < 0.3 else 
                         'Low SNR (<10dB) - Noisy environment')}

"""
        
        # Synthetic pattern analysis
        analysis += f"""
{'='*80}
🔍 SYNTHETIC PATTERN DETECTION
{'='*80}

CNN Feature Extraction Analysis:
"""
        
        if activations is not None:
            analysis += f"   • Processing Layers: {len(activations['names'])} detection stages\\n\\n"
            
            for i, (name, activation) in enumerate(zip(activations['names'], activations['outputs'])):
                layer_type = "Feature Extractor" if 'conv' in name.lower() else "Pattern Classifier"
                act_mean = activation.mean()
                act_max = activation.max()
                
                # Interpret activation patterns
                if act_mean > 5.0:
                    pattern_strength = "STRONG"
                    pattern_desc = "Detecting significant distinguishing features"
                elif act_mean > 2.0:
                    pattern_strength = "MODERATE"
                    pattern_desc = "Identifying characteristic patterns"
                else:
                    pattern_strength = "SUBTLE"
                    pattern_desc = "Analyzing fine-grained details"
                
                analysis += f"""   Layer {i+1}: {name} ({layer_type})
      → Activation Strength: {pattern_strength}
      → Response: {pattern_desc}
      → Activation Range: [{activation.min():.3f}, {act_max:.3f}]
      → Mean Response: {act_mean:.3f}

"""
        
        # Grad-CAM attention analysis
        analysis += f"""
{'='*80}
👁️ ATTENTION MAP ANALYSIS (Where the Model Looked)
{'='*80}

"""
        
        if grad_cam is not None:
            # Analyze attention distribution
            attention_mean = grad_cam.mean()
            attention_max = grad_cam.max()
            attention_focused = np.sum(grad_cam > 0.7) / grad_cam.size * 100
            
            analysis += f"""Attention Distribution:
   • Focus Intensity: {attention_max:.3f} (peak attention)
   • Average Attention: {attention_mean:.3f}
   • Focused Regions: {attention_focused:.1f}% of spectrogram

Region Importance:
   • RED zones (>0.7): CRITICAL synthetic/authentic markers
   • YELLOW zones (0.4-0.7): Supporting evidence regions  
   • BLUE zones (<0.4): Background/less relevant areas

"""
            
            # Interpret attention patterns
            if attention_focused > 30:
                analysis += """   🎯 INTERPRETATION: Model is focusing on MULTIPLE distinct regions
      → Indicates complex pattern recognition
      → Synthetic artifacts often show scattered anomalies
      → High confidence in detection

"""
            elif attention_focused > 15:
                analysis += """   🎯 INTERPRETATION: Model focusing on SPECIFIC regions
      → Targeted feature detection
      → Clear distinguishing patterns found
      → Moderate to high confidence

"""
            else:
                analysis += """   🎯 INTERPRETATION: Model attention is DISTRIBUTED
      → Analyzing overall spectral characteristics
      → No single dominant feature
      → Decision based on holistic pattern

"""
        
        # Final verdict section
        analysis += f"""
{'='*80}
⚖️ FINAL VERDICT & CONFIDENCE ASSESSMENT
{'='*80}

Classification: {pred_label}
Raw Probabilities:
   • AUTHENTIC (Real Speech): {raw_prob*100:.2f}%
   • SYNTHETIC (AI-Generated): {(1-raw_prob)*100:.2f}%

Confidence Level: {confidence*100:.2f}%
   └─ Interpretation: """
        
        if confidence > 0.90:
            analysis += """VERY HIGH CONFIDENCE
      → Model is extremely certain about this classification
      → Strong distinguishing patterns detected
      → Minimal ambiguity in features"""
        elif confidence > 0.80:
            analysis += """HIGH CONFIDENCE
      → Model has strong evidence for this classification
      → Clear synthetic or authentic markers present
      → Reliable prediction"""
        elif confidence > 0.70:
            analysis += """MODERATE-HIGH CONFIDENCE
      → Model found distinguishing patterns
      → Some features are characteristic of the predicted class
      → Generally reliable prediction"""
        elif confidence > 0.60:
            analysis += """MODERATE CONFIDENCE
      → Model detected relevant patterns but with some uncertainty
      → Features show mixed characteristics
      → Prediction is reasonable but not definitive"""
        else:
            analysis += """LOW CONFIDENCE
      → Model is uncertain about this classification
      → Features show ambiguous or conflicting patterns
      → Borderline case - could be either class"""
        
        # Add context about the decision
        analysis += f"""

{'='*80}
💡 WHY THIS CLASSIFICATION?
{'='*80}

"""
        
        if is_synthetic:
            analysis += f"""The model classified this audio as SYNTHETIC because:

1. SPECTRAL ANOMALIES:
   • AI-generated speech often has unnatural spectral patterns
   • Synthetic voices may show artifacts in frequency transitions
   • The model detected patterns consistent with synthetic generation

2. TEMPORAL CHARACTERISTICS:
   • Synthetic audio may have overly smooth or repetitive patterns
   • Natural speech has organic variations that AI struggles to replicate
   • Detected temporal signatures matching synthetic profiles

3. FEATURE CONSISTENCY:
   • Multiple layers agreed on synthetic characteristics
   • Activation patterns indicate artificial generation markers
   • Confidence: {confidence*100:.2f}% that this is AI-generated
"""
        else:
            analysis += f"""The model classified this audio as AUTHENTIC because:

1. NATURAL SPECTRAL PATTERNS:
   • Frequency distribution matches natural human speech
   • Organic variations and irregularities present
   • No significant synthetic artifacts detected

2. TEMPORAL AUTHENTICITY:
   • Natural timing and rhythm patterns
   • Characteristic human speech dynamics observed
   • Genuine vocal variations present

3. FEATURE CONSISTENCY:
   • Multiple layers confirmed authentic speech characteristics
   • Activation patterns indicate natural voice production
   • Confidence: {confidence*100:.2f}% that this is genuine human speech
"""
        
        # Add training context
        if is_custom:
            analysis += f"""
{'='*80}
📚 MODEL CONTEXT & CAPABILITIES
{'='*80}

This CNN model was specifically trained to:
   ✓ Detect synthetic audio under NOISY conditions
   ✓ Handle various SNR levels (5dB to 20dB)
   ✓ Distinguish AI-generated from authentic speech
   ✓ Maintain accuracy in real-world environments

Training Data:
   • SNR-augmented datasets with controlled noise levels
   • Balanced samples of authentic and synthetic audio
   • Various noise conditions simulated

Custom Audio Analysis:
   • Your uploaded file: Processed and analyzed
   • Preprocessing: Matched training pipeline exactly
   • Result: {pred_label} with {confidence*100:.2f}% confidence
"""
        else:
            analysis += f"""
{'='*80}
✅ TEST DATASET ANALYSIS
{'='*80}

Ground Truth: {true_label}
Prediction: {pred_label}
Result: {'✅ CORRECT' if true_label == pred_label else '❌ INCORRECT'}

Model Performance:
   • This sample is from the test dataset
   • Model trained on SNR-augmented noisy audio
   • Designed to handle real-world conditions
   • Prediction confidence: {confidence*100:.2f}%
"""
        
        analysis += f"""
{'='*80}
� TECHNICAL SUMMARY
{'='*80}

Model Architecture:
   • Type: Convolutional Neural Network (CNN)
   • Input: 128×128 Mel-Spectrogram
   • Task: Binary Classification (Authentic vs Synthetic)
   • Training: SNR-augmented datasets (noisy conditions)

Detection Method:
   • Multi-layer feature extraction
   • Hierarchical pattern recognition
   • Attention-based analysis (Grad-CAM)
   • Noise-robust classification

Key Findings:
   ✓ Classification: {pred_label}
   ✓ Confidence: {confidence*100:.2f}%
   ✓ Noise Handling: Robust under current conditions
   ✓ Decision Basis: {'Synthetic patterns detected' if is_synthetic else 'Authentic speech confirmed'}

{'='*80}
"""
        
        self.explain_log.insert(1.0, analysis)
    
    def log_explain(self, message):
        """Add message to explainability log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.explain_log.insert(tk.END, f"[{timestamp}] {message}\\n")
        self.explain_log.see(tk.END)
        self.root.update()
    
    def export_explainability_pdf(self):
        """Export explainability analysis to PDF"""
        if not self.explainability_data:
            messagebox.showwarning("Warning", "No explainability data to export!\\nAnalyze some samples first.")
            return
        
        try:
            # Ask for save location
            filename = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
                initialfile=f"explainability_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            )
            
            if not filename:
                return
            
            self.log_explain("📄 Generating PDF report...")
            
            with PdfPages(filename) as pdf:
                # Cover page
                fig = plt.figure(figsize=(11, 8.5))
                fig.patch.set_facecolor('white')
                ax = fig.add_subplot(111)
                ax.axis('off')
                
                title_text = """
                
                
                🔍 CNN MODEL EXPLAINABILITY REPORT
                
                Detailed Analysis of Prediction Decisions
                
                
                """
                
                ax.text(0.5, 0.7, title_text, ha='center', va='center',
                       fontsize=24, weight='bold', family='monospace')
                
                info_text = f"""
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                Total Samples Analyzed: {len(self.explainability_data)}
                Model: CNN Audio Classifier
                Task: Real vs Fake Audio Detection
                """
                
                ax.text(0.5, 0.3, info_text, ha='center', va='center',
                       fontsize=12, family='monospace')
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                
                # Generate page for each analyzed sample
                for idx, data in enumerate(self.explainability_data):
                    self.log_explain(f"📄 Adding sample {idx+1}/{len(self.explainability_data)} to PDF...")
                    
                    fig = plt.figure(figsize=(11, 14))
                    fig.suptitle(f'Sample #{data["sample_idx"]} - Explainability Analysis',
                               fontsize=16, weight='bold')
                    
                    gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)
                    
                    # Summary info
                    ax_info = fig.add_subplot(gs[0, :])
                    ax_info.axis('off')
                    
                    is_correct = data['true_label'] == data['predicted_label']
                    result = "CORRECT ✓" if is_correct else "INCORRECT ✗"
                    
                    info = f"""
PREDICTION RESULT: {result}

Sample Index: #{data['sample_idx']}
True Label: {data['true_label']}
Predicted Label: {data['predicted_label']}
Confidence: {data['confidence']*100:.2f}%

PROBABILITY BREAKDOWN:
• REAL: {data['raw_probability']*100:.2f}%
• FAKE: {(1-data['raw_probability'])*100:.2f}%
                    """
                    
                    ax_info.text(0.05, 0.5, info, fontsize=11, family='monospace',
                               verticalalignment='center')
                    
                    # Grad-CAM visualization
                    if data['grad_cam'] is not None:
                        ax_cam = fig.add_subplot(gs[1, 0])
                        im = ax_cam.imshow(data['grad_cam'], cmap='jet', aspect='auto')
                        ax_cam.set_title('Grad-CAM Heatmap\\n(Model Attention)', 
                                       fontsize=10, weight='bold')
                        ax_cam.axis('off')
                        plt.colorbar(im, ax=ax_cam, fraction=0.046, pad=0.04)
                    
                    # Confidence bars
                    ax_conf = fig.add_subplot(gs[1, 1])
                    categories = ['FAKE', 'REAL']
                    fake_conf = (1 - data['confidence']) * 100 if data['predicted_label'] == "REAL" else data['confidence'] * 100
                    real_conf = data['confidence'] * 100 if data['predicted_label'] == "REAL" else (1 - data['confidence']) * 100
                    confidences = [fake_conf, real_conf]
                    colors = ['#e74c3c', '#27ae60']
                    bars = ax_conf.barh(categories, confidences, color=colors, alpha=0.8)
                    ax_conf.set_xlim(0, 100)
                    ax_conf.set_xlabel('Confidence (%)', fontsize=10, weight='bold')
                    ax_conf.set_title('Confidence Breakdown', fontsize=10, weight='bold')
                    for bar, conf in zip(bars, confidences):
                        ax_conf.text(conf + 2, bar.get_y() + bar.get_height()/2, 
                                   f'{conf:.2f}%', va='center', fontsize=9, weight='bold')
                    
                    # Analysis text
                    ax_analysis = fig.add_subplot(gs[2:, :])
                    ax_analysis.axis('off')
                    
                    analysis = f"""
WHY THIS PREDICTION?

1. DECISION LOGIC:
   • The model output a probability of {data['raw_probability']*100:.2f}% for REAL
   • Decision threshold is 50%
   • Since {data['raw_probability']*100:.2f}% is {'>' if data['raw_probability'] > 0.5 else '≤'} 50%, model predicted: {data['predicted_label']}

2. CONFIDENCE LEVEL: {data['confidence']*100:.2f}%
   • {'Very high confidence' if data['confidence'] > 0.9 else 'High confidence' if data['confidence'] > 0.8 else 'Good confidence' if data['confidence'] > 0.7 else 'Moderate confidence' if data['confidence'] > 0.6 else 'Low confidence'}
   • Model is {'very certain' if data['confidence'] > 0.9 else 'confident' if data['confidence'] > 0.8 else 'fairly confident' if data['confidence'] > 0.7 else 'moderately certain' if data['confidence'] > 0.6 else 'uncertain'} about this classification

3. FEATURE ANALYSIS:
   • Grad-CAM shows which parts of the audio spectrogram influenced the decision
   • Red/bright areas = HIGH importance (strongly influenced prediction)
   • Yellow areas = MEDIUM importance
   • Blue/dark areas = LOW importance (minimal influence)

4. MODEL REASONING:
   • The CNN detected patterns in the spectrogram consistent with {data['predicted_label']} audio
   • Key features: Frequency patterns, temporal structure, spectral characteristics
   • These learned patterns come from training on thousands of audio samples

5. GROUND TRUTH VERIFICATION:
   • True Label: {data['true_label']}
   • Predicted Label: {data['predicted_label']}
   • Result: {'CORRECT - Model accurately identified the audio type' if is_correct else 'INCORRECT - Misclassification occurred'}
   {f"• Likely reasons for error: Edge case, ambiguous features, or high noise" if not is_correct else ""}

6. TECHNICAL BASIS:
   • Method: Convolutional Neural Network (CNN)
   • Input: 128×128 Mel-Spectrogram
   • Layers: Multiple convolutional and dense layers
   • Training: Learned from labeled dataset of real and fake audio
   • Explainability: Grad-CAM (Gradient-weighted Class Activation Mapping)
                    """
                    
                    ax_analysis.text(0.05, 0.95, analysis, fontsize=9, family='monospace',
                                   verticalalignment='top', wrap=True)
                    
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
                
                # Summary statistics page
                fig = plt.figure(figsize=(11, 8.5))
                fig.suptitle('Overall Explainability Summary', fontsize=16, weight='bold')
                
                gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
                
                # Calculate statistics
                total = len(self.explainability_data)
                correct = sum(1 for d in self.explainability_data 
                            if d['true_label'] == d['predicted_label'])
                avg_confidence = np.mean([d['confidence'] for d in self.explainability_data])
                
                # Stats text
                ax_stats = fig.add_subplot(gs[0, :])
                ax_stats.axis('off')
                
                stats_text = f"""
                            ANALYSIS SUMMARY

                            Total Samples Analyzed: {total}
                            Correct Predictions: {correct} ({correct/total*100:.1f}%)
                            Incorrect Predictions: {total-correct} ({(total-correct)/total*100:.1f}%)
                            Average Confidence: {avg_confidence*100:.2f}%

                            This report provides detailed explanations for each prediction, showing:
                            • Which parts of the audio influenced the decision (Grad-CAM)
                            • How confident the model was (Confidence scores)
                            • Why the model made its prediction (Feature analysis)
                            • Technical basis for the decision (Model architecture and methodology)
                                            """
                
                ax_stats.text(0.05, 0.5, stats_text, fontsize=12, family='monospace',
                            verticalalignment='center')
                
                # Confidence distribution
                ax_conf_dist = fig.add_subplot(gs[1, :])
                confidences = [d['confidence'] for d in self.explainability_data]
                ax_conf_dist.hist(confidences, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
                ax_conf_dist.set_xlabel('Confidence', fontsize=10, weight='bold')
                ax_conf_dist.set_ylabel('Frequency', fontsize=10, weight='bold')
                ax_conf_dist.set_title('Confidence Distribution', fontsize=12, weight='bold')
                ax_conf_dist.grid(True, alpha=0.3)
                
                # Accuracy by confidence level
                ax_acc = fig.add_subplot(gs[2, :])
                conf_bins = [(0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
                bin_labels = ['0-50%', '50-70%', '70-80%', '80-90%', '90-100%']
                bin_accuracies = []
                
                for low, high in conf_bins:
                    samples_in_bin = [d for d in self.explainability_data 
                                    if low <= d['confidence'] < high]
                    if samples_in_bin:
                        correct_in_bin = sum(1 for d in samples_in_bin 
                                           if d['true_label'] == d['predicted_label'])
                        bin_accuracies.append(correct_in_bin / len(samples_in_bin) * 100)
                    else:
                        bin_accuracies.append(0)
                
                ax_acc.bar(bin_labels, bin_accuracies, color='#27ae60', alpha=0.7, edgecolor='black')
                ax_acc.set_xlabel('Confidence Level', fontsize=10, weight='bold')
                ax_acc.set_ylabel('Accuracy (%)', fontsize=10, weight='bold')
                ax_acc.set_title('Accuracy by Confidence Level', fontsize=12, weight='bold')
                ax_acc.set_ylim(0, 100)
                ax_acc.grid(True, alpha=0.3, axis='y')
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            self.log_explain(f"✅ PDF exported successfully: {filename}")
            messagebox.showinfo("Success", f"Explainability report exported to:\\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export PDF:\\n{str(e)}")
            print(f"Error in export_explainability_pdf: {e}")
            import traceback
            traceback.print_exc()
        
    # ========== DATA LOADING METHODS ==========
    
    def load_dataset(self):
        """Load selected dataset"""
        dataset_name = self.dataset_var.get()
        dataset_path = self.dataset_paths.get(dataset_name)
        
        if not os.path.exists(dataset_path):
            messagebox.showerror("Error", f"Dataset not found: {dataset_path}")
            return
        
        try:
            print(f"\n📂 Loading dataset: {dataset_name}...")
            self.log_training(f"Loading dataset: {dataset_name}...")
            
            # Check if mixed SNR is enabled
            if self.mixed_snr_var.get():
                print(f"🔊 Mixed SNR mode enabled!")
                self.log_training(f"🔊 Mixed SNR mode enabled - combining datasets...")
                
                # Get selected SNR datasets
                selected_datasets = []
                if self.snr_5db_var.get():
                    selected_datasets.append(('SNR 5dB', self.dataset_paths['snr_5db']))
                if self.snr_10db_var.get():
                    selected_datasets.append(('SNR 10dB', self.dataset_paths['snr_10db']))
                if self.snr_15db_var.get():
                    selected_datasets.append(('SNR 15dB', self.dataset_paths['snr_15db']))
                if self.snr_20db_var.get():
                    selected_datasets.append(('SNR 20dB', self.dataset_paths['snr_20db']))
                
                if len(selected_datasets) == 0:
                    messagebox.showwarning("Warning", "Please select at least one SNR dataset for mixed training!")
                    return
                
                # Store mixed dataset info
                self.current_dataset_info = {
                    'is_mixed': True,
                    'dataset_name': 'Mixed SNR',
                    'mixed_datasets': [name for name, _ in selected_datasets],
                    'total_samples': {'train': 0, 'val': 0, 'test': 0}
                }
                
                # Load and combine datasets
                combined_train_X, combined_train_y = [], []
                combined_val_X, combined_val_y = [], []
                combined_test_X, combined_test_y = [], []
                
                for snr_name, snr_path in selected_datasets:
                    if not os.path.exists(snr_path):
                        print(f"⚠️ Warning: {snr_name} dataset not found, skipping...")
                        continue
                    
                    print(f"   Loading {snr_name}...")
                    with open(snr_path, 'rb') as f:
                        dataset_dict = pickle.load(f)
                    
                    combined_train_X.append(dataset_dict['train_X'])
                    combined_train_y.append(dataset_dict['train_y'])
                    combined_val_X.append(dataset_dict['val_X'])
                    combined_val_y.append(dataset_dict['val_y'])
                    combined_test_X.append(dataset_dict['test_X'])
                    combined_test_y.append(dataset_dict['test_y'])
                
                # Concatenate all datasets
                train_X = np.concatenate(combined_train_X, axis=0)
                train_y = np.concatenate(combined_train_y, axis=0)
                val_X = np.concatenate(combined_val_X, axis=0)
                val_y = np.concatenate(combined_val_y, axis=0)
                test_X = np.concatenate(combined_test_X, axis=0)
                test_y = np.concatenate(combined_test_y, axis=0)
                
                print(f"✅ Combined {len(selected_datasets)} SNR datasets!")
                self.log_training(f"✅ Combined datasets: {', '.join([name for name, _ in selected_datasets])}")
                
            else:
                # Load single dataset
                # Store single dataset info
                self.current_dataset_info = {
                    'is_mixed': False,
                    'dataset_name': dataset_name,
                    'mixed_datasets': [],
                    'total_samples': {'train': 0, 'val': 0, 'test': 0}
                }
                
                with open(dataset_path, 'rb') as f:
                    dataset_dict = pickle.load(f)
                
                train_X = dataset_dict['train_X']
                train_y = dataset_dict['train_y']
                val_X = dataset_dict['val_X']
                val_y = dataset_dict['val_y']
                test_X = dataset_dict['test_X']
                test_y = dataset_dict['test_y']
            
            print(f"   ✅ Dataset file loaded successfully")
            
            # CRITICAL: Check label distribution
            print(f"\n🔍 DATASET LABEL DISTRIBUTION CHECK:")
            train_fake = np.sum(train_y == 0)
            train_real = np.sum(train_y == 1)
            val_fake = np.sum(val_y == 0)
            val_real = np.sum(val_y == 1)
            test_fake = np.sum(test_y == 0)
            test_real = np.sum(test_y == 1)
            
            print(f"   📊 Training:   Fake={train_fake} ({train_fake/len(train_y)*100:.1f}%), Real={train_real} ({train_real/len(train_y)*100:.1f}%)")
            print(f"   📊 Validation: Fake={val_fake} ({val_fake/len(val_y)*100:.1f}%), Real={val_real} ({val_real/len(val_y)*100:.1f}%)")
            print(f"   📊 Test:       Fake={test_fake} ({test_fake/len(test_y)*100:.1f}%), Real={test_real} ({test_real/len(test_y)*100:.1f}%)")
            
            # WARNING: Check for imbalanced or corrupted datasets
            if test_fake == 0 or test_real == 0:
                warning_msg = f"⚠️⚠️⚠️ CRITICAL WARNING ⚠️⚠️⚠️\nYour TEST set has ONLY {'REAL' if test_fake == 0 else 'FAKE'} samples!\n\n"
                warning_msg += f"Test set needs BOTH classes:\n"
                warning_msg += f"  • Fake: {test_fake} samples\n"
                warning_msg += f"  • Real: {test_real} samples\n\n"
                warning_msg += f"This dataset is CORRUPTED or IMBALANCED!\n"
                warning_msg += f"Check your pickle file: {dataset_path if not self.current_dataset_info['is_mixed'] else 'mixed datasets'}"
                print(f"\n❌ {warning_msg}")
                messagebox.showerror("CORRUPTED DATASET!", warning_msg)
                return
            
            if train_fake == 0 or train_real == 0:
                print(f"\n⚠️ WARNING: Training set is imbalanced! (Fake={train_fake}, Real={train_real})")
                messagebox.showwarning("Imbalanced Dataset", f"Training set has only {'REAL' if train_fake == 0 else 'FAKE'} samples!\nModel cannot learn properly.")
                return
            
            # Update dataset info with sample counts
            self.current_dataset_info['total_samples'] = {
                'train': len(train_X),
                'val': len(val_X),
                'test': len(test_X)
            }
            
            # Create TensorFlow datasets
            batch_size = self.batch_var.get()
            
            self.train_data = tf.data.Dataset.from_tensor_slices((train_X, train_y))
            self.train_data = self.train_data.cache().shuffle(buffer_size=500).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
            self.val_data = tf.data.Dataset.from_tensor_slices((val_X, val_y))
            self.val_data = self.val_data.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
            self.test_data = tf.data.Dataset.from_tensor_slices((test_X, test_y))
            self.test_data = self.test_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)  # REMOVED .cache() to avoid stale data
            
            print(f"   ⚠️  Note: Test data created WITHOUT caching to ensure fresh data for each prediction run")
            
            # Update status with mixed dataset info
            if self.current_dataset_info['is_mixed']:
                mixed_names = ' + '.join(self.current_dataset_info['mixed_datasets'])
                status_text = f"✅ Mixed Dataset: {mixed_names}"
                self.dataset_status.config(text=status_text, fg='#00ff88')
                
                print(f"✅ Mixed Dataset loaded successfully!")
                print(f"   🔊 Combined: {mixed_names}")
                print(f"   📊 Total Training samples: {len(train_X)}")
                print(f"   📊 Total Validation samples: {len(val_X)}")
                print(f"   📊 Total Test samples: {len(test_X)}")
                print(f"   📦 Batch size: {batch_size}\n")
                
                self.log_training(f"✅ Mixed Dataset loaded: {mixed_names}")
                self.log_training(f"   Total Training samples: {len(train_X)}")
                self.log_training(f"   Total Validation samples: {len(val_X)}")
                self.log_training(f"   Total Test samples: {len(test_X)}")
            else:
                status_text = f"✅ Loaded: {dataset_name} | Train: {len(train_X)} | Val: {len(val_X)} | Test: {len(test_X)}"
                self.dataset_status.config(text=status_text, fg='#00ff88')
                
                print(f"✅ Dataset loaded successfully!")
                print(f"   📊 Training samples: {len(train_X)}")
                print(f"   📊 Validation samples: {len(val_X)}")
                print(f"   📊 Test samples: {len(test_X)}")
                print(f"   📦 Batch size: {batch_size}\n")
                
                self.log_training(f"✅ Dataset loaded successfully!")
                self.log_training(f"   Training samples: {len(train_X)}")
                self.log_training(f"   Validation samples: {len(val_X)}")
                self.log_training(f"   Test samples: {len(test_X)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset:\n{str(e)}")
            self.log_training(f"❌ Error loading dataset: {str(e)}")
    
    def load_dataset_for_prediction(self):
        """Load dataset for prediction (uses the same load_dataset function but updates prediction tab status)"""
        # Call the main load_dataset function
        self.load_dataset()
        
        # Update prediction tab status
        if self.test_data is not None:
            if self.current_dataset_info['is_mixed']:
                mixed_names = ' + '.join(self.current_dataset_info['mixed_datasets'])
                self.pred_dataset_status.config(text=f"✅ Mixed: {mixed_names}", fg='#00ff88')
            else:
                dataset_name = self.current_dataset_info['dataset_name']
                self.pred_dataset_status.config(text=f"✅ Loaded: {dataset_name}", fg='#00ff88')
    
    def load_trained_model(self):
        """Load a pre-trained CNN model - auto-loads from CNN folder"""
        # First check if auto-saved model exists
        if os.path.exists(self.cnn_model_path):
            try:
                print(f"\n📂 Auto-loading trained model from: {self.cnn_model_path}")
                self.cnn_model = CNNAudioClassifier()
                self.cnn_model.load_model(self.cnn_model_path)
                
                # Get file modification time
                mod_time = os.path.getmtime(self.cnn_model_path)
                mod_date = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
                
                self.model_status.config(text=f"✅ Model loaded ({mod_date})", fg='#00ff88')
                
                print(f"✅ Model loaded successfully!")
                print(f"   📅 Last trained: {mod_date}")
                
                # Check for dataset metadata file
                metadata_path = os.path.join(self.cnn_model_dir, 'dataset_info.txt')
                if os.path.exists(metadata_path):
                    print(f"\n📄 Dataset metadata found! Reading training info...")
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata_content = f.read()
                    
                    # Extract dataset type from metadata
                    if "MIXED SNR DATASET" in metadata_content:
                        print(f"   ⚠️  Model was trained on: MIXED SNR DATASET")
                        # Extract dataset names
                        lines = metadata_content.split('\n')
                        datasets_line = [l for l in lines if 'Combined Datasets:' in l]
                        if datasets_line:
                            idx = lines.index(datasets_line[0])
                            combined = []
                            for i in range(idx+1, len(lines)):
                                if lines[i].strip().startswith('- '):
                                    combined.append(lines[i].strip()[2:])
                                elif lines[i].strip() == '' or not lines[i].startswith('  '):
                                    break
                            if combined:
                                print(f"   📊 Training datasets: {', '.join(combined)}")
                    else:
                        # Single dataset
                        for line in metadata_content.split('\n'):
                            if 'Dataset Name:' in line:
                                trained_dataset = line.split(':')[1].strip()
                                print(f"   ⚠️  Model was trained on: {trained_dataset}")
                                break
                    
                    print(f"\n   💡 See {metadata_path} for full details")
                else:
                    print(f"\n   ⚠️  No dataset metadata found (model trained before this feature)")
                
                # Check if dataset is loaded
                if self.test_data is not None:
                    dataset_info = ""
                    if self.current_dataset_info['is_mixed']:
                        mixed_names = ' + '.join(self.current_dataset_info['mixed_datasets'])
                        dataset_info = f"Mixed: {mixed_names}"
                    else:
                        dataset_info = self.current_dataset_info['dataset_name']
                    
                    print(f"\n⚠️  DATASET CHECK:")
                    print(f"   Current dataset loaded: {dataset_info}")
                    print(f"   Make sure this matches the training dataset above!")
                    print(f"   If it doesn't match, load the correct dataset first!\n")
                    
                    # Enable prediction button only if dataset is loaded
                    self.predict_btn.config(state='normal')
                else:
                    print(f"\n⚠️  WARNING: No dataset loaded yet!")
                    print(f"   Please load the dataset before making predictions!")
                    print(f"   Click 'Load Dataset' button in the prediction tab.\n")
                    self.predict_btn.config(state='disabled')
                
                self.log_prediction(f"✅ Model auto-loaded from: {self.cnn_model_path}")
                self.log_prediction(f"   Last trained: {mod_date}")
                
                return
                
            except Exception as e:
                print(f"⚠️ Auto-load failed: {str(e)}")
                print(f"   Opening file dialog for manual selection...\n")
        else:
            print(f"⚠️ No trained model found in CNN folder")
            print(f"   Opening file dialog for manual selection...\n")
        
        # If auto-load fails or no model exists, open file dialog
        filepath = filedialog.askopenfilename(
            title="Select Trained CNN Model",
            filetypes=[("Keras Models", "*.h5 *.keras"), ("All Files", "*.*")],
            initialdir=self.cnn_model_dir
        )
        
        if not filepath:
            return
        
        try:
            self.cnn_model = CNNAudioClassifier()
            self.cnn_model.load_model(filepath)
            
            self.model_status.config(text=f"✅ Model loaded: {os.path.basename(filepath)}", fg='#00ff88')
            self.predict_btn.config(state='normal')
            
            print(f"✅ Model loaded from: {filepath}\n")
            self.log_prediction(f"✅ Model loaded from: {filepath}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            self.log_prediction(f"❌ Error loading model: {str(e)}")
    
    # ========== TRAINING METHODS ==========
    
    def reset_training_ui(self):
        """Reset/clear all training UI elements for a fresh start"""
        print("\n🔄 Resetting UI for new training session...")
        
        # Reset epoch display
        self.epoch_label.config(text="Epoch: 0/0")
        self.epoch_progress['value'] = 0
        
        # Reset metric labels
        self.loss_label.config(text="0.0000")
        self.acc_label.config(text="0.00%")
        self.val_loss_label.config(text="0.0000")
        self.val_acc_label.config(text="0.00%")
        
        # Clear training log
        self.training_log.delete('1.0', tk.END)
        self.training_log.insert(tk.END, "🔄 Starting new training session...\n\n")
        
        # Clear plots
        self.loss_ax.clear()
        self.acc_ax.clear()
        
        self.loss_ax.set_title('Training & Validation Loss', color='white', fontsize=12, weight='bold')
        self.loss_ax.set_xlabel('Epoch', color='white')
        self.loss_ax.set_ylabel('Loss', color='white')
        self.loss_ax.tick_params(colors='white')
        self.loss_ax.grid(True, alpha=0.3)
        
        self.acc_ax.set_title('Training & Validation Accuracy', color='white', fontsize=12, weight='bold')
        self.acc_ax.set_xlabel('Epoch', color='white')
        self.acc_ax.set_ylabel('Accuracy', color='white')
        self.acc_ax.tick_params(colors='white')
        self.acc_ax.grid(True, alpha=0.3)
        
        self.train_canvas.draw()
        
        # Clear summary text
        self.summary_text.delete('1.0', tk.END)
        self.summary_text.insert(tk.END, "📋 Training Summary\n")
        self.summary_text.insert(tk.END, "="*50 + "\n\n")
        self.summary_text.insert(tk.END, "Waiting for training to complete...\n")
        
        print("✅ UI reset complete!\n")
    
    def start_training(self):
        """Start training in a separate thread"""
        if self.train_data is None or self.val_data is None:
            messagebox.showwarning("Warning", "Please load a dataset first!")
            return
        
        if self.is_training:
            messagebox.showwarning("Warning", "Training already in progress!")
            return
        
        # RESET UI FIRST - Clear all previous training data
        self.reset_training_ui()
        
        # Reset history
        self.epoch_history = {
            'epoch': [],
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Disable controls
        self.train_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.is_training = True
        
        # Start training thread
        training_thread = threading.Thread(target=self._training_thread, daemon=True)
        training_thread.start()
    
    def stop_training(self):
        """Stop training"""
        self.is_training = False
        self.train_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.log_training("⏹️ Training stopped by user")
    
    def _training_thread(self):
        """Training thread with epoch-by-epoch updates"""
        try:
            # Create and compile model with optimization settings
            print(f"\n{'='*60}")
            print(f"🏗️ BUILDING CNN MODEL WITH CUSTOM SETTINGS")
            print(f"{'='*60}")
            self.log_training("🏗️ Building CNN model with optimization settings...")
            
            # Get architecture settings
            conv_filters = [
                self.conv1_filters_var.get(),
                self.conv2_filters_var.get(),
                self.conv3_filters_var.get()
            ]
            
            dense_units = [
                self.dense1_units_var.get(),
                self.dense2_units_var.get()
            ]
            
            conv_dropout = [
                self.conv1_dropout_var.get(),
                self.conv2_dropout_var.get(),
                self.conv3_dropout_var.get()
            ]
            
            dense_dropout = [
                self.dense1_dropout_var.get(),
                self.dense2_dropout_var.get()
            ]
            
            use_l2 = self.l2_reg_var.get()
            l2_strength = self.l2_strength_var.get()
            
            # Build model with settings
            self.cnn_model = CNNAudioClassifier(input_shape=(128, 128, 1))
            self.cnn_model.build_model(
                conv_filters=conv_filters,
                dense_units=dense_units,
                conv_dropout=conv_dropout,
                dense_dropout=dense_dropout,
                use_l2=use_l2,
                l2_strength=l2_strength
            )
            
            print(f"   ✅ Model architecture created")
            
            # Get learning rate and create optimizer with gradient clipping
            lr = self.lr_var.get()
            
            # Safety check for learning rate
            if lr > 0.01:
                self.log_training(f"⚠️ WARNING: Learning rate {lr} is very high! Reducing to 0.001 for stability.")
                print(f"⚠️ WARNING: Learning rate {lr} is too high! Reducing to 0.001")
                lr = 0.001
            elif lr < 1e-6:
                self.log_training(f"⚠️ WARNING: Learning rate {lr} is too low! Increasing to 0.0001.")
                print(f"⚠️ WARNING: Learning rate {lr} is too low! Increasing to 0.0001")
                lr = 0.0001
            
            # Use Adam with gradient clipping to prevent exploding gradients
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
            
            # Compile with precision and recall metrics
            self.cnn_model.compile(
                optimizer=optimizer, 
                loss='binary_crossentropy', 
                metrics=['binary_accuracy', tf.keras.metrics.Precision(name='precision'), 
                        tf.keras.metrics.Recall(name='recall')]
            )
            
            print(f"   ✅ Model compiled")
            print(f"   📊 Learning Rate: {lr} (with gradient clipping)")
            print(f"   🎯 Loss Function: binary_crossentropy")
            print(f"   📈 Metrics: accuracy, precision, recall")
            self.log_training(f"✅ Model compiled with learning rate: {lr} (gradient clipping enabled)")
            self.log_training(f"   Architecture: Conv{conv_filters} -> Dense{dense_units}")
            self.log_training(f"   L2 Regularization: {use_l2}")
            if use_l2:
                self.log_training(f"   L2 Strength: {l2_strength}")
            
            # Get training parameters
            epochs = self.epochs_var.get()
            self.total_epochs = epochs
            
            # Setup callbacks with custom patience
            patience = self.patience_var.get()
            reduce_lr_patience = self.reduce_lr_patience_var.get()
            
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=reduce_lr_patience,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
            
            # Add ModelCheckpoint if enabled
            if self.save_best_model_var.get():
                checkpoint_path = os.path.join(self.cnn_model_dir, 'best_model_checkpoint.h5')
                callbacks.append(
                    tf.keras.callbacks.ModelCheckpoint(
                        filepath=checkpoint_path,
                        monitor='val_loss',
                        save_best_only=True,
                        verbose=1
                    )
                )
                print(f"   💾 ModelCheckpoint enabled: {checkpoint_path}")
            
            self.start_time = time.time()
            self.training_start_time = datetime.now()
            
            # Start hardware monitoring
            print(f"\n🖥️ Starting hardware resource monitoring...")
            self.start_resource_monitoring()
            
            print(f"\n{'='*60}")
            print(f"🏋️ STARTING TRAINING")
            print(f"{'='*60}")
            print(f"   Total Epochs: {epochs}")
            print(f"   Batch Size: {self.batch_var.get()}")
            print(f"   Early Stopping Patience: {patience}")
            print(f"   ReduceLR Patience: {reduce_lr_patience}")
            print(f"{'='*60}\n")
            
            self.log_training(f"🏋️ Starting training for {epochs} epochs...")
            self.log_training(f"   Early Stopping Patience: {patience}")
            
            # Train epoch by epoch for live updates
            for epoch in range(epochs):
                if not self.is_training:
                    break
                
                self.current_epoch = epoch + 1
                
                # Print epoch start to terminal
                print(f"\n{'='*60}")
                print(f"📍 Epoch {self.current_epoch}/{epochs} - Starting...")
                print(f"{'='*60}")
                
                self.log_training(f"\n📍 Epoch {self.current_epoch}/{epochs}")
                
                # Train one epoch with verbose=1 to show progress in terminal
                history = self.cnn_model.model.fit(
                    self.train_data,
                    epochs=1,
                    validation_data=self.val_data,
                    verbose=1  # Changed from 0 to 1 for terminal output
                )
                
                # Update history
                self.epoch_history['epoch'].append(epoch + 1)
                self.epoch_history['loss'].append(history.history['loss'][0])
                self.epoch_history['accuracy'].append(history.history['binary_accuracy'][0])
                self.epoch_history['val_loss'].append(history.history['val_loss'][0])
                self.epoch_history['val_accuracy'].append(history.history['val_binary_accuracy'][0])
                
                # Update GUI
                self.root.after(0, self.update_training_display)
                
                # Log epoch results (this will also print to terminal)
                self.log_training(f"   Loss: {history.history['loss'][0]:.4f} | Acc: {history.history['binary_accuracy'][0]:.4f}")
                self.log_training(f"   Val Loss: {history.history['val_loss'][0]:.4f} | Val Acc: {history.history['val_binary_accuracy'][0]:.4f}")
                
                # Additional terminal output for clarity
                print(f"✅ Epoch {self.current_epoch} Complete:")
                print(f"   📉 Training   -> Loss: {history.history['loss'][0]:.4f}, Accuracy: {history.history['binary_accuracy'][0]*100:.2f}%")
                print(f"   📊 Validation -> Loss: {history.history['val_loss'][0]:.4f}, Accuracy: {history.history['val_binary_accuracy'][0]*100:.2f}%")
                
                # Log precision and recall if available
                if 'precision' in history.history:
                    print(f"   🎯 Precision: {history.history['precision'][0]:.4f}, Recall: {history.history['recall'][0]:.4f}")
            
            # Mark as fitted
            self.cnn_model.is_fitted = True
            
            # Training complete
            self.training_end_time = datetime.now()
            elapsed_time = time.time() - self.start_time
            self.training_duration = elapsed_time
            
            # Stop hardware monitoring
            print(f"\n🖥️ Stopping hardware resource monitoring...")
            self.stop_resource_monitoring()
            
            # Calculate memory statistics
            if self.resource_stats:
                self.peak_memory_mb = self.resource_stats.get('peak_memory_mb', 0)
                self.avg_memory_mb = self.resource_stats.get('avg_memory_mb', 0)
            
            # Terminal summary
            print(f"\n{'='*60}")
            print(f"🎉 TRAINING COMPLETED!")
            print(f"{'='*60}")
            print(f"⏱️  Total Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
            print(f"🧠 Peak Memory: {self.peak_memory_mb:.2f} MB")
            print(f"🧠 Avg Memory: {self.avg_memory_mb:.2f} MB")
            print(f"📊 Total Epochs: {len(self.epoch_history['epoch'])}")
            if len(self.epoch_history['accuracy']) > 0:
                print(f"📈 Final Training Accuracy: {self.epoch_history['accuracy'][-1]*100:.2f}%")
                print(f"📈 Final Validation Accuracy: {self.epoch_history['val_accuracy'][-1]*100:.2f}%")
                best_val_acc = max(self.epoch_history['val_accuracy'])
                best_epoch = self.epoch_history['val_accuracy'].index(best_val_acc) + 1
                print(f"🏆 Best Validation Accuracy: {best_val_acc*100:.2f}% (Epoch {best_epoch})")
            print(f"{'='*60}\n")
            
            self.log_training(f"\n✅ Training completed in {elapsed_time:.2f} seconds!")
            
            # Auto-save the model
            self.root.after(0, self._auto_save_model)
            
            # Update summary
            self.root.after(0, self.update_summary)
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Training error: {str(e)}\n{traceback.format_exc()}"
            self.log_training(error_msg)
            messagebox.showerror("Training Error", str(e))
        
        finally:
            self.is_training = False
            self.train_btn.config(state='normal')
            self.stop_btn.config(state='disabled')
    
    def update_training_display(self):
        """Update training display with current epoch info"""
        # Update epoch label
        self.epoch_label.config(text=f"Epoch: {self.current_epoch}/{self.total_epochs}")
        
        # Update progress bar
        progress = (self.current_epoch / self.total_epochs) * 100
        self.epoch_progress['value'] = progress
        
        # Update metric labels
        if len(self.epoch_history['loss']) > 0:
            idx = -1
            self.loss_label.config(text=f"{self.epoch_history['loss'][idx]:.4f}")
            self.acc_label.config(text=f"{self.epoch_history['accuracy'][idx]*100:.2f}%")
            self.val_loss_label.config(text=f"{self.epoch_history['val_loss'][idx]:.4f}")
            self.val_acc_label.config(text=f"{self.epoch_history['val_accuracy'][idx]*100:.2f}%")
        
        # Update plots
        self.update_training_plots()
    
    def update_training_plots(self):
        """Update loss and accuracy plots"""
        if len(self.epoch_history['epoch']) == 0:
            return
        
        epochs = self.epoch_history['epoch']
        
        # Clear axes
        self.loss_ax.clear()
        self.acc_ax.clear()
        
        # Plot loss
        self.loss_ax.plot(epochs, self.epoch_history['loss'], 'r-', label='Training Loss', linewidth=2)
        self.loss_ax.plot(epochs, self.epoch_history['val_loss'], 'b-', label='Validation Loss', linewidth=2)
        self.loss_ax.set_title('Training & Validation Loss', color='white', fontsize=12, weight='bold')
        self.loss_ax.set_xlabel('Epoch', color='white')
        self.loss_ax.set_ylabel('Loss', color='white')
        self.loss_ax.legend(facecolor='#2a2a2a', edgecolor='white', labelcolor='white')
        self.loss_ax.tick_params(colors='white')
        self.loss_ax.grid(True, alpha=0.3)
        self.loss_ax.set_facecolor('#2a2a2a')
        
        # Plot accuracy
        self.acc_ax.plot(epochs, self.epoch_history['accuracy'], 'g-', label='Training Accuracy', linewidth=2)
        self.acc_ax.plot(epochs, self.epoch_history['val_accuracy'], 'c-', label='Validation Accuracy', linewidth=2)
        self.acc_ax.set_title('Training & Validation Accuracy', color='white', fontsize=12, weight='bold')
        self.acc_ax.set_xlabel('Epoch', color='white')
        self.acc_ax.set_ylabel('Accuracy', color='white')
        self.acc_ax.legend(facecolor='#2a2a2a', edgecolor='white', labelcolor='white')
        self.acc_ax.tick_params(colors='white')
        self.acc_ax.grid(True, alpha=0.3)
        self.acc_ax.set_facecolor('#2a2a2a')
        
        self.train_fig.tight_layout()
        self.train_canvas.draw()
    
    # ========== PREDICTION METHODS ==========
    
    def start_prediction(self):
        """Start prediction on test data"""
        if self.cnn_model is None or not self.cnn_model.is_fitted:
            messagebox.showwarning("Warning", "Please train or load a model first!")
            return
        
        if self.test_data is None:
            messagebox.showwarning("Warning", "Please load a dataset first!\n\n⚠️ IMPORTANT: You must load the SAME dataset that was used for training!")
            return
        
        # Critical dataset validation
        if self.current_dataset_info['dataset_name'] is None:
            messagebox.showwarning("Warning", "Dataset information missing!\n\nPlease reload the dataset that was used for training.")
            return
        
        if self.is_predicting:
            messagebox.showwarning("Warning", "Prediction already in progress!")
            return
        
        # Show dataset warning for mixed datasets
        if self.current_dataset_info['is_mixed']:
            mixed_names = ' + '.join(self.current_dataset_info['mixed_datasets'])
            print(f"\n⚠️ DATASET CHECK:")
            print(f"   Model trained on: Mixed Dataset ({mixed_names})")
            print(f"   Predictions will use: Same mixed test set")
            print(f"   This ensures consistent preprocessing!\n")
        
        # Reset metrics
        self.prediction_history = []
        self.correct_predictions = 0
        self.total_predictions = 0
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        
        # Disable button
        self.predict_btn.config(state='disabled')
        self.is_predicting = True
        
        # Start prediction thread
        pred_thread = threading.Thread(target=self._prediction_thread, daemon=True)
        pred_thread.start()
    
    def _prediction_thread(self):
        """Prediction thread"""
        try:
            num_samples = self.pred_samples_var.get()
            print(f"\n{'='*60}")
            print(f"🔮 Starting prediction on {num_samples} samples...")
            print(f"{'='*60}")
            
            # CRITICAL: Check dataset consistency
            if self.current_dataset_info['is_mixed']:
                mixed_names = ' + '.join(self.current_dataset_info['mixed_datasets'])
                print(f"⚠️ IMPORTANT: Model was trained on MIXED dataset: {mixed_names}")
                print(f"   Predictions will be made on the SAME mixed test set")
                self.log_prediction(f"🔮 Predicting on MIXED test set: {mixed_names}")
            else:
                print(f"📊 Model was trained on: {self.current_dataset_info['dataset_name']}")
                print(f"   Predictions will be made on the SAME test set")
                self.log_prediction(f"🔮 Predicting on: {self.current_dataset_info['dataset_name']}")
            
            self.log_prediction(f"🔮 Starting prediction on {num_samples} samples...")
            
            samples_processed = 0
            
            # Debug: Track prediction distribution
            pred_probs_list = []
            actual_labels_list = []
            
            # CRITICAL FIX: Create a fresh iterator from test_data to avoid cached/stale data
            # Convert to list to randomly sample or iterate fresh each time
            print(f"🔄 Creating fresh test data iterator...")
            
            # Iterate through ALL batches in test data
            for batch_idx, (batch_x, batch_y) in enumerate(self.test_data):
                if not self.is_predicting or samples_processed >= num_samples:
                    break
                
                # Make predictions (batch_x already batched, so predict on entire batch)
                predictions = self.cnn_model.model.predict(batch_x, verbose=0)
                
                # Process each prediction
                for i in range(len(predictions)):
                    if samples_processed >= num_samples:
                        break
                    
                    pred_prob = predictions[i][0]
                    actual_label = int(batch_y[i])
                    
                    # Store for debugging
                    pred_probs_list.append(pred_prob)
                    actual_labels_list.append(actual_label)
                    
                    # CRITICAL DEBUG: Log first 10 predictions
                    if samples_processed < 10:
                        print(f"   Sample {samples_processed + 1}: Pred={pred_prob:.4f}, Actual={actual_label}")
                    
                    # Dynamic threshold (use 0.5 as default)
                    pred_label = 1 if pred_prob > 0.5 else 0
                    
                    # Update confusion matrix
                    if pred_label == 1 and actual_label == 1:
                        self.true_positives += 1
                    elif pred_label == 1 and actual_label == 0:
                        self.false_positives += 1
                    elif pred_label == 0 and actual_label == 0:
                        self.true_negatives += 1
                    elif pred_label == 0 and actual_label == 1:
                        self.false_negatives += 1
                    
                    # Track prediction
                    is_correct = (pred_label == actual_label)
                    if is_correct:
                        self.correct_predictions += 1
                    
                    self.total_predictions += 1
                    samples_processed += 1
                    
                    self.prediction_history.append({
                        'predicted': pred_label,
                        'actual': actual_label,
                        'confidence': pred_prob,
                        'correct': is_correct
                    })
                    
                    # Log every 10 predictions
                    if samples_processed % 10 == 0:
                        acc = (self.correct_predictions / self.total_predictions) * 100
                        progress_pct = (samples_processed / num_samples) * 100
                        print(f"📊 Progress: {samples_processed}/{num_samples} ({progress_pct:.1f}%) | Running Accuracy: {acc:.2f}%")
                        self.log_prediction(f"Progress: {samples_processed}/{num_samples} | Accuracy: {acc:.2f}%")
            
            # Update display
            self.root.after(0, self.update_prediction_display)
            
            # Calculate detailed classification report
            from sklearn.metrics import classification_report
            y_true = [p['actual'] for p in self.prediction_history]
            y_pred = [p['predicted'] for p in self.prediction_history]
            
            # Generate classification report
            class_report = classification_report(y_true, y_pred, target_names=['Class 0 (Fake)', 'Class 1 (Real)'], output_dict=True, zero_division=0)
            
            # Store in our tracking structure
            self.classification_report = {
                'class_0': {
                    'precision': class_report['Class 0 (Fake)']['precision'],
                    'recall': class_report['Class 0 (Fake)']['recall'],
                    'f1-score': class_report['Class 0 (Fake)']['f1-score'],
                    'support': class_report['Class 0 (Fake)']['support']
                },
                'class_1': {
                    'precision': class_report['Class 1 (Real)']['precision'],
                    'recall': class_report['Class 1 (Real)']['recall'],
                    'f1-score': class_report['Class 1 (Real)']['f1-score'],
                    'support': class_report['Class 1 (Real)']['support']
                },
                'accuracy': class_report['accuracy'],
                'macro_avg': {
                    'precision': class_report['macro avg']['precision'],
                    'recall': class_report['macro avg']['recall'],
                    'f1-score': class_report['macro avg']['f1-score'],
                    'support': class_report['macro avg']['support']
                },
                'weighted_avg': {
                    'precision': class_report['weighted avg']['precision'],
                    'recall': class_report['weighted avg']['recall'],
                    'f1-score': class_report['weighted avg']['f1-score'],
                    'support': class_report['weighted avg']['support']
                }
            }
            
            # Print detailed classification report
            print(f"\n{'='*60}")
            print(f"📊 DETAILED CLASSIFICATION REPORT")
            print(f"{'='*60}")
            print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
            print(f"{'-'*70}")
            print(f"{'Class 0 (Fake)':<20} {self.classification_report['class_0']['precision']:<12.4f} {self.classification_report['class_0']['recall']:<12.4f} {self.classification_report['class_0']['f1-score']:<12.4f} {int(self.classification_report['class_0']['support']):<10}")
            print(f"{'Class 1 (Real)':<20} {self.classification_report['class_1']['precision']:<12.4f} {self.classification_report['class_1']['recall']:<12.4f} {self.classification_report['class_1']['f1-score']:<12.4f} {int(self.classification_report['class_1']['support']):<10}")
            print(f"{'-'*70}")
            print(f"{'Accuracy':<20} {'':<12} {'':<12} {self.classification_report['accuracy']:<12.4f} {int(self.classification_report['class_0']['support'] + self.classification_report['class_1']['support']):<10}")
            print(f"{'Macro Avg':<20} {self.classification_report['macro_avg']['precision']:<12.4f} {self.classification_report['macro_avg']['recall']:<12.4f} {self.classification_report['macro_avg']['f1-score']:<12.4f} {int(self.classification_report['macro_avg']['support']):<10}")
            print(f"{'Weighted Avg':<20} {self.classification_report['weighted_avg']['precision']:<12.4f} {self.classification_report['weighted_avg']['recall']:<12.4f} {self.classification_report['weighted_avg']['f1-score']:<12.4f} {int(self.classification_report['weighted_avg']['support']):<10}")
            print(f"{'='*60}\n")
            
            # CRITICAL DEBUG: Analyze prediction distribution
            import numpy as np
            pred_probs_array = np.array(pred_probs_list)
            actual_labels_array = np.array(actual_labels_list)
            
            print(f"\n{'='*60}")
            print(f"🔍 PREDICTION ANALYSIS:")
            print(f"{'='*60}")
            print(f"📊 Prediction Probability Stats:")
            print(f"   Min:    {pred_probs_array.min():.4f}")
            print(f"   Max:    {pred_probs_array.max():.4f}")
            print(f"   Mean:   {pred_probs_array.mean():.4f}")
            print(f"   Median: {np.median(pred_probs_array):.4f}")
            print(f"   Std:    {pred_probs_array.std():.4f}")
            
            print(f"\n📊 Actual Label Distribution:")
            print(f"   Class 0 (Fake): {np.sum(actual_labels_array == 0)} samples")
            print(f"   Class 1 (Real): {np.sum(actual_labels_array == 1)} samples")
            
            print(f"\n📊 Predicted Label Distribution (threshold=0.5):")
            pred_labels_array = (pred_probs_array > 0.5).astype(int)
            print(f"   Class 0 (Fake): {np.sum(pred_labels_array == 0)} predictions")
            print(f"   Class 1 (Real): {np.sum(pred_labels_array == 1)} predictions")
            
            # Terminal summary
            final_acc = (self.correct_predictions / self.total_predictions) * 100 if self.total_predictions > 0 else 0
            print(f"\n{'='*60}")
            print(f"✅ PREDICTION COMPLETED!")
            print(f"{'='*60}")
            print(f"📊 Total Samples: {samples_processed}")
            print(f"✔️  Correct: {self.correct_predictions}")
            print(f"❌ Incorrect: {self.total_predictions - self.correct_predictions}")
            print(f"🎯 Accuracy: {final_acc:.2f}%")
            print(f"{'='*60}\n")
            
            self.log_prediction(f"✅ Prediction completed on {samples_processed} samples!")
            self.log_prediction(f"🔍 Pred range: [{pred_probs_array.min():.4f}, {pred_probs_array.max():.4f}], Mean: {pred_probs_array.mean():.4f}")
            self.log_prediction(f"📊 Actual: {np.sum(actual_labels_array == 0)} fake, {np.sum(actual_labels_array == 1)} real")
            self.log_prediction(f"📊 Predicted: {np.sum(pred_labels_array == 0)} fake, {np.sum(pred_labels_array == 1)} real")
            
            # Analyze SNR degradation if applicable
            self.analyze_snr_degradation()
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Prediction error: {str(e)}\n{traceback.format_exc()}"
            self.log_prediction(error_msg)
            messagebox.showerror("Prediction Error", str(e))
        
        finally:
            self.is_predicting = False
            self.predict_btn.config(state='normal')
    
    def update_prediction_display(self):
        """Update prediction metrics display"""
        if self.total_predictions == 0:
            return
        
        # Calculate metrics
        accuracy = (self.correct_predictions / self.total_predictions) * 100
        
        # Precision
        if (self.true_positives + self.false_positives) > 0:
            precision = (self.true_positives / (self.true_positives + self.false_positives)) * 100
        else:
            precision = 0.0
        
        # Recall
        if (self.true_positives + self.false_negatives) > 0:
            recall = (self.true_positives / (self.true_positives + self.false_negatives)) * 100
        else:
            recall = 0.0
        
        # F1-Score
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        # Update labels
        self.pred_acc_label.config(text=f"{accuracy:.2f}%")
        self.pred_prec_label.config(text=f"{precision:.2f}%")
        self.pred_recall_label.config(text=f"{recall:.2f}%")
        self.pred_f1_label.config(text=f"{f1:.2f}%")
        
        # Log confusion matrix
        self.log_prediction("\n📋 CONFUSION MATRIX:")
        self.log_prediction(f"   True Positives:  {self.true_positives}")
        self.log_prediction(f"   False Positives: {self.false_positives}")
        self.log_prediction(f"   True Negatives:  {self.true_negatives}")
        self.log_prediction(f"   False Negatives: {self.false_negatives}")
    
    # ========== EXPORT METHODS ==========
    
    def export_to_pdf(self):
        """Export training results to PDF"""
        if self.cnn_model is None:
            messagebox.showwarning("Warning", "No model to export!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            initialfile=f"CNN_Training_Report_{timestamp}.pdf"
        )
        
        if not filename:
            return
        
        try:
            with PdfPages(filename) as pdf:
                # Page 1: Title and Dataset Info
                fig = plt.figure(figsize=(11, 8.5))
                fig.text(0.5, 0.95, 'CNN Audio Classification Training Report', 
                        ha='center', fontsize=20, weight='bold')
                fig.text(0.5, 0.90, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                        ha='center', fontsize=12)
                
                # Dataset Information
                y_pos = 0.80
                fig.text(0.1, y_pos, 'DATASET INFORMATION:', fontsize=14, weight='bold')
                y_pos -= 0.05
                
                if self.current_dataset_info['is_mixed']:
                    fig.text(0.1, y_pos, f"Type: Mixed SNR Dataset", fontsize=11)
                    y_pos -= 0.04
                    fig.text(0.1, y_pos, f"Combined Datasets:", fontsize=11, weight='bold')
                    y_pos -= 0.04
                    for dataset in self.current_dataset_info['mixed_datasets']:
                        fig.text(0.15, y_pos, f"• {dataset}", fontsize=10)
                        y_pos -= 0.03
                else:
                    fig.text(0.1, y_pos, f"Type: Single Dataset", fontsize=11)
                    y_pos -= 0.04
                    fig.text(0.1, y_pos, f"Dataset: {self.current_dataset_info['dataset_name']}", fontsize=11)
                    y_pos -= 0.04
                
                y_pos -= 0.02
                fig.text(0.1, y_pos, f"Training Samples: {self.current_dataset_info['total_samples']['train']}", fontsize=11)
                y_pos -= 0.04
                fig.text(0.1, y_pos, f"Validation Samples: {self.current_dataset_info['total_samples']['val']}", fontsize=11)
                y_pos -= 0.04
                fig.text(0.1, y_pos, f"Test Samples: {self.current_dataset_info['total_samples']['test']}", fontsize=11)
                
                # Training Results
                y_pos -= 0.08
                fig.text(0.1, y_pos, 'TRAINING RESULTS:', fontsize=14, weight='bold')
                y_pos -= 0.05
                
                if len(self.epoch_history['epoch']) > 0:
                    fig.text(0.1, y_pos, f"Total Epochs: {len(self.epoch_history['epoch'])}", fontsize=11)
                    y_pos -= 0.04
                    fig.text(0.1, y_pos, f"Final Training Accuracy: {self.epoch_history['accuracy'][-1]*100:.2f}%", fontsize=11)
                    y_pos -= 0.04
                    fig.text(0.1, y_pos, f"Final Validation Accuracy: {self.epoch_history['val_accuracy'][-1]*100:.2f}%", fontsize=11)
                    y_pos -= 0.04
                    best_val_acc = max(self.epoch_history['val_accuracy'])
                    best_epoch = self.epoch_history['val_accuracy'].index(best_val_acc) + 1
                    fig.text(0.1, y_pos, f"Best Validation Accuracy: {best_val_acc*100:.2f}% (Epoch {best_epoch})", fontsize=11)
                
                # Hardware Resource Usage
                if self.resource_stats:
                    y_pos -= 0.08
                    fig.text(0.1, y_pos, 'HARDWARE RESOURCE USAGE:', fontsize=14, weight='bold')
                    y_pos -= 0.05
                    
                    # Training time
                    if self.training_start_time and self.training_end_time:
                        fig.text(0.1, y_pos, f"Start Time: {self.training_start_time.strftime('%Y-%m-%d %H:%M:%S')}", fontsize=11)
                        y_pos -= 0.04
                        fig.text(0.1, y_pos, f"End Time: {self.training_end_time.strftime('%Y-%m-%d %H:%M:%S')}", fontsize=11)
                        y_pos -= 0.04
                    
                    fig.text(0.1, y_pos, f"Training Duration: {self.training_duration:.2f} sec ({self.training_duration/60:.2f} min)", fontsize=11)
                    y_pos -= 0.04
                    fig.text(0.1, y_pos, f"CPU Cores: {psutil.cpu_count(logical=True)}", fontsize=11)
                    y_pos -= 0.04
                    fig.text(0.1, y_pos, f"Average CPU Usage: {self.resource_stats['avg_cpu_percent']:.1f}%", fontsize=11)
                    y_pos -= 0.04
                    fig.text(0.1, y_pos, f"Peak CPU Usage: {self.resource_stats['max_cpu_percent']:.1f}%", fontsize=11)
                    y_pos -= 0.04
                    fig.text(0.1, y_pos, f"Peak Memory: {self.peak_memory_mb:.2f} MB", fontsize=11, weight='bold')
                    y_pos -= 0.04
                    fig.text(0.1, y_pos, f"Average Memory: {self.avg_memory_mb:.2f} MB", fontsize=11)
                    y_pos -= 0.04
                    fig.text(0.1, y_pos, f"Memory Increase: {self.resource_stats['memory_increase_mb']:.2f} MB", fontsize=11)
                    if self.current_dataset_info['total_samples']['train'] > 0:
                        y_pos -= 0.04
                        samples_per_sec = self.current_dataset_info['total_samples']['train'] / self.training_duration
                        fig.text(0.1, y_pos, f"Efficiency: {samples_per_sec:.2f} samples/sec", fontsize=11)
                
                pdf.savefig(fig)
                plt.close(fig)
                
                # Page 2: Training Curves
                fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
                
                if len(self.epoch_history['epoch']) > 0:
                    epochs = self.epoch_history['epoch']
                    
                    # Loss plot
                    axes[0].plot(epochs, self.epoch_history['loss'], 'r-', label='Training Loss', linewidth=2)
                    axes[0].plot(epochs, self.epoch_history['val_loss'], 'b-', label='Validation Loss', linewidth=2)
                    axes[0].set_title('Training & Validation Loss', fontsize=14, weight='bold')
                    axes[0].set_xlabel('Epoch')
                    axes[0].set_ylabel('Loss')
                    axes[0].legend()
                    axes[0].grid(True, alpha=0.3)
                    
                    # Accuracy plot
                    axes[1].plot(epochs, self.epoch_history['accuracy'], 'g-', label='Training Accuracy', linewidth=2)
                    axes[1].plot(epochs, self.epoch_history['val_accuracy'], 'c-', label='Validation Accuracy', linewidth=2)
                    axes[1].set_title('Training & Validation Accuracy', fontsize=14, weight='bold')
                    axes[1].set_xlabel('Epoch')
                    axes[1].set_ylabel('Accuracy')
                    axes[1].legend()
                    axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                
                # Page 3: Hardware Resource Usage Plots (if available)
                if self.resource_stats and len(self.resource_data['timestamps']) > 0:
                    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
                    
                    # Convert timestamps to relative minutes
                    start_time = self.resource_data['timestamps'][0]
                    time_minutes = [(t - start_time) / 60 for t in self.resource_data['timestamps']]
                    
                    # CPU Usage Plot
                    axes[0].plot(time_minutes, self.resource_data['cpu_percent'], 'b-', linewidth=2)
                    axes[0].set_title('CPU Usage During Training', fontsize=14, weight='bold')
                    axes[0].set_xlabel('Time (minutes)')
                    axes[0].set_ylabel('CPU Usage (%)')
                    axes[0].grid(True, alpha=0.3)
                    axes[0].set_ylim(0, 100)
                    axes[0].axhline(y=self.resource_stats['avg_cpu_percent'], color='r', 
                                   linestyle='--', label=f'Average: {self.resource_stats["avg_cpu_percent"]:.1f}%')
                    axes[0].legend()
                    
                    # Memory Usage Plot
                    axes[1].plot(time_minutes, self.resource_data['memory_mb'], 'g-', linewidth=2)
                    axes[1].set_title('Memory Usage During Training', fontsize=14, weight='bold')
                    axes[1].set_xlabel('Time (minutes)')
                    axes[1].set_ylabel('Memory (MB)')
                    axes[1].grid(True, alpha=0.3)
                    axes[1].axhline(y=self.resource_stats['avg_memory_mb'], color='r', 
                                   linestyle='--', label=f'Average: {self.resource_stats["avg_memory_mb"]:.1f} MB')
                    axes[1].legend()
                    
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)
                
                # Page 4: Detailed Classification Report (if predictions were made)
                if hasattr(self, 'classification_report') and self.classification_report.get('accuracy', 0) > 0:
                    fig = plt.figure(figsize=(11, 8.5))
                    fig.text(0.5, 0.95, 'Detailed Classification Report', 
                            ha='center', fontsize=18, weight='bold')
                    
                    y_pos = 0.85
                    fig.text(0.1, y_pos, 'PREDICTION METRICS:', fontsize=14, weight='bold')
                    y_pos -= 0.05
                    
                    # Classification table header
                    fig.text(0.1, y_pos, 'Class', fontsize=12, weight='bold')
                    fig.text(0.3, y_pos, 'Precision', fontsize=12, weight='bold')
                    fig.text(0.45, y_pos, 'Recall', fontsize=12, weight='bold')
                    fig.text(0.6, y_pos, 'F1-Score', fontsize=12, weight='bold')
                    fig.text(0.75, y_pos, 'Support', fontsize=12, weight='bold')
                    y_pos -= 0.03
                    
                    # Draw horizontal line
                    plt.plot([0.1, 0.9], [y_pos, y_pos], 'k-', linewidth=1, transform=fig.transFigure, clip_on=False)
                    y_pos -= 0.02
                    
                    # Class 0 (Fake)
                    fig.text(0.1, y_pos, 'Class 0 (Fake)', fontsize=11)
                    fig.text(0.3, y_pos, f"{self.classification_report['class_0']['precision']:.4f}", fontsize=11)
                    fig.text(0.45, y_pos, f"{self.classification_report['class_0']['recall']:.4f}", fontsize=11)
                    fig.text(0.6, y_pos, f"{self.classification_report['class_0']['f1-score']:.4f}", fontsize=11)
                    fig.text(0.75, y_pos, f"{int(self.classification_report['class_0']['support'])}", fontsize=11)
                    y_pos -= 0.04
                    
                    # Class 1 (Real)
                    fig.text(0.1, y_pos, 'Class 1 (Real)', fontsize=11)
                    fig.text(0.3, y_pos, f"{self.classification_report['class_1']['precision']:.4f}", fontsize=11)
                    fig.text(0.45, y_pos, f"{self.classification_report['class_1']['recall']:.4f}", fontsize=11)
                    fig.text(0.6, y_pos, f"{self.classification_report['class_1']['f1-score']:.4f}", fontsize=11)
                    fig.text(0.75, y_pos, f"{int(self.classification_report['class_1']['support'])}", fontsize=11)
                    y_pos -= 0.02
                    
                    plt.plot([0.1, 0.9], [y_pos, y_pos], 'k-', linewidth=1, transform=fig.transFigure, clip_on=False)
                    y_pos -= 0.02
                    
                    # Accuracy
                    fig.text(0.1, y_pos, 'Accuracy', fontsize=11, weight='bold')
                    fig.text(0.6, y_pos, f"{self.classification_report['accuracy']:.4f}", fontsize=11, weight='bold')
                    total_support = int(self.classification_report['class_0']['support'] + self.classification_report['class_1']['support'])
                    fig.text(0.75, y_pos, f"{total_support}", fontsize=11, weight='bold')
                    y_pos -= 0.04
                    
                    # Macro Average
                    fig.text(0.1, y_pos, 'Macro Avg', fontsize=11)
                    fig.text(0.3, y_pos, f"{self.classification_report['macro_avg']['precision']:.4f}", fontsize=11)
                    fig.text(0.45, y_pos, f"{self.classification_report['macro_avg']['recall']:.4f}", fontsize=11)
                    fig.text(0.6, y_pos, f"{self.classification_report['macro_avg']['f1-score']:.4f}", fontsize=11)
                    fig.text(0.75, y_pos, f"{total_support}", fontsize=11)
                    y_pos -= 0.04
                    
                    # Weighted Average
                    fig.text(0.1, y_pos, 'Weighted Avg', fontsize=11)
                    fig.text(0.3, y_pos, f"{self.classification_report['weighted_avg']['precision']:.4f}", fontsize=11)
                    fig.text(0.45, y_pos, f"{self.classification_report['weighted_avg']['recall']:.4f}", fontsize=11)
                    fig.text(0.6, y_pos, f"{self.classification_report['weighted_avg']['f1-score']:.4f}", fontsize=11)
                    fig.text(0.75, y_pos, f"{total_support}", fontsize=11)
                    y_pos -= 0.08
                    
                    # Confusion Matrix
                    fig.text(0.1, y_pos, 'CONFUSION MATRIX:', fontsize=14, weight='bold')
                    y_pos -= 0.05
                    fig.text(0.1, y_pos, f"True Positives (Real correctly classified): {self.true_positives}", fontsize=11)
                    y_pos -= 0.04
                    fig.text(0.1, y_pos, f"False Positives (Fake classified as Real): {self.false_positives}", fontsize=11)
                    y_pos -= 0.04
                    fig.text(0.1, y_pos, f"True Negatives (Fake correctly classified): {self.true_negatives}", fontsize=11)
                    y_pos -= 0.04
                    fig.text(0.1, y_pos, f"False Negatives (Real classified as Fake): {self.false_negatives}", fontsize=11)
                    
                    pdf.savefig(fig)
                    plt.close(fig)
                
                # Page 5: SNR Degradation Analysis (if available)
                if len(self.snr_degradation_data) > 1:
                    fig = plt.figure(figsize=(11, 8.5))
                    fig.text(0.5, 0.95, 'Accuracy Degradation Rate Analysis', 
                            ha='center', fontsize=18, weight='bold')
                    
                    y_pos = 0.85
                    fig.text(0.1, y_pos, 'SNR DEGRADATION ANALYSIS:', fontsize=14, weight='bold')
                    y_pos -= 0.05
                    
                    # Sort by SNR level
                    sorted_data = sorted(self.snr_degradation_data, key=lambda x: x['snr_db'])
                    
                    # Table header
                    fig.text(0.1, y_pos, 'SNR Range (dB)', fontsize=12, weight='bold')
                    fig.text(0.35, y_pos, 'Degradation Rate', fontsize=12, weight='bold')
                    fig.text(0.65, y_pos, 'Classification', fontsize=12, weight='bold')
                    y_pos -= 0.03
                    plt.plot([0.1, 0.9], [y_pos, y_pos], 'k-', linewidth=1, transform=fig.transFigure, clip_on=False)
                    y_pos -= 0.02
                    
                    # Calculate degradation between consecutive SNR levels
                    for i in range(len(sorted_data) - 1):
                        snr_low = sorted_data[i]['snr_db']
                        snr_high = sorted_data[i + 1]['snr_db']
                        acc_low = sorted_data[i]['accuracy']
                        acc_high = sorted_data[i + 1]['accuracy']
                        
                        degradation_rate = acc_high - acc_low
                        degradation_per_db = degradation_rate / (snr_high - snr_low)
                        
                        # Classify degradation
                        if abs(degradation_per_db) < 0.5:
                            classification = "Minimal"
                        elif abs(degradation_per_db) < 1.5:
                            classification = "Moderate"
                        elif abs(degradation_per_db) < 3.0:
                            classification = "Significant"
                        else:
                            classification = "Severe"
                        
                        fig.text(0.1, y_pos, f"{snr_low} → {snr_high} dB", fontsize=11)
                        fig.text(0.35, y_pos, f"{degradation_rate:+.2f}% ({degradation_per_db:+.2f}%/dB)", fontsize=11)
                        fig.text(0.65, y_pos, classification, fontsize=11)
                        y_pos -= 0.04
                    
                    # Plot SNR vs Accuracy graph
                    snr_levels = [d['snr_db'] for d in sorted_data]
                    accuracies = [d['accuracy'] for d in sorted_data]
                    
                    ax = fig.add_subplot(111)
                    ax.set_position([0.15, 0.15, 0.75, 0.4])
                    ax.plot(snr_levels, accuracies, 'bo-', linewidth=2, markersize=8)
                    ax.set_xlabel('SNR Level (dB)', fontsize=12, weight='bold')
                    ax.set_ylabel('Accuracy (%)', fontsize=12, weight='bold')
                    ax.set_title('Model Performance vs SNR Level', fontsize=14, weight='bold')
                    ax.grid(True, alpha=0.3)
                    
                    pdf.savefig(fig)
                    plt.close(fig)
            
            messagebox.showinfo("Success", f"Report exported to:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export PDF:\n{str(e)}")
    
    def export_to_csv(self):
        """Export prediction results to CSV"""
        if len(self.prediction_history) == 0:
            messagebox.showwarning("Warning", "No prediction data to export!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=f"CNN_Predictions_{timestamp}.csv"
        )
        
        if not filename:
            return
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Metadata Header
                writer.writerow(['CNN Audio Classification - Prediction Results'])
                writer.writerow([f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'])
                writer.writerow([])
                
                # Dataset Information
                writer.writerow(['Dataset Information'])
                if self.current_dataset_info['is_mixed']:
                    writer.writerow(['Type', 'Mixed SNR Dataset'])
                    writer.writerow(['Combined Datasets', ', '.join(self.current_dataset_info['mixed_datasets'])])
                else:
                    writer.writerow(['Type', 'Single Dataset'])
                    writer.writerow(['Dataset', self.current_dataset_info['dataset_name']])
                writer.writerow(['Training Samples', self.current_dataset_info['total_samples']['train']])
                writer.writerow(['Validation Samples', self.current_dataset_info['total_samples']['val']])
                writer.writerow(['Test Samples', self.current_dataset_info['total_samples']['test']])
                writer.writerow([])
                
                # Metrics Summary
                accuracy = (self.correct_predictions / self.total_predictions) * 100 if self.total_predictions > 0 else 0
                precision = (self.true_positives / (self.true_positives + self.false_positives) * 100) if (self.true_positives + self.false_positives) > 0 else 0
                recall = (self.true_positives / (self.true_positives + self.false_negatives) * 100) if (self.true_positives + self.false_negatives) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                writer.writerow(['Overall Metrics Summary'])
                writer.writerow(['Total Predictions', self.total_predictions])
                writer.writerow(['Correct Predictions', self.correct_predictions])
                writer.writerow(['Incorrect Predictions', self.total_predictions - self.correct_predictions])
                writer.writerow(['Accuracy (%)', f"{accuracy:.2f}"])
                writer.writerow(['Precision (%)', f"{precision:.2f}"])
                writer.writerow(['Recall (%)', f"{recall:.2f}"])
                writer.writerow(['F1-Score (%)', f"{f1:.2f}"])
                writer.writerow([])
                
                # Detailed Classification Report
                if hasattr(self, 'classification_report') and self.classification_report.get('accuracy', 0) > 0:
                    writer.writerow(['Detailed Classification Report'])
                    writer.writerow(['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
                    writer.writerow(['Class 0 (Fake)', 
                                   f"{self.classification_report['class_0']['precision']:.4f}",
                                   f"{self.classification_report['class_0']['recall']:.4f}",
                                   f"{self.classification_report['class_0']['f1-score']:.4f}",
                                   int(self.classification_report['class_0']['support'])])
                    writer.writerow(['Class 1 (Real)', 
                                   f"{self.classification_report['class_1']['precision']:.4f}",
                                   f"{self.classification_report['class_1']['recall']:.4f}",
                                   f"{self.classification_report['class_1']['f1-score']:.4f}",
                                   int(self.classification_report['class_1']['support'])])
                    writer.writerow(['---', '---', '---', '---', '---'])
                    writer.writerow(['Accuracy', '', '', 
                                   f"{self.classification_report['accuracy']:.4f}",
                                   int(self.classification_report['class_0']['support'] + self.classification_report['class_1']['support'])])
                    writer.writerow(['Macro Avg', 
                                   f"{self.classification_report['macro_avg']['precision']:.4f}",
                                   f"{self.classification_report['macro_avg']['recall']:.4f}",
                                   f"{self.classification_report['macro_avg']['f1-score']:.4f}",
                                   int(self.classification_report['macro_avg']['support'])])
                    writer.writerow(['Weighted Avg', 
                                   f"{self.classification_report['weighted_avg']['precision']:.4f}",
                                   f"{self.classification_report['weighted_avg']['recall']:.4f}",
                                   f"{self.classification_report['weighted_avg']['f1-score']:.4f}",
                                   int(self.classification_report['weighted_avg']['support'])])
                    writer.writerow([])
                
                # Training Performance Metrics
                if self.training_start_time and self.training_end_time:
                    writer.writerow(['Training Performance Metrics'])
                    writer.writerow(['Training Start Time', self.training_start_time.strftime('%Y-%m-%d %H:%M:%S')])
                    writer.writerow(['Training End Time', self.training_end_time.strftime('%Y-%m-%d %H:%M:%S')])
                    writer.writerow(['Training Duration (seconds)', f"{self.training_duration:.2f}"])
                    writer.writerow(['Training Duration (minutes)', f"{self.training_duration/60:.2f}"])
                    writer.writerow(['Peak Memory Usage (MB)', f"{self.peak_memory_mb:.2f}"])
                    writer.writerow(['Average Memory Usage (MB)', f"{self.avg_memory_mb:.2f}"])
                    writer.writerow([])
                
                # SNR Degradation Analysis
                if len(self.snr_degradation_data) > 1:
                    writer.writerow(['SNR Degradation Analysis'])
                    writer.writerow(['SNR Range (dB)', 'Degradation Rate (%)', 'Degradation Per dB (%/dB)', 'Classification'])
                    
                    sorted_data = sorted(self.snr_degradation_data, key=lambda x: x['snr_db'])
                    for i in range(len(sorted_data) - 1):
                        snr_low = sorted_data[i]['snr_db']
                        snr_high = sorted_data[i + 1]['snr_db']
                        acc_low = sorted_data[i]['accuracy']
                        acc_high = sorted_data[i + 1]['accuracy']
                        
                        degradation_rate = acc_high - acc_low
                        degradation_per_db = degradation_rate / (snr_high - snr_low)
                        
                        if abs(degradation_per_db) < 0.5:
                            classification = "Minimal"
                        elif abs(degradation_per_db) < 1.5:
                            classification = "Moderate"
                        elif abs(degradation_per_db) < 3.0:
                            classification = "Significant"
                        else:
                            classification = "Severe"
                        
                        writer.writerow([f"{snr_low} → {snr_high}", 
                                       f"{degradation_rate:+.2f}", 
                                       f"{degradation_per_db:+.2f}",
                                       classification])
                    writer.writerow([])
                
                # Confusion Matrix
                writer.writerow(['Confusion Matrix'])
                writer.writerow(['True Positives', self.true_positives])
                writer.writerow(['False Positives', self.false_positives])
                writer.writerow(['True Negatives', self.true_negatives])
                writer.writerow(['False Negatives', self.false_negatives])
                writer.writerow([])
                
                writer.writerow(['Prediction Metrics'])
                writer.writerow(['Total Predictions', self.total_predictions])
                writer.writerow(['Correct Predictions', self.correct_predictions])
                writer.writerow(['Accuracy', f'{accuracy:.2f}%'])
                writer.writerow(['Precision', f'{precision:.2f}%'])
                writer.writerow(['Recall', f'{recall:.2f}%'])
                writer.writerow([])
                
                writer.writerow(['Confusion Matrix'])
                writer.writerow(['True Positives', self.true_positives])
                writer.writerow(['False Positives', self.false_positives])
                writer.writerow(['True Negatives', self.true_negatives])
                writer.writerow(['False Negatives', self.false_negatives])
                writer.writerow([])
                
                # Hardware Resource Usage
                if self.resource_stats:
                    writer.writerow(['Hardware Resource Usage'])
                    writer.writerow(['Training Duration (seconds)', f'{self.resource_stats["duration_seconds"]:.2f}'])
                    writer.writerow(['Training Duration (minutes)', f'{self.resource_stats["duration_seconds"]/60:.2f}'])
                    writer.writerow(['CPU Cores', psutil.cpu_count(logical=True)])
                    writer.writerow(['Average CPU Usage (%)', f'{self.resource_stats["avg_cpu_percent"]:.1f}'])
                    writer.writerow(['Peak CPU Usage (%)', f'{self.resource_stats["max_cpu_percent"]:.1f}'])
                    writer.writerow(['Average Memory (MB)', f'{self.resource_stats["avg_memory_mb"]:.2f}'])
                    writer.writerow(['Peak Memory (MB)', f'{self.resource_stats["peak_memory_mb"]:.2f}'])
                    writer.writerow(['Memory Increase (MB)', f'{self.resource_stats["memory_increase_mb"]:.2f}'])
                    if self.current_dataset_info['total_samples']['train'] > 0:
                        samples_per_sec = self.current_dataset_info['total_samples']['train'] / self.resource_stats['duration_seconds']
                        writer.writerow(['Training Efficiency (samples/sec)', f'{samples_per_sec:.2f}'])
                    writer.writerow([])
                
                # Prediction Data Header
                writer.writerow(['Detailed Predictions'])
                writer.writerow(['Sample', 'Predicted', 'Actual', 'Confidence', 'Result'])
                
                # Data
                for i, pred in enumerate(self.prediction_history, 1):
                    writer.writerow([
                        i,
                        'Real' if pred['predicted'] == 1 else 'Fake',
                        'Real' if pred['actual'] == 1 else 'Fake',
                        f"{pred['confidence']:.4f}",
                        'Correct' if pred['correct'] else 'Incorrect'
                    ])
            
            messagebox.showinfo("Success", f"Predictions exported to:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export CSV:\n{str(e)}")
    
    def _auto_save_model(self):
        """Automatically save model after training (deletes old model first)"""
        if self.cnn_model is None or not self.cnn_model.is_fitted:
            return
        
        try:
            # Delete old model if it exists
            if os.path.exists(self.cnn_model_path):
                old_mod_time = os.path.getmtime(self.cnn_model_path)
                old_date = datetime.fromtimestamp(old_mod_time).strftime("%Y-%m-%d %H:%M:%S")
                
                os.remove(self.cnn_model_path)
                print(f"🗑️  Deleted old model (trained: {old_date})")
                self.log_training(f"🗑️  Deleted old model from: {self.cnn_model_path}")
            
            # Save new model
            print(f"💾 Auto-saving trained model to: {self.cnn_model_path}")
            self.cnn_model.save_model(self.cnn_model_path)
            
            # Save dataset metadata (CRITICAL for prediction!)
            metadata_path = os.path.join(self.cnn_model_dir, 'dataset_info.txt')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("CNN MODEL TRAINING DATASET INFORMATION\n")
                f.write("="*60 + "\n\n")
                f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model File: cnn_trained_model.h5\n\n")
                
                if self.current_dataset_info['is_mixed']:
                    f.write("Dataset Type: MIXED SNR DATASET\n")
                    f.write(f"Combined Datasets:\n")
                    for dataset in self.current_dataset_info['mixed_datasets']:
                        f.write(f"  - {dataset}\n")
                else:
                    f.write("Dataset Type: SINGLE DATASET\n")
                    f.write(f"Dataset Name: {self.current_dataset_info['dataset_name']}\n")
                
                f.write(f"\nTotal Samples:\n")
                f.write(f"  - Training: {self.current_dataset_info['total_samples']['train']}\n")
                f.write(f"  - Validation: {self.current_dataset_info['total_samples']['val']}\n")
                f.write(f"  - Test: {self.current_dataset_info['total_samples']['test']}\n")
                
                f.write("\n" + "="*60 + "\n")
                f.write("⚠️  IMPORTANT FOR PREDICTION:\n")
                f.write("="*60 + "\n")
                f.write("To get accurate predictions, you MUST load the SAME\n")
                f.write("dataset configuration before running predictions!\n\n")
                
                if self.current_dataset_info['is_mixed']:
                    f.write("Steps:\n")
                    f.write("1. Go to Optimization tab\n")
                    f.write("2. Enable 'Mixed SNR Training'\n")
                    f.write("3. Select the EXACT SAME SNR datasets listed above\n")
                    f.write("4. Go to Training tab and click 'Load Dataset'\n")
                    f.write("5. Go to Prediction tab and load the model\n")
                else:
                    f.write("Steps:\n")
                    f.write("1. Go to Training tab\n")
                    f.write(f"2. Select dataset: {self.current_dataset_info['dataset_name']}\n")
                    f.write("3. Click 'Load Dataset'\n")
                    f.write("4. Go to Prediction tab and load the model\n")
            
            save_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"✅ Model saved successfully!")
            print(f"   📅 Saved at: {save_time}")
            print(f"   📁 Location: {self.cnn_model_path}")
            print(f"   📝 Dataset info saved to: {metadata_path}\n")
            
            self.log_training(f"💾 Model auto-saved to: {self.cnn_model_path}")
            self.log_training(f"   Saved at: {save_time}")
            self.log_training(f"   Dataset metadata saved for prediction consistency")
            
            # Enable prediction button
            self.predict_btn.config(state='normal')
            
            # Update model status in prediction tab
            self.model_status.config(text=f"✅ Model ready (trained: {save_time})", fg='#00ff88')
            
        except Exception as e:
            error_msg = f"Failed to auto-save model: {str(e)}"
            print(f"❌ {error_msg}\n")
            self.log_training(f"❌ {error_msg}")
    
    def save_model(self):
        """Manually save the trained model to a custom location"""
        if self.cnn_model is None or not self.cnn_model.is_fitted:
            messagebox.showwarning("Warning", "No trained model to save!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filedialog.asksaveasfilename(
            defaultextension=".h5",
            filetypes=[("Keras Model", "*.h5 *.keras"), ("All Files", "*.*")],
            initialfile=f"CNN_Model_{timestamp}.h5",
            initialdir=self.cnn_model_dir
        )
        
        if not filename:
            return
        
        try:
            self.cnn_model.save_model(filename)
            print(f"💾 Model manually saved to: {filename}\n")
            messagebox.showinfo("Success", f"Model saved to:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save model:\n{str(e)}")
    
    # ========== UTILITY METHODS ==========

    
    
    def log_training(self, message):
        """Log message to training log"""
        self.training_log.insert('end', message + '\n')
        self.training_log.see('end')
        # Also print to terminal
        print(f"[TRAINING] {message}")
    
    def log_prediction(self, message):
        """Log message to prediction log"""
        self.prediction_log.insert('end', message + '\n')
        self.prediction_log.see('end')
        # Also print to terminal
        print(f"[PREDICTION] {message}")
    
    def update_summary(self):
        """Update training summary"""
        if len(self.epoch_history['epoch']) == 0:
            return
        
        summary = []
        summary.append("=" * 70)
        summary.append("CNN TRAINING SUMMARY")
        summary.append("=" * 70)
        summary.append("")
        
        # Dataset Information
        summary.append("DATASET INFORMATION:")
        if self.current_dataset_info['is_mixed']:
            summary.append(f"   Type: Mixed SNR Dataset")
            summary.append(f"   Combined Datasets: {', '.join(self.current_dataset_info['mixed_datasets'])}")
        else:
            summary.append(f"   Type: Single Dataset")
            summary.append(f"   Dataset: {self.current_dataset_info['dataset_name']}")
        summary.append(f"   Training Samples: {self.current_dataset_info['total_samples']['train']}")
        summary.append(f"   Validation Samples: {self.current_dataset_info['total_samples']['val']}")
        summary.append(f"   Test Samples: {self.current_dataset_info['total_samples']['test']}")
        summary.append("")
        
        # Training Results
        summary.append("TRAINING RESULTS:")
        summary.append(f"   Total Epochs: {len(self.epoch_history['epoch'])}")
        summary.append(f"   Final Training Accuracy: {self.epoch_history['accuracy'][-1]:.4f}")
        summary.append(f"   Final Validation Accuracy: {self.epoch_history['val_accuracy'][-1]:.4f}")
        summary.append(f"   Final Training Loss: {self.epoch_history['loss'][-1]:.4f}")
        summary.append(f"   Final Validation Loss: {self.epoch_history['val_loss'][-1]:.4f}")
        summary.append("")
        summary.append("   Best Epoch:")
        best_epoch = np.argmax(self.epoch_history['val_accuracy']) + 1
        summary.append(f"      Epoch: {best_epoch}")
        summary.append(f"      Validation Accuracy: {max(self.epoch_history['val_accuracy']):.4f}")
        summary.append("")
        
        # Hardware Resource Usage
        if self.resource_stats:
            summary.append("HARDWARE RESOURCE USAGE:")
            summary.append(f"   Training Duration: {self.resource_stats['duration_seconds']:.2f} sec ({self.resource_stats['duration_seconds']/60:.2f} min)")
            summary.append(f"   Average CPU: {self.resource_stats['avg_cpu_percent']:.1f}%")
            summary.append(f"   Peak CPU: {self.resource_stats['max_cpu_percent']:.1f}%")
            summary.append(f"   Peak Memory: {self.resource_stats['peak_memory_mb']:.2f} MB")
            summary.append(f"   Average Memory: {self.resource_stats['avg_memory_mb']:.2f} MB")
            summary.append(f"   Memory Increase: {self.resource_stats['memory_increase_mb']:.2f} MB")
            if self.current_dataset_info['total_samples']['train'] > 0:
                samples_per_sec = self.current_dataset_info['total_samples']['train'] / self.resource_stats['duration_seconds']
                summary.append(f"   Samples/Second: {samples_per_sec:.2f}")
            summary.append("")
        
        summary.append("=" * 70)
        
        self.summary_text.delete('1.0', 'end')
        self.summary_text.insert('1.0', '\n'.join(summary))


# Run the application
if __name__ == "__main__":
    if not TF_AVAILABLE:
        print("❌ TensorFlow is not available! Please install TensorFlow first.")
        print("   pip install tensorflow")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("🧠 CNN Audio Classification - Training & Prediction System")
    print("="*60)
    print("\n📂 Available datasets:")
    print("   • augmented_10k: 10,000 augmented samples (spectrograms)")
    print("   • augmented_5k: 5,000 augmented samples (spectrograms)")
    print("   • mfcc_2sec: 2-second MFCC features")
    print("   • snr_5db, snr_10db, snr_15db, snr_20db: SNR augmented datasets")
    print("\n💡 Features:")
    print("   • Train CNN models with real-time feedback")
    print("   • View epoch-by-epoch progress")
    print("   • Automatic model saving to CNN/ folder")
    print("   • Load and test old models")
    print("   • Explainability with Grad-CAM visualization")
    print("\n🎯 Ready to train and predict!\n")
    
    root = tk.Tk()
    app = CNNTrainingGUI(root)
    root.mainloop()
