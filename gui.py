import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from datetime import datetime
import queue
import time
import random
import joblib
import sys
import os
import psutil
import csv
from matplotlib.backends.backend_pdf import PdfPages

class AdvancedSimulationAudioGUI:
    def __init__(self, root, loaded_models, loaded_datasets):
        self.root = root
        self.root.title("🎵 Advanced Audio Classification - Live Prediction Feedback")
        self.root.geometry("1600x1000")  # Increased size
        self.root.configure(bg='#1a1a1a')  # Dark theme
        self.models = loaded_models  # Dictionary of all models
        self.model = None  # Will be set when user selects a model
        self.datasets = loaded_datasets
        # Initialize variables
        self.log_queue = queue.Queue()
        self.selected_model = tk.StringVar(value="rf_model_reduced")
        self.selected_dataset = tk.StringVar(value="test_aug_snr_twenty")
        self.selected_val_dataset = tk.StringVar(value="val_aug_snr_twenty")
        
        # Enhanced simulation variables
        self.is_processing = False
        self.current_operation = ""
        self.operation_progress = 0
        self.prediction_speed = tk.DoubleVar(value=1.0)
        self.auto_mode = tk.BooleanVar(value=False)
        
        # Live prediction tracking
        self.correct_predictions = 0
        self.total_predictions = 0
        self.recent_predictions = []  # Store recent prediction results
        self.prediction_history = []  # Store all predictions for analysis
        self.current_sample_label = ""
        self.current_prediction = ""
        self.current_confidence = 0.0
        
        # Performance tracking
        self.performance_history = []
        self.real_time_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        
        # Visual enhancement variables
        self.animation_frame = 0
        self.pulse_direction = 1

        # Check available models and datasets from notebook
        self.populate_dropdowns_from_data()

        # Create the enhanced interface
        self.create_advanced_interface()

        # Start the updaters
        self.update_logs()
        self.update_simulation_display()
        self.update_animations()

    def create_advanced_interface(self):
        """Create enhanced interface with dark theme and improved visuals"""
        # Configure dark theme style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TNotebook", background='#2c3e50', borderwidth=0)
        style.configure("TNotebook.Tab", background='#34495e', foreground='white', 
                       padding=[20, 10], font=('Arial', 12, 'bold'))
        style.map("TNotebook.Tab", background=[('selected', '#3498db')])
        
        # Create main notebook for different tabs
        self.main_notebook = ttk.Notebook(self.root)
        self.main_notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create enhanced tabs
        self.create_control_tab()
        self.create_live_prediction_tab()
        self.create_training_tab()
        self.create_monitoring_tab()
        self.create_results_tab()
        
    def create_control_tab(self):
        """Create enhanced control tab with dark theme"""
        control_frame = tk.Frame(self.main_notebook, bg='#2c3e50')
        self.main_notebook.add(control_frame, text="🎮 Control Center")
        
        # Create scrollable area with dark theme
        canvas = tk.Canvas(control_frame, bg='#2c3e50', highlightthickness=0)
        scrollbar = ttk.Scrollbar(control_frame, orient="vertical", command=canvas.yview)
        self.control_content = tk.Frame(canvas, bg='#2c3e50')
        
        self.control_content.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.control_content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Add enhanced control sections
        self.create_enhanced_model_selection()
        self.create_enhanced_operation_controls()
        self.create_enhanced_simulation_settings()
        
    def create_live_prediction_tab(self):
        """Create live prediction feedback tab with SCROLLABLE container"""
        pred_frame = tk.Frame(self.main_notebook, bg='#1a1a1a')
        self.main_notebook.add(pred_frame, text="🔮 Live Predictions")
        
        # Create SCROLLABLE container
        pred_canvas = tk.Canvas(pred_frame, bg='#1a1a1a', highlightthickness=0)
        pred_scrollbar = ttk.Scrollbar(pred_frame, orient="vertical", command=pred_canvas.yview)
        self.pred_content = tk.Frame(pred_canvas, bg='#1a1a1a')
        
        self.pred_content.bind(
            "<Configure>",
            lambda e: pred_canvas.configure(scrollregion=pred_canvas.bbox("all"))
        )
        
        pred_canvas.create_window((0, 0), window=self.pred_content, anchor="nw")
        pred_canvas.configure(yscrollcommand=pred_scrollbar.set)
        
        pred_canvas.pack(side="left", fill="both", expand=True)
        pred_scrollbar.pack(side="right", fill="y")
        
        # Enable mouse wheel scrolling
        def _on_pred_mousewheel(event):
            pred_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        pred_canvas.bind_all("<MouseWheel>", _on_pred_mousewheel)
        
        # Header with animated status
        self.create_animated_header(self.pred_content)
        
        # Live prediction display
        self.create_live_prediction_display(self.pred_content)
        
        # Progress and statistics
        self.create_enhanced_progress_section(self.pred_content)
        
        # Recent predictions feed
        self.create_prediction_feed(self.pred_content)
        
        # Live visualization
        self.create_enhanced_live_visualization(self.pred_content)
        
    def create_animated_header(self, parent):
        """Create animated header with pulsing effects"""
        header_frame = tk.Frame(parent, bg='#0f3460', height=100, relief='raised', bd=3)
        header_frame.pack(fill='x', padx=10, pady=10)
        header_frame.pack_propagate(False)
        
        # Main title with gradient effect simulation
        self.status_title = tk.Label(header_frame, text="🎵 LIVE AUDIO CLASSIFICATION PREDICTOR", 
                                    font=('Arial', 20, 'bold'), fg='#00d4ff', bg='#0f3460')
        self.status_title.pack(pady=5)
        
        # Animated status with pulsing effect
        self.main_status = tk.Label(header_frame, text="🟢 SYSTEM READY", 
                                   font=('Arial', 16, 'bold'), fg='#00ff88', bg='#0f3460')
        self.main_status.pack()
        
        # Live counter display
        self.live_counter = tk.Label(header_frame, text="🎯 Predictions: 0/0 | Accuracy: 0%", 
                                    font=('Arial', 14, 'bold'), fg='#ffaa00', bg='#0f3460')
        self.live_counter.pack()
        
    def create_live_prediction_display(self, parent):
        """Create live prediction display with visual feedback"""
        display_frame = tk.LabelFrame(parent, text="🔍 Current Prediction Analysis", 
                                     font=('Arial', 16, 'bold'), bg='#2c3e50', fg='white',
                                     padx=20, pady=20)
        display_frame.pack(fill='x', padx=10, pady=10)
        
        # Current sample info
        sample_frame = tk.Frame(display_frame, bg='#34495e', relief='raised', bd=2)
        sample_frame.pack(fill='x', pady=10)
        
        self.current_sample_display = tk.Label(sample_frame, text="🎵 Analyzing: [No sample loaded]", 
                                              font=('Arial', 14, 'bold'), fg='#ecf0f1', bg='#34495e')
        self.current_sample_display.pack(pady=10)
        
        # Prediction result with color coding
        result_frame = tk.Frame(display_frame, bg='#2c3e50')
        result_frame.pack(fill='x', pady=10)
        
        # Left side - Prediction
        pred_side = tk.Frame(result_frame, bg='#3498db', relief='raised', bd=3, width=200)
        pred_side.pack(side='left', fill='both', expand=True, padx=5)
        pred_side.pack_propagate(False)
        
        tk.Label(pred_side, text="🤖 MODEL SAYS:", font=('Arial', 12, 'bold'), 
                bg='#3498db', fg='white').pack(pady=5)
        self.prediction_display = tk.Label(pred_side, text="[Waiting...]", 
                                          font=('Arial', 16, 'bold'), bg='#3498db', fg='white')
        self.prediction_display.pack(pady=5)
        
        self.confidence_display = tk.Label(pred_side, text="Confidence: 0%", 
                                          font=('Arial', 12), bg='#3498db', fg='white')
        self.confidence_display.pack(pady=2)
        
        # Right side - Actual
        actual_side = tk.Frame(result_frame, bg='#e74c3c', relief='raised', bd=3, width=200)
        actual_side.pack(side='right', fill='both', expand=True, padx=5)
        actual_side.pack_propagate(False)
        
        tk.Label(actual_side, text="✅ ACTUAL LABEL:", font=('Arial', 12, 'bold'), 
                bg='#e74c3c', fg='white').pack(pady=5)
        self.actual_display = tk.Label(actual_side, text="[Waiting...]", 
                                      font=('Arial', 16, 'bold'), bg='#e74c3c', fg='white')
        self.actual_display.pack(pady=5)
        
        # Result indicator
        self.result_indicator = tk.Label(display_frame, text="⏳ Ready to start prediction...", 
                                        font=('Arial', 18, 'bold'), fg='#f39c12', bg='#2c3e50')
        self.result_indicator.pack(pady=15)
        
    def create_enhanced_progress_section(self, parent):
        """Create enhanced progress section with live statistics"""
        progress_frame = tk.LabelFrame(parent, text="📊 Live Progress & Statistics", 
                                      font=('Arial', 16, 'bold'), bg='#2c3e50', fg='white',
                                      padx=20, pady=15)
        progress_frame.pack(fill='x', padx=10, pady=10)
        
        # Statistics row
        stats_frame = tk.Frame(progress_frame, bg='#2c3e50')
        stats_frame.pack(fill='x', pady=10)
        
        # Correct predictions counter
        correct_frame = tk.Frame(stats_frame, bg='#27ae60', relief='raised', bd=3)
        correct_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        tk.Label(correct_frame, text="✅ CORRECT", font=('Arial', 12, 'bold'), 
                bg='#27ae60', fg='white').pack(pady=5)
        self.correct_counter = tk.Label(correct_frame, text="0", font=('Arial', 24, 'bold'), 
                                       bg='#27ae60', fg='white')
        self.correct_counter.pack()
        
        # Incorrect predictions counter
        incorrect_frame = tk.Frame(stats_frame, bg='#e74c3c', relief='raised', bd=3)
        incorrect_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        tk.Label(incorrect_frame, text="❌ INCORRECT", font=('Arial', 12, 'bold'), 
                bg='#e74c3c', fg='white').pack(pady=5)
        self.incorrect_counter = tk.Label(incorrect_frame, text="0", font=('Arial', 24, 'bold'), 
                                         bg='#e74c3c', fg='white')
        self.incorrect_counter.pack()
        
        # Current accuracy
        accuracy_frame = tk.Frame(stats_frame, bg='#9b59b6', relief='raised', bd=3)
        accuracy_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        tk.Label(accuracy_frame, text="🎯 ACCURACY", font=('Arial', 12, 'bold'), 
                bg='#9b59b6', fg='white').pack(pady=5)
        self.live_accuracy = tk.Label(accuracy_frame, text="0.0%", font=('Arial', 24, 'bold'), 
                                     bg='#9b59b6', fg='white')
        self.live_accuracy.pack()
        
        # Enhanced progress bars
        self.create_enhanced_progress_bars(progress_frame)
        
    def create_enhanced_progress_bars(self, parent):
        """Create enhanced progress bars with animations"""
        bars_frame = tk.Frame(parent, bg='#2c3e50')
        bars_frame.pack(fill='x', pady=15)
        
        # Operation label with animation
        self.operation_label = tk.Label(bars_frame, text="Ready to start operations...", 
                                       font=('Arial', 14, 'bold'), bg='#2c3e50', fg='#ecf0f1')
        self.operation_label.pack(pady=5)
        
        # Main progress bar with custom styling
        style = ttk.Style()
        style.configure("Custom.Horizontal.TProgressbar", 
                       background='#3498db', troughcolor='#34495e', 
                       borderwidth=0, lightcolor='#3498db', darkcolor='#3498db')
        
        self.main_progress = ttk.Progressbar(bars_frame, length=700, mode='determinate',
                                           style="Custom.Horizontal.TProgressbar")
        self.main_progress.pack(fill='x', pady=10)
        
        # Progress percentage with animation
        self.progress_percent = tk.Label(bars_frame, text="0%", 
                                        font=('Arial', 14, 'bold'), bg='#2c3e50', fg='#ecf0f1')
        self.progress_percent.pack()
        
        # Enhanced circular progress
        self.create_enhanced_circular_progress(bars_frame)
        
    def create_enhanced_circular_progress(self, parent):
        """Create enhanced circular progress with animations"""
        circle_frame = tk.Frame(parent, bg='#2c3e50')
        circle_frame.pack(pady=15)
        
        self.circle_canvas = tk.Canvas(circle_frame, width=180, height=180, bg='#2c3e50', 
                                      highlightthickness=0)
        self.circle_canvas.pack()
        
        # Draw base circle with gradient effect
        self.circle_canvas.create_oval(15, 15, 165, 165, outline='#34495e', width=10, tags='base_circle')
        
        # Progress arc with glow effect
        self.progress_arc = self.circle_canvas.create_arc(15, 15, 165, 165, start=90, extent=0,
                                                         outline='#00d4ff', width=10, style='arc', 
                                                         tags='progress_arc')
        
        # Center text with enhanced styling
        self.circle_text = self.circle_canvas.create_text(90, 90, text="0%", 
                                                         font=('Arial', 18, 'bold'),
                                                         fill='#00d4ff', tags='circle_text')
        
    def create_prediction_feed(self, parent):
        """Create live prediction feed with LARGER SQUARE-SHAPED LOG AREA"""
        feed_frame = tk.LabelFrame(parent, text="📝 Live Prediction Feed", 
                                  font=('Arial', 18, 'bold'), bg='#2c3e50', fg='white',
                                  padx=20, pady=20)
        feed_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # SQUARE-SHAPED Feed display with MUCH LARGER FONT (35 lines x 85 width for square shape)
        self.prediction_feed = scrolledtext.ScrolledText(feed_frame, height=35, width=85,
                                                        bg='#1a1a1a', fg='#ecf0f1',
                                                        font=('Consolas', 15, 'bold'),  # INCREASED from 13 to 15
                                                        wrap=tk.WORD, insertbackground='white')
        self.prediction_feed.pack(fill='both', expand=True, pady=10)
        
        # Configure text tags for color coding with LARGER FONT
        self.prediction_feed.tag_configure("correct", foreground="#27ae60", font=('Consolas', 15, 'bold'))
        self.prediction_feed.tag_configure("incorrect", foreground="#e74c3c", font=('Consolas', 15, 'bold'))
        self.prediction_feed.tag_configure("info", foreground="#3498db", font=('Consolas', 15, 'bold'))
        self.prediction_feed.tag_configure("warning", foreground="#f39c12", font=('Consolas', 15, 'bold'))
        
        # Add initial message
        self.add_prediction_log("🚀 Live Prediction Feed Initialized", "info")
        self.add_prediction_log("📊 Ready to track real-time predictions!", "info")
        self.add_prediction_log("🤖 Multiple models available - Switch in Control Center!", "info")
        
    def create_enhanced_live_visualization(self, parent):
        """Create enhanced live visualization"""
        viz_frame = tk.LabelFrame(parent, text="📈 Live Performance Visualization", 
                                 font=('Arial', 16, 'bold'), bg='#2c3e50', fg='white',
                                 padx=15, pady=15)
        viz_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create matplotlib figure with dark theme
        plt.style.use('dark_background')
        self.live_fig, (self.acc_ax, self.conf_ax) = plt.subplots(1, 2, figsize=(14, 6))
        self.live_fig.patch.set_facecolor('#2c3e50')
        
        # Initialize accuracy plot
        self.acc_ax.set_title('Running Accuracy', fontsize=14, weight='bold', color='white')
        self.acc_ax.set_xlabel('Prediction #', color='white')
        self.acc_ax.set_ylabel('Accuracy', color='white')
        self.acc_ax.grid(True, alpha=0.3)
        self.acc_ax.set_facecolor('#1a1a1a')
        
        # Initialize confidence plot
        self.conf_ax.set_title('Prediction Confidence', fontsize=14, weight='bold', color='white')
        self.conf_ax.set_xlabel('Prediction #', color='white')
        self.conf_ax.set_ylabel('Confidence', color='white')
        self.conf_ax.grid(True, alpha=0.3)
        self.conf_ax.set_facecolor('#1a1a1a')
        
        plt.tight_layout()
        
        # Embed in tkinter
        self.live_canvas = FigureCanvasTkAgg(self.live_fig, viz_frame)
        self.live_canvas.draw()
        self.live_canvas.get_tk_widget().pack(fill='both', expand=True)
        
    def create_enhanced_model_selection(self):
        """Create enhanced model selection"""
        selection_frame = tk.LabelFrame(self.control_content, text="🤖 Advanced Model Selection", 
                                       font=('Arial', 16, 'bold'),
                                       bg='#34495e', fg='white', padx=20, pady=20)
        selection_frame.pack(fill='x', padx=10, pady=10)
        
        # Model selection with enhanced styling
        model_frame = tk.Frame(selection_frame, bg='#34495e')
        model_frame.pack(fill='x', pady=10)
        
        tk.Label(model_frame, text="🎯 Select Model:", font=('Arial', 14, 'bold'), 
                bg='#34495e', fg='white').pack(anchor='w', pady=5)
        
        model_combo = ttk.Combobox(model_frame, textvariable=self.selected_model,
                                  values=list(self.available_models.keys()) if hasattr(self, 'available_models') else [],
                                  state='readonly', width=40, font=('Arial', 12))
        model_combo.pack(fill='x', pady=5)
        model_combo.bind('<<ComboboxSelected>>', self.on_model_changed)
        
        # Dataset selection
        dataset_frame = tk.Frame(selection_frame, bg='#34495e')
        dataset_frame.pack(fill='x', pady=10)
        
        tk.Label(dataset_frame, text="📊 Select Dataset:", font=('Arial', 14, 'bold'), 
                bg='#34495e', fg='white').pack(anchor='w', pady=5)
        
        dataset_combo = ttk.Combobox(dataset_frame, textvariable=self.selected_dataset,
                                    values=list(self.available_datasets.keys()) if hasattr(self, 'available_datasets') else [],
                                    state='readonly', width=40, font=('Arial', 12))
        dataset_combo.pack(fill='x', pady=5)
        dataset_combo.bind('<<ComboboxSelected>>', self.on_dataset_changed)
        
        # Feature type indicator
        self.feature_type_frame = tk.Frame(dataset_frame, bg='#2c3e50', relief='raised', bd=2)
        self.feature_type_frame.pack(fill='x', pady=10)
        
        self.feature_type_label = tk.Label(self.feature_type_frame, 
                                          text="📊 Feature Type: Select a dataset", 
                                          font=('Arial', 12, 'bold'), 
                                          bg='#2c3e50', fg='#ecf0f1',
                                          padx=10, pady=5)
        self.feature_type_label.pack()
        
        # Update feature type on initial load
        self.update_feature_type_display()
        
        # Add helpful info panel
        info_frame = tk.LabelFrame(selection_frame, text="ℹ️ Dataset Feature Types", 
                                  font=('Arial', 12, 'bold'),
                                  bg='#34495e', fg='white', padx=15, pady=10)
        info_frame.pack(fill='x', pady=10)
        
        info_text = """
🎵 MFCC Datasets (Recommended for Tree Models):
   • 40 features per sample (vs 16,384 for spectrogram)
   • 400x faster training time
   • Lower memory usage (~2 MB vs ~800 MB)
   • Ideal for: Random Forest, XGBoost, KNN, SVM
   
📈 Spectrogram Datasets (High Detail):
   • 128x128 = 16,384 features per sample
   • More detailed frequency information
   • Longer training time
   • Ideal for: CNN, Deep Learning models
        """
        
        info_label = tk.Label(info_frame, text=info_text, 
                            font=('Consolas', 9), 
                            bg='#34495e', fg='#ecf0f1',
                            justify='left', anchor='w')
        info_label.pack(fill='x')
        
    def on_dataset_changed(self, event=None):
        """Called when dataset selection changes"""
        self.update_feature_type_display()
        
    def update_feature_type_display(self):
        """Update the feature type indicator based on selected dataset"""
        selected = self.selected_dataset.get()
        
        if selected and selected != 'none':
            if 'mfcc_' in selected:
                feature_text = "🎵 Feature Type: MFCC (40 coefficients) - Fast Training!"
                bg_color = '#27ae60'  # Green
            else:
                feature_text = "📈 Feature Type: Spectrogram (128x128) - High Detail"
                bg_color = '#3498db'  # Blue
            
            self.feature_type_label.config(text=feature_text)
            self.feature_type_frame.config(bg=bg_color)
            self.feature_type_label.config(bg=bg_color)
        else:
            self.feature_type_label.config(text="📊 Feature Type: Select a dataset")
            self.feature_type_frame.config(bg='#2c3e50')
            self.feature_type_label.config(bg='#2c3e50')
        
    def create_enhanced_operation_controls(self):
        """Create enhanced operation controls"""
        control_frame = tk.LabelFrame(self.control_content, text="🎮 Enhanced Operation Controls", 
                                     font=('Arial', 16, 'bold'),
                                     bg='#34495e', fg='white', padx=20, pady=20)
        control_frame.pack(fill='x', padx=10, pady=10)
        
        # Large enhanced buttons
        button_frame = tk.Frame(control_frame, bg='#34495e')
        button_frame.pack(fill='x', pady=15)
        
        # Start prediction button with glow effect simulation
        self.predict_btn = tk.Button(button_frame, text="🚀 START LIVE PREDICTION", 
                                    command=self.start_live_prediction,
                                    bg='#00d4ff', fg='#1a1a1a', font=('Arial', 16, 'bold'),
                                    height=3, width=30, relief='raised', bd=5)
        self.predict_btn.pack(pady=10)
        
        # Control buttons row
        control_row = tk.Frame(control_frame, bg='#34495e')
        control_row.pack(fill='x', pady=10)
        
        self.stop_btn = tk.Button(control_row, text="⏹️ STOP", command=self.stop_simulation,
                                 bg='#e74c3c', fg='white', font=('Arial', 14, 'bold'),
                                 height=2, width=12, relief='raised', bd=3)
        self.stop_btn.pack(side='left', padx=5)
        
        self.reset_btn = tk.Button(control_row, text="🔄 RESET", command=self.reset_simulation,
                                  bg='#9b59b6', fg='white', font=('Arial', 14, 'bold'),
                                  height=2, width=12, relief='raised', bd=3)
        self.reset_btn.pack(side='left', padx=5)
        
    def create_enhanced_simulation_settings(self):
        """Create enhanced simulation settings"""
        settings_frame = tk.LabelFrame(self.control_content, text="⚙️ Advanced Simulation Settings", 
                                      font=('Arial', 16, 'bold'),
                                      bg='#34495e', fg='white', padx=20, pady=20)
        settings_frame.pack(fill='x', padx=10, pady=10)
        
        # Speed control with enhanced styling
        speed_frame = tk.Frame(settings_frame, bg='#34495e')
        speed_frame.pack(fill='x', pady=10)
        
        tk.Label(speed_frame, text="🚀 Prediction Speed:", font=('Arial', 14, 'bold'), 
                bg='#34495e', fg='white').pack(anchor='w')
        
        self.speed_scale = tk.Scale(speed_frame, from_=0.1, to=10.0, resolution=0.1,
                                   orient='horizontal', variable=self.prediction_speed,
                                   font=('Arial', 12), bg='#34495e', fg='white',
                                   troughcolor='#2c3e50', activebackground='#3498db',
                                   length=300)
        self.speed_scale.pack(fill='x', pady=5)
        
        # Sample size control
        size_frame = tk.Frame(settings_frame, bg='#34495e')
        size_frame.pack(fill='x', pady=10)
        
        tk.Label(size_frame, text="📊 Sample Size:", font=('Arial', 14, 'bold'), 
                bg='#34495e', fg='white').pack(anchor='w')
        
        self.sample_size = tk.StringVar(value="50")
        size_combo = ttk.Combobox(size_frame, textvariable=self.sample_size,
                                 values=["25", "50", "100", "200"],
                                 state='readonly', width=15, font=('Arial', 12))
        size_combo.pack(anchor='w', pady=5)
        
    def start_live_prediction(self):
        """Start enhanced live prediction with feedback"""
        if self.is_processing:
            return
            
        self.is_processing = True
        self.correct_predictions = 0
        self.total_predictions = 0
        self.recent_predictions = []
        self.prediction_history = []
        
        self.main_status.config(text="🔥 LIVE PREDICTION ACTIVE", fg='#00ff88')
        self.predict_btn.config(state='disabled', bg='#95a5a6')
        
        # Clear feed
        self.prediction_feed.delete(1.0, tk.END)
        self.add_prediction_log("🚀 Starting Live Prediction Session...", "info")
        self.add_prediction_log("📊 Loading samples and generating predictions...", "info")
        
        # Start prediction thread
        threading.Thread(target=self._live_prediction_thread, daemon=True).start()
        
    def _live_prediction_thread(self):
        """Enhanced live prediction thread with detailed feedback using real datasets"""
        try:
            total_samples = int(self.sample_size.get())
            self.add_prediction_log(f"🎯 Processing {total_samples} samples...", "info")
            
            # Get selected dataset
            selected_key = self.selected_dataset.get()
            
            # Parse the dataset key (e.g., 'snr_5_test' -> dataset='snr_5', split='test')
            if selected_key == 'none' or not self.datasets:
                self.root.after(0, lambda: self.add_prediction_log("❌ No dataset selected!", "incorrect"))
                self.is_processing = False
                return
                
            # Split the key to get dataset name and split type
            parts = selected_key.rsplit('_', 1)
            if len(parts) != 2:
                self.root.after(0, lambda: self.add_prediction_log("❌ Invalid dataset key!", "incorrect"))
                self.is_processing = False
                return
                
            dataset_name = parts[0]
            split_type = parts[1]  # 'train', 'val', or 'test'
            
            # Get the actual dataset
            if dataset_name not in self.datasets:
                self.root.after(0, lambda: self.add_prediction_log(f"❌ Dataset '{dataset_name}' not found!", "incorrect"))
                self.is_processing = False
                return
                
            dataset = self.datasets[dataset_name][split_type]
            
            if dataset is None:
                self.root.after(0, lambda: self.add_prediction_log(f"❌ Dataset split '{split_type}' is None!", "incorrect"))
                self.is_processing = False
                return
            
            # Convert dataset to list of samples
            import numpy as np
            samples_x = []
            samples_y = []
            
            self.root.after(0, lambda: self.add_prediction_log(f"📦 Extracting samples from dataset...", "info"))
            
            # Unbatch the dataset first, then take samples
            sample_count = 0
            for batch_x, batch_y in dataset.unbatch():
                if sample_count >= total_samples:
                    break
                samples_x.append(batch_x.numpy())
                samples_y.append(batch_y.numpy())
                sample_count += 1
            
            if len(samples_x) == 0:
                self.root.after(0, lambda: self.add_prediction_log("❌ No samples found in dataset!", "incorrect"))
                self.is_processing = False
                return
            
            self.root.after(0, lambda sl=len(samples_x), dn=dataset_name, st=split_type: 
                          self.add_prediction_log(f"✅ Loaded {sl} samples from {dn} ({st})", "correct"))
            
            # Process each sample
            for sample_num in range(len(samples_x)):
                if not self.is_processing:
                    break
                    
                # Update current sample display with dataset info
                dataset_display = f"{dataset_name} ({split_type})"
                self.root.after(0, lambda s=sample_num+1, t=len(samples_x), d=dataset_display: 
                               self.update_current_sample(s, t, d))
                
                # Get actual sample and label
                sample_x = samples_x[sample_num]
                actual_label = int(samples_y[sample_num])
                
                # Make prediction using the model
                # Flatten the sample for Random Forest
                sample_flat = sample_x.reshape(1, -1)
                
                # Get prediction
                pred_proba = self.model.predict(sample_flat)
                confidence = float(pred_proba[0][0]) if pred_proba.shape[1] == 1 else float(pred_proba[0])
                predicted_label = 1 if confidence > 0.5 else 0
                
                # Determine if prediction is correct
                is_correct = (predicted_label == actual_label)
                
                # Update counters
                self.total_predictions += 1
                if is_correct:
                    self.correct_predictions += 1
                
                # Store prediction data
                prediction_data = {
                    'sample': sample_num + 1,
                    'actual': actual_label,
                    'predicted': predicted_label,
                    'confidence': confidence,
                    'correct': is_correct,
                    'timestamp': datetime.now(),
                    'dataset': dataset_display
                }
                
                self.prediction_history.append(prediction_data)
                self.recent_predictions.append(is_correct)
                
                # Keep only recent predictions (last 10)
                if len(self.recent_predictions) > 10:
                    self.recent_predictions.pop(0)
                
                # Update displays
                self.root.after(0, lambda pd=prediction_data, sn=sample_num+1, ts=len(samples_x): 
                               self.update_prediction_displays(pd, sn, ts))
                
                # Add detailed log entry
                self.root.after(0, lambda pd=prediction_data: self.log_prediction_result(pd))
                
                # Update progress
                progress = ((sample_num + 1) / len(samples_x)) * 100
                self.operation_progress = progress
                self.root.after(0, lambda p=progress: self.update_progress_displays(p))
                
                # Simulate processing time with speed control
                time.sleep(0.5 / self.prediction_speed.get())
                
            # Prediction complete
            if self.is_processing:
                self.root.after(0, self.prediction_session_complete)
                
        except Exception as e:
            import traceback
            error_msg = f"❌ Error: {e}\n{traceback.format_exc()}"
            self.root.after(0, lambda: self.add_prediction_log(error_msg, "incorrect"))
            self.is_processing = False
            
    def update_current_sample(self, sample_num, total_samples, dataset_info=None):
        """Update current sample display with detailed information"""
        if dataset_info:
            # Determine feature type
            feature_icon = "🎵" if "mfcc_" in dataset_info.lower() else "📈"
            feature_type = "MFCC" if "mfcc_" in dataset_info.lower() else "Spectrogram"
            text = f"{feature_icon} Sample #{sample_num:04d}/{total_samples} | {feature_type} | Dataset: {dataset_info}"
        else:
            text = f"🎵 Analyzing Sample #{sample_num:04d} of {total_samples}"
        self.current_sample_display.config(text=text)
        
    def update_prediction_displays(self, prediction_data, sample_num, total_samples):
        """Update all prediction displays with enhanced visuals"""
        # Update current prediction display
        actual_text = "🎵 REAL AUDIO" if prediction_data['actual'] == 1 else "🎭 FAKE AUDIO"
        predicted_text = "🎵 REAL AUDIO" if prediction_data['predicted'] == 1 else "🎭 FAKE AUDIO"
        confidence_text = f"Confidence: {prediction_data['confidence']:.1%}"
        
        self.prediction_display.config(text=predicted_text)
        self.confidence_display.config(text=confidence_text)
        self.actual_display.config(text=actual_text)
        
        # Update result indicator with animation
        if prediction_data['correct']:
            self.result_indicator.config(text="✅ CORRECT PREDICTION!", fg='#00ff88')
            # Animate correct prediction
            self.animate_correct_prediction()
        else:
            self.result_indicator.config(text="❌ INCORRECT PREDICTION!", fg='#ff4757')
            # Animate incorrect prediction
            self.animate_incorrect_prediction()
        
        # Update counters
        self.correct_counter.config(text=str(self.correct_predictions))
        self.incorrect_counter.config(text=str(self.total_predictions - self.correct_predictions))
        
        # Update live accuracy
        accuracy = (self.correct_predictions / self.total_predictions) * 100
        self.live_accuracy.config(text=f"{accuracy:.1f}%")
        
        # Update live counter in header
        self.live_counter.config(text=f"🎯 Predictions: {self.correct_predictions}/{self.total_predictions} | Accuracy: {accuracy:.1f}%")
        
        # Update visualizations
        self.update_live_visualizations()
        
    def log_prediction_result(self, prediction_data):
        """Log detailed prediction result"""
        sample_num = prediction_data['sample']
        actual_text = "Real" if prediction_data['actual'] == 1 else "Fake"
        predicted_text = "Real" if prediction_data['predicted'] == 1 else "Fake"
        confidence = prediction_data['confidence']
        dataset_info = prediction_data.get('dataset', 'Unknown')
        
        if prediction_data['correct']:
            log_msg = f"✅ Sample #{sample_num:04d} [{dataset_info}]: {predicted_text} (Conf: {confidence:.1%}) - CORRECT!"
            self.add_prediction_log(log_msg, "correct")
        else:
            log_msg = f"❌ Sample #{sample_num:04d} [{dataset_info}]: Pred={predicted_text}, Actual={actual_text} (Conf: {confidence:.1%})"
            self.add_prediction_log(log_msg, "incorrect")
            
    def update_progress_displays(self, progress):
        """Update progress displays"""
        self.operation_label.config(text=f"Processing... {progress:.1f}% Complete")
        self.main_progress['value'] = progress
        self.progress_percent.config(text=f"{progress:.1f}%")
        self.update_enhanced_circular_progress(progress)
        
    def update_enhanced_circular_progress(self, progress):
        """Update enhanced circular progress with glow effect"""
        # Update arc extent
        extent = (progress / 100) * 360
        self.circle_canvas.delete('progress_arc')
        
        # Create glow effect with multiple arcs
        colors = ['#004d7a', '#0066a2', '#007cc7', '#3498db', '#00d4ff']
        widths = [16, 14, 12, 10, 8]
        
        for i, (color, width) in enumerate(zip(colors, widths)):
            self.circle_canvas.create_arc(15, 15, 165, 165, start=90, extent=-extent,
                                         outline=color, width=width, style='arc', 
                                         tags=f'glow_arc_{i}')
        
        # Update center text
        self.circle_canvas.delete('circle_text')
        self.circle_text = self.circle_canvas.create_text(90, 90, text=f"{progress:.0f}%", 
                                                         font=('Arial', 18, 'bold'),
                                                         fill='#00d4ff', tags='circle_text')
        
    def update_live_visualizations(self):
        """Update live visualizations with enhanced graphics"""
        if len(self.prediction_history) < 2:
            return
            
        # Calculate running accuracy
        running_accuracy = []
        correct_count = 0
        
        for i, pred in enumerate(self.prediction_history):
            if pred['correct']:
                correct_count += 1
            accuracy = correct_count / (i + 1)
            running_accuracy.append(accuracy)
        
        # Update accuracy plot
        self.acc_ax.clear()
        sample_numbers = list(range(1, len(running_accuracy) + 1))
        
        # Plot with enhanced styling
        self.acc_ax.plot(sample_numbers, running_accuracy, 'o-', color='#00d4ff', 
                        linewidth=3, markersize=6, alpha=0.8)
        self.acc_ax.fill_between(sample_numbers, running_accuracy, alpha=0.3, color='#00d4ff')
        
        self.acc_ax.set_title('Running Accuracy', fontsize=14, weight='bold', color='white')
        self.acc_ax.set_xlabel('Prediction #', color='white')
        self.acc_ax.set_ylabel('Accuracy', color='white')
        self.acc_ax.grid(True, alpha=0.3)
        self.acc_ax.set_facecolor('#1a1a1a')
        self.acc_ax.set_ylim(0, 1)
        
        # Update confidence plot
        self.conf_ax.clear()
        confidences = [pred['confidence'] for pred in self.prediction_history]
        colors = ['#27ae60' if pred['correct'] else '#e74c3c' for pred in self.prediction_history]
        
        bars = self.conf_ax.bar(sample_numbers, confidences, color=colors, alpha=0.7, width=0.8)
        self.conf_ax.axhline(y=0.5, color='#f39c12', linestyle='--', linewidth=2, alpha=0.8)
        
        self.conf_ax.set_title('Prediction Confidence', fontsize=14, weight='bold', color='white')
        self.conf_ax.set_xlabel('Prediction #', color='white')
        self.conf_ax.set_ylabel('Confidence', color='white')
        self.conf_ax.grid(True, alpha=0.3)
        self.conf_ax.set_facecolor('#1a1a1a')
        self.conf_ax.set_ylim(0, 1)
        
        plt.tight_layout()
        self.live_canvas.draw()
        
    def animate_correct_prediction(self):
        """Animate correct prediction with green flash"""
        original_bg = self.result_indicator.cget('bg')
        self.result_indicator.config(bg='#27ae60')
        self.root.after(200, lambda: self.result_indicator.config(bg=original_bg))
        
    def animate_incorrect_prediction(self):
        """Animate incorrect prediction with red flash"""
        original_bg = self.result_indicator.cget('bg')
        self.result_indicator.config(bg='#e74c3c')
        self.root.after(200, lambda: self.result_indicator.config(bg=original_bg))
        
    def prediction_session_complete(self):
        """Handle prediction session completion with detailed summary"""
        self.is_processing = False
        self.main_status.config(text="🏁 PREDICTION SESSION COMPLETE", fg='#00ff88')
        self.predict_btn.config(state='normal', bg='#00d4ff')
        
        # Calculate final statistics
        total = self.total_predictions
        correct = self.correct_predictions
        incorrect = total - correct
        accuracy = (correct / total * 100) if total > 0 else 0
        
        # Add comprehensive summary to log
        self.add_prediction_log("=" * 60, "info")
        self.add_prediction_log("🏁 PREDICTION SESSION SUMMARY", "info")
        self.add_prediction_log("=" * 60, "info")
        self.add_prediction_log(f"📊 Total Samples Processed: {total}", "info")
        self.add_prediction_log(f"✅ Correct Predictions: {correct}", "correct")
        self.add_prediction_log(f"❌ Incorrect Predictions: {incorrect}", "incorrect")
        self.add_prediction_log(f"🎯 Final Accuracy: {accuracy:.2f}%", "info")
        
        # Accuracy assessment
        if accuracy >= 90:
            self.add_prediction_log("🏆 EXCELLENT PERFORMANCE!", "correct")
        elif accuracy >= 80:
            self.add_prediction_log("✅ Good Performance!", "correct")
        elif accuracy >= 70:
            self.add_prediction_log("⚠️ Moderate Performance", "warning")
        else:
            self.add_prediction_log("⚠️ Performance Needs Improvement", "warning")
        
        # Model-specific insights
        model_name = self.selected_model.get()
        self.add_prediction_log(f"🤖 Model Used: {model_name}", "info")
        
        # Confidence analysis
        if self.prediction_history:
            avg_confidence = np.mean([pred['confidence'] for pred in self.prediction_history])
            self.add_prediction_log(f"📊 Average Confidence: {avg_confidence:.1%}", "info")
            
        self.add_prediction_log("=" * 60, "info")
        
        # Update results tab
        self.update_results_tab()
        
        # Show summary popup
        self.show_session_summary()
        
    def show_session_summary(self):
        """Show detailed session summary popup"""
        total = self.total_predictions
        correct = self.correct_predictions
        accuracy = (correct / total * 100) if total > 0 else 0
        
        summary_text = f"""
🎵 AUDIO CLASSIFICATION SESSION COMPLETE 🎵

📊 FINAL RESULTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total Samples: {total}
Correct Predictions: {correct}
Incorrect Predictions: {total - correct}
Final Accuracy: {accuracy:.2f}%

🤖 Model: {self.selected_model.get()}
📁 Dataset: {self.selected_dataset.get()}

{('🏆 EXCELLENT PERFORMANCE!' if accuracy >= 90 else
  '✅ Good Performance!' if accuracy >= 80 else
  '⚠️ Moderate Performance' if accuracy >= 70 else
  '⚠️ Needs Improvement')}
        """
        
        messagebox.showinfo("Session Complete", summary_text)
        
    def add_prediction_log(self, message, tag_type="info"):
        """Add message to prediction log with enhanced formatting"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        self.prediction_feed.insert(tk.END, formatted_message)
        
        # Apply color coding
        start_line = float(self.prediction_feed.index(tk.END)) - 1.0
        self.prediction_feed.tag_add(tag_type, f"{start_line:.1f}", tk.END)
        
        # Auto-scroll to bottom
        self.prediction_feed.see(tk.END)
        
    def stop_simulation(self):
        """Stop current simulation"""
        self.is_processing = False
        self.main_status.config(text="🔴 SIMULATION STOPPED", fg='#ff4757')
        self.predict_btn.config(state='normal', bg='#00d4ff')
        self.add_prediction_log("🛑 Prediction session stopped by user", "warning")
        
    def reset_simulation(self):
        """Reset simulation state"""
        self.is_processing = False
        self.correct_predictions = 0
        self.total_predictions = 0
        self.recent_predictions = []
        self.prediction_history = []
        self.operation_progress = 0
        
        # Reset displays
        self.main_status.config(text="🟢 SYSTEM READY", fg='#00ff88')
        self.live_counter.config(text="🎯 Predictions: 0/0 | Accuracy: 0%")
        self.current_sample_display.config(text="🎵 Analyzing: [No sample loaded]")
        self.prediction_display.config(text="[Waiting...]")
        self.confidence_display.config(text="Confidence: 0%")
        self.actual_display.config(text="[Waiting...]")
        self.result_indicator.config(text="⏳ Ready to start prediction...", fg='#f39c12')
        
        # Reset counters
        self.correct_counter.config(text="0")
        self.incorrect_counter.config(text="0")
        self.live_accuracy.config(text="0.0%")
        
        # Reset progress
        self.operation_label.config(text="Ready to start operations...")
        self.main_progress['value'] = 0
        self.progress_percent.config(text="0%")
        self.update_enhanced_circular_progress(0)
        
        # Clear visualizations
        self.acc_ax.clear()
        self.conf_ax.clear()
        self.live_canvas.draw()
        
        # Re-enable controls
        self.predict_btn.config(state='normal', bg='#00d4ff')
        
        # Clear and reset log
        self.prediction_feed.delete(1.0, tk.END)
        self.add_prediction_log("🔄 System Reset Complete", "info")
        self.add_prediction_log("📊 Ready for new prediction session!", "info")
    
    def create_training_tab(self):
        """Create training tab for model training with SCROLLABLE area"""
        train_frame = tk.Frame(self.main_notebook, bg='#2c3e50')
        self.main_notebook.add(train_frame, text="🎓 Model Training")
        
        # Header (fixed at top)
        header = tk.Frame(train_frame, bg='#0f3460', height=80, relief='raised', bd=3)
        header.pack(fill='x', padx=10, pady=10)
        header.pack_propagate(False)
        
        tk.Label(header, text="🎓 MODEL TRAINING CENTER", 
                font=('Arial', 20, 'bold'), fg='#00d4ff', bg='#0f3460').pack(pady=5)
        self.training_status = tk.Label(header, text="⏸️ READY TO TRAIN", 
                                       font=('Arial', 14, 'bold'), fg='#f39c12', bg='#0f3460')
        self.training_status.pack()
        
        # Create SCROLLABLE container for all training content
        scroll_container = tk.Frame(train_frame, bg='#2c3e50')
        scroll_container.pack(fill='both', expand=True)
        
        train_canvas = tk.Canvas(scroll_container, bg='#2c3e50', highlightthickness=0)
        train_scrollbar = ttk.Scrollbar(scroll_container, orient="vertical", command=train_canvas.yview)
        self.training_content = tk.Frame(train_canvas, bg='#2c3e50')
        
        self.training_content.bind(
            "<Configure>",
            lambda e: train_canvas.configure(scrollregion=train_canvas.bbox("all"))
        )
        
        train_canvas.create_window((0, 0), window=self.training_content, anchor="nw")
        train_canvas.configure(yscrollcommand=train_scrollbar.set)
        
        train_canvas.pack(side="left", fill="both", expand=True)
        train_scrollbar.pack(side="right", fill="y")
        
        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            train_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        train_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Training configuration (now inside scrollable area)
        config_frame = tk.LabelFrame(self.training_content, text="⚙️ Training Configuration", 
                                    font=('Arial', 16, 'bold'), bg='#34495e', fg='white',
                                    padx=20, pady=20)
        config_frame.pack(fill='x', padx=10, pady=10)
        
        # Model type selection
        model_type_frame = tk.Frame(config_frame, bg='#34495e')
        model_type_frame.pack(fill='x', pady=10)
        
        tk.Label(model_type_frame, text="🤖 Model Type to Train:", 
                font=('Arial', 14, 'bold'), bg='#34495e', fg='white').pack(anchor='w', pady=5)
        
        self.train_model_type = tk.StringVar(value="RandomForest")
        model_type_combo = ttk.Combobox(model_type_frame, textvariable=self.train_model_type,
                                       values=list(self.models.keys()) if hasattr(self, 'models') else [],
                                       state='readonly', width=40, font=('Arial', 12))
        model_type_combo.pack(fill='x', pady=5)
        
        # Training dataset selection
        train_dataset_frame = tk.Frame(config_frame, bg='#34495e')
        train_dataset_frame.pack(fill='x', pady=10)
        
        tk.Label(train_dataset_frame, text="📊 Training Dataset:", 
                font=('Arial', 14, 'bold'), bg='#34495e', fg='white').pack(anchor='w', pady=5)
        
        self.train_dataset_var = tk.StringVar(value="safe_5000_train")
        train_combo = ttk.Combobox(train_dataset_frame, textvariable=self.train_dataset_var,
                                  values=list(self.available_datasets.keys()) if hasattr(self, 'available_datasets') else [],
                                  state='readonly', width=40, font=('Arial', 12))
        train_combo.pack(fill='x', pady=5)
        
        # Validation dataset selection
        val_dataset_frame = tk.Frame(config_frame, bg='#34495e')
        val_dataset_frame.pack(fill='x', pady=10)
        
        tk.Label(val_dataset_frame, text="📊 Validation Dataset:", 
                font=('Arial', 14, 'bold'), bg='#34495e', fg='white').pack(anchor='w', pady=5)
        
        self.val_dataset_var = tk.StringVar(value="safe_5000_val")
        val_combo = ttk.Combobox(val_dataset_frame, textvariable=self.val_dataset_var,
                                values=list(self.available_datasets.keys()) if hasattr(self, 'available_datasets') else [],
                                state='readonly', width=40, font=('Arial', 12))
        val_combo.pack(fill='x', pady=5)
        
        # ==================== HYPERPARAMETER CONFIGURATION ====================
        # Create container for all model-specific parameters
        hyperparams_container = tk.LabelFrame(config_frame, text="🎛️ Hyperparameter Tuning (Anti-Overfitting)", 
                                             font=('Arial', 14, 'bold'), bg='#2c3e50', fg='#00d4ff',
                                             padx=15, pady=15, relief='raised', bd=3)
        hyperparams_container.pack(fill='x', pady=15)
        
        # Add info label
        info_label = tk.Label(hyperparams_container, 
                             text="⚠️ Adjust these parameters to reduce overfitting. Lower values = Less overfitting!", 
                             font=('Arial', 11, 'bold'), bg='#2c3e50', fg='#f39c12', wraplength=700)
        info_label.pack(pady=5)
        
        # ========== RANDOM FOREST PARAMETERS ==========
        rf_frame = tk.LabelFrame(hyperparams_container, text="🌳 Random Forest Parameters", 
                                font=('Arial', 12, 'bold'), bg='#34495e', fg='white',
                                padx=15, pady=10)
        rf_frame.pack(fill='x', pady=10)
        
        # N_estimators
        rf_row1 = tk.Frame(rf_frame, bg='#34495e')
        rf_row1.pack(fill='x', pady=5)
        tk.Label(rf_row1, text="🌲 Number of Trees:", 
                font=('Arial', 11, 'bold'), bg='#34495e', fg='white', width=25, anchor='w').pack(side='left')
        self.rf_n_estimators_var = tk.StringVar(value="100")
        ttk.Combobox(rf_row1, textvariable=self.rf_n_estimators_var,
                    values=["50", "100", "150", "200", "300"],
                    width=15, font=('Arial', 10)).pack(side='left', padx=5)
        tk.Label(rf_row1, text="More trees = Better but slower", 
                font=('Arial', 9), bg='#34495e', fg='#95a5a6').pack(side='left', padx=5)
        
        # Max depth (CRITICAL for overfitting!)
        rf_row2 = tk.Frame(rf_frame, bg='#34495e')
        rf_row2.pack(fill='x', pady=5)
        tk.Label(rf_row2, text="📏 Max Depth:", 
                font=('Arial', 11, 'bold'), bg='#34495e', fg='#e74c3c', width=25, anchor='w').pack(side='left')
        self.rf_max_depth_var = tk.StringVar(value="10")
        ttk.Combobox(rf_row2, textvariable=self.rf_max_depth_var,
                    values=["5", "8", "10", "15", "20", "None"],
                    width=15, font=('Arial', 10)).pack(side='left', padx=5)
        tk.Label(rf_row2, text="⚠️ LOWER = Less overfitting! Try 5-10", 
                font=('Arial', 9, 'bold'), bg='#34495e', fg='#e74c3c').pack(side='left', padx=5)
        
        # Min samples split (CRITICAL!)
        rf_row3 = tk.Frame(rf_frame, bg='#34495e')
        rf_row3.pack(fill='x', pady=5)
        tk.Label(rf_row3, text="🔀 Min Samples Split:", 
                font=('Arial', 11, 'bold'), bg='#34495e', fg='#e74c3c', width=25, anchor='w').pack(side='left')
        self.rf_min_samples_split_var = tk.StringVar(value="20")
        ttk.Combobox(rf_row3, textvariable=self.rf_min_samples_split_var,
                    values=["2", "10", "20", "50", "100"],
                    width=15, font=('Arial', 10)).pack(side='left', padx=5)
        tk.Label(rf_row3, text="⚠️ HIGHER = Less overfitting! Try 20-50", 
                font=('Arial', 9, 'bold'), bg='#34495e', fg='#e74c3c').pack(side='left', padx=5)
        
        # Min samples leaf (CRITICAL!)
        rf_row4 = tk.Frame(rf_frame, bg='#34495e')
        rf_row4.pack(fill='x', pady=5)
        tk.Label(rf_row4, text="🍃 Min Samples Leaf:", 
                font=('Arial', 11, 'bold'), bg='#34495e', fg='#e74c3c', width=25, anchor='w').pack(side='left')
        self.rf_min_samples_leaf_var = tk.StringVar(value="10")
        ttk.Combobox(rf_row4, textvariable=self.rf_min_samples_leaf_var,
                    values=["1", "5", "10", "25", "50"],
                    width=15, font=('Arial', 10)).pack(side='left', padx=5)
        tk.Label(rf_row4, text="⚠️ HIGHER = Less overfitting! Try 10-25", 
                font=('Arial', 9, 'bold'), bg='#34495e', fg='#e74c3c').pack(side='left', padx=5)
        
        # ========== KNN PARAMETERS ==========
        knn_frame = tk.LabelFrame(hyperparams_container, text="� KNN Parameters", 
                                 font=('Arial', 12, 'bold'), bg='#34495e', fg='white',
                                 padx=15, pady=10)
        knn_frame.pack(fill='x', pady=10)
        
        # N neighbors
        knn_row1 = tk.Frame(knn_frame, bg='#34495e')
        knn_row1.pack(fill='x', pady=5)
        tk.Label(knn_row1, text="👥 Number of Neighbors:", 
                font=('Arial', 11, 'bold'), bg='#34495e', fg='#e74c3c', width=25, anchor='w').pack(side='left')
        self.knn_n_neighbors_var = tk.StringVar(value="15")
        ttk.Combobox(knn_row1, textvariable=self.knn_n_neighbors_var,
                    values=["5", "10", "15", "20", "25", "30"],
                    width=15, font=('Arial', 10)).pack(side='left', padx=5)
        tk.Label(knn_row1, text="⚠️ HIGHER = Less overfitting! Try 15-25", 
                font=('Arial', 9, 'bold'), bg='#34495e', fg='#e74c3c').pack(side='left', padx=5)
        
        # Weights
        knn_row2 = tk.Frame(knn_frame, bg='#34495e')
        knn_row2.pack(fill='x', pady=5)
        tk.Label(knn_row2, text="⚖️ Weights:", 
                font=('Arial', 11, 'bold'), bg='#34495e', fg='white', width=25, anchor='w').pack(side='left')
        self.knn_weights_var = tk.StringVar(value="uniform")
        ttk.Combobox(knn_row2, textvariable=self.knn_weights_var,
                    values=["uniform", "distance"],
                    width=15, font=('Arial', 10)).pack(side='left', padx=5)
        tk.Label(knn_row2, text="uniform = Less overfitting", 
                font=('Arial', 9), bg='#34495e', fg='#95a5a6').pack(side='left', padx=5)
        
        # ========== SVM PARAMETERS ==========
        svm_frame = tk.LabelFrame(hyperparams_container, text="🎯 SVM Parameters", 
                                 font=('Arial', 12, 'bold'), bg='#34495e', fg='white',
                                 padx=15, pady=10)
        svm_frame.pack(fill='x', pady=10)
        
        # Kernel
        svm_row1 = tk.Frame(svm_frame, bg='#34495e')
        svm_row1.pack(fill='x', pady=5)
        tk.Label(svm_row1, text="🔮 Kernel:", 
                font=('Arial', 11, 'bold'), bg='#34495e', fg='white', width=25, anchor='w').pack(side='left')
        self.svm_kernel_var = tk.StringVar(value="linear")  # Changed to linear for less overfitting
        ttk.Combobox(svm_row1, textvariable=self.svm_kernel_var,
                    values=["linear", "rbf", "poly"],
                    width=15, font=('Arial', 10)).pack(side='left', padx=5)
        tk.Label(svm_row1, text="✅ linear = BEST for small datasets!", 
                font=('Arial', 9, 'bold'), bg='#34495e', fg='#2ecc71').pack(side='left', padx=5)
        
        # C (Regularization)
        svm_row2 = tk.Frame(svm_frame, bg='#34495e')
        svm_row2.pack(fill='x', pady=5)
        tk.Label(svm_row2, text="⚖️ Regularization (C):", 
                font=('Arial', 11, 'bold'), bg='#34495e', fg='#e74c3c', width=25, anchor='w').pack(side='left')
        self.svm_C_var = tk.StringVar(value="0.1")  # Changed to 0.1 for strong regularization
        ttk.Combobox(svm_row2, textvariable=self.svm_C_var,
                    values=["0.01", "0.05", "0.1", "0.5", "1.0", "5.0", "10.0"],
                    width=15, font=('Arial', 10)).pack(side='left', padx=5)
        tk.Label(svm_row2, text="❗ DEFAULT: 0.1 (Strong regularization)", 
                font=('Arial', 9, 'bold'), bg='#34495e', fg='#2ecc71').pack(side='left', padx=5)
        
        # ========== XGBOOST PARAMETERS ==========
        xgb_frame = tk.LabelFrame(hyperparams_container, text="🚀 XGBoost Parameters", 
                                 font=('Arial', 12, 'bold'), bg='#34495e', fg='white',
                                 padx=15, pady=10)
        xgb_frame.pack(fill='x', pady=10)
        
        # N estimators
        xgb_row1 = tk.Frame(xgb_frame, bg='#34495e')
        xgb_row1.pack(fill='x', pady=5)
        tk.Label(xgb_row1, text="🌲 Number of Estimators:", 
                font=('Arial', 11, 'bold'), bg='#34495e', fg='white', width=25, anchor='w').pack(side='left')
        self.xgb_n_estimators_var = tk.StringVar(value="100")
        ttk.Combobox(xgb_row1, textvariable=self.xgb_n_estimators_var,
                    values=["50", "100", "150", "200"],
                    width=15, font=('Arial', 10)).pack(side='left', padx=5)
        tk.Label(xgb_row1, text="More = Better but slower", 
                font=('Arial', 9), bg='#34495e', fg='#95a5a6').pack(side='left', padx=5)
        
        # Max depth
        xgb_row2 = tk.Frame(xgb_frame, bg='#34495e')
        xgb_row2.pack(fill='x', pady=5)
        tk.Label(xgb_row2, text="📏 Max Depth:", 
                font=('Arial', 11, 'bold'), bg='#34495e', fg='#e74c3c', width=25, anchor='w').pack(side='left')
        self.xgb_max_depth_var = tk.StringVar(value="6")
        ttk.Combobox(xgb_row2, textvariable=self.xgb_max_depth_var,
                    values=["3", "4", "5", "6", "8", "10"],
                    width=15, font=('Arial', 10)).pack(side='left', padx=5)
        tk.Label(xgb_row2, text="⚠️ LOWER = Less overfitting! Try 3-6", 
                font=('Arial', 9, 'bold'), bg='#34495e', fg='#e74c3c').pack(side='left', padx=5)
        
        # Learning rate
        xgb_row3 = tk.Frame(xgb_frame, bg='#34495e')
        xgb_row3.pack(fill='x', pady=5)
        tk.Label(xgb_row3, text="📈 Learning Rate:", 
                font=('Arial', 11, 'bold'), bg='#34495e', fg='#e74c3c', width=25, anchor='w').pack(side='left')
        self.xgb_learning_rate_var = tk.StringVar(value="0.1")
        ttk.Combobox(xgb_row3, textvariable=self.xgb_learning_rate_var,
                    values=["0.01", "0.05", "0.1", "0.2", "0.3"],
                    width=15, font=('Arial', 10)).pack(side='left', padx=5)
        tk.Label(xgb_row3, text="⚠️ LOWER = Less overfitting! Try 0.01-0.1", 
                font=('Arial', 9, 'bold'), bg='#34495e', fg='#e74c3c').pack(side='left', padx=5)
        
        # Training controls
        control_frame = tk.Frame(config_frame, bg='#34495e')
        control_frame.pack(fill='x', pady=20)
        
        self.train_btn = tk.Button(control_frame, text="🚀 START TRAINING", 
                                   command=self.start_training,
                                   bg='#27ae60', fg='white', font=('Arial', 16, 'bold'),
                                   height=2, width=25, relief='raised', bd=5)
        self.train_btn.pack(side='left', padx=10)
        
        self.save_model_btn = tk.Button(control_frame, text="💾 SAVE MODEL", 
                                        command=self.save_trained_model,
                                        bg='#3498db', fg='white', font=('Arial', 16, 'bold'),
                                        height=2, width=25, relief='raised', bd=5,
                                        state='disabled')
        self.save_model_btn.pack(side='left', padx=10)
        
        # Training progress (now in scrollable area)
        progress_frame = tk.LabelFrame(self.training_content, text="📊 Training Progress", 
                                      font=('Arial', 16, 'bold'), bg='#34495e', fg='white',
                                      padx=20, pady=20)
        progress_frame.pack(fill='x', padx=10, pady=10)
        
        self.training_progress_label = tk.Label(progress_frame, text="Waiting to start training...", 
                                               font=('Arial', 14, 'bold'), bg='#34495e', fg='#ecf0f1')
        self.training_progress_label.pack(pady=10)
        
        self.training_progress_bar = ttk.Progressbar(progress_frame, length=700, mode='indeterminate',
                                                     style="Custom.Horizontal.TProgressbar")
        self.training_progress_bar.pack(fill='x', pady=10)
        
        # Training log with LARGER SQUARE-SHAPED AREA (now in scrollable area)
        log_frame = tk.LabelFrame(self.training_content, text="📝 Training Log", 
                                 font=('Arial', 18, 'bold'), bg='#34495e', fg='white',
                                 padx=20, pady=20)
        log_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # SQUARE-SHAPED log display (30 lines x 85 width for square shape)
        self.training_log = scrolledtext.ScrolledText(log_frame, height=30, width=85,
                                                      bg='#1a1a1a', fg='#ecf0f1',
                                                      font=('Consolas', 15, 'bold'),  # INCREASED from 12 to 15
                                                      wrap=tk.WORD, insertbackground='white')
        self.training_log.pack(fill='both', expand=True, pady=10)
        
        # Configure text tags with LARGER FONT
        self.training_log.tag_configure("success", foreground="#27ae60", font=('Consolas', 15, 'bold'))
        self.training_log.tag_configure("error", foreground="#e74c3c", font=('Consolas', 15, 'bold'))
        self.training_log.tag_configure("info", foreground="#3498db", font=('Consolas', 15, 'bold'))
        self.training_log.tag_configure("warning", foreground="#f39c12", font=('Consolas', 15, 'bold'))
        
        self.add_training_log("🎓 Training Center Initialized", "info")
        self.add_training_log("📊 Configure parameters and click 'START TRAINING' to begin", "info")
        
        # Store trained model
        self.trained_model = None
    
    def add_training_log(self, message, tag_type="info"):
        """Add message to training log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}\n"
        
        self.training_log.insert(tk.END, full_message, tag_type)
        self.training_log.see(tk.END)
        self.training_log.update_idletasks()
    
    def start_training(self):
        """Start model training in a separate thread"""
        if self.is_processing:
            messagebox.showwarning("Training In Progress", "A training session is already running!")
            return
        
        self.is_processing = True
        self.train_btn.config(state='disabled', bg='#95a5a6')
        self.training_status.config(text="🔥 TRAINING IN PROGRESS", fg='#27ae60')
        
        self.training_log.delete(1.0, tk.END)
        self.add_training_log("🚀 Starting Training Session...", "info")
        
        # Start training thread
        threading.Thread(target=self._training_thread, daemon=True).start()
    
    def _training_thread(self):
        """Training thread"""
        try:
            # Get selected datasets
            train_key = self.train_dataset_var.get()
            val_key = self.val_dataset_var.get()
            
            # Parse dataset keys
            train_parts = train_key.rsplit('_', 1)
            val_parts = val_key.rsplit('_', 1)
            
            if len(train_parts) != 2 or len(val_parts) != 2:
                self.root.after(0, lambda: self.add_training_log("❌ Invalid dataset selection!", "error"))
                self.is_processing = False
                return
            
            train_dataset_name = train_parts[0]
            val_dataset_name = val_parts[0]
            
            # Get datasets
            train_dataset = self.datasets[train_dataset_name]['train']
            val_dataset = self.datasets[val_dataset_name]['val']
            
            if train_dataset is None or val_dataset is None:
                self.root.after(0, lambda: self.add_training_log("❌ Dataset not found!", "error"))
                self.is_processing = False
                return
            
            self.root.after(0, lambda: self.add_training_log(f"✅ Training dataset: {train_dataset_name}", "success"))
            self.root.after(0, lambda: self.add_training_log(f"✅ Validation dataset: {val_dataset_name}", "success"))
            
            # Detect and display feature type
            is_mfcc = train_dataset_name.startswith('mfcc_')
            feature_type = "MFCC (40 features)" if is_mfcc else "SPECTROGRAM (~16k features)"
            self.root.after(0, lambda ft=feature_type: self.add_training_log(f"🔍 Feature Type: {ft}", "info"))
            self.root.after(0, lambda: self.add_training_log(f"💡 Remember: Use same feature type for prediction!", "warning"))
            
            # Start progress bar
            self.root.after(0, lambda: self.training_progress_bar.start(10))
            
            # Get selected model type
            model_type = self.train_model_type.get()
            
            # Create model based on type with CONFIGURED HYPERPARAMETERS
            if model_type == 'RandomForest':
                # Get Random Forest hyperparameters
                rf_n_est = int(self.rf_n_estimators_var.get())
                rf_max_depth_str = self.rf_max_depth_var.get()
                rf_max_depth = None if rf_max_depth_str == "None" else int(rf_max_depth_str)
                rf_min_split = int(self.rf_min_samples_split_var.get())
                rf_min_leaf = int(self.rf_min_samples_leaf_var.get())
                
                self.root.after(0, lambda: self.add_training_log(f"🌲 Creating Random Forest with Anti-Overfitting Settings:", "info"))
                self.root.after(0, lambda: self.add_training_log(f"   • n_estimators={rf_n_est}", "info"))
                self.root.after(0, lambda: self.add_training_log(f"   • max_depth={rf_max_depth}", "warning"))
                self.root.after(0, lambda: self.add_training_log(f"   • min_samples_split={rf_min_split}", "warning"))
                self.root.after(0, lambda: self.add_training_log(f"   • min_samples_leaf={rf_min_leaf}", "warning"))
                
                from Models.RandomForrest import RandomForestFullFeatures
                model = RandomForestFullFeatures(
                    n_estimators=rf_n_est,
                    max_depth=rf_max_depth,
                    min_samples_split=rf_min_split,
                    min_samples_leaf=rf_min_leaf,
                    random_state=42
                )
            elif model_type == 'KNN':
                # Get KNN hyperparameters
                knn_neighbors = int(self.knn_n_neighbors_var.get())
                knn_weights = self.knn_weights_var.get()
                
                self.root.after(0, lambda: self.add_training_log(f"🔍 Creating KNN with Anti-Overfitting Settings:", "info"))
                self.root.after(0, lambda: self.add_training_log(f"   • n_neighbors={knn_neighbors}", "warning"))
                self.root.after(0, lambda: self.add_training_log(f"   • weights={knn_weights}", "info"))
                
                from Models.KNN import KNNAudioClassifier
                model = KNNAudioClassifier(
                    n_neighbors=knn_neighbors, 
                    weights=knn_weights, 
                    algorithm='auto'
                )
            elif model_type == 'SVM':
                # Get SVM hyperparameters
                svm_kernel = self.svm_kernel_var.get()
                svm_C = float(self.svm_C_var.get())
                
                self.root.after(0, lambda: self.add_training_log(f"⚡ Creating SVM with Anti-Overfitting Settings:", "info"))
                self.root.after(0, lambda: self.add_training_log(f"   • kernel={svm_kernel}", "info"))
                self.root.after(0, lambda: self.add_training_log(f"   • C={svm_C}", "warning"))
                self.root.after(0, lambda: self.add_training_log(f"   • class_weight=balanced (auto)", "info"))
                
                # Warn about overfitting risk
                if svm_C >= 1.0 or svm_kernel != 'linear':
                    self.root.after(0, lambda: self.add_training_log(f"⚠️  WARNING: High overfitting risk detected!", "error"))
                    if svm_C >= 1.0:
                        self.root.after(0, lambda: self.add_training_log(f"   • C={svm_C} is high - try C=0.1 for better generalization", "error"))
                    if svm_kernel != 'linear':
                        self.root.after(0, lambda: self.add_training_log(f"   • {svm_kernel} kernel is complex - try 'linear' for small datasets", "error"))
                else:
                    self.root.after(0, lambda: self.add_training_log(f"✅ Good anti-overfitting configuration!", "success"))
                
                from Models.SVM import SVMAudioClassifier
                model = SVMAudioClassifier(
                    kernel=svm_kernel, 
                    C=svm_C, 
                    gamma='scale'
                )
            elif model_type == 'XGBoost':
                # Get XGBoost hyperparameters
                xgb_n_est = int(self.xgb_n_estimators_var.get())
                xgb_max_depth = int(self.xgb_max_depth_var.get())
                xgb_lr = float(self.xgb_learning_rate_var.get())
                
                self.root.after(0, lambda: self.add_training_log(f"🚀 Creating XGBoost with Anti-Overfitting Settings:", "info"))
                self.root.after(0, lambda: self.add_training_log(f"   • n_estimators={xgb_n_est}", "info"))
                self.root.after(0, lambda: self.add_training_log(f"   • max_depth={xgb_max_depth}", "warning"))
                self.root.after(0, lambda: self.add_training_log(f"   • learning_rate={xgb_lr}", "warning"))
                
                from Models.XgBoost import XGBoostAudioClassifier
                model = XGBoostAudioClassifier(
                    n_estimators=xgb_n_est,
                    max_depth=xgb_max_depth,
                    learning_rate=xgb_lr
                )
            else:
                self.root.after(0, lambda: self.add_training_log(f"❌ Unknown model type: {model_type}", "error"))
                self.is_processing = False
                return
            
            self.root.after(0, lambda: self.add_training_log("✅ Model created successfully", "success"))
            self.root.after(0, lambda: self.add_training_log("🔧 Compiling model...", "info"))
            
            model.compile()
            
            self.root.after(0, lambda: self.add_training_log("✅ Model compiled", "success"))
            self.root.after(0, lambda: self.add_training_log("🎓 Starting training process...", "info"))
            self.root.after(0, lambda: self.training_progress_label.config(text="Training model... This may take several minutes..."))
            
            # Initialize resource monitoring
            process = psutil.Process(os.getpid())
            start_time = time.time()
            start_memory = process.memory_info().rss / (1024 * 1024)  # MB
            
            self.root.after(0, lambda: self.add_training_log("📊 Resource Monitoring Started", "info"))
            self.root.after(0, lambda: self.add_training_log(f"   • Initial Memory: {start_memory:.2f} MB", "info"))
            
            # Train the model
            history = model.fit(train_dataset, validation_data=val_dataset)
            
            # Calculate resource usage
            end_time = time.time()
            end_memory = process.memory_info().rss / (1024 * 1024)  # MB
            training_duration = end_time - start_time
            memory_used = end_memory - start_memory
            cpu_percent = process.cpu_percent(interval=0.1)
            
            # Log resource metrics
            self.root.after(0, lambda: self.add_training_log("📊 Resource Usage Summary:", "success"))
            self.root.after(0, lambda: self.add_training_log(f"   ⏱️  Training Duration: {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)", "info"))
            self.root.after(0, lambda: self.add_training_log(f"   🧠 Memory Change: {memory_used:+.2f} MB (Peak: {end_memory:.2f} MB)", "info"))
            self.root.after(0, lambda: self.add_training_log(f"   ⚡ CPU Usage: {cpu_percent:.1f}%", "info"))
            
            # Stop progress bar
            self.root.after(0, lambda: self.training_progress_bar.stop())
            self.root.after(0, lambda: self.training_progress_bar.config(value=100))
            
            # Store the trained model
            self.trained_model = model
            
            # Extract metrics
            train_acc = history.history['binary_accuracy'][0]
            val_acc = history.history['val_binary_accuracy'][0] if history.history['val_binary_accuracy'][0] else 0
            
            self.root.after(0, lambda: self.add_training_log("=" * 60, "info"))
            self.root.after(0, lambda: self.add_training_log("🎉 TRAINING COMPLETE!", "success"))
            self.root.after(0, lambda: self.add_training_log("=" * 60, "info"))
            self.root.after(0, lambda: self.add_training_log(f"📊 Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)", "success"))
            self.root.after(0, lambda: self.add_training_log(f"📊 Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)", "success"))
            self.root.after(0, lambda: self.add_training_log("💾 You can now save the model!", "info"))
            self.root.after(0, lambda: self.add_training_log("=" * 60, "info"))
            
            # Enable save button and update status
            self.root.after(0, lambda: self.save_model_btn.config(state='normal'))
            self.root.after(0, lambda: self.training_status.config(text="✅ TRAINING COMPLETE", fg='#00ff88'))
            self.root.after(0, lambda: self.training_progress_label.config(text=f"Training Complete! Accuracy: {val_acc*100:.2f}%"))
            
            # Update the main model reference
            self.model = model
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Training Error: {e}\n{traceback.format_exc()}"
            self.root.after(0, lambda: self.add_training_log(error_msg, "error"))
            self.root.after(0, lambda: self.training_progress_bar.stop())
        
        finally:
            self.is_processing = False
            self.root.after(0, lambda: self.train_btn.config(state='normal', bg='#27ae60'))
    
    def save_trained_model(self):
        """Save the trained model"""
        if self.trained_model is None:
            messagebox.showwarning("No Model", "No trained model to save!")
            return
        
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Get model type for filename
            model_type = self.train_model_type.get().lower()
            filename = f"{model_type}_model_trained_{timestamp}.joblib"
            
            self.add_training_log(f"💾 Saving {model_type.upper()} model to {filename}...", "info")
            
            filepath = self.trained_model.save_model(filename)
            
            self.add_training_log(f"✅ Model saved successfully to {filepath}!", "success")
            messagebox.showinfo("Success", f"Model saved successfully!\n\nFile: {filepath}")
            
            # Update the models dictionary with newly trained model
            self.models[self.train_model_type.get()] = self.trained_model
            
            # Update dropdown to show trained status
            self.populate_dropdowns_from_data()
            
        except Exception as e:
            self.add_training_log(f"❌ Error saving model: {e}", "error")
            messagebox.showerror("Error", f"Failed to save model:\n{e}")
        
    def create_monitoring_tab(self):
        """Create enhanced monitoring tab with SCROLLABLE container"""
        monitor_frame = tk.Frame(self.main_notebook, bg='#2c3e50')
        self.main_notebook.add(monitor_frame, text="📈 System Monitor")
        
        # Create SCROLLABLE container
        monitor_canvas = tk.Canvas(monitor_frame, bg='#2c3e50', highlightthickness=0)
        monitor_scrollbar = ttk.Scrollbar(monitor_frame, orient="vertical", command=monitor_canvas.yview)
        self.monitor_content = tk.Frame(monitor_canvas, bg='#2c3e50')
        
        self.monitor_content.bind(
            "<Configure>",
            lambda e: monitor_canvas.configure(scrollregion=monitor_canvas.bbox("all"))
        )
        
        monitor_canvas.create_window((0, 0), window=self.monitor_content, anchor="nw")
        monitor_canvas.configure(yscrollcommand=monitor_scrollbar.set)
        
        monitor_canvas.pack(side="left", fill="both", expand=True)
        monitor_scrollbar.pack(side="right", fill="y")
        
        # Enable mouse wheel scrolling
        def _on_monitor_mousewheel(event):
            monitor_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        monitor_canvas.bind_all("<MouseWheel>", _on_monitor_mousewheel)
        
        # Add enhanced monitoring features (simplified for space)
        self.create_system_resources_monitor(self.monitor_content)
        
    def create_system_resources_monitor(self, parent):
        """Create system resources monitor"""
        resource_frame = tk.LabelFrame(parent, text="💻 System Resources", 
                                      font=('Arial', 16, 'bold'), bg='#34495e', fg='white',
                                      padx=20, pady=20)
        resource_frame.pack(fill='x', padx=10, pady=10)
        
        # CPU usage
        tk.Label(resource_frame, text="CPU Usage:", font=('Arial', 14, 'bold'), 
                bg='#34495e', fg='white').pack(anchor='w')
        self.cpu_progress = ttk.Progressbar(resource_frame, length=500, mode='determinate')
        self.cpu_progress.pack(fill='x', pady=5)
        self.cpu_label = tk.Label(resource_frame, text="0%", font=('Arial', 12), 
                                 bg='#34495e', fg='white')
        self.cpu_label.pack(anchor='w')
        
        # Memory usage
        tk.Label(resource_frame, text="Memory Usage:", font=('Arial', 14, 'bold'), 
                bg='#34495e', fg='white').pack(anchor='w', pady=(15,0))
        self.memory_progress = ttk.Progressbar(resource_frame, length=500, mode='determinate')
        self.memory_progress.pack(fill='x', pady=5)
        self.memory_label = tk.Label(resource_frame, text="0 MB", font=('Arial', 12), 
                                    bg='#34495e', fg='white')
        self.memory_label.pack(anchor='w')
        
    def create_results_tab(self):
        """Create enhanced results tab with comprehensive analysis and SCROLLABLE container"""
        results_frame = tk.Frame(self.main_notebook, bg='#2c3e50')
        self.main_notebook.add(results_frame, text="📊 Results & Analysis")
        
        # Header (fixed at top)
        header_frame = tk.Frame(results_frame, bg='#0f3460', height=80, relief='raised', bd=3)
        header_frame.pack(fill='x', padx=10, pady=10)
        header_frame.pack_propagate(False)
        
        tk.Label(header_frame, text="📊 COMPREHENSIVE RESULTS & ANALYSIS", 
                font=('Arial', 20, 'bold'), fg='#00d4ff', bg='#0f3460').pack(pady=5)
        self.results_status = tk.Label(header_frame, text="⏸️ No session data yet", 
                                      font=('Arial', 14, 'bold'), fg='#f39c12', bg='#0f3460')
        self.results_status.pack()
        
        # Export buttons row
        export_frame = tk.Frame(results_frame, bg='#2c3e50')
        export_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Button(export_frame, text="📄 Export to PDF", 
                 command=self.export_to_pdf,
                 bg='#e74c3c', fg='white', font=('Arial', 12, 'bold'),
                 height=1, width=20, relief='raised', bd=3).pack(side='left', padx=5)
        
        tk.Button(export_frame, text="📊 Export to CSV", 
                 command=self.export_to_csv,
                 bg='#27ae60', fg='white', font=('Arial', 12, 'bold'),
                 height=1, width=20, relief='raised', bd=3).pack(side='left', padx=5)
        
        tk.Button(export_frame, text="📋 Export Summary", 
                 command=self.export_summary_txt,
                 bg='#3498db', fg='white', font=('Arial', 12, 'bold'),
                 height=1, width=20, relief='raised', bd=3).pack(side='left', padx=5)
        
        # Create SCROLLABLE container
        results_canvas = tk.Canvas(results_frame, bg='#2c3e50', highlightthickness=0)
        results_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=results_canvas.yview)
        self.results_content = tk.Frame(results_canvas, bg='#2c3e50')
        
        self.results_content.bind(
            "<Configure>",
            lambda e: results_canvas.configure(scrollregion=results_canvas.bbox("all"))
        )
        
        results_canvas.create_window((0, 0), window=self.results_content, anchor="nw")
        results_canvas.configure(yscrollcommand=results_scrollbar.set)
        
        results_canvas.pack(side="left", fill="both", expand=True)
        results_scrollbar.pack(side="right", fill="y")
        
        # Enable mouse wheel scrolling
        def _on_results_mousewheel(event):
            results_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        results_canvas.bind_all("<MouseWheel>", _on_results_mousewheel)
        
        # Metrics summary
        metrics_frame = tk.LabelFrame(self.results_content, text="📈 Session Metrics", 
                                     font=('Arial', 16, 'bold'), bg='#34495e', fg='white',
                                     padx=20, pady=20)
        metrics_frame.pack(fill='x', padx=10, pady=10)
        
        # Create metrics grid
        metrics_grid = tk.Frame(metrics_frame, bg='#34495e')
        metrics_grid.pack(fill='x', pady=10)
        
        # Accuracy
        acc_box = tk.Frame(metrics_grid, bg='#9b59b6', relief='raised', bd=3)
        acc_box.grid(row=0, column=0, padx=10, pady=5, sticky='ew')
        tk.Label(acc_box, text="🎯 Accuracy", font=('Arial', 12, 'bold'), 
                bg='#9b59b6', fg='white').pack(pady=5)
        self.result_accuracy = tk.Label(acc_box, text="0.00%", font=('Arial', 20, 'bold'), 
                                       bg='#9b59b6', fg='white')
        self.result_accuracy.pack(pady=5)
        
        # Precision
        prec_box = tk.Frame(metrics_grid, bg='#3498db', relief='raised', bd=3)
        prec_box.grid(row=0, column=1, padx=10, pady=5, sticky='ew')
        tk.Label(prec_box, text="📏 Precision", font=('Arial', 12, 'bold'), 
                bg='#3498db', fg='white').pack(pady=5)
        self.result_precision = tk.Label(prec_box, text="0.00%", font=('Arial', 20, 'bold'), 
                                        bg='#3498db', fg='white')
        self.result_precision.pack(pady=5)
        
        # Recall
        rec_box = tk.Frame(metrics_grid, bg='#e74c3c', relief='raised', bd=3)
        rec_box.grid(row=0, column=2, padx=10, pady=5, sticky='ew')
        tk.Label(rec_box, text="� Recall", font=('Arial', 12, 'bold'), 
                bg='#e74c3c', fg='white').pack(pady=5)
        self.result_recall = tk.Label(rec_box, text="0.00%", font=('Arial', 20, 'bold'), 
                                      bg='#e74c3c', fg='white')
        self.result_recall.pack(pady=5)
        
        # F1-Score
        f1_box = tk.Frame(metrics_grid, bg='#27ae60', relief='raised', bd=3)
        f1_box.grid(row=0, column=3, padx=10, pady=5, sticky='ew')
        tk.Label(f1_box, text="⚖️ F1-Score", font=('Arial', 12, 'bold'), 
                bg='#27ae60', fg='white').pack(pady=5)
        self.result_f1 = tk.Label(f1_box, text="0.00%", font=('Arial', 20, 'bold'), 
                                 bg='#27ae60', fg='white')
        self.result_f1.pack(pady=5)
        
        metrics_grid.columnconfigure(0, weight=1)
        metrics_grid.columnconfigure(1, weight=1)
        metrics_grid.columnconfigure(2, weight=1)
        metrics_grid.columnconfigure(3, weight=1)
        
        # Confusion Matrix visualization
        confusion_frame = tk.LabelFrame(results_frame, text="🔢 Confusion Matrix", 
                                       font=('Arial', 16, 'bold'), bg='#34495e', fg='white',
                                       padx=20, pady=20)
        confusion_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create matplotlib figure for confusion matrix
        plt.style.use('dark_background')
        self.results_fig, self.results_ax = plt.subplots(figsize=(8, 6))
        self.results_fig.patch.set_facecolor('#34495e')
        self.results_ax.set_facecolor('#2c3e50')
        
        # Initial empty plot
        self.results_ax.text(0.5, 0.5, 'No data yet\nRun a prediction session first', 
                           ha='center', va='center', fontsize=16, color='white',
                           transform=self.results_ax.transAxes)
        self.results_ax.axis('off')
        
        plt.tight_layout()
        
        # Embed in tkinter
        self.results_canvas = FigureCanvasTkAgg(self.results_fig, confusion_frame)
        self.results_canvas.draw()
        self.results_canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def update_results_tab(self):
        """Update results tab with session data"""
        if len(self.prediction_history) == 0:
            return
        
        # Calculate metrics
        y_true = [p['actual'] for p in self.prediction_history]
        y_pred = [p['predicted'] for p in self.prediction_history]
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Update metric labels
        self.result_accuracy.config(text=f"{accuracy*100:.2f}%")
        self.result_precision.config(text=f"{precision*100:.2f}%")
        self.result_recall.config(text=f"{recall*100:.2f}%")
        self.result_f1.config(text=f"{f1*100:.2f}%")
        self.results_status.config(text=f"✅ Session Complete: {len(self.prediction_history)} samples", fg='#00ff88')
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        self.results_ax.clear()
        im = self.results_ax.imshow(cm, interpolation='nearest', cmap='Blues')
        self.results_ax.figure.colorbar(im, ax=self.results_ax)
        
        # Labels
        classes = ['Fake', 'Real']
        tick_marks = np.arange(len(classes))
        self.results_ax.set_xticks(tick_marks)
        self.results_ax.set_yticks(tick_marks)
        self.results_ax.set_xticklabels(classes, color='white', fontsize=12)
        self.results_ax.set_yticklabels(classes, color='white', fontsize=12)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                self.results_ax.text(j, i, format(cm[i, j], 'd'),
                                   ha="center", va="center",
                                   color="white" if cm[i, j] > thresh else "black",
                                   fontsize=20, weight='bold')
        
        self.results_ax.set_ylabel('True Label', color='white', fontsize=14, weight='bold')
        self.results_ax.set_xlabel('Predicted Label', color='white', fontsize=14, weight='bold')
        self.results_ax.set_title('Confusion Matrix', color='white', fontsize=16, weight='bold', pad=20)
        
        plt.tight_layout()
        self.results_canvas.draw()
        
    # In gui.py
    def populate_dropdowns_from_data(self):
        """Populates dropdowns from the model and datasets passed during init."""
        self.available_models = {}
        self.available_datasets = {}

        # 1. Add all loaded models
        if self.models and len(self.models) > 0:
            for model_key, model_obj in self.models.items():
                trained_status = "✅ Trained" if hasattr(model_obj, 'is_fitted') and model_obj.is_fitted else "⚠️ Not Trained"
                self.available_models[model_key] = f"{model_key} - {trained_status}"
            
            # Set default to the first trained model or first model
            default_model = None
            for key, model_obj in self.models.items():
                if hasattr(model_obj, 'is_fitted') and model_obj.is_fitted:
                    default_model = key
                    break
            
            if default_model is None:
                default_model = list(self.models.keys())[0]
            
            self.selected_model.set(default_model)
            self.model = self.models[default_model]  # Set the active model
        else:
            self.available_models['none'] = "No Models Available"
            self.selected_model.set('none')

        # 2. Add all loaded datasets
        if self.datasets:
            # Separate MFCC and spectrogram datasets for better organization
            mfcc_datasets = {}
            spectrogram_datasets = {}
            
            for key, data in self.datasets.items():
                if data['train'] is not None:
                    # Determine if it's MFCC or spectrogram based
                    is_mfcc = key.startswith('mfcc_')
                    
                    # Get display name
                    if is_mfcc:
                        # Remove 'mfcc_' prefix for display
                        base_name = key.replace('mfcc_', '', 1)
                        display_name = f"🎵 MFCC - {base_name}"
                    else:
                        display_name = data.get('info', {}).get('dataset_name', key)
                        if 'snr_' in key:
                            display_name = f"📈 Spectrogram - {display_name}"
                        else:
                            display_name = f"📈 {display_name}"
                    
                    # Add all three splits with feature type indicator
                    feature_type = "MFCC" if is_mfcc else "Spectrogram"
                    dataset_dict = mfcc_datasets if is_mfcc else spectrogram_datasets
                    
                    dataset_dict[f'{key}_train'] = f"{display_name} (Train)"
                    dataset_dict[f'{key}_val'] = f"{display_name} (Validation)"
                    dataset_dict[f'{key}_test'] = f"{display_name} (Test)"
            
            # Add spectrogram datasets first, then MFCC datasets
            self.available_datasets.update(spectrogram_datasets)
            self.available_datasets.update(mfcc_datasets)

            # Set a default dataset - prefer MFCC for faster training
            default_ds = None
            if mfcc_datasets:
                # Try to use MFCC train_test_val test set first
                if 'mfcc_train_test_val_test' in self.available_datasets:
                    default_ds = 'mfcc_train_test_val_test'
                else:
                    default_ds = list(mfcc_datasets.keys())[0]
            elif spectrogram_datasets:
                if 'snr_5_test' in self.available_datasets:
                    default_ds = 'snr_5_test'
                else:
                    default_ds = list(spectrogram_datasets.keys())[0]
            
            if default_ds:
                self.selected_dataset.set(default_ds)
        else:
            self.available_datasets['none'] = "No Datasets Loaded"
            self.selected_dataset.set('none')
    
    def on_model_changed(self, event=None):
        """Handle model selection change"""
        selected_key = self.selected_model.get()
        if selected_key in self.models:
            self.model = self.models[selected_key]
            trained_status = "✅ Trained" if hasattr(self.model, 'is_fitted') and self.model.is_fitted else "⚠️ Not Trained"
            print(f"🔄 Model switched to: {selected_key} - {trained_status}")
            
            # Show warning if model is not trained
            if not (hasattr(self.model, 'is_fitted') and self.model.is_fitted):
                messagebox.showwarning(
                    "Model Not Trained",
                    f"⚠️ The selected model '{selected_key}' is not trained yet!\n\n"
                    f"Please go to the '🎓 Model Training' tab to train this model first."
                )
            
    def update_animations(self):
        """Update various animations"""
        # Animate header pulsing effect
        self.animation_frame += self.pulse_direction
        if self.animation_frame >= 20:
            self.pulse_direction = -1
        elif self.animation_frame <= 0:
            self.pulse_direction = 1
            
        # Apply pulsing effect to title
        if hasattr(self, 'status_title'):
            alpha = 0.7 + (self.animation_frame / 40)  # Varies between 0.7 and 1.2
            # Simulate alpha effect by changing font weight
            if self.animation_frame > 10:
                self.status_title.config(font=('Arial', 20, 'bold'))
            else:
                self.status_title.config(font=('Arial', 19, 'bold'))
        
        # Schedule next animation update
        self.root.after(100, self.update_animations)
        
    def update_simulation_display(self):
        """Update simulation displays periodically"""
        # Simulate system resource usage
        if self.is_processing:
            cpu_usage = random.uniform(60, 90)
            memory_usage = random.uniform(3000, 7000)
        else:
            cpu_usage = random.uniform(10, 30)
            memory_usage = random.uniform(1500, 3000)
            
        if hasattr(self, 'cpu_progress'):
            self.cpu_progress['value'] = cpu_usage
            self.cpu_label.config(text=f"{cpu_usage:.1f}%")
            
            self.memory_progress['value'] = (memory_usage / 8000) * 100
            self.memory_label.config(text=f"{memory_usage:.0f} MB")
        
        # Schedule next update
        self.root.after(1000, self.update_simulation_display)
        
    def update_logs(self):
        """Update log display"""
        # Process any queued log messages
        self.root.after(100, self.update_logs)
    
    def export_to_pdf(self):
        """Export comprehensive analysis to PDF"""
        if self.total_predictions == 0:
            messagebox.showwarning("No Data", "No prediction data available to export!")
            return
        
        try:
            # Ask for save location
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf")],
                initialfile=f"Audio_Classification_Report_{timestamp}.pdf"
            )
            
            if not filename:
                return
            
            # Create PDF with matplotlib
            with PdfPages(filename) as pdf:
                # Page 1: Summary and Metrics
                fig = plt.figure(figsize=(11, 8.5))
                fig.suptitle('Audio Classification Analysis Report', fontsize=20, weight='bold')
                
                # Add metadata
                report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
                plt.figtext(0.5, 0.92, f'Generated: {report_date}', ha='center', fontsize=10)
                
                # Session Information
                ax1 = plt.subplot(3, 2, 1)
                ax1.axis('off')
                session_info = f"""
SESSION INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model: {self.selected_model.get()}
Dataset: {self.selected_dataset.get()}
Total Predictions: {self.total_predictions}
Date: {report_date}
                """
                ax1.text(0.1, 0.5, session_info, fontsize=11, family='monospace',
                        verticalalignment='center')
                
                # Performance Metrics
                ax2 = plt.subplot(3, 2, 2)
                ax2.axis('off')
                accuracy = (self.correct_predictions / self.total_predictions * 100) if self.total_predictions > 0 else 0
                metrics_info = f"""
PERFORMANCE METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Correct: {self.correct_predictions}
✗ Incorrect: {self.total_predictions - self.correct_predictions}
🎯 Accuracy: {accuracy:.2f}%
                """
                ax2.text(0.1, 0.5, metrics_info, fontsize=11, family='monospace',
                        verticalalignment='center')
                
                # Accuracy Chart
                ax3 = plt.subplot(3, 2, (3, 4))
                if len(self.prediction_history) > 0:
                    running_accuracy = []
                    correct_count = 0
                    for i, pred in enumerate(self.prediction_history):
                        if pred['correct']:
                            correct_count += 1
                        running_accuracy.append(correct_count / (i + 1))
                    
                    sample_numbers = list(range(1, len(running_accuracy) + 1))
                    ax3.plot(sample_numbers, running_accuracy, 'b-', linewidth=2)
                    ax3.fill_between(sample_numbers, running_accuracy, alpha=0.3)
                    ax3.set_xlabel('Prediction Number', fontsize=10)
                    ax3.set_ylabel('Running Accuracy', fontsize=10)
                    ax3.set_title('Running Accuracy Over Time', fontsize=12, weight='bold')
                    ax3.grid(True, alpha=0.3)
                    ax3.set_ylim(0, 1)
                
                # Confidence Distribution
                ax4 = plt.subplot(3, 2, (5, 6))
                if len(self.prediction_history) > 0:
                    confidences = [pred['confidence'] for pred in self.prediction_history]
                    colors = ['green' if pred['correct'] else 'red' for pred in self.prediction_history]
                    sample_nums = list(range(1, len(confidences) + 1))
                    ax4.bar(sample_nums, confidences, color=colors, alpha=0.6)
                    ax4.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, label='Threshold')
                    ax4.set_xlabel('Prediction Number', fontsize=10)
                    ax4.set_ylabel('Confidence', fontsize=10)
                    ax4.set_title('Prediction Confidence Distribution', fontsize=12, weight='bold')
                    ax4.legend()
                    ax4.grid(True, alpha=0.3)
                    ax4.set_ylim(0, 1)
                
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                pdf.savefig(fig)
                plt.close(fig)
                
                # Page 2: Detailed Prediction Table (if we have data)
                if len(self.prediction_history) > 0:
                    fig2 = plt.figure(figsize=(11, 8.5))
                    fig2.suptitle('Detailed Prediction Results', fontsize=16, weight='bold')
                    ax = fig2.add_subplot(111)
                    ax.axis('off')
                    
                    # Create table data
                    table_data = [['#', 'Actual', 'Predicted', 'Confidence', 'Result']]
                    for i, pred in enumerate(self.prediction_history[:50], 1):  # First 50 predictions
                        actual = 'Real' if pred['actual'] == 1 else 'Fake'
                        predicted = 'Real' if pred['predicted'] == 1 else 'Fake'
                        confidence = f"{pred['confidence']:.2%}"
                        result = '✓ Correct' if pred['correct'] else '✗ Wrong'
                        table_data.append([str(i), actual, predicted, confidence, result])
                    
                    # Create table
                    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                                   colWidths=[0.1, 0.2, 0.2, 0.2, 0.2])
                    table.auto_set_font_size(False)
                    table.set_fontsize(9)
                    table.scale(1, 2)
                    
                    # Style header row
                    for i in range(5):
                        table[(0, i)].set_facecolor('#3498db')
                        table[(0, i)].set_text_props(weight='bold', color='white')
                    
                    # Color code results
                    for i in range(1, len(table_data)):
                        if '✓' in table_data[i][4]:
                            table[(i, 4)].set_facecolor('#d5f4e6')
                        else:
                            table[(i, 4)].set_facecolor('#fadbd8')
                    
                    plt.figtext(0.5, 0.95, f'Showing first 50 of {len(self.prediction_history)} predictions',
                              ha='center', fontsize=10)
                    
                    pdf.savefig(fig2)
                    plt.close(fig2)
            
            messagebox.showinfo("Export Successful", 
                              f"Report exported successfully to:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export PDF:\n{str(e)}")
    
    def export_to_csv(self):
        """Export detailed prediction results to CSV"""
        if self.total_predictions == 0:
            messagebox.showwarning("No Data", "No prediction data available to export!")
            return
        
        try:
            # Ask for save location
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                initialfile=f"Audio_Classification_Results_{timestamp}.csv"
            )
            
            if not filename:
                return
            
            # Write CSV
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Header information
                writer.writerow(['Audio Classification Results'])
                writer.writerow(['Generated:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                writer.writerow(['Model:', self.selected_model.get()])
                writer.writerow(['Dataset:', self.selected_dataset.get()])
                writer.writerow([])
                
                # Summary statistics
                accuracy = (self.correct_predictions / self.total_predictions * 100) if self.total_predictions > 0 else 0
                writer.writerow(['SUMMARY STATISTICS'])
                writer.writerow(['Total Predictions', self.total_predictions])
                writer.writerow(['Correct Predictions', self.correct_predictions])
                writer.writerow(['Incorrect Predictions', self.total_predictions - self.correct_predictions])
                writer.writerow(['Accuracy (%)', f'{accuracy:.2f}'])
                
                if self.prediction_history:
                    avg_confidence = np.mean([p['confidence'] for p in self.prediction_history])
                    writer.writerow(['Average Confidence', f'{avg_confidence:.2%}'])
                
                writer.writerow([])
                
                # Detailed results
                writer.writerow(['DETAILED PREDICTION RESULTS'])
                writer.writerow(['Sample #', 'Actual Label', 'Predicted Label', 'Confidence', 
                               'Result', 'Dataset Info'])
                
                for i, pred in enumerate(self.prediction_history, 1):
                    actual = 'Real Audio' if pred['actual'] == 1 else 'Fake Audio'
                    predicted = 'Real Audio' if pred['predicted'] == 1 else 'Fake Audio'
                    confidence = f"{pred['confidence']:.4f}"
                    result = 'Correct' if pred['correct'] else 'Incorrect'
                    dataset_info = pred.get('dataset', 'N/A')
                    
                    writer.writerow([i, actual, predicted, confidence, result, dataset_info])
            
            messagebox.showinfo("Export Successful", 
                              f"Results exported successfully to:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export CSV:\n{str(e)}")
    
    def export_summary_txt(self):
        """Export comprehensive summary as formatted text file"""
        if self.total_predictions == 0:
            messagebox.showwarning("No Data", "No prediction data available to export!")
            return
        
        try:
            # Ask for save location
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt")],
                initialfile=f"Audio_Classification_Summary_{timestamp}.txt"
            )
            
            if not filename:
                return
            
            # Calculate statistics
            accuracy = (self.correct_predictions / self.total_predictions * 100) if self.total_predictions > 0 else 0
            
            # Build comprehensive summary
            summary = []
            summary.append("=" * 80)
            summary.append("AUDIO CLASSIFICATION ANALYSIS SUMMARY")
            summary.append("=" * 80)
            summary.append("")
            summary.append(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M:%S %p')}")
            summary.append("")
            
            summary.append("-" * 80)
            summary.append("SESSION INFORMATION")
            summary.append("-" * 80)
            summary.append(f"Model Used:           {self.selected_model.get()}")
            summary.append(f"Dataset:              {self.selected_dataset.get()}")
            summary.append(f"Total Predictions:    {self.total_predictions}")
            summary.append("")
            
            summary.append("-" * 80)
            summary.append("PERFORMANCE METRICS")
            summary.append("-" * 80)
            summary.append(f"Correct Predictions:     {self.correct_predictions}")
            summary.append(f"Incorrect Predictions:   {self.total_predictions - self.correct_predictions}")
            summary.append(f"Accuracy:                {accuracy:.2f}%")
            summary.append("")
            
            if self.prediction_history:
                avg_confidence = np.mean([p['confidence'] for p in self.prediction_history])
                max_confidence = np.max([p['confidence'] for p in self.prediction_history])
                min_confidence = np.min([p['confidence'] for p in self.prediction_history])
                
                summary.append("-" * 80)
                summary.append("CONFIDENCE STATISTICS")
                summary.append("-" * 80)
                summary.append(f"Average Confidence:   {avg_confidence:.2%}")
                summary.append(f"Maximum Confidence:   {max_confidence:.2%}")
                summary.append(f"Minimum Confidence:   {min_confidence:.2%}")
                summary.append("")
            
            # Performance assessment
            summary.append("-" * 80)
            summary.append("PERFORMANCE ASSESSMENT")
            summary.append("-" * 80)
            if accuracy >= 90:
                assessment = "EXCELLENT - Model shows outstanding performance!"
            elif accuracy >= 80:
                assessment = "GOOD - Model performs well with room for improvement"
            elif accuracy >= 70:
                assessment = "MODERATE - Consider model tuning or additional training"
            else:
                assessment = "NEEDS IMPROVEMENT - Review model parameters and training data"
            summary.append(f"Overall Rating: {assessment}")
            summary.append("")
            
            # Sample predictions
            summary.append("-" * 80)
            summary.append("SAMPLE PREDICTIONS (First 20)")
            summary.append("-" * 80)
            summary.append(f"{'#':<5} {'Actual':<12} {'Predicted':<12} {'Confidence':<12} {'Result':<10}")
            summary.append("-" * 80)
            
            for i, pred in enumerate(self.prediction_history[:20], 1):
                actual = 'Real' if pred['actual'] == 1 else 'Fake'
                predicted = 'Real' if pred['predicted'] == 1 else 'Fake'
                confidence = f"{pred['confidence']:.2%}"
                result = 'Correct' if pred['correct'] else 'Incorrect'
                summary.append(f"{i:<5} {actual:<12} {predicted:<12} {confidence:<12} {result:<10}")
            
            summary.append("")
            summary.append("=" * 80)
            summary.append("END OF REPORT")
            summary.append("=" * 80)
            
            # Write to file
            with open(filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(summary))
            
            messagebox.showinfo("Export Successful", 
                              f"Summary exported successfully to:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export summary:\n{str(e)}")

# Launch the Advanced Simulation GUI
def launch_advanced_simulation_gui(model, datasets_dict):
    """Launch the advanced simulation GUI with live prediction feedback"""
    try:
        root = tk.Tk()
        app = AdvancedSimulationAudioGUI(root, model, datasets_dict)

        # Center window
        root.update_idletasks()
        x = (root.winfo_screenwidth() - root.winfo_width()) // 2
        y = (root.winfo_screenheight() - root.winfo_height()) // 2
        root.geometry(f'+{x}+{y}')
        
        print("🚀 Launching Advanced Audio Classification Simulation GUI...")
        print("🎮 Enhanced Features:")
        print("   • Live prediction feedback with correct/incorrect indicators")
        print("   • Real-time accuracy tracking and visual counters")
        print("   • Enhanced dark theme with animated elements")
        print("   • Detailed prediction logging with color coding")
        print("   • Live performance visualization and graphs")
        print("   • Comprehensive session summaries")
        print("   • System resource monitoring")
        print("📊 Perfect for demonstrating AI model performance!")
        
        root.mainloop()
        
    except Exception as e:
        print(f"❌ Error launching advanced GUI: {e}")

# Launch the advanced simulation GUI