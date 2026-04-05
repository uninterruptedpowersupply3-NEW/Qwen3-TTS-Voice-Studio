import sys
import os
import time
import datetime
import numpy as np
import torch
import soundfile as sf
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QTextEdit, 
                             QPushButton, QGroupBox, QFormLayout, QDoubleSpinBox, 
                             QSpinBox, QCheckBox, QMessageBox, QScrollArea,
                             QTabWidget, QFileDialog)
from PyQt6.QtCore import QThread, pyqtSignal, Qt

# --- HARDWARE OPTIMIZATIONS ---
# Enable TF32 for NVIDIA Ampere+ GPUs (RTX 3000/4000 series)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ==========================================
# THREAD WORKERS (Keeps GUI from freezing)
# ==========================================

class ModelLoaderThread(QThread):
    finished = pyqtSignal(object, object)
    error = pyqtSignal(str)

    def run(self):
        print("\n" + "="*50)
        print("[SYSTEM LOG] Starting model initialization...")
        print("="*50)
        try:
            from qwen_tts import Qwen3TTSModel
            
            # 1. Force global BF16 so the models don't bloat into FP32 and crash your VRAM.
            torch.set_default_dtype(torch.bfloat16)

            # 2. Flash Attention 2 setup
            try:
                import flash_attn
                attn_impl = "flash_attention_2"
                print("[SYSTEM LOG] Flash Attention 2 module found. Using flash_attention_2.")
            except ImportError:
                attn_impl = "sdpa"
                print("[SYSTEM LOG] Flash Attention 2 NOT found. Falling back to PyTorch SDPA.")

            print(f"\n[SYSTEM LOG] Loading Qwen3-TTS-12Hz-1.7B-VoiceDesign...")
            print(f"[SYSTEM LOG] Dtype: bfloat16 | TF32: Enabled | Attn: {attn_impl}")
            
            # THE FIX: Strictly use "cuda:0". 
            # Do NOT use "auto", as Qwen's custom code leaves "meta tensors" behind.
            try:
                print("[SYSTEM LOG] Checking local cache for Base TTS model...")
                base_model = Qwen3TTSModel.from_pretrained(
                    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                    device_map="cuda:0",
                    dtype=torch.bfloat16,
                    attn_implementation=attn_impl,
                    local_files_only=True
                )
                print("[SYSTEM LOG] Base model loaded successfully from local cache.")
            except Exception as e_offline:
                print(f"[SYSTEM LOG] Local base model not found (Error: {e_offline}). Attempting online download...")
                try:
                    base_model = Qwen3TTSModel.from_pretrained(
                        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                        device_map="cuda:0",
                        dtype=torch.bfloat16,
                        attn_implementation=attn_impl
                    )
                    print("[SYSTEM LOG] Base model downloaded and loaded successfully.")
                except Exception as e_online:
                    raise RuntimeError(f"Could not download model. Please reconnect to internet.\nOffline Error: {e_offline}\nOnline Error: {e_online}")

            # Return None for design_model so it only loads dynamically on demand
            self.finished.emit(None, base_model)
        except Exception as e:
            print(f"\n[ERROR] Model load failed: {e}")
            self.error.emit(str(e))
            
class VoiceDesignThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, model, base_model, instruct, ref_text, kwargs):
        super().__init__()
        self.model = model
        self.base_model = base_model
        self.instruct = instruct
        self.ref_text = ref_text
        self.kwargs = kwargs

    def run(self):
        try:
            print("\n" + "="*50)
            print("[GENERATION LOG] Initiating Voice Design...")
            print("="*50)
            
            if getattr(self, 'model', None) is None:
                print("[SYSTEM LOG] Dynamically loading VoiceDesign model to save VRAM...")
                from qwen_tts import Qwen3TTSModel
                try:
                    import flash_attn
                    attn_impl = "flash_attention_2"
                except ImportError:
                    attn_impl = "sdpa"
                try:
                    self.model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", device_map="cuda:0", dtype=torch.bfloat16, attn_implementation=attn_impl, local_files_only=True)
                except Exception:
                    self.model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", device_map="cuda:0", dtype=torch.bfloat16, attn_implementation=attn_impl)

            print(f"Prompt: {self.instruct}")
            print(f"Parameters: {self.kwargs}")
            
            t0 = time.time()
            ref_wavs, sr = self.model.generate_voice_design(
                text=self.ref_text,
                language="English",
                instruct=self.instruct,
                **self.kwargs
            )
            print(f"\n[GENERATION LOG] Voice design rendered in {time.time()-t0:.2f}s.")
            
            print("[SYSTEM LOG] Unloading VoiceDesign to free VRAM immediately...")
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            
            ref_audio = ref_wavs[0]
            sf.write("reference_voice.wav", ref_audio, sr)
            print("[GENERATION LOG] Saved master audio reference to 'reference_voice.wav'.")

            print("[GENERATION LOG] Extracting acoustic tensors (voice_clone_prompt) for cache...")
            t0 = time.time()
            prompt = self.base_model.create_voice_clone_prompt(
                ref_audio=(ref_audio, sr),
                ref_text=self.ref_text,
            )
            print(f"[GENERATION LOG] Feature extraction complete in {time.time()-t0:.2f}s.")
            
            # Automatically save the embedding file as a backup
            torch.save(prompt, "reference_voice_embedding.pt")
            print("[GENERATION LOG] Saved master tensor embedding to 'reference_voice_embedding.pt'.")

            self.finished.emit(prompt)
        except Exception as e:
            print(f"\n[ERROR] Voice design failed: {e}")
            self.error.emit(str(e))


class AudioCloneThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, base_model, audio_path, transcript):
        super().__init__()
        self.base_model = base_model
        self.audio_path = audio_path
        self.transcript = transcript

    def run(self):
        try:
            print("\n" + "="*50)
            print(f"[CLONE LOG] Cloning voice from: {self.audio_path}")
            print("="*50)
            
            t0 = time.time()
            prompt = self.base_model.create_voice_clone_prompt(
                ref_audio=self.audio_path,
                ref_text=self.transcript,
            )
            print(f"[CLONE LOG] Feature extraction complete in {time.time()-t0:.2f}s.")
            
            self.finished.emit(prompt)
        except Exception as e:
            print(f"\n[ERROR] Audio cloning failed: {e}")
            self.error.emit(str(e))


class BatchCloneThread(QThread):
    finished = pyqtSignal(float)
    error = pyqtSignal(str)

    def __init__(self, base_model, prompt, sentences, kwargs, batch_size=4):
        super().__init__()
        self.base_model = base_model
        self.prompt = prompt
        self.sentences = sentences
        self.kwargs = kwargs
        self.batch_size = batch_size

    def run(self):
        try:
            print("\n" + "="*50)
            print(f"[BATCH LOG] Starting batch generation for {len(self.sentences)} audio lines...")
            print("="*50)
            print(f"Parameters: {self.kwargs}")
            
            t0 = time.time()
            wavs = []
            sr = 24000 # default fallback
            
            # Process using dynamic batch size to avoid OOM or speed up
            for i in range(0, len(self.sentences), self.batch_size):
                batch_sentences = self.sentences[i:i+self.batch_size]
                languages = ["English"] * len(batch_sentences)
                
                print(f"[BATCH LOG] Generating audio segment [{i+1}-{min(i+self.batch_size, len(self.sentences))}/{len(self.sentences)}]...")
                b_wavs, b_sr = self.base_model.generate_voice_clone(
                    text=batch_sentences,
                    language=languages,
                    voice_clone_prompt=self.prompt,
                    **self.kwargs
                )
                wavs.extend(b_wavs)
                sr = b_sr
                torch.cuda.empty_cache() # clear intermediate memory after every batch segment
                
            t1 = time.time()
            
            print(f"\n[BATCH LOG] Generation completed in {t1-t0:.2f}s.")
            print("[BATCH LOG] Concatenating and Exporting payload...")
            
            os.makedirs("audio_exports", exist_ok=True)
            if len(wavs) > 0:
                final_wav = np.concatenate(wavs, axis=0)
                
                # Format: time_date_firstline.wav
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                # Clean up the first line to be safe for filenames
                safe_first_line = ''.join(c for c in self.sentences[0] if c.isalnum() or c in ' -_')[:30].strip()
                safe_first_line = safe_first_line.replace(' ', '_')
                
                filename = f"audio_exports/{timestamp}_{safe_first_line}.wav"
                sf.write(filename, final_wav, sr)
                print(f" -> Saved combined audio: {filename}")
                
            self.finished.emit(t1-t0)
        except Exception as e:
            print(f"\n[ERROR] Batch generation failed: {e}")
            self.error.emit(str(e))


# ==========================================
# PYQT6 MAIN APPLICATION
# ==========================================

class VoiceStudioApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qwen3-TTS Voice Studio Pro")
        self.resize(1100, 850)

        # State Variables
        self.design_model = None
        self.base_model = None
        self.reusable_voice_prompt = None
        self.selected_audio_file = None
        self.selected_pt_file = None
        
        self.init_ui()
        self.start_model_loader()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # ---------------------------------------------------------
        # LEFT PANEL: Parameters
        # ---------------------------------------------------------
        param_scroll = QScrollArea()
        param_scroll.setWidgetResizable(True)
        param_scroll.setFixedWidth(350)
        param_widget = QWidget()
        param_layout = QFormLayout(param_widget)

        param_group = QGroupBox("Generation Variables")
        
        self.val_max_new_tokens = QSpinBox(); self.val_max_new_tokens.setRange(100, 8192); self.val_max_new_tokens.setValue(2048)
        self.val_temperature = QDoubleSpinBox(); self.val_temperature.setRange(0.1, 2.0); self.val_temperature.setSingleStep(0.1); self.val_temperature.setValue(0.9)
        self.val_top_p = QDoubleSpinBox(); self.val_top_p.setRange(0.1, 1.0); self.val_top_p.setSingleStep(0.05); self.val_top_p.setValue(1.0)
        self.val_top_k = QSpinBox(); self.val_top_k.setRange(1, 100); self.val_top_k.setValue(50)
        self.val_rep_penalty = QDoubleSpinBox(); self.val_rep_penalty.setRange(1.0, 2.0); self.val_rep_penalty.setSingleStep(0.05); self.val_rep_penalty.setValue(1.05)
        self.val_do_sample = QCheckBox(); self.val_do_sample.setChecked(True)
        self.val_batch_size = QSpinBox(); self.val_batch_size.setRange(1, 100); self.val_batch_size.setValue(4)

        self.val_sub_do_sample = QCheckBox(); self.val_sub_do_sample.setChecked(True)
        self.val_sub_temp = QDoubleSpinBox(); self.val_sub_temp.setRange(0.1, 2.0); self.val_sub_temp.setSingleStep(0.1); self.val_sub_temp.setValue(0.9)
        self.val_sub_top_p = QDoubleSpinBox(); self.val_sub_top_p.setRange(0.1, 1.0); self.val_sub_top_p.setSingleStep(0.05); self.val_sub_top_p.setValue(1.0)
        self.val_sub_top_k = QSpinBox(); self.val_sub_top_k.setRange(1, 100); self.val_sub_top_k.setValue(50)

        param_layout.addRow("Max New Tokens:", self.val_max_new_tokens)
        param_layout.addRow("Temperature:", self.val_temperature)
        param_layout.addRow("Top P:", self.val_top_p)
        param_layout.addRow("Top K:", self.val_top_k)
        param_layout.addRow("Repetition Penalty:", self.val_rep_penalty)
        param_layout.addRow("Do Sample:", self.val_do_sample)
        param_layout.addRow("Batch Size (Parallel items):", self.val_batch_size)
        param_layout.addRow("--- Subtalker Settings ---", QLabel(""))
        param_layout.addRow("Subtalker Do Sample:", self.val_sub_do_sample)
        param_layout.addRow("Subtalker Temp:", self.val_sub_temp)
        param_layout.addRow("Subtalker Top P:", self.val_sub_top_p)
        param_layout.addRow("Subtalker Top K:", self.val_sub_top_k)

        param_group.setLayout(param_layout)
        wrapper_layout = QVBoxLayout(param_widget)
        wrapper_layout.addWidget(param_group)
        wrapper_layout.addStretch()
        param_scroll.setWidget(param_widget)

        # ---------------------------------------------------------
        # RIGHT PANEL: Execution
        # ---------------------------------------------------------
        exec_layout = QVBoxLayout()

        self.lbl_status = QLabel("Status: Waiting to load models... (Check Terminal)")
        self.lbl_status.setStyleSheet("color: blue; font-weight: bold; font-size: 14px;")
        exec_layout.addWidget(self.lbl_status)

        # --- STEP 1: OBTAIN MASTER VOICE (Tabbed Interface) ---
        step1_group = QGroupBox("Step 1: Obtain Master Voice")
        step1_layout = QVBoxLayout()
        self.tabs = QTabWidget()

        # TAB A: Voice Design
        tab_design = QWidget()
        design_vbox = QVBoxLayout()
        design_vbox.addWidget(QLabel("Voice Design Prompt (Describe the voice):"))
        self.txt_instruct = QLineEdit("A clear, professional female voice speaking calmly and at a measured pace.")
        design_vbox.addWidget(self.txt_instruct)
        design_vbox.addWidget(QLabel("Reference Text (What should the AI say in the master clone sample?):"))
        self.txt_ref_design = QLineEdit("Hello, welcome to the audio generation system. Let me demonstrate my tone.")
        design_vbox.addWidget(self.txt_ref_design)
        self.btn_design = QPushButton("Generate & Cache Master Voice")
        self.btn_design.setEnabled(False)
        self.btn_design.clicked.connect(self.on_design_clicked)
        design_vbox.addWidget(self.btn_design)
        design_vbox.addStretch()
        tab_design.setLayout(design_vbox)

        # TAB B: Clone Audio File
        tab_clone = QWidget()
        clone_vbox = QVBoxLayout()
        clone_vbox.addWidget(QLabel("Upload a 10-15s clean audio clip (.wav or .mp3) of the target voice:"))
        
        file_hlayout = QHBoxLayout()
        self.lbl_audio_file = QLabel("No file selected.")
        self.lbl_audio_file.setStyleSheet("color: gray;")
        btn_browse_audio = QPushButton("Browse Audio...")
        btn_browse_audio.clicked.connect(self.browse_audio_file)
        file_hlayout.addWidget(self.lbl_audio_file)
        file_hlayout.addWidget(btn_browse_audio)
        clone_vbox.addLayout(file_hlayout)

        clone_vbox.addWidget(QLabel("Transcript (You MUST type exactly what is spoken in the audio clip):"))
        self.txt_ref_clone = QLineEdit("")
        clone_vbox.addWidget(self.txt_ref_clone)
        
        self.btn_clone_audio = QPushButton("Extract & Cache Voice from Audio")
        self.btn_clone_audio.setEnabled(False)
        self.btn_clone_audio.clicked.connect(self.on_clone_audio_clicked)
        clone_vbox.addWidget(self.btn_clone_audio)
        clone_vbox.addStretch()
        tab_clone.setLayout(clone_vbox)

        # TAB C: Load Embedding (.pt)
        tab_embed = QWidget()
        embed_vbox = QVBoxLayout()
        embed_vbox.addWidget(QLabel("Already saved a voice? Load the .pt tensor file directly:"))
        
        pt_hlayout = QHBoxLayout()
        self.lbl_pt_file = QLabel("No file selected.")
        self.lbl_pt_file.setStyleSheet("color: gray;")
        btn_browse_pt = QPushButton("Browse .pt File...")
        btn_browse_pt.clicked.connect(self.browse_pt_file)
        pt_hlayout.addWidget(self.lbl_pt_file)
        pt_hlayout.addWidget(btn_browse_pt)
        embed_vbox.addLayout(pt_hlayout)

        self.btn_load_pt = QPushButton("Load Embedding to RAM")
        self.btn_load_pt.setEnabled(False) # <-- FIX 1: Lock this button on startup
        self.btn_load_pt.clicked.connect(self.on_load_pt_clicked)
        embed_vbox.addWidget(self.btn_load_pt)
        embed_vbox.addStretch()
        tab_embed.setLayout(embed_vbox)

        # Add tabs to Step 1 Group
        self.tabs.addTab(tab_design, "Design from Text")
        self.tabs.addTab(tab_clone, "Clone from Audio")
        self.tabs.addTab(tab_embed, "Load .pt Embedding")
        step1_layout.addWidget(self.tabs)

        # Save Embedding Button (Global to Step 1)
        self.btn_save_pt = QPushButton("💾 Save Current Master Voice as .pt file")
        self.btn_save_pt.setEnabled(False)
        self.btn_save_pt.clicked.connect(self.save_current_embedding)
        step1_layout.addWidget(self.btn_save_pt)
        
        step1_group.setLayout(step1_layout)
        exec_layout.addWidget(step1_group)

        # --- STEP 2: BATCH PROCESSING ---
        batch_group = QGroupBox("Step 2: Batch Audio Generation (Reuses Master Voice)")
        batch_vbox = QVBoxLayout()
        
        batch_vbox.addWidget(QLabel("Audio Script (One sentence per line):"))
        self.txt_script = QTextEdit()
        self.txt_script.setPlainText("This is the first generated line.\nAnd this is the second line for the batch process.\nEverything is working perfectly.")
        batch_vbox.addWidget(self.txt_script)

        self.btn_batch = QPushButton("🚀 Batch Generate Script")
        self.btn_batch.setEnabled(False)
        self.btn_batch.setStyleSheet("font-weight: bold; padding: 10px;")
        self.btn_batch.clicked.connect(self.on_batch_clicked)
        batch_vbox.addWidget(self.btn_batch)

        batch_group.setLayout(batch_vbox)
        exec_layout.addWidget(batch_group)

        # Assemble Main Layout
        main_layout.addWidget(param_scroll)
        main_layout.addLayout(exec_layout)

    def get_generation_kwargs(self):
        """Collects all UI variables into a dict for Hugging Face generation"""
        return {
            "max_new_tokens": self.val_max_new_tokens.value(),
            "temperature": self.val_temperature.value(),
            "top_p": self.val_top_p.value(),
            "top_k": self.val_top_k.value(),
            "repetition_penalty": self.val_rep_penalty.value(),
            "do_sample": self.val_do_sample.isChecked(),
            "subtalker_dosample": self.val_sub_do_sample.isChecked(),
            "subtalker_temperature": self.val_sub_temp.value(),
            "subtalker_top_p": self.val_sub_top_p.value(),
            "subtalker_top_k": self.val_sub_top_k.value(),
        }

    # --- FILE BROWSERS ---
    def browse_audio_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav *.mp3 *.flac)")
        if file_path:
            self.selected_audio_file = file_path
            self.lbl_audio_file.setText(os.path.basename(file_path))
            self.lbl_audio_file.setStyleSheet("color: black;")
            # Enable the clone button only if models are loaded
            if self.base_model:
                self.btn_clone_audio.setEnabled(True)

    def browse_pt_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Embedding File", "", "PyTorch Tensors (*.pt *.pth)")
        if file_path:
            self.selected_pt_file = file_path
            self.lbl_pt_file.setText(os.path.basename(file_path))
            self.lbl_pt_file.setStyleSheet("color: black;")

    # --- ACTION HANDLERS ---
    def start_model_loader(self):
        self.lbl_status.setText("Status: Loading Models (Check terminal for Hugging Face download progress)...")
        self.loader_thread = ModelLoaderThread()
        self.loader_thread.finished.connect(self.on_models_loaded)
        self.loader_thread.error.connect(self.on_error)
        self.loader_thread.start()

    def on_models_loaded(self, design_model, base_model):
        self.design_model = design_model
        self.base_model = base_model
        self.lbl_status.setText("Status: Models Loaded. Ready.")
        self.btn_design.setEnabled(True)
        self.btn_design.setText("Generate & Cache Master Voice")
        self.btn_load_pt.setEnabled(True) # <-- FIX 2: Unlock it safely here
        if self.selected_audio_file:
            self.btn_clone_audio.setEnabled(True)
        if getattr(self, 'reusable_voice_prompt', None) is not None:
            self.btn_batch.setEnabled(True)

    # ACTION: Voice Design (Tab A)
    def on_design_clicked(self):
        self.btn_design.setEnabled(False)
        self.lbl_status.setText("Status: Designing Voice (Check terminal)...")
        
        kwargs = self.get_generation_kwargs()
        instruct = self.txt_instruct.text()
        ref_text = self.txt_ref_design.text()
        
        self.design_thread = VoiceDesignThread(self.design_model, self.base_model, instruct, ref_text, kwargs)
        self.design_thread.finished.connect(self.on_master_voice_ready)
        self.design_thread.error.connect(self.on_error)
        self.design_thread.start()

    # ACTION: Clone Audio (Tab B)
    def on_clone_audio_clicked(self):
        if not self.selected_audio_file:
            QMessageBox.warning(self, "Missing File", "Please select an audio file first.")
            return
            
        transcript = self.txt_ref_clone.text().strip()
        if not transcript:
            QMessageBox.warning(self, "Missing Transcript", "You must type the transcript of the audio file.")
            return

        self.btn_clone_audio.setEnabled(False)
        self.lbl_status.setText("Status: Extracting Voice from Audio (Check terminal)...")
        
        self.clone_thread = AudioCloneThread(self.base_model, self.selected_audio_file, transcript)
        self.clone_thread.finished.connect(self.on_master_voice_ready)
        self.clone_thread.error.connect(self.on_error)
        self.clone_thread.start()

    # ACTION: Load .pt (Tab C)
    def on_load_pt_clicked(self):
        if not self.selected_pt_file:
            QMessageBox.warning(self, "Missing File", "Please select a .pt file first.")
            return
        try:
            self.lbl_status.setText("Status: Loading Tensor to RAM...")
            # Added weights_only=False to allow complex custom objects to load
            self.reusable_voice_prompt = torch.load(self.selected_pt_file, weights_only=False) 
            print(f"[SYSTEM LOG] Loaded custom embedding from {self.selected_pt_file}")
            self.on_master_voice_ready(self.reusable_voice_prompt)
        except Exception as e:
            self.on_error(f"Failed to load .pt file: {e}")

    # CALLBACK: Voice is ready (Unified for Tab A, B, and C)
    def on_master_voice_ready(self, prompt_tensor):
        self.reusable_voice_prompt = prompt_tensor
        self.lbl_status.setText("Status: Master Voice Cached! Ready for Batching.")
        
        # Re-enable buttons
        self.btn_design.setEnabled(True)
        if self.selected_audio_file:
            self.btn_clone_audio.setEnabled(True)
            
        # Enable downstream actions
        self.btn_save_pt.setEnabled(True)
        self.btn_batch.setEnabled(True)

    # ACTION: Save Current Embedding
    def save_current_embedding(self):
        if self.reusable_voice_prompt is None:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Embedding", "my_custom_voice.pt", "PyTorch Tensors (*.pt)")
        if file_path:
            try:
                torch.save(self.reusable_voice_prompt, file_path)
                QMessageBox.information(self, "Saved", f"Voice embedding saved successfully to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", str(e))

    # ACTION: Run Batch
    def on_batch_clicked(self):
        script_text = self.txt_script.toPlainText().strip()
        sentences = [line.strip() for line in script_text.split('\n') if line.strip()]
        
        if not sentences:
            QMessageBox.warning(self, "Warning", "Script is empty.")
            return

        self.btn_batch.setEnabled(False)
        self.lbl_status.setText("Status: Processing Batch (Check terminal)...")
        
        if self.design_model is not None:
            print("[SYSTEM LOG] Unloading VoiceDesign model to free VRAM for batch processing...")
            del self.design_model
            self.design_model = None
            torch.cuda.empty_cache()
            self.btn_design.setEnabled(False)
            self.btn_design.setText("⚠ Model Unloaded (Restart app to redesign)")
        
        kwargs = self.get_generation_kwargs()
        batch_size = self.val_batch_size.value()
        
        self.batch_thread = BatchCloneThread(self.base_model, self.reusable_voice_prompt, sentences, kwargs, batch_size=batch_size)
        self.batch_thread.finished.connect(self.on_batch_finished)
        self.batch_thread.error.connect(self.on_error)
        self.batch_thread.start()

    def on_batch_finished(self, duration):
        self.lbl_status.setText(f"Status: Batch Generation Complete! ({duration:.2f}s) Reloading original state...")
        QMessageBox.information(self, "Success", f"Batch exported successfully to /audio_exports/")
        self.btn_batch.setEnabled(True)
        
        print("[SYSTEM LOG] Unloading Base model after batch process...")
        if getattr(self, 'base_model', None) is not None:
            del self.base_model
            self.base_model = None
            torch.cuda.empty_cache()
            
        print("[SYSTEM LOG] Reloading models for idle state...")
        self.start_model_loader()

    def on_error(self, err_msg):
        self.lbl_status.setText("Status: Error occurred.")
        QMessageBox.critical(self, "Error", str(err_msg))
        
        # Recover button states
        self.btn_design.setEnabled(True if self.design_model else False)
        if self.selected_audio_file and self.base_model:
            self.btn_clone_audio.setEnabled(True)
        self.btn_batch.setEnabled(True if self.reusable_voice_prompt else False)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VoiceStudioApp()
    window.show()
    sys.exit(app.exec())