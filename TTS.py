import sys
import os
import time
import torch
import soundfile as sf
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QTextEdit, 
                             QPushButton, QGroupBox, QFormLayout, QDoubleSpinBox, 
                             QSpinBox, QCheckBox, QMessageBox, QScrollArea)
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
            
            # Flash Attention 2 with Native PyTorch SDPA Fallback
            try:
                import flash_attn
                attn_impl = "flash_attention_2"
                print("[SYSTEM LOG] Flash Attention 2 module found. Using flash_attention_2.")
            except ImportError:
                attn_impl = "sdpa"
                print("[SYSTEM LOG] Flash Attention 2 NOT found. Falling back to PyTorch SDPA.")

            print(f"\n[SYSTEM LOG] Loading Qwen3-TTS-12Hz-1.7B-VoiceDesign...")
            print(f"[SYSTEM LOG] Dtype: bfloat16 | TF32: Enabled | Attn: {attn_impl}")
            print("[SYSTEM LOG] (If weights are missing, Hugging Face will download them now...)")
            
            design_model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                device_map="cuda:0",
                dtype=torch.bfloat16,
                attn_implementation=attn_impl
            )
            print("[SYSTEM LOG] VoiceDesign model loaded successfully.")

            print(f"\n[SYSTEM LOG] Loading Qwen3-TTS-12Hz-1.7B-Base...")
            base_model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                device_map="cuda:0",
                dtype=torch.bfloat16,
                attn_implementation=attn_impl
            )
            print("[SYSTEM LOG] Base model loaded successfully.")

            # Torch Compile (Graph Optimization)
            print("\n[SYSTEM LOG] Attempting torch.compile for graph optimization...")
            try:
                design_model = torch.compile(design_model, mode="reduce-overhead")
                base_model = torch.compile(base_model, mode="reduce-overhead")
                print("[SYSTEM LOG] Torch compilation successful.")
            except Exception as e:
                print(f"[SYSTEM LOG] Torch compile skipped/failed (Safe to ignore): {e}")

            self.finished.emit(design_model, base_model)
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
            
            ref_audio = ref_wavs[0]
            sf.write("reference_voice.wav", ref_audio, sr)
            print("[GENERATION LOG] Saved master reference to 'reference_voice.wav'.")

            print("[GENERATION LOG] Extracting acoustic tensors (voice_clone_prompt) for cache...")
            t0 = time.time()
            prompt = self.base_model.create_voice_clone_prompt(
                ref_audio=(ref_audio, sr),
                ref_text=self.ref_text,
            )
            print(f"[GENERATION LOG] Feature extraction complete in {time.time()-t0:.2f}s.")
            
            self.finished.emit(prompt)
        except Exception as e:
            print(f"\n[ERROR] Voice design failed: {e}")
            self.error.emit(str(e))


class BatchCloneThread(QThread):
    finished = pyqtSignal(float)
    error = pyqtSignal(str)

    def __init__(self, base_model, prompt, sentences, kwargs):
        super().__init__()
        self.base_model = base_model
        self.prompt = prompt
        self.sentences = sentences
        self.kwargs = kwargs

    def run(self):
        try:
            print("\n" + "="*50)
            print(f"[BATCH LOG] Starting batch generation for {len(self.sentences)} audio lines...")
            print("="*50)
            print(f"Parameters: {self.kwargs}")
            
            languages = ["English"] * len(self.sentences)
            
            t0 = time.time()
            wavs, sr = self.base_model.generate_voice_clone(
                text=self.sentences,
                language=languages,
                voice_clone_prompt=self.prompt,
                **self.kwargs
            )
            t1 = time.time()
            
            print(f"\n[BATCH LOG] Generation completed in {t1-t0:.2f}s.")
            print("[BATCH LOG] Exporting payload...")
            
            os.makedirs("audio_exports", exist_ok=True)
            for i, w in enumerate(wavs):
                filename = f"audio_exports/line_{i+1:03d}.wav"
                sf.write(filename, w, sr)
                print(f" -> Saved: {filename}")
                
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
        self.resize(1000, 800)

        # State Variables
        self.design_model = None
        self.base_model = None
        self.reusable_voice_prompt = None
        
        self.init_ui()
        self.start_model_loader()

    def init_ui(self):
        # Main Widget and Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- LEFT PANEL: Parameters ---
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

        # --- RIGHT PANEL: Execution ---
        exec_layout = QVBoxLayout()

        self.lbl_status = QLabel("Status: Waiting to load models... (Check Terminal)")
        self.lbl_status.setStyleSheet("color: blue; font-weight: bold;")
        exec_layout.addWidget(self.lbl_status)

        # 1. Voice Design
        design_group = QGroupBox("1. Voice Design (Create Custom Persona)")
        design_vbox = QVBoxLayout()
        
        design_vbox.addWidget(QLabel("Voice Design Prompt:"))
        self.txt_instruct = QLineEdit("A clear, professional female voice speaking calmly and at a measured pace.")
        design_vbox.addWidget(self.txt_instruct)

        design_vbox.addWidget(QLabel("Reference Text (Master clone sample):"))
        self.txt_ref = QLineEdit("Hello, welcome to the audio generation system. Let me demonstrate my tone.")
        design_vbox.addWidget(self.txt_ref)

        self.btn_design = QPushButton("Generate Master Voice")
        self.btn_design.setEnabled(False)
        self.btn_design.clicked.connect(self.on_design_clicked)
        design_vbox.addWidget(self.btn_design)
        
        design_group.setLayout(design_vbox)
        exec_layout.addWidget(design_group)

        # 2. Batch Processing
        batch_group = QGroupBox("2. Batch Audio Generation (Reuses Master Voice)")
        batch_vbox = QVBoxLayout()
        
        batch_vbox.addWidget(QLabel("Audio Script (One sentence per line):"))
        self.txt_script = QTextEdit()
        self.txt_script.setPlainText("This is the first generated line.\nAnd this is the second line for the batch process.\nEverything is working perfectly.")
        batch_vbox.addWidget(self.txt_script)

        self.btn_batch = QPushButton("Batch Generate Script")
        self.btn_batch.setEnabled(False)
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

    # --- Worker Thread Triggers & Callbacks ---
    
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

    def on_design_clicked(self):
        self.btn_design.setEnabled(False)
        self.lbl_status.setText("Status: Designing Voice (Check terminal)...")
        
        kwargs = self.get_generation_kwargs()
        instruct = self.txt_instruct.text()
        ref_text = self.txt_ref.text()
        
        self.design_thread = VoiceDesignThread(self.design_model, self.base_model, instruct, ref_text, kwargs)
        self.design_thread.finished.connect(self.on_design_finished)
        self.design_thread.error.connect(self.on_error)
        self.design_thread.start()

    def on_design_finished(self, prompt_tensor):
        self.reusable_voice_prompt = prompt_tensor
        self.lbl_status.setText("Status: Voice Designed & Cached. Ready for Batching.")
        self.btn_design.setEnabled(True)
        self.btn_batch.setEnabled(True)

    def on_batch_clicked(self):
        script_text = self.txt_script.toPlainText().strip()
        sentences = [line.strip() for line in script_text.split('\n') if line.strip()]
        
        if not sentences:
            QMessageBox.warning(self, "Warning", "Script is empty.")
            return

        self.btn_batch.setEnabled(False)
        self.lbl_status.setText("Status: Processing Batch (Check terminal)...")
        
        kwargs = self.get_generation_kwargs()
        
        self.batch_thread = BatchCloneThread(self.base_model, self.reusable_voice_prompt, sentences, kwargs)
        self.batch_thread.finished.connect(self.on_batch_finished)
        self.batch_thread.error.connect(self.on_error)
        self.batch_thread.start()

    def on_batch_finished(self, duration):
        self.lbl_status.setText(f"Status: Batch Generation Complete! ({duration:.2f}s)")
        QMessageBox.information(self, "Success", f"Batch exported successfully to /audio_exports/")
        self.btn_batch.setEnabled(True)

    def on_error(self, err_msg):
        self.lbl_status.setText("Status: Error occurred.")
        QMessageBox.critical(self, "Error", str(err_msg))
        self.btn_design.setEnabled(True if self.design_model else False)
        self.btn_batch.setEnabled(True if self.reusable_voice_prompt else False)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VoiceStudioApp()
    window.show()
    sys.exit(app.exec())