import gradio as gr
import vosk
import numpy as np
import pyaudio
from TTS.api import TTS
from TTS.utils.radam import RAdam
import torch
import torch.serialization
import requests
import json
import os
import whisper
import soundfile as sf
import tempfile
import threading
import time
import librosa
import re
from piper import PiperVoice
import subprocess
import wave
import psutil
import logging
import sys
from faster_whisper import WhisperModel  # For Faster Whisper STT
import argostranslate.package
import argostranslate.translate  # For offline translation

# Redirect stderr to stdout for logging
sys.stderr = sys.stdout

# Set up logging to capture all output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/sun/voice-llm-venv/voice_llm.log', mode='a'),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logging.info("Logging initialized successfully.")

# Suppress ALSA/JACK warnings
os.environ["ALSA_LOG_LEVEL"] = "0"
os.environ["PULSE_LOG"] = "0"
os.environ["PA_ALSA_PLUGHW"] = "1"  # Force ALSA to use PulseAudio
os.environ["ALSA_NO_JACK"] = "1"  # Suppress JACK errors

# Configure ALSA to use PulseAudio
alsa_config = """
pcm.!default {
    type plug
    slave {
        pcm "pulse"
    }
}
ctl.!default {
    type pulse
}
pcm.pulse {
    type pulse
}
ctl.pulse {
    type pulse
}
"""
with open(os.path.expanduser("~/.asoundrc"), "w") as f:
    f.write(alsa_config)

# Allowlist RAdam for PyTorch loading
torch.serialization.add_safe_globals([RAdam])

OLLAMA_URL = "http://localhost:11434/api/generate"

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
PORCUPINE_RATE = 16000
DEVICE_INDEX = 16  # Use Device 16: pulse for input
OUTPUT_DEVICE_INDEX = None  # Will be set dynamically

conversation_history = []
stop_flag = False
wav_files = []  # Track generated WAV files

# Device selection for models (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

def log_gpu_memory():
    """Log GPU memory usage if using CUDA."""
    if device == "cuda":
        allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB
        logging.info(f"GPU Memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

# Check if PulseAudio is running
def is_pulseaudio_running():
    try:
        result = subprocess.run(["pulseaudio", "--check"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except Exception as e:
        logging.error(f"Error checking PulseAudio status: {e}")
        return False

pulseaudio_available = is_pulseaudio_running()
logging.info(f"PulseAudio available: {pulseaudio_available}")

# List available audio devices and store their info
def list_audio_devices():
    p = pyaudio.PyAudio()
    devices = []
    output_devices = []
    default_output_device = None
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        device_name = device_info['name']
        if "pulse" in device_name.lower() and not pulseaudio_available:
            continue
        devices.append({
            "index": i,
            "name": device_name,
            "input_channels": device_info['maxInputChannels'],
            "output_channels": device_info['maxOutputChannels'],
            "default_sample_rate": int(device_info.get('defaultSampleRate', 44100))  # Ensure integer
        })
        logging.info(f"Device {i}: {device_name}, Input Channels: {device_info['maxInputChannels']}, Output Channels: {device_info['maxOutputChannels']}")
        if device_info['maxOutputChannels'] > 0:
            output_devices.append(f"{i}: {device_name}")
            if i == p.get_default_output_device_info()['index']:
                default_output_device = f"{i}: {device_name}"
    p.terminate()
    return devices, output_devices, default_output_device

devices, output_devices, default_output_device = list_audio_devices()
if default_output_device:
    OUTPUT_DEVICE_INDEX = int(default_output_device.split(":")[0])
    logging.info(f"Set default output device to: {default_output_device}")
else:
    OUTPUT_DEVICE_INDEX = 16  # Fallback to Device 16: pulse
    logging.warning("No default output device found. Falling back to Device 16: pulse")

# Vosk model setup (CPU only)
VOSK_MODEL_DIR = "/home/sun/voice-llm-venv/vosk-model/"
vosk_models = [d for d in os.listdir(VOSK_MODEL_DIR) if os.path.isdir(os.path.join(VOSK_MODEL_DIR, d))]
DEFAULT_VOSK_MODEL = "vosk-model-small-en-us-0.15"
current_vosk_model_name = None
current_vosk_model = None

# Whisper models
whisper_models = ["tiny", "base", "small", "medium", "large"]
current_whisper_model_name = "tiny"
whisper_model = None  # Lazy load

# Faster Whisper setup
faster_whisper_models = ["tiny", "base", "small", "medium", "large"]
current_faster_whisper_model_name = "tiny"
faster_whisper_model = None  # Lazy load

# TTS engines
tts_engines = ["coqui", "piper", "espeak", "openutau"]
current_tts_engine = "piper"

# Coqui TTS
tts_models = [
    "tts_models/en/ljspeech/glow-tts",
    "tts_models/en/vctk/vits",
    "tts_models/es/mai/tacotron2-DDC",  # Spanish
    "tts_models/fr/mai/tacotron2-DDC",  # French
    "tts_models/de/thorsten/tacotron2-DDC"  # German
]
current_tts_model_name = None
coqui_tts = None  # Lazy load
vctk_speakers = ["p225", "p226", "p227", "p228"]

# Piper TTS
PIPER_VOICE_DIR = "/home/sun/voice-llm-venv/piper/"
piper_voices = [f for f in os.listdir(PIPER_VOICE_DIR) if f.endswith('.onnx')]
if not piper_voices:
    logging.warning("No Piper voice models found in %s. Please download voice models.", PIPER_VOICE_DIR)
    piper_voices = ["en_US-lessac-medium.onnx"]
current_piper_voice = piper_voices[0]
piper_voice = None  # Lazy load

# eSpeak voices
espeak_voices = ["en-us", "en-uk", "es", "fr", "de", "it", "zh"]
current_espeak_voice = "en-us"

# OpenUtau setup
openutau_path = "/home/sun/voice-llm-venv/openutau/OpenUtau"  # Adjust path after installation
voicebank_path = "/home/sun/voice-llm-venv/openutau/voicebanks/teto"  # Adjust after voicebank setup

# Offline Translation setup
translation_languages = ["es", "fr", "de", "it", "zh"]
current_translation_lang = "es"
translation_pairs = {}

def install_translation_packages():
    from_code = "en"
    for to_code in translation_languages:
        try:
            argostranslate.package.update_package_index()
            available_packages = argostranslate.package.get_available_packages()
            package = next(
                filter(lambda x: x.from_code == from_code and x.to_code == to_code, available_packages), None
            )
            if package and not any(pkg.from_code == from_code and pkg.to_code == to_code for pkg in argostranslate.package.get_installed_packages()):
                logging.info(f"Installing translation package for English to {to_code}")
                package_path = package.download()
                argostranslate.package.install_from_path(package_path)
            else:
                logging.info(f"Translation package for English to {to_code} already installed")
        except Exception as e:
            logging.error(f"Failed to install translation package for {to_code}: {e}")

try:
    install_translation_packages()
except Exception as e:
    logging.error(f"Error during translation package installation: {e}")

for to_code in translation_languages:
    try:
        package = next(
            filter(lambda x: x.from_code == "en" and x.to_code == to_code, argostranslate.package.get_installed_packages()), None
        )
        if package:
            translation_pairs[to_code] = package
            logging.info(f"Initialized offline translation for English to {to_code}")
    except Exception as e:
        logging.error(f"Failed to initialize translation for English to {to_code}: {e}")

# Ollama model options
def get_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        return [model["name"] for model in json.loads(response.text)["models"]]
    except Exception:
        return ["smollm2:135m"]

ollama_models = get_ollama_models()

# Model loading/unloading functions
def load_vosk_model(model_name):
    global current_vosk_model, current_vosk_model_name
    if current_vosk_model_name == model_name:
        return
    if current_vosk_model:
        logging.info(f"Unloading Vosk model: {current_vosk_model_name}")
        current_vosk_model = None
    logging.info(f"Loading Vosk model: {model_name}")
    current_vosk_model = vosk.Model(os.path.join(VOSK_MODEL_DIR, model_name))
    current_vosk_model_name = model_name
    mem_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    logging.info(f"Memory usage after loading Vosk model: {mem_usage:.2f} MB")

def unload_vosk_model():
    global current_vosk_model, current_vosk_model_name
    if current_vosk_model:
        logging.info(f"Unloading Vosk model: {current_vosk_model_name}")
        current_vosk_model = None
        current_vosk_model_name = None
        mem_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        logging.info(f"Memory usage after unloading Vosk model: {mem_usage:.2f} MB")

def load_whisper_model(model_name):
    global whisper_model, current_whisper_model_name
    if current_whisper_model_name == model_name and whisper_model:
        return
    if whisper_model:
        logging.info(f"Unloading Whisper model: {current_whisper_model_name}")
        whisper_model = None
        torch.cuda.empty_cache()
    logging.info(f"Loading Whisper model: {model_name}")
    try:
        whisper_model = whisper.load_model(model_name, device=device)
    except torch.cuda.OutOfMemoryError:
        logging.warning("GPU out of memory. Falling back to CPU.")
        whisper_model = whisper.load_model(model_name, device="cpu")
    current_whisper_model_name = model_name
    mem_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    logging.info(f"Memory usage after loading Whisper model: {mem_usage:.2f} MB")
    log_gpu_memory()

def unload_whisper_model():
    global whisper_model, current_whisper_model_name
    if whisper_model:
        logging.info(f"Unloading Whisper model: {current_whisper_model_name}")
        whisper_model = None
        current_whisper_model_name = None
        torch.cuda.empty_cache()
        log_gpu_memory()

def load_faster_whisper_model(model_name):
    global faster_whisper_model, current_faster_whisper_model_name
    if current_faster_whisper_model_name == model_name and faster_whisper_model:
        return
    if faster_whisper_model:
        logging.info(f"Unloading Faster Whisper model: {current_faster_whisper_model_name}")
        faster_whisper_model = None
        torch.cuda.empty_cache()
    logging.info(f"Loading Faster Whisper model: {model_name}")
    try:
        faster_whisper_model = WhisperModel(model_name, device=device, compute_type="float16")
    except torch.cuda.OutOfMemoryError:
        logging.warning("GPU out of memory. Falling back to CPU.")
        faster_whisper_model = WhisperModel(model_name, device="cpu", compute_type="int8")
    current_faster_whisper_model_name = model_name
    mem_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    logging.info(f"Memory usage after loading Faster Whisper model: {mem_usage:.2f} MB")
    log_gpu_memory()

def unload_faster_whisper_model():
    global faster_whisper_model, current_faster_whisper_model_name
    if faster_whisper_model:
        logging.info(f"Unloading Faster Whisper model: {current_faster_whisper_model_name}")
        faster_whisper_model = None
        current_faster_whisper_model_name = None
        torch.cuda.empty_cache()
        log_gpu_memory()

def load_tts_model(model_name, speaker_id=None):
    global coqui_tts, current_tts_model_name
    if current_tts_model_name == model_name and coqui_tts:
        return
    if coqui_tts:
        logging.info(f"Unloading TTS model: {current_tts_model_name}")
        coqui_tts = None
        torch.cuda.empty_cache()
    logging.info(f"Loading TTS model: {model_name}")
    try:
        coqui_tts = TTS(model_name=model_name, progress_bar=False, gpu=(device == "cuda"))
    except torch.cuda.OutOfMemoryError:
        logging.warning("GPU out of memory. Falling back to CPU.")
        coqui_tts = TTS(model_name=model_name, progress_bar=False, gpu=False)
    current_tts_model_name = model_name
    mem_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    logging.info(f"Memory usage after loading TTS model: {mem_usage:.2f} MB")
    log_gpu_memory()

def unload_tts_model():
    global coqui_tts, current_tts_model_name
    if coqui_tts:
        logging.info(f"Unloading TTS model: {current_tts_model_name}")
        coqui_tts = None
        current_tts_model_name = None
        torch.cuda.empty_cache()
        log_gpu_memory()

def load_piper_voice(voice_name):
    global piper_voice, current_piper_voice
    if current_piper_voice == voice_name and piper_voice:
        return
    if piper_voice:
        logging.info(f"Unloading Piper voice: {current_piper_voice}")
        piper_voice = None
    logging.info(f"Loading Piper voice: {voice_name}")
    piper_voice = PiperVoice.load(os.path.join(PIPER_VOICE_DIR, voice_name))
    current_piper_voice = voice_name
    mem_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    logging.info(f"Memory usage after loading Piper voice: {mem_usage:.2f} MB")

def unload_piper_voice():
    global piper_voice, current_piper_voice
    if piper_voice:
        logging.info(f"Unloading Piper voice: {current_piper_voice}")
        piper_voice = None
        current_piper_voice = None
        mem_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        logging.info(f"Memory usage after unloading Piper voice: {mem_usage:.2f} MB")

def load_openutau():
    global openutau_path, voicebank_path
    if not os.path.exists(openutau_path):
        logging.error(f"OpenUtau not found at {openutau_path}. Please install it.")
        raise FileNotFoundError(f"OpenUtau binary not found at {openutau_path}")
    if not os.path.exists(voicebank_path):
        logging.error(f"Voicebank not found at {voicebank_path}. Please install a voicebank.")
        raise FileNotFoundError(f"Voicebank not found at {voicebank_path}")

def synthesize_openutau(text, output_path):
    logging.info("Starting OpenUtau synthesis...")
    ust_content = f"""[#0000]
Length=480
Lyric={text}
NoteNum=60
[#TRACKEND]"""
    with tempfile.NamedTemporaryFile(suffix=".ust", delete=False) as ust_file:
        ust_file.write(ust_content.encode())
        ust_file_path = ust_file.name
    cmd = [openutau_path, "-i", ust_file_path, "-o", output_path, "-v", voicebank_path]
    subprocess.run(cmd, check=True)
    os.remove(ust_file_path)
    logging.info("OpenUtau synthesis completed.")

def set_stop_flag():
    global stop_flag
    stop_flag = True
    return "Processing interrupted."

def clear_history():
    global conversation_history
    conversation_history = []
    return "Conversation history cleared.", None, ""

def timeout(func, args=(), kwargs={}, timeout_duration=30, default=None):
    result = [default]
    exception = [None]
    stop_event = threading.Event()

    def wrapper():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e
        finally:
            stop_event.set()

    thread = threading.Thread(target=wrapper)
    thread.daemon = True
    thread.start()
    stop_event.wait(timeout_duration)

    if not stop_event.is_set():
        logging.error(f"Function {func.__name__} timed out after {timeout_duration} seconds")
        return default
    if exception[0]:
        raise exception[0]
    return result[0]

def make_audio_sing(audio_data, sample_rate, pitch_shift=1.2, tempo_shift=0.8):
    try:
        audio_data = audio_data.astype(np.float64)
        audio_data = librosa.effects.pitch_shift(audio_data, sr=sample_rate, n_steps=pitch_shift * 4)
        audio_data = librosa.effects.time_stretch(audio_data, rate=tempo_shift)
        return np.clip(audio_data, -32768, 32767).astype(np.int16)
    except Exception as e:
        logging.error(f"Error in make_audio_sing: {str(e)}")
        return audio_data

def talk_to_llm(audio, amplitude_factor=16.0, ollama_model="smollm2:135m", 
                stt_architecture="whisper", vosk_model_name=DEFAULT_VOSK_MODEL, 
                whisper_model_name="tiny", faster_whisper_model_name="tiny",
                tts_model_name="tts_models/en/ljspeech/glow-tts", 
                tts_engine="piper", espeak_voice="en-us", piper_voice_name=current_piper_voice, 
                speaker_id=None, output_device=None, translate_to=None, mode="talk"):
    global stop_flag, current_tts_engine, wav_files
    stop_flag = False

    # Load STT model
    if stt_architecture == "vosk":
        unload_whisper_model()
        unload_faster_whisper_model()
        load_vosk_model(vosk_model_name)
    elif stt_architecture == "whisper":
        unload_vosk_model()
        unload_faster_whisper_model()
        load_whisper_model(whisper_model_name)
    elif stt_architecture == "faster_whisper":
        unload_vosk_model()
        unload_whisper_model()
        load_faster_whisper_model(faster_whisper_model_name)

    # Load TTS engine
    if tts_engine == "coqui":
        unload_piper_voice()
        load_tts_model(tts_model_name, speaker_id)
        current_tts_engine = "coqui"
    elif tts_engine == "piper":
        unload_tts_model()
        load_piper_voice(piper_voice_name)
        current_tts_engine = "piper"
    elif tts_engine == "espeak":
        unload_tts_model()
        unload_piper_voice()
        current_tts_engine = "espeak"
        current_espeak_voice = espeak_voice
    elif tts_engine == "openutau":
        unload_tts_model()
        unload_piper_voice()
        load_openutau()
        current_tts_engine = "openutau"

    if audio is None:
        return "Error: No audio captured.", None

    sample_rate, audio = audio if isinstance(audio, tuple) else (RATE, audio)
    logging.info(f"Sample rate: {sample_rate}, Audio shape: {audio.shape}")
    audio = np.mean(audio, axis=1) if len(audio.shape) > 1 else audio
    if sample_rate != PORCUPINE_RATE:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sample_rate, target_sr=PORCUPINE_RATE)
        sample_rate = PORCUPINE_RATE
    audio = audio * amplitude_factor
    audio = np.clip(audio, -32768, 32767)
    logging.info(f"Audio max amplitude: {np.max(np.abs(audio))}")
    logging.info(f"Audio length (samples): {len(audio)}")

    if stop_flag:
        return "Interrupted during transcription.", None

    # Transcribe audio
    if stt_architecture == "vosk":
        audio_data = (audio * 32768).astype(np.int16).tobytes()
        rec = vosk.KaldiRecognizer(current_vosk_model, sample_rate)
        rec.SetWords(True)
        rec.SetMaxAlternatives(10)
        text = json.loads(rec.Result()).get("text", "") if rec.AcceptWaveform(audio_data) else json.loads(rec.PartialResult()).get("partial", "")
    elif stt_architecture == "whisper":
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            sf.write(temp_file.name, audio, sample_rate)
            wav_files.append(temp_file.name)
            text = whisper_model.transcribe(temp_file.name, language="en")["text"]
    elif stt_architecture == "faster_whisper":
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            sf.write(temp_file.name, audio, sample_rate)
            wav_files.append(temp_file.name)
            segments, _ = faster_whisper_model.transcribe(temp_file.name, language="en")
            text = " ".join(segment.text for segment in segments)
    else:
        return "Error: Unsupported STT architecture.", None

    if not text:
        return "No speech detected.", None

    logging.info(f"Transcribed text: {text}")
    if stop_flag:
        return "Interrupted after transcription.", None

    # Unload STT models
    if stt_architecture == "vosk":
        unload_vosk_model()
    elif stt_architecture == "whisper":
        unload_whisper_model()
    elif stt_architecture == "faster_whisper":
        unload_faster_whisper_model()

    # Translate if requested
    if translate_to and translate_to != "None":
        package = translation_pairs.get(translate_to)
        translated = argostranslate.translate.translate(text, from_code="en", to_code=translate_to) if package else text
        logging.info(f"Translated text ({translate_to}): {translated}")
    else:
        translated = text

    conversation_history.append({"role": "user", "content": text})
    history_text = "\n".join([f"{entry['role']}: {entry['content']}" for entry in conversation_history[-2:]])
    prompt = f"You are a helpful voice assistant. Focus on the user's latest message and provide a concise, relevant response. Here is the recent conversation history for context:\n{history_text}\nRespond to the latest user message."
    response = requests.post(OLLAMA_URL, json={"model": ollama_model, "prompt": prompt, "stream": False})
    logging.info(f"Ollama response status: {response.status_code}")
    logging.info(f"Ollama response text: {response.text}")

    if stop_flag:
        return "Interrupted after Ollama response.", None

    try:
        response_text = json.loads(response.text)["response"]
    except (KeyError, json.JSONDecodeError) as e:
        return f"Error: Ollama API failed - {str(e)}.", None

    response_text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', response_text)
    conversation_history.append({"role": "assistant", "content": response_text})

    output_path = "output.wav"
    if stop_flag:
        return "Interrupted before TTS generation.", None

    tts_text = translated if translate_to and translate_to != "None" else response_text

    def synthesize_audio():
        if tts_engine == "coqui":
            logging.info("Starting Coqui TTS synthesis...")
            coqui_tts.tts_to_file(text=tts_text, file_path=output_path, speaker=speaker_id if tts_model_name == "tts_models/en/vctk/vits" else None)
            logging.info("Coqui TTS synthesis completed.")
        elif tts_engine == "piper":
            logging.info("Starting Piper TTS synthesis...")
            with wave.open(output_path, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(22050)
                piper_voice.synthesize(text=tts_text, wav_file=wav_file)
            logging.info("Piper TTS synthesis completed.")
        elif tts_engine == "espeak":
            logging.info("Starting eSpeak TTS synthesis...")
            subprocess.run(["espeak", "-v", espeak_voice, "-p", "80", "-s", "120", "-w", output_path, tts_text] if mode == "sing" else ["espeak", "-v", espeak_voice, "-w", output_path, tts_text], check=True)
            logging.info("eSpeak TTS synthesis completed.")
        elif tts_engine == "openutau":
            synthesize_openutau(tts_text, output_path)

    timeout(synthesize_audio, timeout_duration=30, default=False)
    if not os.path.exists(output_path):
        return "Error: TTS output file was not generated.", None
    wav_files.append(output_path)
    mem_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    logging.info(f"Memory usage after TTS synthesis: {mem_usage:.2f} MB")
    log_gpu_memory()

    if mode == "sing" and tts_engine in ["coqui", "piper"]:
        audio_data, original_rate = sf.read(output_path)
        audio_data = make_audio_sing(audio_data, original_rate)
        sf.write(output_path, audio_data, original_rate)

    if tts_engine == "coqui":
        unload_tts_model()
    elif tts_engine == "piper":
        unload_piper_voice()

    output_device_index = int(output_device.split(":")[0]) if output_device else OUTPUT_DEVICE_INDEX
    device_info = next((d for d in devices if d["index"] == output_device_index), None)
    target_rate = device_info['default_sample_rate'] if device_info else 44100

    audio_data, original_rate = sf.read(output_path)
    channels = 1 if len(audio_data.shape) == 1 else audio_data.shape[1]
    if audio_data.dtype != np.int16:
        audio_data = (audio_data * 32767).astype(np.int16)
    if channels > 1:
        audio_data = np.mean(audio_data, axis=1).astype(np.int16)
    if original_rate != target_rate:
        logging.info(f"Resampling audio from {original_rate} Hz to {target_rate} Hz")
        audio_data = librosa.resample(audio_data.astype(np.float32), orig_sr=original_rate, target_sr=target_rate).astype(np.int16)

    p = pyaudio.PyAudio()
    stream_out = p.open(
        format=FORMAT,
        channels=1,
        rate=int(target_rate),
        output=True,
        output_device_index=output_device_index,
        frames_per_buffer=CHUNK
    )
    for i in range(0, len(audio_data), CHUNK):
        if stop_flag or not stream_out.is_active():
            break
        stream_out.write(audio_data[i:i + CHUNK].tobytes())
    stream_out.stop_stream()
    stream_out.close()
    p.terminate()

    return response_text, output_path

def clear_all_wav_files():
    global wav_files
    for wav_file in wav_files[:]:
        try:
            if os.path.exists(wav_file):
                os.remove(wav_file)
                logging.info(f"Deleted WAV file: {wav_file}")
        except Exception as e:
            logging.error(f"Error deleting WAV file {wav_file}: {e}")
    wav_files = []
    return "All WAV files cleared."

def clear_wav_file(wav_file):
    global wav_files
    try:
        if os.path.exists(wav_file):
            os.remove(wav_file)
            logging.info(f"Deleted WAV file: {wav_file}")
            wav_files.remove(wav_file)
        return f"Cleared WAV file: {wav_file}"
    except Exception as e:
        logging.error(f"Error deleting WAV file {wav_file}: {e}")
        return f"Error clearing WAV file {wav_file}: {e}"

# Gradio interface
with gr.Blocks() as interface:
    gr.Markdown("Configure settings, click 'Record', speak, and hear the response! (Offline after downloads)")
    with gr.Row():
        audio_input = gr.Audio(sources=["microphone"], type="numpy", label="Speak Here")
        output_device_dropdown = gr.Dropdown(choices=output_devices, value=default_output_device, label="Output Device")
    with gr.Row():
        response_text = gr.Textbox(label="Response", lines=10)
    with gr.Row():
        amplitude_slider = gr.Slider(minimum=1, maximum=32, value=16, step=1, label="Amplitude Factor")
        ollama_dropdown = gr.Dropdown(choices=ollama_models, value="smollm2:135m", label="Ollama Model")
    with gr.Row():
        stt_dropdown = gr.Dropdown(choices=["vosk", "whisper", "faster_whisper"], value="whisper", label="STT Architecture")
        vosk_dropdown = gr.Dropdown(choices=vosk_models, value=DEFAULT_VOSK_MODEL, label="Vosk Model", visible=True)
        whisper_dropdown = gr.Dropdown(choices=whisper_models, value="tiny", label="Whisper Model", visible=True)
        faster_whisper_dropdown = gr.Dropdown(choices=faster_whisper_models, value="tiny", label="Faster Whisper Model", visible=False)
    with gr.Row():
        tts_engine_dropdown = gr.Dropdown(choices=tts_engines, value="piper", label="TTS Engine")
        tts_dropdown = gr.Dropdown(choices=tts_models, value="tts_models/en/ljspeech/glow-tts", label="Coqui TTS Voice", visible=False)
        speaker_dropdown = gr.Dropdown(choices=vctk_speakers, value=None, label="Speaker (VCTK/VITS)", visible=False)
        espeak_voice_dropdown = gr.Dropdown(choices=espeak_voices, value="en-us", label="eSpeak Voice", visible=False)
        piper_voice_dropdown = gr.Dropdown(choices=piper_voices, value=current_piper_voice, label="Piper Voice", visible=True)
    with gr.Row():
        translate_dropdown = gr.Dropdown(choices=["None"] + translation_languages, value="None", label="Translate To (TTS)")
        mode_dropdown = gr.Dropdown(choices=["talk", "sing"], value="talk", label="Mode")
        status_text = gr.Textbox(label="Status")
    audio_output = gr.Audio(type="filepath", label="Listen")
    history_display = gr.Textbox(label="Conversation History", value="", interactive=False)
    with gr.Row():
        submit_button = gr.Button("Submit", variant="primary")
        interrupt_button = gr.Button("Interrupt", variant="stop")
        clear_history_button = gr.Button("Clear History", variant="secondary")
    with gr.Row():
        clear_all_wav_button = gr.Button("Clear All WAV Files")
        wav_file_dropdown = gr.Dropdown(choices=[], label="Select WAV File to Clear")
        clear_wav_button = gr.Button("Clear Selected WAV File")

    def update_stt_visibility(stt_architecture):
        return (
            gr.update(visible=stt_architecture == "vosk"),
            gr.update(visible=stt_architecture == "whisper"),
            gr.update(visible=stt_architecture == "faster_whisper")
        )

    stt_dropdown.change(fn=update_stt_visibility, inputs=stt_dropdown, outputs=[vosk_dropdown, whisper_dropdown, faster_whisper_dropdown])

    def update_tts_visibility(tts_engine):
        global current_tts_engine
        current_tts_engine = tts_engine
        return (
            gr.update(visible=tts_engine == "coqui"),
            gr.update(visible=tts_engine == "coqui" and tts_dropdown.value == "tts_models/en/vctk/vits"),
            gr.update(visible=tts_engine == "espeak"),
            gr.update(visible=tts_engine == "piper")
        )

    tts_engine_dropdown.change(fn=update_tts_visibility, inputs=tts_engine_dropdown, outputs=[tts_dropdown, speaker_dropdown, espeak_voice_dropdown, piper_voice_dropdown])

    def update_speaker_visibility(tts_model):
        return gr.update(visible=tts_model == "tts_models/en/vctk/vits")

    tts_dropdown.change(fn=update_speaker_visibility, inputs=tts_dropdown, outputs=speaker_dropdown)

    def update_wav_dropdown():
        return gr.update(choices=wav_files)

    interrupt_button.click(fn=set_stop_flag, outputs=response_text)
    clear_history_button.click(fn=clear_history, outputs=[response_text, audio_output, history_display])
    submit_button.click(
        fn=talk_to_llm,
        inputs=[audio_input, amplitude_slider, ollama_dropdown, stt_dropdown, vosk_dropdown, whisper_dropdown, faster_whisper_dropdown, tts_dropdown, tts_engine_dropdown, espeak_voice_dropdown, piper_voice_dropdown, speaker_dropdown, output_device_dropdown, translate_dropdown, mode_dropdown],
        outputs=[response_text, audio_output]
    ).then(fn=lambda: "\n".join([f"{entry['role']}: {entry['content']}" for entry in conversation_history]), outputs=history_display).then(
        fn=update_wav_dropdown, outputs=wav_file_dropdown
    )
    clear_all_wav_button.click(fn=clear_all_wav_files, outputs=response_text).then(
        fn=update_wav_dropdown, outputs=wav_file_dropdown
    )
    clear_wav_button.click(fn=clear_wav_file, inputs=wav_file_dropdown, outputs=response_text).then(
        fn=update_wav_dropdown, outputs=wav_file_dropdown
    )

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)