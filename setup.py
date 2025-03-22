import os
import sys
import venv
import subprocess
import argparse
import requests
import zipfile
import shutil
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).resolve().parent
VENV_DIR = BASE_DIR / "voice-llm-venv"
PIPER_VOICE_DIR = BASE_DIR / "voice-llm-venv" / "piper"
VOSK_MODEL_DIR = BASE_DIR / "voice-llm-venv" / "vosk-model"

# Define Piper voices to download (expanded selection)
PIPER_VOICES = [
    ("en_US-lessac-medium", "https://github.com/rhasspy/piper/releases/download/v1.2.0/voice-en-us-lessac-medium.tar.gz"),
    ("en_US-libritts_r-medium", "https://github.com/rhasspy/piper/releases/download/v1.2.0/voice-en-us-libritts_r-medium.tar.gz"),
    ("en_GB-alan-medium", "https://github.com/rhasspy/piper/releases/download/v1.2.0/voice-en-gb-alan-medium.tar.gz"),
    ("es_ES-davefx-medium", "https://github.com/rhasspy/piper/releases/download/v1.2.0/voice-es-es-davefx-medium.tar.gz"),
    ("fr_FR-siwis-medium", "https://github.com/rhasspy/piper/releases/download/v1.2.0/voice-fr-fr-siwis-medium.tar.gz"),
    ("de_DE-thorsten-medium", "https://github.com/rhasspy/piper/releases/download/v1.2.0/voice-de-de-thorsten-medium.tar.gz"),
    ("it_IT-riccardo_fasol-medium", "https://github.com/rhasspy/piper/releases/download/v1.2.0/voice-it-it-riccardo_fasol-medium.tar.gz"),
    ("zh_CN-huayan-medium", "https://github.com/rhasspy/piper/releases/download/v1.2.0/voice-zh-cn-huayan-medium.tar.gz"),
]

# Define Vosk model to download
VOSK_MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
VOSK_MODEL_NAME = "vosk-model-small-en-us-0.15"

# Define dependencies
DEPENDENCIES = [
    "gradio",
    "vosk",
    "numpy",
    "pyaudio",
    "tts",
    "torch",
    "requests",
    "soundfile",
    "librosa",
    "faster-whisper",
    "argostranslate",
    "psutil",
]

def create_venv():
    """Create a virtual environment if it doesn't exist."""
    if not VENV_DIR.exists():
        print(f"Creating virtual environment at {VENV_DIR}")
        venv.create(VENV_DIR, with_pip=True)
    else:
        print(f"Virtual environment already exists at {VENV_DIR}")

def activate_venv():
    """Return the path to the pip executable in the virtual environment."""
    if sys.platform == "win32":
        pip_path = VENV_DIR / "Scripts" / "pip.exe"
    else:
        pip_path = VENV_DIR / "bin" / "pip"
    return pip_path

def install_dependencies(pip_path):
    """Install Python dependencies using the virtual environment's pip."""
    print("Installing dependencies...")
    subprocess.check_call([pip_path, "install", "--upgrade", "pip"])
    subprocess.check_call([pip_path, "install"] + DEPENDENCIES)

def download_file(url, dest_path):
    """Download a file from a URL to the destination path."""
    print(f"Downloading {url} to {dest_path}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

def extract_tar_gz(file_path, extract_path):
    """Extract a .tar.gz file to the specified path."""
    print(f"Extracting {file_path} to {extract_path}")
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=extract_path)
    os.remove(file_path)

def extract_zip(file_path, extract_path):
    """Extract a .zip file to the specified path."""
    print(f"Extracting {file_path} to {extract_path}")
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)
    os.remove(file_path)

def preload_piper_voices():
    """Download and extract Piper voices."""
    if not PIPER_VOICE_DIR.exists():
        PIPER_VOICE_DIR.mkdir(parents=True)

    for voice_name, url in PIPER_VOICES:
        voice_tar_path = PIPER_VOICE_DIR / f"{voice_name}.tar.gz"
        try:
            download_file(url, voice_tar_path)
            extract_tar_gz(voice_tar_path, PIPER_VOICE_DIR)
            # Piper extracts to a subdirectory; move files to the main directory
            extracted_dir = PIPER_VOICE_DIR / f"voice-{voice_name}"
            if extracted_dir.exists():
                for item in extracted_dir.iterdir():
                    shutil.move(str(item), PIPER_VOICE_DIR)
                extracted_dir.rmdir()
        except Exception as e:
            print(f"Failed to download or extract Piper voice {voice_name}: {e}")

def preload_vosk_model():
    """Download and extract the Vosk model."""
    if not VOSK_MODEL_DIR.exists():
        VOSK_MODEL_DIR.mkdir(parents=True)

    vosk_zip_path = VOSK_MODEL_DIR / f"{VOSK_MODEL_NAME}.zip"
    try:
        download_file(VOSK_MODEL_URL, vosk_zip_path)
        extract_zip(vosk_zip_path, VOSK_MODEL_DIR)
    except Exception as e:
        print(f"Failed to download or extract Vosk model: {e}")

def preload_argos_models():
    """Preload Argos Translate models."""
    print("Preloading Argos Translate models...")
    import argostranslate.package
    import argostranslate.translate

    translation_languages = ["es", "fr", "de", "it", "zh"]
    from_code = "en"
    for to_code in translation_languages:
        try:
            argostranslate.package.update_package_index()
            available_packages = argostranslate.package.get_available_packages()
            package = next(
                filter(
                    lambda x: x.from_code == from_code and x.to_code == to_code,
                    available_packages
                ),
                None
            )
            if package:
                package_path = package.download()
                argostranslate.package.install_from_path(package_path)
                print(f"Installed Argos Translate model for English to {to_code}")
            else:
                print(f"No Argos Translate model available for English to {to_code}")
        except Exception as e:
            print(f"Failed to preload Argos Translate model for {to_code}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Setup script for Voice-to-Voice AI tool")
    parser.add_argument("--preload", action="store_true", help="Preload models (Piper voices, Vosk model, Argos Translate models)")
    args = parser.parse_args()

    # Create and activate virtual environment
    create_venv()
    pip_path = activate_venv()

    # Install dependencies
    install_dependencies(pip_path)

    # Preload models if requested
    if args.preload:
        preload_piper_voices()
        preload_vosk_model()
        preload_argos_models()

    print("Setup complete! To run the application:")
    if sys.platform == "win32":
        print(f"  cd {BASE_DIR}")
        print(f"  {VENV_DIR}\\Scripts\\activate")
        print("  python voice_llm.py")
    else:
        print(f"  cd {BASE_DIR}")
        print(f"  source {VENV_DIR}/bin/activate")
        print("  python voice_llm.py")

if __name__ == "__main__":
    main()