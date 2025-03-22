# VoiceLLM

An offline voice-to-voice AI assistant with speech-to-text, text-to-speech, translation, and singing capabilities.

## Setup
1. Clone the repository:
   bash
   git clone https://github.com/FortezzaJ/VoiceLLM.git
   cd VoiceLLM

2. Run the setup script:
   bash
   python setup.py --preload

3. Activate the virtual environment:
   bash
   source voice-llm-venv/bin/activate

4. Launch the app:
   bash
   python voice_llm.py

   Requirements

    Python 3.10
    Linux
    GPU (optional, NVIDIA RTX 3060 recommended)
    System packages: portaudio19-dev
    
    OpenUtau
    bash
    source <(curl -s https://raw.githubusercontent.com/HitCoder9768/OpenUtau-Installer-Linux/main/install.sh)


Features

    STT: Vosk, Whisper, Faster Whisper
    TTS: Coqui, Piper, eSpeak, OpenUtau (singing)
    Translation: English to Spanish, French, German, Italian, Chinese
    LLM: Ollama (e.g., smollm2:135m)
    Gradio Interface
