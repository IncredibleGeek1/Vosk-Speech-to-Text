# debug_utils.py
import os
import json
import logging

# Path to debug config JSON file
CONFIG_DIR = "config"
DEBUG_CONFIG_PATH = os.path.join(CONFIG_DIR, "debug_config.json")

# Set up logging to a file
logging.basicConfig(level=logging.DEBUG, filename="dictation_script.log", format="%(asctime)s - %(levelname)s - %(message)s")

def load_debug_config():
    """Load debug settings from debug_config.json. Create a default config if it doesn't exist."""
    if not os.path.exists(DEBUG_CONFIG_PATH):
        print(f"Debug config file not found at {DEBUG_CONFIG_PATH}. Creating default config.")
        default_config = {
            "debug_model_loading": {
                "cli_enabled": True,
                "gui_enabled": True,
                "description": "Logs Vosk model loading steps (e.g., 'Loading Vosk model...', 'Vosk model loaded successfully.')."
            },
            "debug_recognizer_init": {
                "cli_enabled": True,
                "gui_enabled": True,
                "description": "Logs Vosk recognizer initialization steps (e.g., 'Initializing Vosk recognizer...')."
            },
            "debug_audio_stream": {
                "cli_enabled": True,
                "gui_enabled": True,
                "description": "Logs audio stream setup and status (e.g., 'Starting audio stream with sample rate 48000, device index 1...')."
            },
            "debug_audio_processing": {
                "cli_enabled": False,
                "gui_enabled": False,
                "description": "Logs audio processing steps, such as retrieving audio data and feeding it to the Vosk recognizer (e.g., 'Retrieved audio data from queue.')."
            },
            "debug_audio_level": {
                "cli_enabled": True,
                "gui_enabled": False,
                "description": "Logs raw audio amplitude levels to monitor input volume (e.g., 'Raw amplitude: 1234')."
            },
            "debug_audio_clipping": {
                "cli_enabled": True,
                "gui_enabled": False,
                "description": "Logs when audio clipping occurs (e.g., 'Audio clipping detected: amplitude 32767 exceeds max 32767')."
            },
            "debug_silence_threshold": {
                "cli_enabled": True,
                "gui_enabled": False,
                "description": "Logs when audio blocks are below the silence threshold (e.g., 'Block below silence threshold (max amplitude: 123)')."
            },
            "debug_transcription": {
                "cli_enabled": True,
                "gui_enabled": True,
                "description": "Logs transcription results, including partial and final transcriptions (e.g., 'Transcribed: Hello', 'Vosk Result: {...}')."
            },
            "debug_commands": {
                "cli_enabled": True,
                "gui_enabled": True,
                "description": "Logs command detection and execution (e.g., 'Detected command: stop listening', 'Executing command action: cmd_stop_listening')."
            },
            "debug_typing": {
                "cli_enabled": True,
                "gui_enabled": True,
                "description": "Logs text typing actions (e.g., 'Typed: Hello')."
            },
            "debug_status_updates": {
                "cli_enabled": True,
                "gui_enabled": True,
                "description": "Logs general status updates (e.g., 'Dictation stopped.', 'Audio saved to output_gigaspeech.wav')."
            },
            "debug_audio_queue": {
                "cli_enabled": False,
                "gui_enabled": False,
                "description": "Logs when the audio queue is empty and waiting for data (e.g., 'Queue empty, waiting for audio data...')."
            }
        }
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(DEBUG_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=4)
        return default_config

    try:
        with open(DEBUG_CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error loading debug config from {DEBUG_CONFIG_PATH}: {e}")
        raise

# Load the debug configuration at startup
DEBUG_CONFIG = load_debug_config()

def log_message(category, message, gui=None):
    """
    Log a message if the specified debug category is enabled for CLI or GUI.

    Args:
        category (str): The debug category (e.g., 'debug_audio_processing').
        message (str): The message to log.
        gui: The GUI object (if available) to update the status bar.
    """
    # Check if the category exists and get its settings
    category_settings = DEBUG_CONFIG.get(category, {})
    cli_enabled = category_settings.get("cli_enabled", False)
    gui_enabled = category_settings.get("gui_enabled", False)

    # Log to CLI if enabled
    if cli_enabled:
        print(message)

    # Always log to the file for debugging purposes
    logging.debug(f"[{category}] {message}")

    # Update the GUI status bar if enabled and a GUI object is provided
    if gui_enabled and gui is not None:
        try:
            import dictation_pygui_shared_v6_test as gui_module
            gui_module.dpg.set_value("status_text", message)
        except Exception as e:
            logging.error(f"Error updating GUI status: {e}")