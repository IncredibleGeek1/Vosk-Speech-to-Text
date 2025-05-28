#MIT License

#Copyright (c) 2025 IncredibleGeek

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.



# -*- coding: utf-8 -*-
import queue
import sys
import sounddevice as sd
import vosk
import numpy as np
from scipy.signal import resample
import os
import wave
import json
import time
import words_to_numbers_v7 as words_to_numbers
import keyboard  # For typing and commands
import subprocess  # For launching applications
from spellchecker import SpellChecker  # For autocorrect
import nltk  # For tokenization to separate commands from dictation
import pyautogui  # For mouse clicks and screenshots
import logging  # For logging audio amplitudes and debugging
import debug_utils # For Debugging (configure via json) 
import threading

# Import the GUI module
import dictation_pygui_shared_v10 as gui_module

# Set up logging
logging.basicConfig(level=logging.DEBUG, filename="dictation_script.log", format="%(asctime)s - %(levelname)s - %(message)s")

# Download NLTK data (run once, automatically on first run)
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    logging.error(f"Error downloading NLTK data: {e}")
    print(f"Error downloading NLTK data: {e}")
    print("Please ensure you have an internet connection and try again.")
    sys.exit(1)

# Suppress Vosk logging for cleaner output
vosk.SetLogLevel(-1)

# Queue for audio data
q = queue.Queue()

# Configuration (will be set by GUI)
MODEL_PATH = None
MIC_SAMPLERATE = None
MIC_CHANNELS = 1
MIC_BITDEPTH = None
VOSK_SAMPLERATE = 16000
WAV_FILE = "output_gigaspeech.wav"
TRANSCRIPTION_FILE = "dictation_output_gigaspeech.txt"
SILENCE_THRESHOLD = None
BLOCKSIZE = 32000
PRE_SCALE_FACTOR = None
SILENCE_AMPLITUDE_THRESHOLD = None
RELATIVE_SENSITIVITY = None
STARTUP_DELAY = 5
COMMAND_DEBOUNCE_TIME = 1.0

# Path to commands and debug config JSON files
CONFIG_DIR = "config"
COMMANDS_JSON_PATH = os.path.join(CONFIG_DIR, "commands.json")
DEBUG_CONFIG_PATH = os.path.join(CONFIG_DIR, "debug_config.json")

# Paths to map JSON files (in the config folder)
FRACTIONS_MAP_PATH = os.path.join(CONFIG_DIR, "fractions_map.json")
F_KEYS_MAP_PATH = os.path.join(CONFIG_DIR, "f_keys_map.json")
FUNCTIONS_MAP_PATH = os.path.join(CONFIG_DIR, "functions_map.json")
SYMBOLS_MAP_PATH = os.path.join(CONFIG_DIR, "symbols_map.json")
GOOGLE_NUMBERS_PATH = os.path.join(CONFIG_DIR, "numbers_map.json")
LARGE_NUMBERS_MAP_PATH = os.path.join(CONFIG_DIR, "large_numbers_map.json")

# Dictionary to convert spoken numbers to digits (0 to 1 million)
NUMBERS_MAP = {
    "negative": "-", "minus": "-",
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19",
    "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60",
    "seventy": "70", "eighty": "80", "ninety": "90",
    "hundred": "100",
    "thousand": "1000",
    "million": "1000000"
}

# Load debug configuration
def load_debug_config():
    """Load debug settings from debug_config.json."""
    if not os.path.exists(DEBUG_CONFIG_PATH):
        logging.warning(f"Debug config file not found at {DEBUG_CONFIG_PATH}. Creating default config.")
        default_config = {
            "debug_audio_queue": False,
            "debug_audio_callback": False,
            "debug_status_updates": True
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
        logging.error(f"Error loading debug config from {DEBUG_CONFIG_PATH}: {e}")
        print(f"Error loading debug config from {DEBUG_CONFIG_PATH}: {e}")
        sys.exit(1)

DEBUG_CONFIG = load_debug_config()

# Load maps from JSON files
def load_json_map(file_path, map_name):
    """Load a mapping from a JSON file, with error handling."""
    if not os.path.exists(file_path):
        logging.error(f"{map_name} file not found at {file_path}")
        print(f"Error: {map_name} file not found at {file_path}")
        sys.exit(1)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        if not isinstance(mapping, dict):
            logging.error(f"{map_name} at {file_path} must be a JSON object (dictionary)")
            print(f"Error: {map_name} at {file_path} must be a JSON object (dictionary)")
            sys.exit(1)
        return mapping
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse {map_name} at {file_path}: {e}")
        print(f"Error: Failed to parse {map_name} at {file_path}: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to load {map_name} from {file_path}: {e}")
        print(f"Error: Failed to load {map_name} from {file_path}: {e}")
        sys.exit(1)

FRACTION_MAP = load_json_map(FRACTIONS_MAP_PATH, "fractions map")
F_KEYS_MAP = load_json_map(F_KEYS_MAP_PATH, "f-keys map")
FUNCTIONS_MAP = load_json_map(FUNCTIONS_MAP_PATH, "functions map")
SYMBOLS_MAP = load_json_map(SYMBOLS_MAP_PATH, "symbols map")
GOOGLE_NUMBERS = load_json_map(GOOGLE_NUMBERS_PATH, "google numbers map")
LARGE_NUMBERS_MAP = load_json_map(LARGE_NUMBERS_MAP_PATH, "large numbers map")

# List to store audio data for WAV file
audio_buffer = []

# Store the last word's end time for silence detection
last_word_end_time = 0.0

# Track the last command and its timestamp for debouncing
last_command = None
last_command_time = 0.0

# Track if a command was recently executed to skip dictation
skip_dictation = False

# Track the last dictated text for "scratch that", "correct <word>", etc.
last_dictated_text = ""
last_dictated_length = 0

# Initialize spell checker for "correct <word>"
spell = SpellChecker()

# Track caps lock state
caps_lock_on = False

# Track number lock state (for typing digits)
number_lock_on = False

# Track the last processed command to prevent dictation of command phrases
last_processed_command = None

# Track if the audio stream has started to display message once
STREAM_STARTED = False

# Track if feeding audio message has been displayed
FEEDING_AUDIO_STARTED = False

# Global flag to control dictation loop (will be set by GUI)
dictating = True

def parse_number_sequence(words):
    """Parse a sequence of number words into a single string of digits, handling negatives."""
    is_negative = False
    start_idx = 0
    i = 0
    
    if words and words[0].lower() in ["negative", "minus"]:
        is_negative = True
        start_idx = 1
        i = 1
    
    text = " ".join(words[start_idx:])
    if not text:
        return "", i
    
    result = words_to_numbers.convert_numbers(text, FRACTION_MAP, SYMBOLS_MAP, GOOGLE_NUMBERS, LARGE_NUMBERS_MAP)
    
    if result is None:
        return "", len(words)
    
    number = int(result)
    if is_negative:
        number = -number
    return str(number), len(words)

def normalize_text(text):
    """Normalize text by replacing hyphens with spaces and ensuring consistent spacing."""
    text = text.replace("-", " ")
    text = " ".join(text.split())
    return text

def convert_numbers(text):
    """Convert spoken numbers to digits if number lock is on."""
    global number_lock_on
    if not number_lock_on:
        return text
    words = text.split()
    converted_words = []
    for word in words:
        word_lower = word.lower()
        if word_lower in NUMBERS_MAP:
            converted_words.append(NUMBERS_MAP[word_lower])
        else:
            converted_words.append(word)
    return " ".join(converted_words)

def load_commands():
    """Load commands from JSON file and tokenize them."""
    if not os.path.exists(COMMANDS_JSON_PATH):
        logging.error(f"Commands JSON file not found at {COMMANDS_JSON_PATH}")
        print(f"Commands JSON file not found at {COMMANDS_JSON_PATH}")
        sys.exit(1)
    try:
        with open(COMMANDS_JSON_PATH, "r", encoding="utf-8") as f:
            commands_data = json.load(f)
    except Exception as e:
        logging.error(f"Error loading commands from {COMMANDS_JSON_PATH}: {e}")
        print(f"Error loading commands from {COMMANDS_JSON_PATH}: {e}")
        sys.exit(1)
    
    simple_commands = commands_data.get("simple_commands", {})
    tokenized_simple_commands = {}
    for cmd, action in simple_commands.items():
        normalized_cmd = normalize_text(cmd.lower())
        tokens = tuple(nltk.word_tokenize(normalized_cmd))
        tokenized_simple_commands[tokens] = (cmd, action)
    
    parameterized_commands = commands_data.get("parameterized_commands", [])
    final_only_commands = ["quote unquote "]
    tokenized_parameterized_partial = []
    tokenized_parameterized_final = []
    for cmd in parameterized_commands:
        normalized_cmd = normalize_text(cmd.lower())
        tokens = tuple(nltk.word_tokenize(normalized_cmd))
        if cmd in final_only_commands:
            tokenized_parameterized_final.append((tokens, cmd))
        else:
            tokenized_parameterized_partial.append((tokens, cmd))
    
    return tokenized_simple_commands, tokenized_parameterized_partial, tokenized_parameterized_final

# Load tokenized commands
try:
    TOKENIZED_SIMPLE_COMMANDS, TOKENIZED_PARAMETERIZED_PARTIAL, TOKENIZED_PARAMETERIZED_FINAL = load_commands()
except Exception as e:
    logging.error(f"Error loading commands from {COMMANDS_JSON_PATH}: {e}")
    print(f"Error loading commands from {COMMANDS_JSON_PATH}: {e}")
    sys.exit(1)

def process_text(text):
    """Handle capitalization and number conversion for transcribed text."""
    global caps_lock_on

    if not text:
        return text

    text = convert_numbers(text)
    words = text.split()
    if not words:
        return text

    if caps_lock_on:
        words = [word.upper() for word in words]
    else:
        words[0] = words[0][0].upper() + words[0][1:] if len(words[0]) > 1 else words[0].upper()

    processed_text = " ".join(words)
    number = words_to_numbers.convert_numbers(processed_text, FRACTION_MAP, SYMBOLS_MAP, GOOGLE_NUMBERS, LARGE_NUMBERS_MAP)
    if number is not None:
        final_text = str(number)
    else:
        final_text = processed_text

    return final_text

def handle_special_phrases(text):
    """Handle special phrases like 'one hundred percent', fractions, symbols, and 'number <number>' sequences."""
    text_lower = text.lower()
    words = text_lower.split()
    i = 0
    result = []

    while i < len(words):
        if words[i] == "number":
            i += 1
            number_sequence, num_words = parse_number_sequence(words[i:])
            if number_sequence:
                actual_idx = i + num_words
                if actual_idx < len(words) and words[actual_idx] == "percent":
                    result.append(number_sequence + "%")
                    i = actual_idx + 1
                else:
                    result.append(number_sequence)
                    i = actual_idx
            continue
        
        first_word = words[i]
        is_potential_number = False
        if first_word in ["negative", "minus"]:
            if i + 1 < len(words) and words[i + 1] in NUMBERS_MAP and NUMBERS_MAP[words[i + 1]].isdigit():
                is_potential_number = True
        elif first_word in NUMBERS_MAP and NUMBERS_MAP[first_word].isdigit():
            is_potential_number = True

        if is_potential_number:
            number_sequence, num_words = parse_number_sequence(words[i:])
            if number_sequence:
                if i + num_words < len(words) and words[i + num_words] == "percent":
                    result.append(number_sequence + "%")
                    i += num_words + 1
                else:
                    result.append(number_sequence)
                    i = num_words
                continue

        number_sequence, num_words = parse_number_sequence(words[i:])
        if number_sequence:
            if i + num_words < len(words) and words[i + num_words] == "percent":
                result.append(number_sequence + "%")
                i += num_words + 1
            else:
                result.append(number_sequence)
                i += num_words
            continue

        found_fraction = False
        for phrase, replacement in FRACTION_MAP.items():
            phrase_lower = phrase.lower()
            if " ".join(words[i:i + len(phrase_lower.split())]).lower() == phrase_lower:
                result.append(replacement)
                i += len(phrase_lower.split())
                found_fraction = True
                break
        if found_fraction:
            continue

        found_symbol = False
        for phrase, replacement in SYMBOLS_MAP.items():
            phrase_lower = phrase.lower()
            if " ".join(words[i:i + len(phrase_lower.split())]).lower() == phrase_lower:
                result.append(replacement)
                i += len(phrase_lower.split())
                found_symbol = True
                break
        if found_symbol:
            continue

        result.append(words[i])
        i += 1

    return " ".join(result)

def update_status(message):
    """Update both the CLI (if enabled) and GUI status bar with the given message."""
    if DEBUG_CONFIG.get("debug_status_updates", True):
        print(message)
    logging.debug(message)
    if gui_module.dpg.does_item_exist("status_text"):
        gui_module.dpg.set_value("status_text", message)

def type_text(text):
    """Type the given text into the active window using the keyboard library."""
    try:
        # Replace newlines with actual Enter key presses to avoid extra spacing
        if text == "\n\n":
            keyboard.press_and_release("enter")
            keyboard.press_and_release("enter")
        else:
            keyboard.write(text)
        update_status(f"Typed: {text}")
    except Exception as e:
        update_status(f"Error typing text: {e}")
        logging.error(f"Error typing text: {e}")

def callback(indata, frames, time_info, status):
    """Audio callback for Vosk processing."""
    global last_dictated_text, last_dictated_length, last_word_end_time, last_command, last_command_time, skip_dictation

    if status:
        update_status(f"Audio callback status: {status}")
        logging.error(f"Audio callback status: {status}")

    if MIC_BITDEPTH == 16:
        dtype = np.int16
        max_value = 32767
    elif MIC_BITDEPTH == 24 or MIC_BITDEPTH == 32:
        dtype = np.int32
        max_value = 8388607 if MIC_BITDEPTH == 24 else 2147483647
    else:
        logging.error(f"Unsupported bit depth: {MIC_BITDEPTH}")
        return

    try:
        indata_array = np.frombuffer(indata, dtype=dtype)
        indata_normalized = indata_array.astype(np.float32) / (max_value + 1)

        if RELATIVE_SENSITIVITY:
            reference_max = 32767
            scale_factor = reference_max / (max_value + 1)
            adjusted_pre_scale = PRE_SCALE_FACTOR * scale_factor
        else:
            adjusted_pre_scale = PRE_SCALE_FACTOR

        indata_normalized = indata_normalized * adjusted_pre_scale
        indata_array = np.clip(indata_normalized * 32767, -32768, 32767).astype(np.int16)

        if MIC_CHANNELS > 1:
            indata_array = indata_array.reshape(-1, MIC_CHANNELS).mean(axis=1).astype(np.int16)

        if MIC_SAMPLERATE != VOSK_SAMPLERATE:
            num_samples_resampled = int(len(indata_array) * VOSK_SAMPLERATE / MIC_SAMPLERATE)
            indata_array = resample(indata_array, num_samples_resampled)
            indata_array = indata_array.astype(np.int16)

        raw_amplitude = np.max(np.abs(indata_array))
        if DEBUG_CONFIG.get("debug_audio_callback", False):
            logging.debug(f"Audio callback - Frames: {frames}, Raw amplitude: {raw_amplitude}")
            if raw_amplitude > 0:
                db_level = 20 * np.log10(raw_amplitude / 32767)
                logging.debug(f"Raw amplitude in dB: {db_level:.2f} dB")

        if raw_amplitude < SILENCE_AMPLITUDE_THRESHOLD:
            if DEBUG_CONFIG.get("debug_audio_callback", False):
                update_status(f"Block below silence threshold (max amplitude: {raw_amplitude})")
                logging.info(f"Block below silence threshold (max amplitude: {raw_amplitude})")
            return

        audio_buffer.append(indata_array)
        indata_bytes = indata_array.tobytes()
        q.put(indata_bytes)
    except Exception as e:
        logging.error(f"Error in audio callback: {e}")
        update_status(f"Error in audio callback: {e}")

def execute_command(action, gui, transcribed_text):
    """Execute the action associated with a command."""
    global caps_lock_on, number_lock_on, last_dictated_text, last_dictated_length

    update_status(f"Executing command action: {action}")
    logging.info(f"Executing command action: {action}")

    if action in ["stop_dictation", "cmd_stop_listening"]:  # Handle both possible actions
        gui.is_dictating = False
        update_status(f"Set gui.is_dictating to {gui.is_dictating}")
        logging.debug(f"Set gui.is_dictating to {gui.is_dictating}")
        gui_module.dpg.set_value("status_text", "Dictation stopped by command.")
    elif action == "caps_lock_on":
        caps_lock_on = True
        gui_module.dpg.set_value("status_text", "Caps lock enabled.")
    elif action == "caps_lock_off":
        caps_lock_on = False
        gui_module.dpg.set_value("status_text", "Caps lock disabled.")
    elif action == "number_lock_on":
        number_lock_on = True
        gui_module.dpg.set_value("status_text", "Number lock enabled.")
    elif action == "number_lock_off":
        number_lock_on = False
        gui_module.dpg.set_value("status_text", "Number lock disabled.")
    elif action == "new_paragraph":
        type_text("\n\n")
        transcribed_text.append("\n\n")
        full_text = "".join(transcribed_text).rstrip()
        gui_module.dpg.set_value("output_text", full_text)
        with open(TRANSCRIPTION_FILE, "a", encoding="utf-8") as f:
            f.write("\n\n")
        gui_module.dpg.set_value("status_text", "Inserted new paragraph.")
    elif action == "scratch_that":
        if last_dictated_text:
            chars_to_delete = len(last_dictated_text) + 1  # +1 for the trailing space
            for _ in range(chars_to_delete):
                keyboard.press_and_release("backspace")
            if transcribed_text and transcribed_text[-1] == last_dictated_text:
                transcribed_text.pop()
            full_text = "".join(transcribed_text).rstrip()
            gui_module.dpg.set_value("output_text", full_text)
            with open(TRANSCRIPTION_FILE, "w", encoding="utf-8") as f:
                f.write(full_text)
            gui_module.dpg.set_value("status_text", "Last dictation removed.")
        else:
            gui_module.dpg.set_value("status_text", "Nothing to scratch.")
    else:
        gui_module.dpg.set_value("status_text", f"Unknown command action: {action}")
        logging.warning(f"Unknown command action: {action}")

def perform_dictation(gui, model_path, bit_depth_value, sample_rate, device_index, pre_scale, relative_sensitivity, silence_threshold, silence_duration):
    global MODEL_PATH, MIC_BITDEPTH, MIC_SAMPLERATE, PRE_SCALE_FACTOR, RELATIVE_SENSITIVITY, SILENCE_AMPLITUDE_THRESHOLD, SILENCE_THRESHOLD, audio_buffer, last_word_end_time, last_command, last_command_time, skip_dictation, last_dictated_text, last_dictated_length, last_processed_command, dictating, STREAM_STARTED, FEEDING_AUDIO_STARTED

    # Update global variables with provided parameters
    MODEL_PATH = model_path
    MIC_BITDEPTH = bit_depth_value
    MIC_SAMPLERATE = int(sample_rate)
    PRE_SCALE_FACTOR = pre_scale
    RELATIVE_SENSITIVITY = bool(relative_sensitivity)
    SILENCE_AMPLITUDE_THRESHOLD = silence_threshold
    SILENCE_THRESHOLD = silence_duration

    # Verify the model path exists
    if not os.path.exists(MODEL_PATH):
        gui_module.dpg.set_value("status_text", f"Vosk model not found at {MODEL_PATH}")
        logging.error(f"Vosk model not found at {MODEL_PATH}")
        return

    # Reset global state
    audio_buffer = []
    last_word_end_time = 0.0
    last_command = None
    last_command_time = 0.0
    skip_dictation = False
    last_dictated_text = ""
    last_dictated_length = 0
    last_processed_command = None

    # Load Vosk model
    update_status("Loading Vosk model...")
    try:
        model = vosk.Model(MODEL_PATH)
        update_status("Vosk model loaded successfully.")
    except Exception as e:
        update_status(f"Failed to load Vosk model: {e}")
        logging.error(f"Failed to load Vosk model: {e}")
        gui_module.dpg.set_value("status_text", f"Failed to load Vosk model: {e}")
        return

    # Initialize Vosk recognizer
    update_status("Initializing Vosk recognizer...")
    try:
        rec = vosk.KaldiRecognizer(model, VOSK_SAMPLERATE)
        rec.SetWords(True)
        update_status("Vosk recognizer initialized successfully.")
    except Exception as e:
        update_status(f"Failed to initialize Vosk recognizer: {e}")
        logging.error(f"Failed to initialize Vosk recognizer: {e}")
        gui_module.dpg.set_value("status_text", f"Failed to initialize Vosk recognizer: {e}")
        return

    # Clear the transcription file
    with open(TRANSCRIPTION_FILE, "w", encoding="utf-8") as f:
        f.write("")

    # Start transcription
    update_status(f"Starting in {STARTUP_DELAY} seconds... Click into a text field (e.g., Notepad) to begin typing.")
    time.sleep(STARTUP_DELAY)
    update_status("Starting live speech-to-text with Vosk (GigaSpeech 0.42 model). Speak to type or use commands!")

    # Start audio stream
    if not STREAM_STARTED:
        update_status(f"Starting audio stream with sample rate {MIC_SAMPLERATE}, device index {device_index}...")
    try:
        stream = sd.RawInputStream(
            samplerate=MIC_SAMPLERATE,
            blocksize=BLOCKSIZE,
            dtype="int32",
            channels=MIC_CHANNELS,
            callback=callback,
            device=device_index
        )
        stream.start()
        if not STREAM_STARTED:
            update_status("Audio stream started successfully.")
            STREAM_STARTED = True
    except Exception as e:
        update_status(f"Failed to start audio stream: {e}")
        logging.error(f"Failed to start audio stream: {e}")
        gui_module.dpg.set_value("status_text", f"Failed to start audio stream: {e}")
        return

    last_partial = ""
    transcribed_text = []

    while gui.is_dictating:
        try:
            data = q.get(timeout=0.1)
        except queue.Empty:
            debug_utils.log_message("debug_audio_queue", "Queue empty, waiting for audio data...", gui)
            continue
            
        if not FEEDING_AUDIO_STARTED:
            update_status("Feeding audio to Vosk recognizer...")
            FEEDING_AUDIO_STARTED = True

        try:
            if rec.AcceptWaveform(data):
                result_dict = json.loads(rec.Result())
                logging.debug(f"Vosk Result: {result_dict}")
                text = result_dict.get("text", "")
                if text:
                    text = handle_special_phrases(text)
                    normalized_text = normalize_text(text.lower())
                    tokens = tuple(nltk.word_tokenize(normalized_text))
                    is_final_command = False
                    for cmd_tokens, command in TOKENIZED_PARAMETERIZED_FINAL:
                        if len(tokens) >= len(cmd_tokens) and tokens[:len(cmd_tokens)] == cmd_tokens:
                            update_status(f"\nDetected command: {text}")
                            last_processed_command = text
                            skip_dictation = True
                            param = text[len(command):].strip().lower()
                            if command == "quote unquote ":
                                update_status(f"Executing action: quote unquote {param}")
                                quoted_text = f'"{param}"'
                                type_text(quoted_text)
                                transcribed_text.append(quoted_text)
                                gui_module.dpg.set_value("output_text", quoted_text)
                            is_final_command = True
                            break
                    if is_final_command:
                        continue
                    if last_processed_command:
                        last_processed_tokens = tuple(nltk.word_tokenize(normalize_text(last_processed_command.lower())))
                        if tokens == last_processed_tokens:
                            update_status(f"Skipping dictation for command (already processed): {text}")
                            last_processed_command = None
                            skip_dictation = False
                            continue
                    is_command = False
                    for cmd_tokens, (cmd, action) in TOKENIZED_SIMPLE_COMMANDS.items():
                        if tokens == cmd_tokens:
                            update_status(f"Detected command: {cmd}")
                            skip_dictation = False
                            is_command = True
                            execute_command(action, gui, transcribed_text)
                            if not gui.is_dictating:
                                break
                            continue
                    if not gui.is_dictating:
                        break
                    if is_command:
                        continue
                    if skip_dictation:
                        skip_dictation = False
                        continue
                    processed_text = process_text(text)
                    last_dictated_text = processed_text
                    last_dictated_length = len(processed_text)
                    type_text(processed_text + " ")
                    transcribed_text.append(processed_text + " ")
                    full_text = "".join(transcribed_text).rstrip()
                    gui_module.dpg.set_value("output_text", full_text)
                    update_status(f"Transcribed: {processed_text}")
                    with open(TRANSCRIPTION_FILE, "a", encoding="utf-8") as f:
                        f.write(processed_text + " ")
            else:
                partial_dict = json.loads(rec.PartialResult())
                partial_text = partial_dict.get("partial", "")
                if partial_text and partial_text != last_partial:
                    last_partial = partial_text
                    update_status(f"Partial: {partial_text}")
        except Exception as e:
            logging.error(f"Error processing audio data: {e}")
            update_status(f"Error processing audio data: {e}")

    # Clean up
    try:
        stream.stop()
        stream.close()
    except Exception as e:
        logging.error(f"Error closing audio stream: {e}")
    STREAM_STARTED = False
    FEEDING_AUDIO_STARTED = False
    update_status("Dictation stopped.")

    # Save the recorded audio to WAV file
    if audio_buffer:
        try:
            audio_data = np.concatenate(audio_buffer)
            with wave.open(WAV_FILE, "wb") as wf:
                wf.setnchannels(MIC_CHANNELS)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(VOSK_SAMPLERATE)
                wf.writeframes(audio_data.tobytes())
            update_status(f"Audio saved to {WAV_FILE}")
            audio_buffer = []  # Clear audio_buffer
        except Exception as e:
            logging.error(f"Error saving audio to WAV: {e}")
            update_status(f"Error saving audio to WAV: {e}")

class CustomDictationGUI(gui_module.DictationGUI):
    """Custom subclass to override start_dictation and stop_dictation."""
    def __init__(self):
        logging.debug("Initializing CustomDictationGUI")
        try:
            super().__init__()
            logging.debug("Parent DictationGUI initialized successfully")
        except Exception as e:
            logging.error(f"Error in DictationGUI.__init__: {e}", exc_info=True)
            raise

    def start_dictation(self, sender, app_data):
        logging.debug("Start dictation button clicked.")

        if self.is_dictating:
            logging.warning("Dictation already running.")
            gui_module.dpg.set_value("status_text", "Dictation already running.")
            return
        if self.is_testing:
            self.stop_audio_test()
        if self.is_recording:
            self.toggle_recording()

        # Gather settings from GUI
        bit_depth = gui_module.dpg.get_value("bit_depth_combo") if gui_module.dpg.does_item_exist("bit_depth_combo") else "int16"
        if " - " in bit_depth:
            bit_depth = bit_depth.split(" - ")[0]
        bit_depth_value = int(bit_depth.replace("int", ""))
        sample_rate = gui_module.dpg.get_value("sample_rate_combo") if gui_module.dpg.does_item_exist("sample_rate_combo") else "16000"
        device_index = self.get_device_index(
            gui_module.dpg.get_value("input_device_combo") if gui_module.dpg.does_item_exist("input_device_combo") else None,
            gui_module.dpg.get_value("host_api_combo") if gui_module.dpg.does_item_exist("host_api_combo") else "MME"
        )
        if device_index is None:
            gui_module.dpg.set_value("status_text", "No valid input device selected.")
            logging.error("No valid input device selected.")
            return

        unit = gui_module.dpg.get_value("unit_combo") if gui_module.dpg.does_item_exist("unit_combo") else "Numbers"
        slider_value = gui_module.dpg.get_value("sensitivity_slider") if gui_module.dpg.does_item_exist("sensitivity_slider") else 0
        if unit == "Numbers":
            pre_scale = self.slider_to_pre_scale(slider_value)
        elif unit == "Percent":
            pre_scale = self.percent_to_pre_scale(slider_value)
        elif unit == "dB":
            db = (slider_value / 100) * 100 - 60
            pre_scale = self.db_to_pre_scale(db)

        relative_sensitivity = 1 if (gui_module.dpg.does_item_exist("relative_sensitivity_check") and gui_module.dpg.get_value("relative_sensitivity_check")) else 0
        if relative_sensitivity:
            dtype, max_value = self.get_dtype_and_max()
            scale_factor = 32767 / (max_value + (1 if dtype != "float32" else 0))
            pre_scale /= scale_factor

        silence_threshold = gui_module.dpg.get_value("silence_input") if gui_module.dpg.does_item_exist("silence_input") else 500.0
        silence_duration = 1.0  # Default silence duration

        # Read and validate model path
        model_path = gui_module.dpg.get_value("model_path_input")
        model_path = os.path.abspath(model_path)
        logging.debug(f"Model path from GUI: {model_path}")
        if not model_path or not os.path.exists(model_path):
            gui_module.dpg.set_value("status_text", f"Vosk model not found at {model_path}.")
            logging.error(f"Vosk model not found at {model_path}.")
            return

        # Start the dictation process
        self.is_dictating = True
        gui_module.dpg.configure_item("start_dictation_button", enabled=False)
        gui_module.dpg.configure_item("stop_dictation_button", enabled=True)
        gui_module.dpg.set_value("status_text", "Starting dictation...")

        # Run dictation in a separate thread to keep GUI responsive
        dictation_thread = threading.Thread(target=perform_dictation, args=(
            self, model_path, bit_depth_value, sample_rate, device_index,
            pre_scale, relative_sensitivity, silence_threshold, silence_duration
        ))
        dictation_thread.daemon = True
        dictation_thread.start()

    def stop_dictation(self, sender=None, app_data=None):
        logging.debug("Stop dictation button clicked.")
        if self.is_dictating:
            self.is_dictating = False
            gui_module.dpg.configure_item("start_dictation_button", enabled=True)
            gui_module.dpg.configure_item("stop_dictation_button", enabled=False)
            gui_module.dpg.set_value("status_text", "Dictation stopped.")
            logging.info("Dictation stopped via GUI or interrupt.")
            if self.is_debug_recording:
                self.save_debug_recording()

def main():
    """Main function to launch the GUI and dictation process."""
    global dictating
    gui = None
    logging.debug("Starting main() function")
    try:
        logging.debug("Instantiating CustomDictationGUI")
        gui = CustomDictationGUI()
        logging.debug("GUI instantiated, starting DearPyGui")
        gui_module.dpg.start_dearpygui()
    except KeyboardInterrupt:
        update_status("Received Ctrl+C. Shutting down gracefully...")
        logging.info("User initiated shutdown with Ctrl+C.")
        dictating = False
        if hasattr(gui, 'is_dictating') and gui.is_dictating:
            gui.is_dictating = False
            gui_module.dpg.set_value("status_text", "Dictation stopped by user.")
            logging.info("Dictation stopped.")
        try:
            gui_module.dpg.stop_dearpygui()
            update_status("GUI closed successfully.")
            logging.info("Dear PyGui GUI closed successfully.")
        except Exception as e:
            update_status(f"Error closing GUI: {e}")
            logging.error(f"Error closing GUI: {e}")
        update_status("Application terminated.")
        logging.info("Application terminated.")
        sys.exit(0)
    except Exception as e:
        update_status(f"Unexpected error: {e}")
        logging.error(f"Unexpected error: {e}", exc_info=True)
        print(f"Error: {e}. Check dictation_script.log for details.")
        sys.exit(1)
    finally:
        try:
            gui_module.dpg.destroy_context()
        except Exception as e:
            logging.error(f"Error destroying DPG context: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s", filename="dictation_script.log")
    try:
        main()
    except Exception as e:
        logging.error(f"Application crashed: {e}", exc_info=True)
        print(f"Error: {e}. Check dictation_script.log for details.")
        input("Press Enter to exit...")