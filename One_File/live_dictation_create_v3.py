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


# _*_ coding: utf-8 -*-
from tkinter import UNITS
import dearpygui.dearpygui as dpg  # For GUI handling
import debug_utils as debug  # For Debugging
import vosk  # For speech recognition
import wave  # For audio file handling
import json  # For JSON handling
import os  # For file handling
import subprocess  # For subprocess handling
import sys  # For subprocess handling
import time  # For time handling
import math  # For audio processing
import numpy as np  # For audio processing
from scipy.signal import resample
import sounddevice as sd  # For audio input/output
import threading  # For threading
import queue  # For command queue
import logging  # For logging
import keyboard  # For keyboard input
import re  # For regex
import msvcrt  # For keyboard input
import selectors  # For keyboard input
from spellchecker import SpellChecker  # For spell checking
from collections import deque  # For audio buffer
from vosk import Model, KaldiRecognizer  # For speech recognition
import nltk  # For tokenization to separate commands from dictation

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s", filename="dictation_gui.log")

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
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
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

# Define mappings for small numbers (0-999)
SMALL_NUMBERS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
    "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19",
    "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50",
    "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
    "hundred": "100", "thousand": "1000"
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
    """Load a mapping from a JSON file."""
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

# Helper function to get the base path for bundled files
def get_base_path():
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

JSON_FILES = {
    "Commands": COMMANDS_JSON_PATH,
    "Fractions": FRACTIONS_MAP_PATH,
    "F-Keys": F_KEYS_MAP_PATH,
    "Functions": FUNCTIONS_MAP_PATH,
    "Symbols": SYMBOLS_MAP_PATH,
    "Google Numbers": GOOGLE_NUMBERS_PATH,
    "Large Numbers": LARGE_NUMBERS_MAP_PATH
}

SAMPLE_RATES = [
    "8000", "11025", "16000", "22050", "32000", "44100", "48000",
    "88200", "96000", "176400", "192000", "352800", "384000"
]

DATA_TYPES = {
    "int8": "8-bit Integer (Fastest, lowest quality)",
    "int16": "16-bit Integer (Standard quality)",
    "int24": "24-bit Integer (High quality)",
    "int32": "32-bit Integer (Highest integer quality)",
    "float32": "32-bit Float (Best for GPU acceleration)"
}

# Convert a phrase representing a number less than 1000 to its numerical value
def words_to_number_less_than_thousand(phrase):
    if not phrase:
        return 0

    words = phrase.replace("-", " ").split()
    if not words:
        return 0

    total = 0
    current = 0

    for word in words:
        if word in SMALL_NUMBERS:
            value = int(SMALL_NUMBERS[word])
            if value == 100:
                current *= 100
            elif value >= 1000:
                current *= value
                total += current
                current = 0
            else:
                current += value
        else:
            return None  # Return None for invalid words

    return total + current

# Main function to convert a word phrase to a number
def convert_numbers(phrase, fraction_map=None, symbs_map=None, google_numbers=None, large_numbers_map=None):
    # Use the pre-loaded LARGE_NUMBERS_MAP
    large_numbers_map = large_numbers_map or LARGE_NUMBERS_MAP

    # Practical upper limit to prevent overflow
    PRACTICAL_LIMIT = 10**303  # Up to centillion (10^303)
    
    # Clean the input phrase
    if not isinstance(phrase, str):
        return None
    
    phrase = phrase.lower().strip()
    if not phrase:
        return None

    phrase = re.sub(r'\s+', ' ', phrase)  # Normalize spaces
    phrase = phrase.replace(" and ", " ")  # Remove "and"
    phrase = phrase.replace(",", "")  # Remove commas

    # Handle negative numbers
    is_negative = False
    if phrase.startswith("negative"):
        is_negative = True
        phrase = phrase[len("negative"):].strip()
        if not phrase:
            return None

    # Split the phrase into sections based on large number scales
    large_scales = sorted(
        large_numbers_map.items(),
        key=lambda x: int(x[1]),
        reverse=True
    )

    # Create a regex pattern to split on large number words
    large_scale_words = "|".join(re.escape(scale) for scale, _ in large_scales)
    pattern = f"\\b({large_scale_words})\\b"
    sections = re.split(pattern, phrase)

    total = 0
    current_section_value = 0
    current_scale = 1  # Default scale for numbers less than the smallest large scale

    # Process each section
    for section in sections:
        section = section.strip()
        if not section:
            continue

        if section in large_numbers_map:
            # This section is a large scale (e.g., "billion")
            scale_value = int(large_numbers_map[section])
            if current_section_value == 0:
                current_section_value = 1  # e.g., "billion" alone means "one billion"
            total += current_section_value * scale_value
            current_section_value = 0
            current_scale = 1  # Reset for the next section
        else:
            # This section is a number phrase (e.g., "one hundred and twenty-three")
            section_value = words_to_number_less_than_thousand(section)
            if section_value is None:
                return None
            current_section_value += section_value

    # Add any remaining value (e.g., numbers less than the smallest large scale)
    total += current_section_value * current_scale

    # Check if the result exceeds the practical limit
    if total > PRACTICAL_LIMIT:
        return None

    return str(-total if is_negative else total)

# Helper function to parse a sequence of number words from a list
def parse_number_sequence(words, fraction_map=None, symbs_map=None, google_numbers=None, large_numbers_map=None):
    """Parse a sequence of number words into a single number, returning the number and the number of words consumed."""
    if not words:
        return None, 0

    # Join words into a phrase and try to parse it
    phrase = " ".join(words)
    number = convert_numbers(phrase, fraction_map, symbs_map, google_numbers, large_numbers_map)
    if number is not None:
        return number, len(words)

    # Try parsing smaller sequences until we find a valid number
    for i in range(len(words), 0, -1):
        sub_phrase = " ".join(words[:i])
        number = convert_numbers(sub_phrase, fraction_map, symbs_map, google_numbers, large_numbers_map)
        if number is not None:
            return number, i
    return None, 0

def normalize_text(text):
    """Normalize text by replacing hyphens with spaces and ensuring consistent spacing."""
    text = text.replace("-", " ")
    text = " ".join(text.split())
    return text

def convert_spoken_numbers_to_digits(text):
    """Convert spoken numbers to digits if number lock is on."""
    global number_lock_on
    if not number_lock_on:
        return text
    words = text.split()
    result = []
    i = 0
    while i < len(words):
        # Try to parse a sequence of words as a number
        number_str, words_consumed = parse_number_sequence(words[i:], FRACTION_MAP, SYMBOLS_MAP, GOOGLE_NUMBERS, LARGE_NUMBERS_MAP)
        if number_str is not None and words_consumed > 0:
            result.append(number_str)
            i += words_consumed
        else:
            result.append(words[i])
            i += 1
    return " ".join(result)

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

class AudioProcessor:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.gui_update_queue = queue.Queue()
        self.audio_stream = None
        self.output_stream = None
        self.is_testing = False
        self.is_recording = False
        self.audio_buffer = deque(maxlen=48000)  # For WaveformDisplay
        self.output_buffer = []
        self.debug_audio_buffer = []
        self.noise_floor = 0
        self.last_noise_update = 0
        self.peak_amplitude = 0
        self.is_debug_recording = False
       #self.recordings_dir = recordings_dir
        self.saved_settings = {}
        self.load_settings()

    def load_settings(self):
        default_settings = {
            "bit_depth": "int24",
            "sample_rate": "48000",
            "pre_scale_factor": 0.002,
            "unit": "Numbers",
            "relative_sensitivity": False,
            "silence_threshold": 10.0,
            "show_peaks": False,
            "host_api": "MME",
            "input_device": None,
            "output_device": None
        }
        base_path = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_path, CONFIG_PATH)
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            config = {}
        
        self.saved_settings = default_settings
        self.saved_settings.update(config.get("audio_settings", {}))
        
        # Update UI elements if they exist
        if dpg.does_item_exist("bit_depth_combo"):
            dpg.set_value("bit_depth_combo", self.saved_settings["bit_depth"])
        if dpg.does_item_exist("sample_rate_combo"):
            dpg.set_value("sample_rate_combo", self.saved_settings["sample_rate"])
        if dpg.does_item_exist("sensitivity_slider"):
            self.set_slider_from_pre_scale(self.saved_settings["pre_scale_factor"])
        if dpg.does_item_exist("unit_combo"):
            dpg.set_value("unit_combo", self.saved_settings["unit"])
        if dpg.does_item_exist("relative_sensitivity_check"):
            dpg.set_value("relative_sensitivity_check", self.saved_settings["relative_sensitivity"])
        if dpg.does_item_exist("silence_input"):
            dpg.set_value("silence_input", self.saved_settings["silence_threshold"])
        if dpg.does_item_exist("show_peaks_check"):
            dpg.set_value("show_peaks_check", self.saved_settings["show_peaks"])
        if dpg.does_item_exist("host_api_combo"):
            dpg.set_value("host_api_combo", self.saved_settings["host_api"])
            self.update_host_api(None, self.saved_settings["host_api"])
        if dpg.does_item_exist("input_device_combo"):
            dpg.set_value("input_device_combo", self.saved_settings["input_device"])
        if dpg.does_item_exist("output_device_combo"):
            dpg.set_value("output_device_combo", self.saved_settings.get("output_device", ""))

    def saved_settings(self):
        slider_value = dpg.get_value("sensitivity_slider") if dpg.does_item_exist("sensitivity_slider") else self.pre_scale_to_slider(self.saved_settings["pre_scale_factor"])
        unit = dpg.get_value("unit_combo") if dpg.does_item_exist("unit_combo") else self.saved_settings["unit"]
        if unit == "Numbers":
            pre_scale_factor = self.slider_to_pre_scale(slider_value)
        elif unit == "Percent":
            pre_scale_factor = self.percent_to_pre_scale(slider_value)
        elif unit == "dB":
            db = (slider_value / 100) * 100 - 60
            pre_scale_factor = self.db_to_pre_scale(db)
        pre_scale_factor = min(pre_scale_factor, 10.0)

        bit_depth = dpg.get_value("bit_depth_combo") if dpg.does_item_exist("bit_depth_combo") else self.saved_settings["bit_depth"]
        if " - " in bit_depth:
            bit_depth = bit_depth.split(" - ")[0]

        self.saved_settings.update({
            "bit_depth": bit_depth,
            "sample_rate": dpg.get_value("sample_rate_combo") if dpg.does_item_exist("sample_rate_combo") else self.saved_settings["sample_rate"],
            "pre_scale_factor": pre_scale_factor,
            "unit": unit,
            "relative_sensitivity": dpg.get_value("relative_sensitivity_check") if dpg.does_item_exist("relative_sensitivity_check") else self.saved_settings["relative_sensitivity"],
            "silence_threshold": dpg.get_value("silence_input") if dpg.does_item_exist("silence_input") else self.saved_settings["silence_threshold"],
            "show_peaks": dpg.get_value("show_peaks_check") if dpg.does_item_exist("show_peaks_check") else self.saved_settings["show_peaks"],
            "host_api": dpg.get_value("host_api_combo") if dpg.does_item_exist("host_api_combo") else self.saved_settings["host_api"],
            "input_device": dpg.get_value("input_device_combo") if dpg.does_item_exist("input_device_combo") else self.saved_settings["input_device"],
            "output_device": dpg.get_value("output_device_combo") if dpg.does_item_exist("output_device_combo") else self.saved_settings["output_device"]
        })

        base_path = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_path, CONFIG_PATH)
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            config = {}
        
        config["audio_settings"] = self.saved_settings
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
            logging.info("Audio settings saved successfully.")
        except Exception as e:
            logging.error(f"Failed to save audio settings: {e}")

    def setup_audio_tab(self):
        with dpg.group(horizontal=True):
            dpg.add_text("Host API:")
            dpg.add_combo(self.get_host_apis(), default_value=self.saved_settings["host_api"], callback=self.update_host_api, tag="host_api_combo")
        
        with dpg.group(horizontal=True):
            dpg.add_text("Input Device:")
            input_devices = self.get_input_devices(self.saved_settings["host_api"])
            dpg.add_combo(input_devices, default_value=self.saved_settings["input_device"], callback=self.update_device, tag="input_device_combo", width=300)
        
        with dpg.group(horizontal=True):
            dpg.add_text("Output Device:")
            output_devices = self.get_output_devices(self.saved_settings["host_api"])
            dpg.add_combo(output_devices, default_value=self.saved_settings["output_device"], callback=self.update_output_device, tag="output_device_combo", width=300)
        
        with dpg.group(horizontal=True):
            dpg.add_text("Data Type:")
            dpg.add_combo(list(DATA_TYPES.keys()), default_value=self.saved_settings["bit_depth"], callback=self.update_bit_depth, tag="bit_depth_combo")
        
        with dpg.group(horizontal=True):
            dpg.add_text("Sample Rate:")
            dpg.add_combo(SAMPLE_RATES, default_value=self.saved_settings["sample_rate"], tag="sample_rate_combo")
        
        with dpg.group(horizontal=True):
            dpg.add_text("Sensitivity Unit:")
            dpg.add_combo(UNITS, default_value=self.saved_settings["unit"], callback=self.update_unit, tag="unit_combo")
        
        with dpg.group(horizontal=True):
            dpg.add_text("Sensitivity (Volume):")
            slider_value = self.pre_scale_to_slider(self.saved_settings["pre_scale_factor"])
            dpg.add_slider_float(default_value=slider_value, min_value=0, max_value=100, callback=self.update_pre_scale_label, tag="sensitivity_slider", width=400)
            dpg.add_button(label=f"{self.saved_settings['pre_scale_factor']:.3f}", tag="pre_scale_label", callback=self.open_manual_sensitivity_input)
        
        dpg.add_checkbox(label="Keep Same Relative Sensitivity", default_value=self.saved_settings["relative_sensitivity"], callback=self.update_relative_sensitivity, tag="relative_sensitivity_check")
        dpg.add_checkbox(label="Show Peaks", default_value=self.saved_settings["show_peaks"], tag="show_peaks_check")
        dpg.add_checkbox(label="Enable Debug Recording", default_value=self.is_debug_recording, callback=lambda s, a: setattr(self, "is_debug_recording", a), tag="debug_recording_check")
        
        with dpg.group(horizontal=True):
            dpg.add_text("Silence Threshold:")
            dpg.add_input_float(default_value=self.saved_settings["silence_threshold"], tag="silence_input")
        
        with dpg.group(horizontal=True):
            dpg.add_button(label="Suggest Settings", callback=self.suggest_settings)
            dpg.add_button(label="Calibrate", callback=self.calibrate)
            dpg.add_button(label="Reset to Defaults", callback=self.reset_settings)
        
        with dpg.group(horizontal=True):
            dpg.add_button(label="Test Audio", tag="test_audio_button", callback=self.toggle_audio_test)
            dpg.add_button(label="Record", tag="record_button", callback=self.toggle_recording)
        
        with dpg.group(horizontal=True):
            dpg.add_text("Audio Level:")
            with dpg.drawlist(width=400, height=20, tag="level_drawlist"):
                dpg.draw_rectangle((0, 0), (400 * 0.6, 20), fill=(0, 255, 0, 255))
                dpg.draw_rectangle((400 * 0.6, 0), (400 * 0.8, 20), fill=(255, 255, 0, 255))
                dpg.draw_rectangle((400 * 0.8, 0), (400, 20), fill=(255, 0, 0, 255))
                dpg.draw_rectangle((0, 0), (0, 20), fill=(0, 0, 255, 255), tag="level_bar")
                dpg.draw_rectangle((0, 0), (0, 20), fill=(255, 255, 255, 50), tag="shadow_bar")
                dpg.draw_circle((380, 10), 10, fill=(255, 0, 0, 128), tag="clipping_indicator")

    def get_host_apis(self):
        try:
            return [host_api['name'] for host_api in sd.query_hostapis()]
        except Exception as e:
            logging.error(f"Failed to query host APIs: {e}")
            return ["MME"]

    def get_input_devices(self, host_api_name):
        host_api_index = next((i for i, h in enumerate(sd.query_hostapis()) if h['name'] == host_api_name), None)
        if host_api_index is None:
            return []
        return [d['name'] for d in sd.query_devices() if d['hostapi'] == host_api_index and d['max_input_channels'] > 0]

    def get_output_devices(self, host_api_name):
        host_api_index = next((i for i, h in enumerate(sd.query_hostapis()) if h['name'] == host_api_name), None)
        if host_api_index is None:
            return []
        return [d['name'] for d in sd.query_devices() if d['hostapi'] == host_api_index and d['max_output_channels'] > 0]

    def get_device_index(self, device_name, host_api_name, is_input=True):
        if not device_name:
            return None
        host_api_index = next((i for i, h in enumerate(sd.query_hostapis()) if h['name'] == host_api_name), None)
        if host_api_index is None:
            return None
        for i, d in enumerate(sd.query_devices()):
            if d['name'] == device_name and d['hostapi'] == host_api_index:
                if (is_input and d['max_input_channels'] > 0) or (not is_input and d['max_output_channels'] > 0):
                    return i
        return None

    def update_host_api(self, sender, app_data):
        host_api = app_data or self.saved_settings["host_api"]
        input_devices = self.get_input_devices(host_api)
        dpg.configure_item("input_device_combo", items=input_devices)
        if input_devices:
            if not dpg.get_value("input_device_combo") or dpg.get_value("input_device_combo") not in input_devices:
                dpg.set_value("input_device_combo", input_devices[0])
        else:
            dpg.set_value("input_device_combo", "")
        self.update_device(None, dpg.get_value("input_device_combo"))

        output_devices = self.get_output_devices(host_api)
        dpg.configure_item("output_device_combo", items=output_devices)
        if output_devices:
            if not dpg.get_value("output_device_combo") or dpg.get_value("output_device_combo") not in output_devices:
                dpg.set_value("output_device_combo", output_devices[0])
        else:
            dpg.set_value("output_device_combo", "")
        self.update_output_device(None, dpg.get_value("output_device_combo"))

    def update_device(self, sender, app_data):
        device_name = app_data
        host_api_name = dpg.get_value("host_api_combo")
        device_index = self.get_device_index(device_name, host_api_name)
        if device_index is None:
            return
        device_info = sd.query_devices()[device_index]
        supported_sample_rates = []
        try:
            for sr in SAMPLE_RATES:
                try:
                    sd.check_input_settings(device=device_index, samplerate=int(sr), channels=2, dtype='int16')
                    supported_sample_rates.append(sr)
                except:
                    continue
        except:
            supported_sample_rates = [str(int(device_info['default_samplerate']))]
        dpg.configure_item("sample_rate_combo", items=supported_sample_rates)
        if dpg.get_value("sample_rate_combo") not in supported_sample_rates:
            dpg.set_value("sample_rate_combo", max(supported_sample_rates, key=int))
        if self.is_testing:
            self.stop_audio_test()
            self.toggle_audio_test()
        self.save_settings()

    def update_output_device(self, sender, app_data):
        if self.is_testing:
            self.stop_audio_test()
            self.toggle_audio_test()
        self.save_settings()

    def update_bit_depth(self, sender, app_data):
        if self.is_testing:
            self.stop_audio_test()
            self.toggle_audio_test()
        self.save_settings()

    def update_sample_rate(self, sender, app_data):
        if self.is_testing:
            self.stop_audio_test()
            self.toggle_audio_test()
        self.save_settings()

    def update_unit(self, sender, app_data):
        unit = app_data
        slider_value = dpg.get_value("sensitivity_slider")
        if unit == "Numbers":
            pre_scale = self.slider_to_pre_scale(slider_value)
            dpg.configure_item("sensitivity_slider", min_value=0, max_value=100, default_value=self.pre_scale_to_slider(pre_scale))
        elif unit == "Percent":
            percent = self.pre_scale_to_percent(self.slider_to_pre_scale(slider_value))
            dpg.configure_item("sensitivity_slider", min_value=0, max_value=100, default_value=percent)
        elif unit == "dB":
            db = self.pre_scale_to_db(self.slider_to_pre_scale(slider_value))
            dpg.configure_item("sensitivity_slider", min_value=0, max_value=100, default_value=(db + 60) / 100 * 100)
        self.update_pre_scale_label(None, dpg.get_value("sensitivity_slider"))
        self.save_settings()

    def update_pre_scale_label(self, sender, app_data):
        unit = dpg.get_value("unit_combo")
        slider_value = app_data if app_data is not None else dpg.get_value("sensitivity_slider")
        if unit == "Numbers":
            pre_scale = self.slider_to_pre_scale(slider_value)
            dpg.set_value("pre_scale_label", f"{pre_scale:.3f}")
        elif unit == "Percent":
            percent = self.pre_scale_to_percent(self.slider_to_pre_scale(slider_value))
            dpg.set_value("pre_scale_label", f"{percent:.1f}%")
        elif unit == "dB":
            db = (slider_value / 100) * 100 - 60
            dpg.set_value("pre_scale_label", f"{db:.1f} dB")

    def update_relative_sensitivity(self, sender, app_data):
        bit_depth = dpg.get_value("bit_depth_combo")
        dtype, max_value = self.get_dtype_and_max(bit_depth)
        slider_value = dpg.get_value("sensitivity_slider")
        unit = dpg.get_value("unit_combo")
        if unit == "Numbers":
            pre_scale = self.slider_to_pre_scale(slider_value)
        elif unit == "Percent":
            pre_scale = self.percent_to_pre_scale(slider_value)
        elif unit == "dB":
            db = (slider_value / 100) * 100 - 60
            pre_scale = self.db_to_pre_scale(db)
        
        if dpg.get_value("relative_sensitivity_check"):
            reference_max = 32767
            scale_factor = reference_max / (max_value + (1 if dtype != "float32" else 0))
            pre_scale /= scale_factor
            pre_scale = min(pre_scale, 1.0)
            adjusted_pre_scale = pre_scale / scale_factor
        else:
            adjusted_pre_scale = pre_scale
        self.set_slider_from_pre_scale(adjusted_pre_scale)
        self.save_settings()

    def slider_to_pre_scale(self, slider_value):
        if slider_value <= 0:
            return 0.0001
        log_min = math.log10(0.0001)
        log_max = math.log10(99.999)
        log_pre_scale = (slider_value / 100) * (log_max - log_min) + log_min
        return 10 ** log_pre_scale

    def pre_scale_to_slider(self, pre_scale):
        if pre_scale <= 0:
            return 0
        log_pre_scale = math.log10(pre_scale)
        log_min = math.log10(0.0001)
        log_max = math.log10(99.999)
        return max(0, min(100, (log_pre_scale - log_min) / (log_max - log_min) * 100))

    def percent_to_pre_scale(self, percent):
        if percent <= 0:
            return 0.0001
        log_min = math.log10(0.0001)
        log_max = math.log10(99.999)
        log_pre_scale = (percent / 100) * (log_max - log_min) + log_min
        return 10 ** log_pre_scale

    def pre_scale_to_percent(self, pre_scale):
        if pre_scale <= 0.0001:
            return 0.0
        if pre_scale >= 10.0:
            return 100.0
        log_pre_scale = math.log10(pre_scale)
        log_min = math.log10(0.0001)
        log_max = math.log10(99.999)
        return (log_pre_scale - log_min) / (log_max - log_min) * 100

    def db_to_pre_scale(self, db):
        return 10 ** (db / 20)

    def pre_scale_to_db(self, pre_scale):
        return -float("inf") if pre_scale <= 0 else 20 * math.log10(pre_scale)

    def get_dtype_and_max(self, bit_depth=None):
        bit_depth = bit_depth or dpg.get_value("bit_depth_combo")
        if " - " in bit_depth:
            bit_depth = bit_depth.split(" - ")[0]
        if bit_depth == "int8":
            return "int8", 127
        elif bit_depth == "int16":
            return "int16", 32767
        elif bit_depth == "int24":
            return "int32", 8388607
        elif bit_depth == "int32":
            return "int32", 2147483647
        elif bit_depth == "float32":
            return "float32", 1.0
        else:
            raise ValueError(f"Invalid bit depth: {bit_depth}")

    def update_shadow(self):
        shadow_position = (self.noise_floor / 32767) * 400
        if dpg.does_item_exist("shadow_bar"):
            dpg.configure_item("shadow_bar", pmax=(shadow_position, 20))

    def update_audio_level(self, data):
        if len(data) == 0:
            level = 0
            shadow = 0
            clipping = False
        else:
            peak = np.max(np.abs(data))
            self.peak_amplitude = max(self.peak_amplitude, peak)
            current_time = time.time()
            if current_time - self.last_noise_update >= 1.0:
                self.noise_floor = min(self.noise_floor, peak) if self.noise_floor > 0 else peak
                self.last_noise_update = current_time
            level = (peak - self.noise_floor) / (32768 - self.noise_floor) if self.noise_floor < 32768 else 0
            level = max(0, min(1, level))
            shadow = (self.peak_amplitude - self.noise_floor) / (32768 - self.noise_floor) if self.noise_floor < 32768 else 0
            shadow = max(0, min(1, shadow))
            clipping = peak >= 32767
        dpg.configure_item("level_bar", pmax=(level * 400, 20))
        dpg.configure_item("shadow_bar", pmax=(shadow * 400, 20))
        dpg.configure_item("clipping_indicator", fill=(255, 0, 0, 255 if clipping else 128))

    def toggle_audio_test(self, sender=None, app_data=None):
        if not self.is_testing:
            self.is_testing = True
            dpg.configure_item("test_audio_button", label="Stop Testing")
            self.start_audio_test()
        else:
            self.is_testing = False
            dpg.configure_item("test_audio_button", label="Test Audio")
            self.stop_audio_test()

    def start_audio_test(self):
        if self.is_testing:
            self.stop_audio_test()
        
        dtype, max_value = self.get_dtype_and_max()

        def input_callback(indata, frames, time_info, status):
            if not self.is_testing:
                return
            indata_array = np.frombuffer(indata, dtype=dtype).reshape(-1, 1)
            indata_left = indata_array[:, 0].astype(np.float32) / (max_value + (1 if dtype != "float32" else 0))
            if dtype == "int8":
                indata_normalized = indata_left * (32767 / 127)
            elif dtype == "int16":
                indata_normalized = indata_left
            elif dtype == "int24" or dtype == "int32":
                indata_normalized = indata_left * (32767 / (8388607 if dtype == "int24" else 2147483647))
            elif dtype == "float32":
                indata_normalized = indata_left * 32767
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")
            unit = dpg.get_value("unit_combo")
            slider_value = dpg.get_value("sensitivity_slider")
            if unit == "Numbers":
                pre_scale = self.slider_to_pre_scale(slider_value)
            elif unit == "Percent":
                pre_scale = self.percent_to_pre_scale(slider_value)
            elif unit == "dB":
                db = (slider_value / 100) * 100 - 60
                pre_scale = self.db_to_pre_scale(db)
            else:
                pre_scale = 1.0
            if dpg.get_value("relative_sensitivity_check"):
                scale_factor = 32767 / (max_value + (1 if dtype != "float32" else 0))
                pre_scale /= scale_factor
            indata_scaled = indata_normalized * pre_scale
            scaled_data = (indata_scaled * 32767).astype(np.int16)
            self.audio_buffer.extend(scaled_data)  # For WaveformDisplay
            max_amplitude = np.max(np.abs(scaled_data))
            self.gui_update_queue.put(("update_level", max_amplitude))
            alpha = 0.1
            self.noise_floor = alpha * max_amplitude + (1 - alpha) * self.noise_floor
            if dpg.get_value("show_peaks_check"):
                if max_amplitude > self.peak_amplitude:
                    self.peak_amplitude = max_amplitude
                    self.noise_floor = self.peak_amplitude
                    self.gui_update_queue.put(("update_shadow", None))
            elif time.time() - self.last_noise_update >= 5:
                self.last_noise_update = time.time()
                self.gui_update_queue.put(("update_shadow", None))
            scaled_data = (indata_scaled * max_value).astype(dtype)
            data_stereo = np.repeat(scaled_data[:, np.newaxis], 2, axis=1)
            if not self.audio_queue.full():
                self.audio_queue.put_nowait(data_stereo)
            if self.is_recording:
                self.audio_buffer.append(data_stereo.flatten().copy())
            if self.is_debug_recording:
                self.debug_audio_buffer.append(data_stereo.flatten().copy())

        def output_callback(outdata, frames, time_info, status):
            if not self.is_testing:
                outdata[:] = np.zeros((frames, 2), dtype=dtype)
                return
            try:
                data = self.audio_queue.get_nowait()
                num_samples = data.shape[0]
                if num_samples >= frames:
                    outdata[:] = data[:frames, :]
                else:
                    outdata[:num_samples, :] = data
                    outdata[num_samples:, :].fill(0)
            except queue.Empty:
                outdata[:] = np.zeros((frames, 2), dtype=dtype)

        try:
            input_device_index = self.get_device_index(dpg.get_value("input_device_combo"), dpg.get_value("host_api_combo"))
            output_device_index = self.get_device_index(dpg.get_value("output_device_combo"), dpg.get_value("host_api_combo"), is_input=False)
            sample_rate = int(dpg.get_value("sample_rate_combo"))
            self.audio_stream = sd.RawInputStream(
                samplerate=sample_rate, blocksize=256, dtype=dtype, channels=1,
                callback=input_callback, device=input_device_index, latency='low'
            )
            self.audio_stream.start()
            self.output_stream = sd.RawOutputStream(
                samplerate=sample_rate, blocksize=256, dtype=dtype, channels=2,
                callback=output_callback, device=output_device_index, latency='low'
            )
            self.output_stream.start()

            def update_gui():
                while not self.gui_update_queue.empty():
                    update_type, value = self.gui_update_queue.get()
                    if update_type == "update_level":
                        max_amplitude = value
                        dpg.configure_item("clipping_indicator", fill=(255, 0, 0, 255 if max_amplitude >= 32767 else 128))
                        level_position = (max_amplitude / 32767) * 400
                        dpg.configure_item("level_bar", pmax=(level_position, 20))
                    elif update_type == "update_shadow":
                        self.update_shadow()
                if self.is_testing:
                    dpg.set_frame_callback(dpg.get_frame_count() + 1, update_gui)
            dpg.set_frame_callback(dpg.get_frame_count() + 1, update_gui)
        except Exception as e:
            dpg.set_value("status_text", f"Failed to start audio test: {e}")
            self.is_testing = False
            dpg.configure_item("test_audio_button", label="Test Audio")

    def stop_audio_test(self):
        self.is_testing = False
        if self.audio_stream:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except Exception as e:
                logging.error(f"Error stopping input stream: {e}")
            self.audio_stream = None
        if self.output_stream:
            try:
                self.output_stream.stop()
                self.output_stream.close()
            except Exception as e:
                logging.error(f"Error stopping output stream: {e}")
            self.output_stream = None
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        while not self.gui_update_queue.empty():
            try:
                self.gui_update_queue.get_nowait()
            except queue.Empty:
                break
        dpg.configure_item("level_bar", pmax=(0, 20))
        dpg.configure_item("clipping_indicator", fill=(255, 0, 0, 128))

    def toggle_recording(self, sender=None, app_data=None):
        if not self.is_recording:
            if not self.is_testing:
                dpg.set_value("status_text", "Start audio testing before recording.")
                return
            if not os.path.exists(self.recordings_dir):
                dpg.set_value("status_text", "No valid recordings directory.")
                return
            self.is_recording = True
            dpg.configure_item("record_button", label="Stop Recording")
            self.audio_buffer.clear()
        else:
            self.is_recording = False
            dpg.configure_item("record_button", label="Record")
            self.save_recording()

    def save_recording(self):
        if not self.audio_buffer:
            dpg.set_value("status_text", "No audio recorded.")
            return
        audio_data = np.concatenate([np.array(self.audio_buffer)])
        bit_depth = dpg.get_value("bit_depth_combo")
        if " - " in bit_depth:
            bit_depth = bit_depth.split(" - ")[0]
        sample_width = 1 if bit_depth == "int8" else 2 if bit_depth == "int16" else 3 if bit_depth == "int24" else 4
        filename = os.path.join(self.recordings_dir, f"recording_{int(time.time())}.wav")
        try:
            with wave.open(filename, "wb") as wf:
                wf.setnchannels(2)
                wf.setsampwidth(sample_width)
                wf.setframerate(int(dpg.get_value("sample_rate_combo")))
                wf.writeframes(audio_data.tobytes())
            dpg.set_value("status_text", f"Recording saved as {filename}")
            logging.info(f"Recording saved as {filename}")
        except Exception as e:
            dpg.set_value("status_text", f"Failed to save recording: {e}")
            logging.error(f"Failed to save recording: {e}")

    def save_debug_recording(self):
        if not self.debug_audio_buffer:
            return
        audio_data = np.concatenate(self.debug_audio_buffer)
        bit_depth = dpg.get_value("bit_depth_combo")
        if " - " in bit_depth:
            bit_depth = bit_depth.split(" - ")[0]
        sample_width = 1 if bit_depth == "int8" else 2 if bit_depth == "int16" else 3 if bit_depth == "int24" else 4
        filename = os.path.join(self.recordings_dir, f"debug_recording_{int(time.time())}.wav")
        try:
            with wave.open(filename, "wb") as wf:
                wf.setnchannels(2)
                wf.setsampwidth(sample_width)
                wf.setframerate(int(dpg.get_value("sample_rate_combo")))
                wf.writeframes(audio_data.tobytes())
            logging.info(f"Debug recording saved as {filename}")
        except Exception as e:
            logging.error(f"Failed to save debug recording: {e}")

    def suggest_settings(self, sender=None, app_data=None):
        dpg.set_value("sample_rate_combo", "16000")
        dpg.set_value("bit_depth_combo", "int16")
        self.set_slider_from_pre_scale(0.002)
        dpg.set_value("unit_combo", "Numbers")
        dpg.set_value("silence_input", 10.0)
        dpg.set_value("relative_sensitivity_check", False)
        self.update_pre_scale_label(None, self.pre_scale_to_slider(0.002))
        self.save_settings()
        dpg.set_value("status_text", "Suggested settings applied.")

    def calibrate(self, sender=None, app_data=None):
        if self.is_testing:
            dpg.set_value("status_text", "Stop audio testing before calibrating.")
            return
        dtype, max_value = self.get_dtype_and_max()
        self.audio_buffer.clear()
        calibration_duration = 5

        def calibration_callback(indata, frames, time_info, status):
            data = np.frombuffer(indata, dtype=dtype).reshape(-1, 2)
            data = data[:, 0].astype(np.float32) / (max_value + (1 if dtype != "float32" else 0))
            data_stereo = np.zeros((len(data), 2), dtype=dtype)
            data_stereo[:, 0] = (data * max_value).astype(dtype)
            self.audio_buffer.append(data_stereo.flatten().copy())

        dpg.set_value("status_text", "Speak normally for 5 seconds.")
        try:
            device_index = self.get_device_index(dpg.get_value("input_device_combo"), dpg.get_value("host_api_combo"))
            stream = sd.RawInputStream(
                samplerate=int(dpg.get_value("sample_rate_combo")), blocksize=32000, dtype=dtype,
                channels=2, callback=calibration_callback, device=device_index
            )
            stream.start()
            time.sleep(calibration_duration)
            stream.stop()
            stream.close()
        except Exception as e:
            dpg.set_value("status_text", f"Calibration failed: {e}")
            logging.error(f"Calibration failed: {e}")
            return

        if not self.audio_buffer:
            dpg.set_value("status_text", "No audio captured.")
            return

        audio_data = np.concatenate([np.array(self.audio_buffer)]).reshape(-1, 2)[:, 0].astype(np.float32) / (max_value + 1)
        peak_amplitude = np.max(np.abs(audio_data))
        target_amplitude = 0.5
        pre_scale = target_amplitude / (peak_amplitude + 1e-10)
        self.set_slider_from_pre_scale(pre_scale)
        self.save_settings()
        dpg.set_value("status_text", f"Sensitivity adjusted to {dpg.get_value('pre_scale_label')}.")

    def open_manual_sensitivity_input(self, sender, app_data):
        unit = self.saved_settings.get("unit", "Numbers")
        slider_value = self.saved_settings.get("pre_scale_factor", 0.002)
        if unit == "Numbers":
            default_value = slider_value
        elif unit == "Percent":
            default_value = self.pre_scale_to_percent(slider_value)
        elif unit == "dB":
            default_value = self.pre_scale_to_db(slider_value)
        with dpg.window(label=f"Set Sensitivity ({unit})", modal=True, width=300, height=150, tag="manual_sensitivity_window"):
            dpg.add_text(f"Enter sensitivity value in {unit}:")
            dpg.add_input_float(default_value=default_value, tag="manual_sensitivity_input", step=0.0001, format="%.4f")
            with dpg.group(horizontal=True):
                dpg.add_button(label="Save", callback=self.save_manual_sensitivity)
                dpg.add_button(label="Cancel", callback=lambda: dpg.delete_item("manual_sensitivity_window"))

    def save_manual_sensitivity(self, sender, app_data):
        unit = self.saved_settings.get("unit", "Numbers")
        input_value = dpg.get_value("manual_sensitivity_input")
        if unit == "Numbers":
            pre_scale = input_value
        elif unit == "Percent":
            pre_scale = self.percent_to_pre_scale(input_value)
        elif unit == "dB":
            pre_scale = self.db_to_pre_scale(input_value)
        self.saved_settings["pre_scale_factor"] = pre_scale
        self.save_settings()
        dpg.delete_item("manual_sensitivity_window")
        slider_value = self.pre_scale_to_slider(pre_scale)
        dpg.set_value("sensitivity_slider", slider_value)

    def set_slider_from_pre_scale(self, pre_scale):
        unit = dpg.get_value("unit_combo")
        if unit == "Numbers":
            slider_value = self.pre_scale_to_slider(pre_scale)
        elif unit == "Percent":
            slider_value = self.pre_scale_to_percent(pre_scale)
        elif unit == "dB":
            db = self.pre_scale_to_db(pre_scale)
            slider_value = (db + 60) / 100 * 100
        dpg.set_value("sensitivity_slider", slider_value)
        self.update_pre_scale_label(None, slider_value)

    def reset_settings(self, sender=None, app_data=None):
        default_settings = {
            "bit_depth": "int24",
            "sample_rate": "48000",
            "pre_scale_factor": 0.002,
            "unit": "Numbers",
            "relative_sensitivity": False,
            "silence_threshold": 10.0,
            "show_peaks": False,
            "host_api": "MME",
            "input_device": None,
            "output_device": None
        }
        self.saved_settings = default_settings
        dpg.set_value("bit_depth_combo", default_settings["bit_depth"])
        dpg.set_value("sample_rate_combo", default_settings["sample_rate"])
        self.set_slider_from_pre_scale(default_settings["pre_scale_factor"])
        dpg.set_value("unit_combo", default_settings["unit"])
        dpg.set_value("relative_sensitivity_check", default_settings["relative_sensitivity"])
        dpg.set_value("silence_input", default_settings["silence_threshold"])
        dpg.set_value("show_peaks_check", default_settings["show_peaks"])
        dpg.set_value("host_api_combo", default_settings["host_api"])
        dpg.set_value("input_device_combo", default_settings["input_device"])
        dpg.set_value("output_device_combo", default_settings.get("output_device", ""))
        self.update_host_api(None, default_settings["host_api"])
        self.save_settings()
        dpg.set_value("status_text", "Settings reset to defaults and saved.")

class WaveformDisplay:
    def __init__(self, AudioProcessor):
        self.AudioProcessor = AudioProcessor
        self.samplerate = int(self.AudioProcessor.saved_settings.get("sample_rate", 48000))
        self.bit_depth = self.AudioProcessor.saved_settings.get("bit_depth", "int24")
        self.pre_scale_factor = self.AudioProcessor.saved_settings.get("pre_scale_factor", 0.002)
        self.relative_sensitivity = self.AudioProcessor.saved_settings.get("relative_sensitivity", False)
        self.channels = 2  # Assume stereo from AudioProcessor
        self.is_running = True
        self.audio_buffer = deque(maxlen=int(self.samplerate * 1.0))  # 1 second of data
        self.update_interval = 1.0 / 30  # Update at 30 FPS
        self.last_update = 0

        # Determine dtype and max_value
        self.dtype, self.max_value = self.get_dtype_and_max()
        
        # DPG setup (use existing context)
        with dpg.window(label="Waveform Display", tag="waveform_window", width=800, height=300, on_close=self.close):
            with dpg.plot(label="Waveform (16-bit PCM, as Vosk hears)", height=200, width=-1):
                dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag="waveform_x_axis")
                dpg.add_plot_axis(dpg.mvYAxis, label="Amplitude", tag="waveform_y_axis")
                dpg.set_axis_limits("waveform_y_axis", -32768, 32768)
                dpg.set_axis_limits("waveform_x_axis", 0, 1.0)
                self.time_axis = np.linspace(0, 1.0, int(self.samplerate * 1.0))
                self.audio_data = np.zeros(int(self.samplerate * 1.0))
                dpg.add_line_series(self.time_axis, self.audio_data, label="Waveform", parent="waveform_y_axis", tag="waveform_series")

        # Schedule the update_waveform method
        dpg.set_frame_callback(dpg.get_frame_count() + 1, self.update_waveform)

    def get_dtype_and_max(self):
        bit_depth = self.bit_depth
        if " - " in bit_depth:
            bit_depth = bit_depth.split(" - ")[0]
        if bit_depth == "int8":
            return "int8", 127
        elif bit_depth == "int16":
            return "int16", 32767
        elif bit_depth == "int24":
            return "int32", 8388607
        elif bit_depth == "int32":
            return "int32", 2147483647
        elif bit_depth == "float32":
            return "float32", 1.0
        else:
            raise ValueError(f"Invalid bit depth: {bit_depth}")

    def update_waveform(self):
        if not self.is_running or not dpg.does_item_exist("waveform_window"):
            return
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            try:
                if self.AudioProcessor.is_testing and self.AudioProcessor.audio_buffer:
                    # Get recent audio data from AudioProcessor (last 1 second)
                    audio_data = np.concatenate(self.AudioProcessor.audio_buffer)[-int(self.samplerate * 1.0):]
                    # Assuming stereo, take left channel
                    audio_data = audio_data[::2][:int(self.samplerate * 1.0)]
                    # Normalize to 16-bit range based on bit depth
                    if self.dtype == "int8":
                        data_normalized = audio_data * (32767 / 127)
                    elif self.dtype == "int16":
                        data_normalized = audio_data
                    elif self.dtype == "int24" or self.dtype == "int32":
                        data_normalized = audio_data * (32767 / (8388607 if self.dtype == "int24" else 2147483647))
                    elif self.dtype == "float32":
                        data_normalized = audio_data * 32767
                    else:
                        raise ValueError(f"Unsupported dtype: {self.dtype}")
                    # Apply pre-scale factor
                    data_scaled = data_normalized * self.pre_scale_factor
                    data_scaled = np.clip(data_scaled, -32768, 32767)
                    self.audio_buffer.clear()
                    self.audio_buffer.extend(data_scaled)
                else:
                    # Clear buffer if not testing
                    self.audio_buffer.clear()
                
                # Update plot data
                self.audio_data = np.array(self.audio_buffer)
                if len(self.audio_data) < len(self.time_axis):
                    self.audio_data = np.pad(self.audio_data, (0, len(self.time_axis) - len(self.audio_data)), mode='constant')
                else:
                    self.audio_data = self.audio_data[-len(self.time_axis):]
                dpg.set_value("waveform_series", [self.time_axis, self.audio_data])
                self.last_update = current_time
            except Exception as e:
                logging.error(f"Error updating waveform: {e}")
        # Schedule the next update
        dpg.set_frame_callback(dpg.get_frame_count() + 1, self.update_waveform)

    def close(self, sender, app_data):
        self.is_running = False
        dpg.delete_item("waveform_window")


class DictationGUI:
    def __init__(self, AudioProcessor):
        self.AudioProcessor = AudioProcessor        
        self.dictation_process = None
        self.monitor_stop_event = None
        self.saved_settings = self.load_config()
        self.model_path = self.saved_settings.get("model_path", None)

        if not self.model_path or not os.path.exists(self.model_path):
            logging.error(f"Vosk model not found at {self.model_path}.")
            raise FileNotFoundError(f"Vosk model not found at {self.model_path}.")

        self.AudioProcessor = AudioProcessor        
        self.waveform = WaveformDisplay(self.AudioProcessor)
        self.default_recordings_dir = os.path.join(os.getcwd(), "Recordings")
        self.recordings_dir = self.default_recordings_dir
        try:
            if not os.path.exists(self.default_recordings_dir):
                os.makedirs(self.default_recordings_dir)
        except Exception as e:
            logging.error(f"Failed to create recordings directory: {e}")
            self.default_recordings_dir = os.getcwd()
            self.recordings_dir = self.default_recordings_dir
        
        self.is_dictating = False
        self.recognizer = None
        self.dictation_stream = None
        self.pa = None
        self.command_queue = queue.Queue()
        self.transcribed_text = []
        self.last_command = ""
        self.last_command_time = 0
        self.COMMAND_DEBOUNCE_TIME = 0.5
        self.spell_checker = SpellChecker()
        self.caps_lock_on = False
        self.command_progress = 0.0
        self.command_progress_max = 100.0
        self.command_status = ""
        self.saved_settings = {}
        self.json_data = {}
        self.commands = {}
        self.load_commands()
        self.theme = "Dark"

    def load_config(self):
        config_path = os.path.join(os.getcwd(), "config.json")
        if not os.path.exists(config_path):
            logging.error(f"Configuration file not found at {config_path}.")
            raise FileNotFoundError(f"Configuration file not found at {config_path}.")
        
        with open(config_path, 'r') as config_file:
            return json.load(config_file)
        
        dpg.create_context()
        dpg.create_viewport(title="Speech-to-Text Dictation Configuration", width=800, height=600)
        
        with dpg.theme(tag="dark_theme"):
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (46, 46, 46))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Button, (74, 74, 74))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (74, 74, 74))
        
        with dpg.theme(tag="light_theme"):
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (200, 200, 200))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (0, 0, 0))
                dpg.add_theme_color(dpg.mvThemeCol_Button, (150, 150, 150))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (150, 150, 150))
        
        with dpg.window(label="Speech-to-Text Dictation Configuration", tag="primary_window"):
            with dpg.tab_bar(tag="main_tab_bar"):
                with dpg.tab(label="Audio Settings", tag="audio_tab"):
                    self.AudioProcessor.setup_audio_tab()
                for tab_name, json_path in JSON_FILES.items():
                    with dpg.tab(label=tab_name, tag=f"{tab_name}_tab"):
                        self.setup_json_tab(tab_name, json_path)
            
            with dpg.group(horizontal=True):
                dpg.add_button(label="Start Dictation", tag="start_dictation_button", callback=self.start_dictation)
                dpg.add_button(label="Stop Dictation", tag="stop_dictation_button", callback=self.stop_dictation, enabled=False)
                dpg.add_button(label="Save Settings", callback=self.save_settings)
                dpg.add_combo(["Dark", "Light"], default_value="Dark", callback=self.apply_theme, tag="theme_combo")
            
            dpg.add_text("Transcription Output:")
            dpg.add_text("", tag="output_text", wrap=780)
            
            dpg.add_text("Command Progress:")
            dpg.add_progress_bar(default_value=0.0, width=780, tag="command_progress_bar")
            dpg.add_text("", tag="command_status_text")
        
        self.apply_theme("Dark")
        self.load_settings()
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("primary_window", True)
        self.create_file_dialog()

    def load_commands(self):
        try:
            base_path = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(base_path, COMMANDS_JSON_PATH)
            if not os.path.exists(full_path):
                logging.warning(f"commands.json not found at {full_path}. Using default commands.")
                self.commands = {"simple_commands": {}, "parameterized_commands": []}
                return
            with open(full_path, "r", encoding="utf-8") as f:
                self.commands = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load commands.json: {e}")
            self.commands = {"simple_commands": {}, "parameterized_commands": []}

    def type_text(self, text):
        try:
            if text == "\n\n":
                keyboard.press_and_release("enter")
                keyboard.press_and_release("enter")
            else:
                keyboard.write(text)
            logging.debug(f"GUI typed: {text}")
            dpg.set_value("status_text", f"Typed: {text}")
        except Exception as e:
            logging.error(f"GUI error typing text: {e}")
            dpg.set_value("status_text", f"Error typing text: {e}")

    def update_command_progress(self, value, status):
        self.command_progress = value
        self.command_status = status
        dpg.set_value("command_progress_bar", value / self.command_progress_max)
        dpg.set_value("command_status_text", status)

    def handle_command(self, action):
        logging.info(f"GUI handling command action: {action}")
        self.update_command_progress(10.0, f"Executing command: {action}")
        try:
            if action == "cmd_select_all":
                keyboard.press_and_release("ctrl+a")
                self.update_command_progress(100.0, "Selected all.")
            elif action == "cmd_select_down":
                keyboard.press("shift")
                keyboard.press_and_release("down")
                keyboard.release("shift")
                self.update_command_progress(100.0, "Selected down.")
            elif action == "cmd_select_up":
                keyboard.press("shift")
                keyboard.press_and_release("up")
                keyboard.release("shift")
                self.update_command_progress(100.0, "Selected up.")
            elif action == "cmd_select_all_up":
                keyboard.press("shift")
                keyboard.press_and_release("home")
                keyboard.release("shift")
                self.update_command_progress(100.0, "Selected to start.")
            elif action == "cmd_select_all_down":
                keyboard.press("shift")
                keyboard.press_and_release("end")
                keyboard.release("shift")
                self.update_command_progress(100.0, "Selected to end.")
            elif action == "cmd_copy":
                keyboard.press_and_release("ctrl+c")
                self.update_command_progress(100.0, "Copied selection.")
            elif action == "cmd_paste":
                keyboard.press_and_release("ctrl+v")
                self.update_command_progress(100.0, "Pasted content.")
            elif action == "cmd_delete":
                keyboard.press_and_release("backspace")
                self.update_command_progress(100.0, "Deleted character.")
            elif action == "cmd_undo":
                keyboard.press_and_release("ctrl+z")
                self.update_command_progress(100.0, "Undo performed.")
            elif action == "cmd_redo":
                keyboard.press_and_release("ctrl+y")
                self.update_command_progress(100.0, "Redo performed.")
            elif action == "cmd_file_properties":
                keyboard.press_and_release("menu")
                self.update_command_progress(100.0, "Opened file properties.")
            elif action == "cmd_save_document":
                keyboard.press_and_release("ctrl+s")
                self.update_command_progress(100.0, "Saved document.")
            elif action == "cmd_open_file":
                self.update_command_progress(100.0, "Open file not implemented.")
            elif action == "cmd_move_up":
                keyboard.press_and_release("up")
                self.update_command_progress(100.0, "Moved cursor up.")
            elif action == "cmd_move_down":
                keyboard.press_and_release("down")
                self.update_command_progress(100.0, "Moved cursor down.")
            elif action == "cmd_move_left":
                keyboard.press_and_release("left")
                self.update_command_progress(100.0, "Moved cursor left.")
            elif action == "cmd_move_right":
                keyboard.press_and_release("right")
                self.update_command_progress(100.0, "Moved cursor right.")
            elif action == "cmd_move_up_paragraph":
                keyboard.press_and_release("ctrl+up")
                self.update_command_progress(100.0, "Moved up paragraph.")
            elif action == "cmd_move_down_paragraph":
                keyboard.press_and_release("ctrl+down")
                self.update_command_progress(100.0, "Moved down paragraph.")
            elif action == "cmd_enter":
                keyboard.press_and_release("enter")
                self.transcribed_text.append("\n")
                full_text = "".join(self.transcribed_text).rstrip()
                dpg.set_value("output_text", full_text)
                with open("dictation_output_gigaspeech.txt", "a", encoding="utf-8") as f:
                    f.write("\n")
                self.update_command_progress(100.0, "Pressed Enter.")
            elif action == "cmd_number_lock":
                global number_lock_on
                number_lock_on = not number_lock_on
                self.update_command_progress(100.0, f"Number lock {'on' if number_lock_on else 'off'}.")
            elif action == "cmd_caps_lock_on":
                if not self.caps_lock_on:
                    keyboard.press_and_release("caps lock")
                    self.caps_lock_on = True
                self.update_command_progress(100.0, "Caps lock on.")
            elif action == "cmd_caps_lock_off":
                if self.caps_lock_on:
                    keyboard.press_and_release("caps lock")
                    self.caps_lock_on = False
                self.update_command_progress(100.0, "Caps lock off.")
            elif action == "cmd_bold":
                keyboard.press_and_release("ctrl+b")
                self.update_command_progress(100.0, "Applied bold.")
            elif action == "cmd_italicize":
                keyboard.press_and_release("ctrl+i")
                self.update_command_progress(100.0, "Applied italic.")
            elif action == "cmd_underline":
                keyboard.press_and_release("ctrl+u")
                self.update_command_progress(100.0, "Applied underline.")
            elif action == "cmd_center":
                keyboard.press_and_release("ctrl+e")
                self.update_command_progress(100.0, "Centered text.")
            elif action == "cmd_left_align":
                keyboard.press_and_release("ctrl+l")
                self.update_command_progress(100.0, "Left aligned.")
            elif action == "cmd_right_align":
                keyboard.press_and_release("ctrl+r")
                self.update_command_progress(100.0, "Right aligned.")
            elif action == "cmd_cut":
                keyboard.press_and_release("ctrl+x")
                self.update_command_progress(100.0, "Cut selection.")
            elif action == "cmd_go_to_beginning":
                keyboard.press_and_release("ctrl+home")
                self.update_command_progress(100.0, "Moved to beginning.")
            elif action == "cmd_go_to_end":
                keyboard.press_and_release("ctrl+end")
                self.update_command_progress(100.0, "Moved to end.")
            elif action == "cmd_go_to_beginning_of_line":
                keyboard.press_and_release("home")
                self.update_command_progress(100.0, "Moved to line start.")
            elif action == "cmd_go_to_end_of_line":
                keyboard.press_and_release("end")
                self.update_command_progress(100.0, "Moved to line end.")
            elif action == "cmd_go_to_address":
                keyboard.press_and_release("ctrl+l")
                self.update_command_progress(100.0, "Focused address bar.")
            elif action == "cmd_refresh_page":
                keyboard.press_and_release("f5")
                self.update_command_progress(100.0, "Refreshed page.")
            elif action == "cmd_go_back":
                keyboard.press_and_release("alt+left")
                self.update_command_progress(100.0, "Went back.")
            elif action == "cmd_go_forward":
                keyboard.press_and_release("alt+right")
                self.update_command_progress(100.0, "Went forward.")
            elif action == "cmd_open_new_tab":
                keyboard.press_and_release("ctrl+t")
                self.update_command_progress(100.0, "Opened new tab.")
            elif action == "cmd_close_tab":
                keyboard.press_and_release("ctrl+w")
                self.update_command_progress(100.0, "Closed tab.")
            elif action == "cmd_next_tab":
                keyboard.press_and_release("ctrl+tab")
                self.update_command_progress(100.0, "Switched to next tab.")
            elif action == "cmd_previous_tab":
                keyboard.press_and_release("ctrl+shift+tab")
                self.update_command_progress(100.0, "Switched to previous tab.")
            elif action == "cmd_shift_tab":
                keyboard.press_and_release("shift+tab")
                self.update_command_progress(100.0, "Shift-tabbed.")
            elif action == "cmd_scratch_that":
                global last_dictated_length
                for _ in range(last_dictated_length):
                    keyboard.press_and_release("backspace")
                if self.transcribed_text:
                    self.transcribed_text.pop()
                full_text = "".join(self.transcribed_text).rstrip()
                dpg.set_value("output_text", full_text)
                with open("dictation_output_gigaspeech.txt", "w", encoding="utf-8") as f:
                    f.write(full_text)
                self.update_command_progress(100.0, "Last dictation removed.")
            elif action == "cmd_click_that":
                subprocess.run(["powershell", "-Command", "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.Cursor]::Position = [System.Windows.Forms.Cursor]::Position; [System.Windows.Forms.Mouse]::LeftClick()"])
                self.update_command_progress(100.0, "Clicked mouse.")
            elif action == "cmd_punctuation_period":
                self.type_text(".")
                self.update_command_progress(100.0, "Typed period.")
            elif action == "cmd_punctuation_comma":
                self.type_text(",")
                self.update_command_progress(100.0, "Typed comma.")
            elif action == "cmd_punctuation_question_mark":
                self.type_text("?")
                self.update_command_progress(100.0, "Typed question mark.")
            elif action == "cmd_punctuation_exclamation":
                self.type_text("!")
                self.update_command_progress(100.0, "Typed exclamation.")
            elif action == "cmd_punctuation_semicolon":
                self.type_text(";")
                self.update_command_progress(100.0, "Typed semicolon.")
            elif action == "cmd_punctuation_colon":
                self.type_text(":")
                self.update_command_progress(100.0, "Typed colon.")
            elif action == "cmd_punctuation_tilde":
                self.type_text("~")
                self.update_command_progress(100.0, "Typed tilde.")
            elif action == "cmd_punctuation_ampersand":
                self.type_text("&")
                self.update_command_progress(100.0, "Typed ampersand.")
            elif action == "cmd_punctuation_percent":
                self.type_text("%")
                self.update_command_progress(100.0, "Typed percent.")
            elif action == "cmd_punctuation_asterisk":
                self.type_text("*")
                self.update_command_progress(100.0, "Typed asterisk.")
            elif action == "cmd_punctuation_parentheses":
                self.type_text("()")
                keyboard.press_and_release("left")
                self.update_command_progress(100.0, "Typed parentheses.")
            elif action == "cmd_punctuation_dash":
                self.type_text("-")
                self.update_command_progress(100.0, "Typed dash.")
            elif action == "cmd_punctuation_underscore":
                self.type_text("_")
                self.update_command_progress(100.0, "Typed underscore.")
            elif action == "cmd_punctuation_plus":
                self.type_text("+")
                self.update_command_progress(100.0, "Typed plus.")
            elif action == "cmd_punctuation_equals":
                self.type_text("=")
                self.update_command_progress(100.0, "Typed equals.")
            elif action == "cmd_press_escape":
                keyboard.press_and_release("escape")
                self.update_command_progress(100.0, "Pressed Escape.")
            elif action == "cmd_screen_shoot":
                keyboard.press_and_release("printscreen")
                self.update_command_progress(100.0, "Captured screenshot.")
            elif action == "cmd_screen_shoot_window":
                keyboard.press_and_release("alt+printscreen")
                self.update_command_progress(100.0, "Captured window screenshot.")
            elif action == "cmd_screen_shoot_monitor":
                try:
                    subprocess.Popen("ms-screenclip:", shell=True)
                    time.sleep(0.5)
                    keyboard.press_and_release("ctrl+n")
                except Exception:
                    try:
                        subprocess.Popen("SnippingTool.exe", shell=True)
                        time.sleep(0.5)
                        keyboard.press_and_release("ctrl+n")
                    except Exception as e:
                        logging.error(f"Error opening Snipping Tool: {e}")
                        self.update_command_progress(100.0, f"Error capturing monitor: {e}")
                else:
                    self.update_command_progress(100.0, "Started monitor screenshot.")
            elif action == "cmd_task_manager":
                keyboard.press_and_release("ctrl+shift+esc")
                self.update_command_progress(100.0, "Opened Task Manager.")
            elif action == "cmd_debug_screen":
                keyboard.press_and_release("ctrl+alt+delete")
                self.update_command_progress(100.0, "Opened debug screen.")
            elif action == "cmd_force_close":
                keyboard.press_and_release("alt+f4")
                self.update_command_progress(100.0, "Force closed window.")
            elif action.startswith("highlight_"):
                param = action[len("highlight_"):].replace("_", " ")
                self._execute_find_and_select(param, highlight=True)
                self.update_command_progress(100.0, f"Highlighted '{param}'.")
            elif action.startswith("find_"):
                param = action[len("find_"):].replace("_", " ")
                self._execute_find_and_select(param, highlight=False)
                self.update_command_progress(100.0, f"Found '{param}'.")
            elif action.startswith("insert_after_"):
                param = action[len("insert_after_"):].replace("_", " ")
                self._execute_find_and_select(param, move_to_end=True)
                self.update_command_progress(100.0, f"Inserted after '{param}'.")
            elif action.startswith("insert_before_"):
                param = action[len("insert_before_"):].replace("_", " ")
                self._execute_find_and_select(param, move_to_end=False)
                self.update_command_progress(100.0, f"Inserted before '{param}'.")
            elif action.startswith("copy_"):
                param = action[len("copy_"):].replace("_", " ")
                self._execute_find_and_select(param, highlight=True)
                keyboard.press_and_release("ctrl+c")
                self.update_command_progress(100.0, f"Copied '{param}'.")
            elif action.startswith("cut_"):
                param = action[len("cut_"):].replace("_", " ")
                self._execute_find_and_select(param, highlight=True)
                keyboard.press_and_release("ctrl+x")
                self.update_command_progress(100.0, f"Cut '{param}'.")
            elif action.startswith("all_caps_"):
                param = action[len("all_caps_"):].replace("_", " ")
                self._execute_find_and_select(param, highlight=True)
                keyboard.press_and_release("delete")
                self.type_text(param.upper())
                self.update_command_progress(100.0, f"Converted '{param}' to all caps.")
            elif action.startswith("press_"):
                param = action[len("press_"):].replace("_", " ")
                keyboard.press_and_release(param)
                self.update_command_progress(100.0, f"Pressed key '{param}'.")
            elif action.startswith("open_"):
                param = action[len("open_"):].replace("_", " ")
                self._execute_open_app(param)
                self.update_command_progress(100.0, f"Opened '{param}'.")
            elif action.startswith("go_to_address_"):
                param = action[len("go_to_address_"):].replace("_", " ")
                self._execute_go_to_address(param)
                self.update_command_progress(100.0, f"Opened URL for '{param}'.")
            elif action.startswith("move_up_") or action.startswith("move_down_") or action.startswith("move_left_") or action.startswith("move_right_"):
                direction = action.split("_")[1]
                try:
                    num = int(action.split("_")[-1])
                except (ValueError, IndexError):
                    num = 1
                for _ in range(num):
                    keyboard.press_and_release(direction)
                self.update_command_progress(100.0, f"Moved {direction} {num} times.")
            elif action.startswith("function_"):
                param = action[len("function_"):].replace("_", "")
                if param in [str(i) for i in range(1, 13)]:
                    keyboard.press_and_release(f"f{param}")
                    self.update_command_progress(100.0, f"Pressed F{param}.")
                else:
                    self.update_command_progress(100.0, f"Invalid function key: {param}")
            elif action.startswith("select_through_"):
                parts = action[len("select_through_"):].split("_through_")
                if len(parts) == 2:
                    word1, word2 = parts
                    self._execute_select_through(word1.replace("_", " "), word2.replace("_", " "))
                    self.update_command_progress(100.0, f"Selected from '{word1}' to '{word2}'.")
                else:
                    self.update_command_progress(100.0, "Invalid select through command.")
            elif action.startswith("correct_"):
                param = action[len("correct_"):].replace("_", " ")
                self._execute_correct(param)
                self.update_command_progress(100.0, f"Corrected '{param}'.")
            else:
                logging.warning(f"GUI unknown command action: {action}")
                self.update_command_progress(100.0, f"Unknown command action: {action}")
        except Exception as e:
            logging.error(f"GUI error handling command {action}: {e}")
            self.update_command_progress(100.0, f"Error handling command: {e}")

    def _execute_find_and_select(self, param, highlight=False, move_to_end=False):
        self.update_command_progress(20.0, f"Finding '{param}'...")
        keyboard.press_and_release("ctrl+f")
        time.sleep(0.1)
        self.update_command_progress(40.0, f"Typing search term '{param}'...")
        self.type_text(param)
        time.sleep(0.1)
        self.update_command_progress(60.0, "Submitting search...")
        keyboard.press_and_release("enter")
        time.sleep(0.1)
        self.update_command_progress(80.0, "Closing search dialog...")
        keyboard.press_and_release("escape")
        if highlight:
            self.update_command_progress(90.0, f"Highlighting '{param}'...")
            keyboard.press("shift")
            for _ in range(len(param)):
                keyboard.press_and_release("right")
            keyboard.release("shift")
        elif move_to_end:
            self.update_command_progress(90.0, f"Moving to end of '{param}'...")
            for _ in range(len(param)):
                keyboard.press_and_release("right")

    def _execute_open_app(self, param):
        if not param:
            logging.error("No application name provided.")
            self.update_command_progress(100.0, "Error: No app name.")
            return
        self.update_command_progress(20.0, f"Looking up application '{param}'...")
        stt_apps = os.environ.get("STT", "")
        if not stt_apps:
            logging.error("STT environment variable not set.")
            self.update_command_progress(100.0, "Error: STT not set.")
            return
        app_dict = {}
        for app in stt_apps.split(";"):
            if not app or "=" not in app:
                continue
            key, value = app.split("=", 1)
            app_dict[key] = value
        app_name = param.replace(" ", "").lower()
        app_path = app_dict.get(app_name)
        if app_path:
            self.update_command_progress(80.0, f"Opening application '{param}'...")
            subprocess.Popen(app_path, shell=True)
        else:
            logging.error(f"Application '{param}' not found.")
            self.update_command_progress(100.0, f"App '{param}' not found.")

    def _execute_go_to_address(self, param):
        self.update_command_progress(20.0, f"Processing URL for '{param}'...")
        domain = "com"
        if "dot org" in param:
            domain = "org"
            param = param.replace("dot org", "").strip()
        elif "dot net" in param:
            domain = "net"
            param = param.replace("dot net", "").strip()
        elif "dot io" in param:
            domain = "io"
            param = param.replace("dot io", "").strip()
        url = f"https://{param}.{domain}"
        self.update_command_progress(80.0, f"Opening URL '{url}'...")
        subprocess.Popen(['start', url], shell=True)

    def _execute_select_through(self, word1, word2):
        self.update_command_progress(20.0, f"Finding first word '{word1}'...")
        keyboard.press_and_release("ctrl+f")
        time.sleep(0.1)
        self.type_text(word1)
        time.sleep(0.1)
        keyboard.press_and_release("enter")
        time.sleep(0.1)
        keyboard.press_and_release("escape")
        self.update_command_progress(50.0, f"Selecting to second word '{word2}'...")
        keyboard.press("shift")
        keyboard.press_and_release("ctrl+f")
        time.sleep(0.1)
        self.type_text(word2)
        time.sleep(0.1)
        keyboard.press_and_release("enter")
        time.sleep(0.1)
        keyboard.press_and_release("escape")
        for _ in range(len(word2)):
            keyboard.press_and_release("right")
        keyboard.release("shift")

    def _execute_correct(self, param):
        self.update_command_progress(20.0, f"Finding word '{param}'...")
        keyboard.press_and_release("ctrl+f")
        time.sleep(0.1)
        self.type_text(param)
        time.sleep(0.1)
        keyboard.press_and_release("enter")
        time.sleep(0.1)
        keyboard.press_and_release("escape")
        self.update_command_progress(50.0, f"Highlighting word '{param}'...")
        keyboard.press("shift")
        for _ in range(len(param)):
            keyboard.press_and_release("right")
        keyboard.release("shift")
        self.update_command_progress(70.0, f"Correcting word '{param}'...")
        keyboard.press_and_release("delete")
        corrected_word = self.spell_checker.correction(param)
        self.update_command_progress(90.0, f"Typing corrected word '{corrected_word}'...")
        self.type_text(corrected_word)

    def send_command(self, action):
        self.command_queue.put(action)
        logging.debug(f"GUI sent command: {action}")
        self.update_command_progress(0.0, f"Queued command: {action}")

    def apply_theme(self, theme, app_data=None):
        selected_theme = app_data if app_data is not None else theme
        self.theme = selected_theme
        if selected_theme == "Dark":
            dpg.bind_theme("dark_theme")
        else:
            dpg.bind_theme("light_theme")

    def load_settings(self):
        self.AudioProcessor.load_settings()
        self.saved_settings = self.AudioProcessor.saved_settings
        dpg.set_value("theme_combo", self.saved_settings.get("theme", "Dark"))
        self.apply_theme(self.saved_settings.get("theme", "Dark"))

    def save_settings(self, sender=None, app_data=None):
        self.AudioProcessor.save_settings()
        self.saved_settings = self.AudioProcessor.saved_settings
        dpg.set_value("status_text", "Settings saved successfully.")

    def setup_json_tab(self, tab_name, json_path):
        with dpg.group():
            with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp, tag=f"{tab_name}_table"):
                dpg.add_table_column(label="Key")
                dpg.add_table_column(label="Value")
                base_path = os.path.dirname(os.path.abspath(__file__))
                full_path = os.path.join(base_path, json_path)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        self.json_data[tab_name] = json.load(f)
                except Exception:
                    self.json_data[tab_name] = {}
                
                for key, value in self.json_data[tab_name].items():
                    with dpg.table_row():
                        dpg.add_text(key, tag=f"{tab_name}_{key}_key")
                        dpg.add_text(value, tag=f"{tab_name}_{key}_value")
            
            with dpg.group(horizontal=True):
                dpg.add_button(label="Add", callback=lambda: self.add_json_entry(tab_name, json_path))
                dpg.add_button(label="Edit", callback=lambda: self.edit_json_entry(tab_name, json_path))
                dpg.add_button(label="Delete", callback=lambda: self.delete_json_entry(tab_name, json_path))

    def add_json_entry(self, tab_name, json_path):
        with dpg.window(label=f"Add Entry to {tab_name}", modal=True, width=300, height=150, tag=f"{tab_name}_add_window"):
            dpg.add_input_text(label="Key", tag=f"{tab_name}_add_key")
            dpg.add_input_text(label="Value", tag=f"{tab_name}_add_value")
            dpg.add_button(label="Save", callback=lambda: self.save_json_entry(tab_name, json_path, True))

    def edit_json_entry(self, tab_name, json_path):
        if not self.json_data[tab_name]:
            dpg.set_value("status_text", "No entry to edit.")
            return
        key = list(self.json_data[tab_name].keys())[0]
        value = self.json_data[tab_name][key]
        with dpg.window(label=f"Edit Entry in {tab_name}", modal=True, width=300, height=150, tag=f"{tab_name}_edit_window"):
            dpg.add_input_text(label="Key", default_value=key, tag=f"{tab_name}_edit_key")
            dpg.add_input_text(label="Value", default_value=value, tag=f"{tab_name}_edit_value")
            dpg.add_button(label="Save", callback=lambda: self.save_json_entry(tab_name, json_path, False))

    def delete_json_entry(self, tab_name, json_path):
        if not self.json_data[tab_name]:
            dpg.set_value("status_text", "No entry to delete.")
            return
        key = list(self.json_data[tab_name].keys())[0]
        del self.json_data[tab_name][key]
        base_path = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_path, json_path)
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(self.json_data[tab_name], f, indent=4)
        dpg.delete_item(f"{tab_name}_table", children_only=True)
        dpg.add_table_column(label="Key", parent=f"{tab_name}_table")
        dpg.add_table_column(label="Value", parent=f"{tab_name}_table")
        for k, v in self.json_data[tab_name].items():
            with dpg.table_row(parent=f"{tab_name}_table"):
                dpg.add_text(k, tag=f"{tab_name}_{k}_key")
                dpg.add_text(v, tag=f"{tab_name}_{k}_value")

    def save_json_entry(self, tab_name, json_path, is_add):
        key = dpg.get_value(f"{tab_name}_add_key" if is_add else f"{tab_name}_edit_key")
        value = dpg.get_value(f"{tab_name}_add_value" if is_add else f"{tab_name}_edit_value")
        if key and value:
            if not is_add:
                old_key = list(self.json_data[tab_name].keys())[0]
                if key != old_key:
                    del self.json_data[tab_name][old_key]
            self.json_data[tab_name][key] = value
            base_path = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(base_path, json_path)
            with open(full_path, "w", encoding="utf-8") as f:
                json.dump(self.json_data[tab_name], f, indent=4)
            dpg.delete_item(f"{tab_name}_table", children_only=True)
            dpg.add_table_column(label="Key", parent=f"{tab_name}_table")
            dpg.add_table_column(label="Value", parent=f"{tab_name}_table")
            for k, v in self.json_data[tab_name].items():
                with dpg.table_row(parent=f"{tab_name}_table"):
                    dpg.add_text(k, tag=f"{tab_name}_{k}_key")
                    dpg.add_text(v, tag=f"{tab_name}_{k}_value")
        dpg.delete_item(f"{tab_name}_add_window" if is_add else f"{tab_name}_edit_window")

    def create_file_dialog(self):
        with dpg.file_dialog(
            directory_selector=True,
            show=False,
            callback=self.on_model_path_selected,
            tag="file_dialog",
            width=700,
            height=400
        ):
            dpg.add_file_extension(".*")

    def set_model_path(self, sender, app_data):
        if not dpg.does_item_exist("file_dialog"):
            self.create_file_dialog()
        dpg.show_item("file_dialog")

    def on_model_path_selected(self, sender, app_data):
        selected_path = app_data["file_path_name"]
        if not selected_path:
            dpg.set_value("status_text", "No directory selected.")
            return
        absolute_path = os.path.abspath(selected_path)
        dpg.set_value("model_path_input", absolute_path)
        base_path = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_path, CONFIG_PATH)
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            config = {}
        config["model_path"] = absolute_path
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
            dpg.set_value("status_text", f"Model path set to: {absolute_path}")
            logging.info(f"Model path set to {absolute_path}")
        except Exception as e:
            dpg.set_value("status_text", f"Failed to save model path: {e}")
            logging.error(f"Failed to save model path: {e}")

    def show_waveform(self, sender, app_data):
        if dpg.does_item_exist("waveform_window"):
            dpg.focus_item("waveform_window")
            return
        self.waveform = WaveformDisplay(self.AudioProcessor)

    def start_dictation(self):
        logging.basicConfig(filename="dictation_gui.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info("Starting dictation process.")

        if self.is_dictating:
            logging.warning("Dictation already running.")
            return

        if self.AudioProcessor.is_testing:
            self.AudioProcessor.toggle_audio_test()
        if self.AudioProcessor.is_recording:
            self.AudioProcessor.toggle_recording()

        bit_depth = self.AudioProcessor.saved_settings.get("bit_depth", "int24")
        if " - " in bit_depth:
            bit_depth = bit_depth.split(" - ")[0]
        bit_depth_value = int(bit_depth.replace("int", ""))
        sample_rate = int(self.AudioProcessor.saved_settings.get("sample_rate", "48000"))
        silence_threshold = self.AudioProcessor.saved_settings.get("silence_threshold", 10.0)
        model_path = self.AudioProcessor.saved_settings.get("model_path", r"C:\Users\MenaBeshai\Downloads\Speech to Text\vosk-model-en-us-0.42-gigaspeech")

        if not os.path.exists(model_path):
            logging.error(f"Vosk model not found at {model_path}.")
            return

        try:
            self.monitor_stop_event = threading.Event()
            self.is_dictating = True
            logging.info("Dictation process started.")

            def monitor_process():
                model = vosk.Model(model_path)
                logging.info("Vosk model loaded successfully.")
                last_status = "Dictation started."

                while not self.monitor_stop_event.is_set():
                    if not self.is_dictating:
                        break

                    # Simulate audio input processing
                    audio_data = self.AudioProcessor.get_audio_data()
                    if audio_data:
                        recognizer = vosk.KaldiRecognizer(model, sample_rate)
                        if recognizer.AcceptWaveform(audio_data):
                            result = json.loads(recognizer.Result())
                            last_status = f"Dictation output: {result.get('text', '')}"
                            logging.info(last_status)
                        else:
                            partial_result = json.loads(recognizer.PartialResult())
                            logging.info(f"Partial output: {partial_result.get('partial', '')}")

                    time.sleep(0.1)

                self.stop_dictation()

            monitor_thread = threading.Thread(target=monitor_process, daemon=True)
            monitor_thread.start()

        except Exception as e:
            logging.error(f"Failed to start dictation: {e}")
            self.is_dictating = False

    def stop_dictation(self):
        if self.is_dictating:
            self.is_dictating = False
            if self.monitor_stop_event:
                self.monitor_stop_event.set()
            logging.info("Dictation stopped.")

    def run(self):
        while dpg.is_dearpygui_running():
            jobs = dpg.get_callback_queue()
            if jobs:
                for job in jobs:
                    try:
                        job()
                    except Exception as e:
                        logging.error(f"GUI callback error: {e}")
            try:
                if not self.command_queue.empty():
                    action = self.command_queue.get_nowait()
                    self.handle_command(action)
            except queue.Empty:
                pass
            if self.AudioProcessor.is_testing:
                self.waveform.update_waveform()
            dpg.render_dearpygui_frame()
        dpg.destroy_context()




class CustomDictationGUI(DictationGUI):
    def __init__(self, audioprocessor):
        super().__init__()
    """Custom subclass to override start_dictation and stop_dictation."""
    def __init__(self, audioprocessor):
        logging.debug("Initializing CustomDictationGUI")
        try:
            super().__init__()
            logging.debug("Parent DictationGUI initialized successfully")
        except Exception as e:
            logging.error(f"Error in DictationGUI.__init__: {e}", exc_info=True)
            raise

    def start_dictation(self, sender, app_data):
        logging.basicConfig(filename="dictation_gui.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.debug("Start dictation button clicked.")

        if self.is_dictating:
            logging.warning("Dictation already running.")
            dpg.set_value("status_text", "Dictation already running.")
            return
        if self.is_testing:
            self.stop_audio_test()
        if self.is_recording:
            self.toggle_recording()

        # Gather settings from GUI
        bit_depth = dpg.get_value("bit_depth_combo") if dpg.does_item_exist("bit_depth_combo") else "int16"
        if " - " in bit_depth:
            bit_depth = bit_depth.split(" - ")[0]
        bit_depth_value = int(bit_depth.replace("int", ""))
        sample_rate = dpg.get_value("sample_rate_combo") if dpg.does_item_exist("sample_rate_combo") else "16000"
        device_index = self.get_device_index(
            dpg.get_value("input_device_combo") if dpg.does_item_exist("input_device_combo") else None,
            dpg.get_value("host_api_combo") if dpg.does_item_exist("host_api_combo") else "MME"
        )
        if device_index is None:
            dpg.set_value("status_text", "No valid input device selected.")
            logging.error("No valid input device selected.")
            return

        unit = dpg.get_value("unit_combo") if dpg.does_item_exist("unit_combo") else "Numbers"
        slider_value = dpg.get_value("sensitivity_slider") if dpg.does_item_exist("sensitivity_slider") else 0
        if unit == "Numbers":
            pre_scale = self.slider_to_pre_scale(slider_value)
        elif unit == "Percent":
            pre_scale = self.percent_to_pre_scale(slider_value)
        elif unit == "dB":
            db = (slider_value / 100) * 100 - 60
            pre_scale = self.db_to_pre_scale(db)

        relative_sensitivity = 1 if (dpg.does_item_exist("relative_sensitivity_check") and dpg.get_value("relative_sensitivity_check")) else 0
        if relative_sensitivity:
            dtype, max_value = self.get_dtype_and_max()
            scale_factor = 32767 / (max_value + (1 if dtype != "float32" else 0))
            pre_scale /= scale_factor

        silence_threshold = dpg.get_value("silence_input") if dpg.does_item_exist("silence_input") else 500.0
        silence_duration = 1.0  # Default silence duration (not used for automatic paragraph breaks anymore)

        # Get device index using the selected Host API
        device_index = self.get_device_index(dpg.get_value("input_device_combo"), dpg.get_value("host_api_combo"))

        # Read and validate model path
        model_path = dpg.get_value("model_path_input")
        model_path = os.path.abspath(model_path)
        logging.debug(f"Model path from GUI: {model_path}")
        if not model_path or not os.path.exists(model_path):
            dpg.set_value("status_text", f"Vosk model not found at {model_path}.")
            logging.error(f"Vosk model not found at {model_path}.")
            return

        # Start the dictation process directly (no subprocess)
        self.is_dictating = True
        dpg.configure_item("start_dictation_button", enabled=False)
        dpg.configure_item("stop_dictation_button", enabled=True)
        dpg.set_value("status_text", "Starting dictation...")

        # Run dictation in a separate thread to keep GUI responsive
        import threading
        dictation_thread = threading.Thread(target=perform_dictation, args=(
            self, model_path, bit_depth_value, sample_rate, device_index,
            pre_scale, relative_sensitivity, silence_threshold, silence_duration
        ))
        dictation_thread.daemon = True
        dictation_thread.start()

        # Gather settings from GUI
        bit_depth = dpg.get_value("bit_depth_combo") if dpg.does_item_exist("bit_depth_combo") else "int16"
        if " - " in bit_depth:
            bit_depth = bit_depth.split(" - ")[0]
        bit_depth_value = int(bit_depth.replace("int", ""))
        sample_rate = dpg.get_value("sample_rate_combo") if dpg.does_item_exist("sample_rate_combo") else "16000"
        device_index = self.get_device_index(
            dpg.get_value("input_device_combo") if dpg.does_item_exist("input_device_combo") else None,
            dpg.get_value("host_api_combo") if dpg.does_item_exist("host_api_combo") else "MME"
        )
        if device_index is None:
            dpg.set_value("status_text", "No valid input device selected.")
            logging.error("No valid input device selected.")
            return

        unit = dpg.get_value("unit_combo") if dpg.does_item_exist("unit_combo") else "Numbers"
        slider_value = dpg.get_value("sensitivity_slider") if dpg.does_item_exist("sensitivity_slider") else 0
        if unit == "Numbers":
            pre_scale = self.slider_to_pre_scale(slider_value)
        elif unit == "Percent":
            pre_scale = self.percent_to_pre_scale(slider_value)
        elif unit == "dB":
            db = (slider_value / 100) * 100 - 60
            pre_scale = self.db_to_pre_scale(db)

        relative_sensitivity = 1 if (dpg.does_item_exist("relative_sensitivity_check") and dpg.get_value("relative_sensitivity_check")) else 0
        if relative_sensitivity:
            dtype, max_value = self.get_dtype_and_max()
            scale_factor = 32767 / (max_value + (1 if dtype != "float32" else 0))
            pre_scale /= scale_factor

        silence_threshold = dpg.get_value("silence_input") if dpg.does_item_exist("silence_input") else 500.0
        silence_duration = 1.0  # Default silence duration

        # Read and validate model path
        model_path = dpg.get_value("model_path_input")
        model_path = os.path.abspath(model_path)
        logging.debug(f"Model path from GUI: {model_path}")
        if not model_path or not os.path.exists(model_path):
            dpg.set_value("status_text", f"Vosk model not found at {model_path}.")
            logging.error(f"Vosk model not found at {model_path}.")
            return

        # Start the dictation process
        self.is_dictating = True
        dpg.configure_item("start_dictation_button", enabled=False)
        dpg.configure_item("stop_dictation_button", enabled=True)
        dpg.set_value("status_text", "Starting dictation...")

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
            dpg.configure_item("start_dictation_button", enabled=True)
            dpg.configure_item("stop_dictation_button", enabled=False)
            dpg.set_value("status_text", "Dictation stopped.")
            logging.info("Dictation stopped via GUI or interrupt.")
            if self.is_debug_recording:
                self.save_debug_recording()

def perform_dictation(gui, model_path, bit_depth, sample_rate, device_index, pre_scale, relative_sensitivity, silence_threshold, silence_duration):
    global dictating, audio_buffer, last_word_end_time, last_command, last_command_time, skip_dictation, last_dictated_text, last_dictated_length, last_processed_command, caps_lock_on, number_lock_on, STREAM_STARTED, FEEDING_AUDIO_STARTED
    # Ensure model_path is a string and uses forward slashes
    if not isinstance(model_path, str):
        model_path = model_path.decode("utf-8") if isinstance(model_path, bytes) else str(model_path)
    model_path = model_path.replace("\\", "/")
    
    # Ensure sample_rate is an integer
    sample_rate = int(sample_rate)

    try:
        print(f"Model path: {model_path}")
        model = Model(model_path)
        recognizer = KaldiRecognizer(model, sample_rate)
        if not recognizer:
            raise ValueError("Failed to create KaldiRecognizer")
    except Exception as e:
        logging.error(f"Failed to initialize Vosk model or recognizer: {e}")
        dpg.set_value("status_text", f"Failed to initialize Vosk model or recognizer: {e}")
        gui.is_dictating = False
        dpg.configure_item("start_dictation_button", enabled=True)
        dpg.configure_item("stop_dictation_button", enabled=False)
        return

    dtype, max_value = gui.get_dtype_and_max(f"int{bit_depth}")
    if relative_sensitivity:
        scale_factor = 32767 / (max_value + (1 if dtype != "float32" else 0))
        pre_scale /= scale_factor

    tokenized_simple_commands, tokenized_parameterized_partial, tokenized_parameterized_final = load_commands()

    def audio_callback(indata, frames, time_info, status):
        global audio_buffer, STREAM_STARTED, FEEDING_AUDIO_STARTED
        try:
            data = np.frombuffer(indata, dtype=dtype).reshape(-1, 1)
            data = data[:, 0].astype(np.float32)
            if dtype == "int8":
                data_normalized = data * (32767 / 127)
            elif dtype == "int16":
                data_normalized = data
            elif dtype == "int24":
                data_normalized = data * (32767 / 8388607)
            elif dtype == "int32":
                data_normalized = data * (32767 / 2147483647)
            elif dtype == "float32":
                data_normalized = data * 32767
            data_scaled = data_normalized * pre_scale
            data_scaled = np.clip(data_scaled, -32768, 32767)
            audio_buffer.extend(data_scaled)
            gui.update_audio_level(data_scaled)
            if not STREAM_STARTED:
                dpg.set_value("status_text", "Audio stream started successfully.")
                STREAM_STARTED = True
            if len(audio_buffer) >= 32000 and not FEEDING_AUDIO_STARTED:
                dpg.set_value("status_text", "Feeding audio to Vosk recognizer...")
                FEEDING_AUDIO_STARTED = True
            q.put(data_scaled.tobytes())
        except Exception as e:
            logging.error(f"Error in audio callback: {e}")

    try:
        stream = sd.RawInputStream(
            samplerate=sample_rate, blocksize=32000, device=device_index,
            dtype=dtype, channels=1, callback=audio_callback
        )
        stream.start()
    except Exception as e:
        logging.error(f"Failed to start audio stream: {e}")
        dpg.set_value("status_text", f"Failed to start audio stream: {e}")
        gui.is_dictating = False
        dpg.configure_item("start_dictation_button", enabled=True)
        dpg.configure_item("stop_dictation_button", enabled=False)
        return

    silence_amplitude_threshold = silence_threshold * 32767 / 100
    dictating = True
    last_amplitude = 0
    silence_start = None

    with open(TRANSCRIPTION_FILE, "w", encoding="utf-8") as f:
        f.write("")

    while dictating and gui.is_dictating:
        try:
            data = q.get(timeout=1.0)
        except queue.Empty:
            continue

        if not data:
            continue

        current_amplitude = np.max(np.abs(np.frombuffer(data, dtype=np.int16)))
        if current_amplitude < silence_amplitude_threshold:
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start >= silence_duration:
                last_word_end_time = time.time()
        else:
            silence_start = None

        if time.time() - last_word_end_time > silence_duration and audio_buffer:
            audio_buffer.clear()

        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "")
            if text:
                last_word_end_time = time.time()
                normalized_text = normalize_text(text)
                tokens = tuple(nltk.word_tokenize(normalized_text))
                current_time = time.time()

                command_executed = False
                for cmd_tokens, (cmd, action) in tokenized_simple_commands.items():
                    if tokens == cmd_tokens:
                        if last_command != cmd or (current_time - last_command_time) >= COMMAND_DEBOUNCE_TIME:
                            last_command = cmd
                            last_command_time = current_time
                            gui.send_command(action)
                            skip_dictation = True
                            last_processed_command = cmd
                            command_executed = True
                        break

                if not command_executed:
                    for cmd_tokens, cmd in tokenized_parameterized_partial + tokenized_parameterized_final:
                        if tokens[:len(cmd_tokens)] == cmd_tokens:
                            param = " ".join(tokens[len(cmd_tokens):])
                            param = param.replace(" ", "_")
                            action = f"{cmd.replace(' ', '_')}_{param}" if param else cmd.replace(" ", "_")
                            if last_command != cmd or (current_time - last_command_time) >= COMMAND_DEBOUNCE_TIME:
                                last_command = cmd
                                last_command_time = current_time
                                gui.send_command(action)
                                skip_dictation = True
                                last_processed_command = cmd
                                command_executed = True
                            break

                if not command_executed and not skip_dictation and normalized_text != last_processed_command:
                    if caps_lock_on:
                        normalized_text = normalized_text.upper()
                    normalized_text = convert_spoken_numbers_to_digits(normalized_text)
                    last_dictated_text = normalized_text + " "
                    last_dictated_length = len(last_dictated_text)
                    gui.transcribed_text.append(last_dictated_text)
                    full_text = "".join(gui.transcribed_text).rstrip()
                    dpg.set_value("output_text", full_text)
                    with open(TRANSCRIPTION_FILE, "a", encoding="utf-8") as f:
                        f.write(last_dictated_text)
                    gui.type_text(last_dictated_text)
                skip_dictation = False
        else:
            partial_result = json.loads(recognizer.PartialResult())
            partial_text = partial_result.get("partial", "")
            if partial_text:
                last_word_end_time = time.time()

    stream.stop()
    stream.close()
    audio_buffer.clear()
    q.queue.clear()

    def perform_dictation(model_path, bit_depth_value, sample_rate, device_index, pre_scale, relative_sensitivity, silence_threshold):
        global last_word_end_time, audio_buffer, last_dictated_text, last_dictated_length, skip_dictation, last_processed_command, dictating

        # Reset globals
        audio_buffer = []
        last_word_end_time = 0.0
        last_dictated_text = ""
        last_dictated_length = 0
        skip_dictation = False
        last_processed_command = None

        # Initialize Vosk model
        if not model_path or not os.path.exists(model_path):
            print(f"Model path {model_path} does not exist. Please provide a valid Vosk model path.")
            sys.exit(1)
        model = Model(model_path)
        rec = KaldiRecognizer(model, VOSK_SAMPLERATE)
        rec.SetWords(True)

        # Determine bit depth for sounddevice
        if bit_depth_value == 16:
            dtype = 'int16'
            max_amplitude = 32767
        elif bit_depth_value == 24:
            dtype = 'int32'  # sounddevice doesn't support int24 directly, so we use int32 and mask
            max_amplitude = 8388607  # 2^23 - 1
        elif bit_depth_value == 32:
            dtype = 'int32'
            max_amplitude = 2147483647  # 2^31 - 1
        else:
            print(f"Unsupported bit depth: {bit_depth_value}")
            sys.exit(1)

        # Apply relative sensitivity
        if relative_sensitivity:
            scale_factor = 32767 / max_amplitude
            pre_scale *= scale_factor


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename="dictation_script.log"
    )
    logging.info("Application started.")

    # Initialize variables
    number_lock_on = False
    last_dictated_length = 0

    # Create an instance of your audio processor
 
    # Create an instance of your DictationGUI
    gui = DictationGUI(AudioProcessor)  # Pass the AudioProcessor here
    gui.run()
    while True:
        try:
            if not gui.is_dictating:
                break
            time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Application interrupted by user.")
            break
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            print(f"Error: {e}")
            logging.error(f"Error: {e}")
            print("An error occurred. Please check the log file for details.")
    pass

    # Check command-line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "waveform":
        try:
            bit_depth = sys.argv[2]
            channels = int(sys.argv[3])
            samplerate = int(sys.argv[4])
            device = int(sys.argv[5])
            pre_scale_factor = float(sys.argv[6])
            relative_sensitivity = sys.argv[7] == "True"
            WaveformDisplay(bit_depth, channels, samplerate, device, pre_scale_factor, relative_sensitivity)
        except IndexError:
            logging.error("Insufficient arguments provided for waveform mode.")
            print("Error: Insufficient arguments for waveform mode.")
            sys.exit(1)
        except ValueError as e:
            logging.error(f"Invalid argument type: {e}")
            print(f"Error: Invalid argument type. {e}")
            sys.exit(1)
    else:
        # Run the Dictation GUI with the audio processor
        gui = DictationGUI(AudioProcessor)  # Pass the AudioProcessor here
        gui.run()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Application crashed: {e}", exc_info=True)
        print(f"Error: {e}. Check dictation_script.log for details.")
        input("Press Enter to exit...")