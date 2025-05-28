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
from scipy.signal import resample # For audio resampling
import sounddevice as sd  # For audio input/output
import threading  # For threading
import queue  # For command queue
import logging  # For logging
import keyboard  # For keyboard input
import pyautogui  # For mouse clicks and screenshots
import re  # For regex
import msvcrt  # For keyboard input
import selectors  # For keyboard input
from live_gigaspeech_dictation_v32 import perform_dictation, type_text, update_status # For dictation
import words_to_numbers_v7 as words_to_numbers # For number conversion
from spellchecker import SpellChecker  # For spell checking
from collections import deque  # For audio buffer
from vosk import Model, KaldiRecognizer  # For speech recognition
import nltk  # For tokenization to separate commands from dictation

import dictation_pygui_shared_v10 as gui_module # For GUI handling

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
DEVICE_INDEX = 1  # MY MIC (Realtek USB Audio)
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
NUMBERS_PATH = os.path.join(CONFIG_DIR, "numbers_map.json")
GOOGLE_NUMBERS_PATH = os.path.join(CONFIG_DIR, "google_numbers.json")
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
NUMBERS_MAP = load_json_map(NUMBERS_PATH, "numbers map")
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



class WaveformDisplay:
    def __init__(self, bit_depth, channels, samplerate, device, pre_scale_factor, relative_sensitivity):
        self.bit_depth = bit_depth
        self.channels = channels
        self.samplerate = samplerate
        self.device = device
        self.pre_scale_factor = pre_scale_factor
        self.relative_sensitivity = relative_sensitivity
        self.is_running = False
        self.audio_buffer = deque(maxlen=int(self.samplerate * 1.0))  # 1 second of data
        self.update_interval = 1.0 / 30  # Update at 30 FPS
        self.last_update = 0

        # Determine dtype and max_value
        self.dtype, self.max_value = self.get_dtype_and_max()
        
        # DPG setup (use existing context)
        with dpg.window(label="Waveform Display", tag="waveform_window", width=800, height=300, on_close=self.close):
            # Plot for waveform
            with dpg.plot(label="Waveform (16-bit PCM, as Vosk hears)", height=200, width=-1):
                dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag="waveform_x_axis")
                dpg.add_plot_axis(dpg.mvYAxis, label="Amplitude", tag="waveform_y_axis")
                dpg.set_axis_limits("waveform_y_axis", -32768, 32768)
                dpg.set_axis_limits("waveform_x_axis", 0, 1.0)
                self.time_axis = np.linspace(0, 1.0, int(self.samplerate * 1.0))
                self.audio_data = np.zeros(int(self.samplerate * 1.0))
                dpg.add_line_series(self.time_axis, self.audio_data, label="Waveform", parent="waveform_y_axis", tag="waveform_series")
        
        self.start_stream()
        # Schedule the update_waveform method to run periodically
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
    
    def start_stream(self):
        self.is_running = True
        def audio_callback(indata, frames, time_info, status):
            if not self.is_running:
                return
            try:
                # Convert raw input to float
                data = np.frombuffer(indata, dtype=self.dtype).reshape(-1, self.channels)
                data = data[:, 0].astype(np.float32)

                # Normalize to 16-bit range based on bit depth
                if self.dtype == "int8":
                    data_normalized = data * (32767 / 127)  # Scale from ±127 to ±32767
                elif self.dtype == "int16":
                    data_normalized = data  # Already in ±32767 range
                elif self.dtype == "int24" or self.dtype == "int32":  # int24 uses int32 dtype
                    if self.max_value == 8388607:  # int24
                        data_normalized = data * (32767 / 8388607)  # Scale from ±8388607 to ±32767
                    else:  # int32
                        data_normalized = data * (32767 / 2147483647)  # Scale from ±2147483647 to ±32767
                elif self.dtype == "float32":
                    data_normalized = data * 32767  # Scale from ±1.0 to ±32767
                else:
                    raise ValueError(f"Unsupported dtype: {self.dtype}")

                # Apply pre-scale factor as a sensitivity adjustment
                pre_scale = self.pre_scale_factor
                data_scaled = data_normalized * pre_scale

                # Ensure the data is within the 16-bit range for display
                data_scaled = np.clip(data_scaled, -32768, 32767)

                # Update waveform buffer
                self.audio_buffer.extend(data_scaled)
            except Exception as e:
                logging.error(f"Error in waveform audio callback: {e}")

        try:
            self.stream = sd.RawInputStream(
                samplerate=self.samplerate, blocksize=32000, device=self.device,
                dtype=self.dtype, channels=self.channels, callback=audio_callback
            )
            self.stream.start()
        except Exception as e:
            logging.error(f"Failed to start waveform stream: {e}")
            self.is_running = False
    
    def update_waveform(self):
        if not self.is_running or not dpg.does_item_exist("waveform_window"):
            return
        # Update the waveform plot at a fixed interval
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            try:
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

    def stop_stream(self):
        self.is_running = False
        if hasattr(self, "stream"):
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logging.error(f"Error stopping waveform stream: {e}")

    def close(self, sender, app_data):
        self.is_running = False # Stop the update loop
        self.stop_stream()
        dpg.delete_item("waveform_window")


class DictationGUI:
    def __init__(self):
        # Audio variables
        self.audio_queue = queue.Queue()
        self.gui_update_queue = queue.Queue()
        self.audio_stream = None
        self.output_stream = None
        self.is_testing = False
        self.is_recording = False       
        self.audio_buffer = []
        self.output_buffer = []
        self.noise_floor = 0
        self.last_noise_update = 0
        self.peak_amplitude = 0
        
        # Dictation variables
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
        
        # Progress variables
        self.command_progress = 0.0
        self.command_progress_max = 100.0
        self.command_status = ""
        
        # Debug recording
        self.is_debug_recording = False
        self.debug_audio_buffer = []
        
        # Recordings directory
        self.default_recordings_dir = os.path.join(os.getcwd(), "Recordings")
        self.recordings_dir = self.default_recordings_dir
        try:
            if not os.path.exists(self.default_recordings_dir):
                os.makedirs(self.default_recordings_dir)
        except Exception as e:
            logging.error(f"Failed to create recordings directory: {e}")
            self.default_recordings_dir = os.getcwd()
            self.recordings_dir = self.default_recordings_dir
        
        # Initialize saved settings with defaults
        self.saved_settings = {}
        
        # JSON data
        self.json_data = {}
        self.commands = {}
        self.load_commands()
        
        # DPG setup
        dpg.create_context()
        dpg.create_viewport(title="Speech-to-Text Dictation Configuration", width=800, height=600)
        
        # Theme setup
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
        
        self.theme = "Dark"
        
        with dpg.window(label="Speech-to-Text Dictation Configuration", tag="primary_window"):
            with dpg.tab_bar(tag="main_tab_bar"):
                with dpg.tab(label="Audio Settings", tag="audio_tab"):
                    self.setup_audio_tab()
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
            
            # Progress bar for command execution
            dpg.add_text("Command Progress:")
            dpg.add_progress_bar(default_value=0.0, width=780, tag="command_progress_bar")
            dpg.add_text("", tag="command_status_text")
        
        self.apply_theme("Dark")
        self.load_settings()
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("primary_window", True)
        self.create_file_dialog()
        self.update_gui()

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

    
    # Load commands from JSON file and tokenize them
def load_commands():
    if not os.path.exists(COMMANDS_JSON_PATH):
        print(f"Commands JSON file not found at {COMMANDS_JSON_PATH}")
        sys.exit(1)
    with open(COMMANDS_JSON_PATH, "r") as f:
        commands_data = json.load(f)
    
    # Tokenize simple commands
    simple_commands = commands_data["simple_commands"]
    tokenized_simple_commands = {}
    for cmd, action in simple_commands.items():
        normalized_cmd = normalize_text(cmd.lower())
        tokens = tuple(nltk.word_tokenize(normalized_cmd))
        tokenized_simple_commands[tokens] = (cmd, action)
    
    # Tokenize parameterized commands (split into partial and final)
    parameterized_commands = commands_data["parameterized_commands"]
    # Commands that should only run on final results
    final_only_commands = ["quote unquote "]  # Add more as needed
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
    print(f"Error loading commands from {COMMANDS_JSON_PATH}: {e}")
    sys.exit(1)

def process_text(text):
    """Handle capitalization and number conversion for transcribed text."""
    global caps_lock_on

    if not text:
        return text

    # Convert spoken numbers to digits if number lock is on
    text = convert_numbers(text)

    # Split the text into words
    words = text.split()
    if not words:
        return text

    # Apply caps lock if enabled
    if caps_lock_on:
        words = [word.upper() for word in words]
    else:
        # Capitalize the first letter of the sentence
        words[0] = words[0][0].upper() + words[0][1:] if len(words[0]) > 1 else words[0].upper()

    # Join the words back together
    processed_text = " ".join(words)
    
    # Try to convert the processed text to a number using words_to_numbers
    number = words_to_numbers.convert_numbers(processed_text, FRACTION_MAP, SYMBOLS_MAP, GOOGLE_NUMBERS, LARGE_NUMBERS_MAP)
    if number is not None:
        final_text = str(number)  # Convert the number to a string for output
    else:
        final_text = processed_text  # If conversion fails, use the original processed text

    return final_text

def handle_special_phrases(text):
    """Handle special phrases like 'one hundred percent', fractions, symbols, and 'number <number>' sequences."""
    text_lower = text.lower()
    words = text_lower.split()
    i = 0
    result = []

    while i < len(words):
        # Handle "number" followed by a sequence of numbers
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
        
        
        # Check if the sequence might be a number phrase
        # Only try to parse if the first word is a number word or in NUMBERS_MAP
        first_word = words[i]
        is_potential_number = False
        if first_word in ["negative", "minus"]:
            # If the first word is "negative" or "minus", check the next word
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
                    i += num_words
                continue

        # Handle any number followed by "percent"
        number_sequence, num_words = parse_number_sequence(words[i:])
        if number_sequence:
            if i + num_words < len(words) and words[i + num_words] == "percent":
                result.append(number_sequence + "%")
                i += num_words + 1
            else:
                result.append(number_sequence)
                i += num_words
            continue

        # Handle fractions from FRACTION_MAP
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

        # Handle symbols from SYMBOLS_MAP
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

        # If no special handling, append the word as-is
        result.append(words[i])
        i += 1

    return " ".join(result)

def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    global last_dictated_text, last_dictated_length, last_word_end_time, last_command, last_command_time, skip_dictation

    if status:
        print(f"Audio callback status: {status}", file=sys.stderr)
    
    # Handle audio based on bit depth (16-bit, 24-bit, or 32-bit)
    if MIC_BITDEPTH == 16:
        dtype = np.int16
        max_value = 32767
    elif MIC_BITDEPTH == 24 or MIC_BITDEPTH == 32:
        dtype = np.int32
        max_value = 8388607 if MIC_BITDEPTH == 24 else 2147483647

    # Convert raw audio data to numpy array
    indata_array = np.frombuffer(indata, dtype=dtype)
    indata_normalized = indata_array.astype(np.float32) / (max_value + 1)

    # Apply pre-scale factor with relative sensitivity
    if RELATIVE_SENSITIVITY:
        reference_max = 32767
        scale_factor = reference_max / (max_value + 1)
        adjusted_pre_scale = PRE_SCALE_FACTOR * scale_factor
    else:
        adjusted_pre_scale = PRE_SCALE_FACTOR

    indata_normalized = indata_normalized * adjusted_pre_scale
    indata_array = np.clip(indata_normalized * 32767, -32768, 32767).astype(np.int16)

    # Convert stereo to mono (average channels)
    if MIC_CHANNELS > 1:
        indata_array = indata_array.reshape(-1, MIC_CHANNELS).mean(axis=1).astype(np.int16)

    # Resample the audio to the model's expected sample rate (16000 Hz)
    if MIC_SAMPLERATE != VOSK_SAMPLERATE:
        num_samples_resampled = int(len(indata_array) * VOSK_SAMPLERATE / MIC_SAMPLERATE)
        indata_array = resample(indata_array, num_samples_resampled)
        indata_array = indata_array.astype(np.int16)

    # Check if the audio block is below the silence threshold
    max_amplitude = np.max(np.abs(indata_array))
    if max_amplitude < SILENCE_AMPLITUDE_THRESHOLD:
        print(f"Block below silence threshold (max amplitude: {max_amplitude})")
        return
    
    # Append to audio buffer for WAV file
    audio_buffer.append(indata_array)
    
    indata_bytes = indata_array.tobytes()
    q.put(indata_bytes)

# Define MODEL_PATH
MODEL_PATH = r"C:\Users\MenaBeshai\Downloads\Speech to Text\vosk-model-en-us-0.42-gigaspeech\Speech_to_Text_Package\stt\vosk-model\vosk-model-en-us-0.42-gigaspeech"

# Load Vosk model
print("Loading Vosk model...")
if not MODEL_PATH:
    print("Error: MODEL_PATH is not defined. Please set MODEL_PATH to the Vosk model directory.")
    sys.exit(1)
if not os.path.exists(MODEL_PATH):
    print(f"Model path {MODEL_PATH} does not exist.")
    sys.exit(1)
model = vosk.Model(MODEL_PATH)

# List available audio devices
print("Available input audio devices:")
devices = sd.query_devices()
for i, device in enumerate(devices):
    if device['max_input_channels'] > 0:
        print(f"{i}: {device['name']}, {device['max_input_channels']} in")

# Clear the transcription file
with open(TRANSCRIPTION_FILE, "w", encoding="utf-8") as f:
    f.write("")

# Give user time to select a text field
print(f"Starting in {STARTUP_DELAY} seconds... Click into a text field to begin typing.")
time.sleep(STARTUP_DELAY)

# Start transcription
print("Starting live speech-to-text with Vosk (GigaSpeech 0.42 model). Speak to type or use commands! Press Ctrl+C to stop.")
rec = vosk.KaldiRecognizer(model, VOSK_SAMPLERATE)
last_partial = ""  # To track the last partial result and reduce spam
with sd.RawInputStream(samplerate=MIC_SAMPLERATE, blocksize=BLOCKSIZE, dtype="int32", channels=MIC_CHANNELS, callback=callback, device=DEVICE_INDEX):
    stop_listening = False
    while True:
        if stop_listening:
            break
        data = q.get()
        if rec.AcceptWaveform(data):
            result_dict = json.loads(rec.Result())
            text = result_dict.get("text", "")
            if text:
                # Handle special phrases first
                text = handle_special_phrases(text)
                # Normalize and tokenize the transcription
                normalized_text = normalize_text(text.lower())
                tokens = tuple(nltk.word_tokenize(normalized_text))
                    
                # Check for parameterized commands that should only run on final results
                is_final_command = False
                for cmd_tokens, command in TOKENIZED_PARAMETERIZED_FINAL:
                    if len(tokens) >= len(cmd_tokens) and tokens[:len(cmd_tokens)] == cmd_tokens:
                        print(f"\nDetected command: {text}")
                        last_processed_command = text
                        skip_dictation = True

                        param = text[len(command):].strip().lower()
                        try:
                            if command == "quote unquote ":
                                print(f"Executing action: quote unquote {param}")
                                keyboard.write(f'"{param}"')
                        except Exception as e:
                            print(f"Error executing command '{command}{param}': {e}")
                        is_final_command = True
                        break
                    
                if is_final_command:
                    continue
                    
                # Check if the transcription matches a command
                is_command = False
                if last_processed_command:
                    last_processed_tokens = tuple(nltk.word_tokenize(normalize_text(last_processed_command.lower())))
                    if tokens == last_processed_tokens:
                        print(f"Skipping dictation for command (already processed): {text}")
                        last_processed_command = None  # Reset after skipping
                        skip_dictation = False  # Reset for the next transcription
                        continue
                    
                # Check for simple commands
                for cmd_tokens, (cmd, _) in TOKENIZED_SIMPLE_COMMANDS.items():
                    if tokens == cmd_tokens:
                        print(f"Skipping dictation for command: {text}")
                        skip_dictation = False  # Reset for the next transcription
                        is_command = True
                        break
                    
                if is_command:
                    continue
                    
                # Process the text (capitalization and numbers)
                processed_text = process_text(text)
                    
                # Add paragraph breaks based on silence, but skip if "new paragraph" command was just executed
                current_time = time.time()
                if last_word_end_time > 0 and not (last_processed_command == "new paragraph"):
                    silence_duration = current_time - last_word_end_time
                    if silence_duration > SILENCE_THRESHOLD:
                        processed_text += "\n\n"
                    
                # Update the last word's end time for silence detection
                if "result" in result_dict and result_dict["result"]:
                    last_word_end_time = result_dict["result"][-1]["end"]

                # Try to convert the processed text to a number using words_to_numbers
                number = words_to_numbers.convert_numbers(processed_text, FRACTION_MAP, SYMBOLS_MAP, GOOGLE_NUMBERS, LARGE_NUMBERS_MAP)
                if number is not None:
                    final_text = str(number)  # Convert the number to a string for output
                else:
                    final_text = processed_text  # If conversion fails, use the original processed text

                print(f"Transcription: {final_text}")

                # Save to file
                with open(TRANSCRIPTION_FILE, "a", encoding="utf-8") as f:
                    f.write(final_text + " ")

                # Type the text (excluding commands that return a result)
                if not any(final_text.startswith(cmd) for cmd in ["\n\n", "\n", " ", "\t"]):
                    keyboard.write(final_text)
                    keyboard.write(" ")
                    # Store the last dictated text for "scratch that", "correct <word>", etc.
                    last_dictated_text = final_text + " "
                    last_dictated_length = len(last_dictated_text)

                skip_dictation = False  # Reset for the next transcription
        else:
            partial_dict = json.loads(rec.PartialResult())
            partial = partial_dict.get("partial", "")
            if partial:
                # Only print partial if it has changed
                if partial != last_partial:
                    print(f"Partial: {partial}", end="\r")
                    last_partial = partial
                    
                # Normalize and tokenize the partial transcription
                normalized_partial = normalize_text(partial.lower())
                partial_tokens = tuple(nltk.word_tokenize(normalized_partial))
                current_time = time.time()

                # Handle commands with parameters (e.g., "highlight <word>", "go to address <word>")
                for cmd_tokens, command in TOKENIZED_PARAMETERIZED_PARTIAL:
                    if len(partial_tokens) >= len(cmd_tokens) and partial_tokens[:len(cmd_tokens)] == cmd_tokens:
                        if last_command == partial and (current_time - last_command_time) < COMMAND_DEBOUNCE_TIME:
                            continue
                            
                        print(f"\nDetected command: {partial}")
                        last_command = partial
                        last_command_time = current_time
                        last_processed_command = partial
                        skip_dictation = True

                        param = partial[len(command):].strip().lower()
                        try:
                            if command == "highlight ":
                                print(f"Executing action: highlight {param}")
                                keyboard.press_and_release("ctrl+f")
                                time.sleep(0.2)
                                keyboard.write(param)
                                time.sleep(0.1)
                                keyboard.press_and_release("enter")
                                time.sleep(0.1)
                                keyboard.press_and_release("escape")
                                keyboard.press("shift")
                                for _ in range(len(param)):
                                    keyboard.press_and_release("right")
                                keyboard.release("shift")
                            elif command == "find ":
                                print(f"Executing action: find {param}")
                                keyboard.press_and_release("ctrl+f")
                                time.sleep(0.2)
                                keyboard.write(param)
                                time.sleep(0.1)
                                keyboard.press_and_release("enter")
                                time.sleep(0.1)
                                keyboard.press_and_release("escape")
                            elif command == "insert after ":
                                print(f"Executing action: insert after {param}")
                                keyboard.press_and_release("ctrl+f")
                                time.sleep(0.2)
                                keyboard.write(param)
                                time.sleep(0.1)
                                keyboard.press_and_release("enter")
                                time.sleep(0.1)
                                keyboard.press_and_release("escape")
                                for _ in range(len(param)):
                                    keyboard.press_and_release("right")
                            elif command == "insert before ":
                                print(f"Executing action: insert before {param}")
                                keyboard.press_and_release("ctrl+f")
                                time.sleep(0.2)
                                keyboard.write(param)
                                time.sleep(0.1)
                                keyboard.press_and_release("enter")
                                time.sleep(0.1)
                                keyboard.press_and_release("escape")
                            elif command == "copy ":
                                print(f"Executing action: copy {param}")
                                keyboard.press_and_release("ctrl+f")
                                time.sleep(0.2)
                                keyboard.write(param)
                                time.sleep(0.1)
                                keyboard.press_and_release("enter")
                                time.sleep(0.1)
                                keyboard.press_and_release("escape")
                                keyboard.press("shift")
                                for _ in range(len(param)):
                                    keyboard.press_and_release("right")
                                keyboard.release("shift")
                                keyboard.press_and_release("ctrl+c")
                            elif command == "cut ":
                                print(f"Executing action: cut {param}")
                                keyboard.press_and_release("ctrl+f")
                                time.sleep(0.2)
                                keyboard.write(param)
                                time.sleep(0.1)
                                keyboard.press_and_release("enter")
                                time.sleep(0.1)
                                keyboard.press_and_release("escape")
                                keyboard.press("shift")
                                for _ in range(len(param)):
                                    keyboard.press_and_release("right")
                                keyboard.release("shift")
                                keyboard.press_and_release("ctrl+x")
                            elif command == "all caps ":
                                print(f"Executing action: all caps {param}")
                                keyboard.press_and_release("ctrl+f")
                                time.sleep(0.2)
                                keyboard.write(param)
                                time.sleep(0.1)
                                keyboard.press_and_release("enter")
                                time.sleep(0.1)
                                keyboard.press_and_release("escape")
                                keyboard.press("shift")
                                for _ in range(len(param)):
                                    keyboard.press_and_release("right")
                                keyboard.release("shift")
                                keyboard.press_and_release("delete")
                                keyboard.write(param.upper())
                            elif command == "press ":
                                print(f"Executing action: press {param}")
                                keyboard.press_and_release(param)
                            elif command == "open ":
                                print(f"Executing action: open {param}")
                                if not param:
                                    print("Error: No application name provided for 'open' command.")
                                    break
                                stt_apps = os.environ.get("STT", "")
                                if not stt_apps:
                                    print("Error: STT environment variable is not set.")
                                    break
                                app_dict = {}
                                for app in stt_apps.split(";"):
                                    if not app:
                                        continue
                                    if "=" not in app:
                                        print(f"Warning: Invalid STT entry '{app}' - missing '=' delimiter. Skipping.")
                                        continue
                                    key, value = app.split("=", 1)
                                    app_dict[key] = value
                                app_name = param.replace(" ", "").lower()
                                app_path = app_dict.get(app_name)
                                if app_path:
                                    try:
                                        subprocess.Popen(app_path, shell=True)
                                    except Exception as e:
                                        print(f"Error launching application '{app_name}': {e}")
                                else:
                                    print(f"Application '{param}' not found in STT environment variable.")
                            elif command == "go to address ":
                                print(f"Executing action: go to address {param}")
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
                                try:
                                    subprocess.Popen(['start', url], shell=True)
                                except Exception as e:
                                    print(f"Error opening URL '{url}': {e}")
                            elif command == "move up " or command == "move down " or command == "move left " or command == "move right ":
                                print(f"Executing action: {command}{param}")
                                try:
                                    num = int(param.split()[0])
                                except (ValueError, IndexError):
                                    num = 1
                                direction = command.split()[1]
                                for _ in range(num):
                                    keyboard.press_and_release(direction)
                            elif command == "function ":
                                print(f"Executing action: function {param}")
                                function_key = param.lower().replace("f", "")
                                if function_key in [str(i) for i in range(1, 13)]:
                                    keyboard.press_and_release(f"f{function_key}")
                                else:
                                    print(f"Invalid function key: {param}")
                        except Exception as e:
                            print(f"Error executing command '{command}{param}': {e}")
                        break

                # Handle "select <word> through <word>"
                if "select " in normalized_partial and " through " in normalized_partial:
                    partial_tokens_list = list(partial_tokens)
                    if "through" in partial_tokens_list:
                        if last_command == partial and (current_time - last_command_time) < COMMAND_DEBOUNCE_TIME:
                            continue
                            
                        print(f"\nDetected command: {partial}")
                        last_command = partial
                        last_command_time = current_time
                        last_processed_command = partial
                        skip_dictation = True

                        try:
                            parts = partial.lower().split(" through ")
                            word1 = parts[0].replace("select ", "").strip()
                            word2 = parts[1].strip()
                            print(f"Executing action: select {word1} through {word2}")
                            keyboard.press_and_release("ctrl+f")
                            time.sleep(0.2)
                            keyboard.write(word1)
                            time.sleep(0.1)
                            keyboard.press_and_release("enter")
                            time.sleep(0.1)
                            keyboard.press_and_release("escape")
                            keyboard.press("shift")
                            keyboard.press_and_release("ctrl+f")
                            time.sleep(0.2)
                            keyboard.write(word2)
                            time.sleep(0.1)
                            keyboard.press_and_release("enter")
                            time.sleep(0.1)
                            keyboard.press_and_release("escape")
                            for _ in range(len(word2)):
                                keyboard.press_and_release("right")
                            keyboard.release("shift")
                        except Exception as e:
                            print(f"Error executing command 'select {word1} through {word2}': {e}")

                # Handle "correct <word>"
                if "correct " in normalized_partial:
                    if partial_tokens[:1] == ("correct",):
                        if last_command == partial and (current_time - last_command_time) < COMMAND_DEBOUNCE_TIME:
                            continue
                            
                        print(f"\nDetected command: {partial}")
                        last_command = partial
                        last_command_time = current_time
                        last_processed_command = partial
                        skip_dictation = True

                        try:
                            word_to_correct = partial.lower().replace("correct ", "").strip()
                            print(f"Executing action: correct {word_to_correct}")
                            keyboard.press_and_release("ctrl+f")
                            time.sleep(0.2)
                            keyboard.write(word_to_correct)
                            time.sleep(0.1)
                            keyboard.press_and_release("enter")
                            time.sleep(0.1)
                            keyboard.press_and_release("escape")
                            keyboard.press("shift")
                            for _ in range(len(word_to_correct)):
                                keyboard.press_and_release("right")
                            keyboard.release("shift")
                            keyboard.press_and_release("delete")
                            corrected_word = spell.correction(word_to_correct)
                            keyboard.write(corrected_word)
                        except Exception as e:
                            print(f"Error executing command 'correct {word_to_correct}': {e}")

                # Handle simple commands
                for cmd_tokens, (command, action) in TOKENIZED_SIMPLE_COMMANDS.items():
                    if partial_tokens == cmd_tokens:
                        if last_command == command and (current_time - last_command_time) < COMMAND_DEBOUNCE_TIME:
                            continue
                            
                        print(f"\nDetected command: {command}")
                        last_command = command
                        last_command_time = current_time
                        last_processed_command = command
                        skip_dictation = True
                            
                        try:
                            if action == "cmd_stop_listening":
                                print("Executing action: stop listening")
                                stop_listening = True
                            elif action == "cmd_select_all":
                                print("Executing action: select all")
                                time.sleep(0.1)  # Small delay to ensure focus
                                keyboard.press("ctrl")
                                keyboard.press_and_release("a")
                                keyboard.release("ctrl")
                            elif action == "cmd_select_down":
                                print("Executing action: select down")
                                time.sleep(0.1)  # Consistent delay
                                keyboard.press("shift")
                                keyboard.press_and_release("down")
                                keyboard.release("shift")
                            elif action == "cmd_select_up":
                                print("Executing action: select up")
                                time.sleep(0.1)  # Consistent delay
                                keyboard.press("shift")
                                keyboard.press_and_release("up")
                                keyboard.release("shift")
                            elif action == "cmd_select_all_up":
                                print("Executing action: select all up")
                                time.sleep(0.1)  # Consistent delay
                                keyboard.press("shift")
                                keyboard.press_and_release("home")
                                keyboard.release("shift")
                            elif action == "cmd_select_all_down":
                                print("Executing action: select all down")
                                time.sleep(0.1)  # Consistent delay
                                keyboard.press("shift")
                                keyboard.press_and_release("end")
                                keyboard.release("shift")
                            elif action == "cmd_copy":
                                print("Executing action: copy")
                                keyboard.press("ctrl")
                                keyboard.press_and_release("c")
                                keyboard.release("ctrl")
                            elif action == "cmd_paste":
                                print("Executing action: paste")
                                keyboard.press("ctrl")
                                keyboard.press_and_release("v")
                                keyboard.release("ctrl")
                            elif action == "cmd_delete":
                                print("Executing action: delete")
                                keyboard.press_and_release("backspace")
                            elif action == "cmd_undo":
                                print("Executing action: undo")
                                keyboard.press("ctrl")
                                keyboard.press_and_release("z")
                                keyboard.release("ctrl")
                            elif action == "cmd_redo":
                                print("Executing action: redo")
                                keyboard.press("ctrl")
                                keyboard.press_and_release("y")
                                keyboard.release("ctrl")
                            elif action == "cmd_file_properties":
                                print("Executing action: file properties")
                                keyboard.press_and_release("menu")
                            elif action == "cmd_save_document":
                                print("Executing action: save document")
                                keyboard.press("ctrl")
                                keyboard.press_and_release("s")
                                keyboard.release("ctrl")
                            elif action == "cmd_open_file":
                                print("Executing action: open file (placeholder)")
                            elif action == "cmd_move_up":
                                print("Executing action: move up")
                                keyboard.press_and_release("up")
                            elif action == "cmd_move_down":
                                print("Executing action: move down")
                                keyboard.press_and_release("down")
                            elif action == "cmd_move_left":
                                print("Executing action: move left")
                                keyboard.press_and_release("left")
                            elif action == "cmd_move_right":
                                print("Executing action: move right")
                                keyboard.press_and_release("right")
                            elif action == "cmd_move_up_paragraph":
                                print("Executing action: move up paragraph")
                                keyboard.press("ctrl")
                                keyboard.press_and_release("up")
                                keyboard.release("ctrl")
                            elif action == "cmd_move_down_paragraph":
                                print("Executing action: move down paragraph")
                                keyboard.press("ctrl")
                                keyboard.press_and_release("down")
                                keyboard.release("ctrl")
                            elif action == "cmd_enter":
                                print("Executing action: enter")
                                keyboard.press_and_release("enter")
                            elif action == "cmd_number_lock":
                                print("Executing action: number lock")
                                number_lock_on = not number_lock_on
                                print(f"Number lock is now {'on' if number_lock_on else 'off'}")
                            elif action == "cmd_caps_lock_on":
                                print("Executing action: caps lock on")
                                if not caps_lock_on:
                                    keyboard.press_and_release("caps lock")
                                    caps_lock_on = True
                            elif action == "cmd_caps_lock_off":
                                print("Executing action: caps lock off")
                                if caps_lock_on:
                                    keyboard.press_and_release("caps lock")
                                    caps_lock_on = False
                            elif action == "cmd_bold":
                                print("Executing action: bold")
                                keyboard.press("ctrl")
                                keyboard.press_and_release("b")
                                keyboard.release("ctrl")
                            elif action == "cmd_italicize":
                                print("Executing action: italicize")
                                keyboard.press("ctrl")
                                keyboard.press_and_release("i")
                                keyboard.release("ctrl")
                            elif action == "cmd_underline":
                                print("Executing action: underline")
                                keyboard.press("ctrl")
                                keyboard.press_and_release("u")
                                keyboard.release("ctrl")
                            elif action == "cmd_center":
                                print("Executing action: center")
                                keyboard.press("ctrl")
                                keyboard.press_and_release("e")
                                keyboard.release("ctrl")
                            elif action == "cmd_left_align":
                                print("Executing action: left align")
                                keyboard.press("ctrl")
                                keyboard.press_and_release("l")
                                keyboard.release("ctrl")
                            elif action == "cmd_right_align":
                                print("Executing action: right align")
                                keyboard.press("ctrl")
                                keyboard.press_and_release("r")
                                keyboard.release("ctrl")
                            elif action == "cmd_cut":
                                print("Executing action: cut")
                                keyboard.press("ctrl")
                                keyboard.press_and_release("x")
                                keyboard.release("ctrl")
                            elif action == "cmd_go_to_beginning":
                                print("Executing action: go to beginning")
                                keyboard.press("ctrl")
                                keyboard.press_and_release("home")
                                keyboard.release("ctrl")
                            elif action == "cmd_go_to_end":
                                print("Executing action: go to end")
                                keyboard.press("ctrl")
                                keyboard.press_and_release("end")
                                keyboard.release("ctrl")
                            elif action == "cmd_go_to_beginning_of_line":
                                print("Executing action: go to beginning of line")
                                keyboard.press_and_release("home")
                            elif action == "cmd_go_to_end_of_line":
                                print("Executing action: go to end of line")
                                keyboard.press_and_release("end")
                            elif action == "cmd_go_to_address":
                                print("Executing action: go to address")
                                keyboard.press("ctrl")
                                keyboard.press_and_release("l")
                                keyboard.release("ctrl")
                            elif action == "cmd_refresh_page":
                                print("Executing action: refresh page")
                                keyboard.press_and_release("f5")
                            elif action == "cmd_go_back":
                                print("Executing action: go back")
                                keyboard.press("alt")
                                keyboard.press_and_release("left")
                                keyboard.release("alt")
                            elif action == "cmd_go_forward":
                                print("Executing action: go forward")
                                keyboard.press("alt")
                                keyboard.press_and_release("right")
                                keyboard.release("alt")
                            elif action == "cmd_open_new_tab":
                                print("Executing action: open new tab")
                                keyboard.press("ctrl")
                                keyboard.press_and_release("t")
                                keyboard.release("ctrl")
                            elif action == "cmd_close_tab":
                                print("Executing action: close tab")
                                keyboard.press("ctrl")
                                keyboard.press_and_release("w")
                                keyboard.release("ctrl")
                            elif action == "cmd_next_tab":
                                print("Executing action: next tab")
                                keyboard.press("ctrl")
                                keyboard.press_and_release("tab")
                                keyboard.release("ctrl")
                            elif action == "cmd_previous_tab":
                                print("Executing action: previous tab")
                                keyboard.press("ctrl")
                                keyboard.press("shift")
                                keyboard.press_and_release("tab")
                                keyboard.release("shift")
                                keyboard.release("ctrl")
                            elif action == "cmd_shift_tab":
                                print("Executing action: shift tab")
                                keyboard.press("shift")
                                keyboard.press_and_release("tab")
                                keyboard.release("shift")
                            elif action == "cmd_scratch_that":
                                print("Executing action: scratch that")
                                for _ in range(last_dictated_length):
                                    keyboard.press_and_release("backspace")
                            elif action == "cmd_click_that":
                                print("Executing action: click that (left mouse click)")
                                pyautogui.click()
                            elif action == "cmd_punctuation_period":
                                print("Executing action: punctuation period")
                                keyboard.write(".")
                            elif action == "cmd_punctuation_comma":
                                print("Executing action: punctuation comma")
                                keyboard.write(",")
                            elif action == "cmd_punctuation_question_mark":
                                print("Executing action: punctuation question mark")
                                keyboard.write("?")
                            elif action == "cmd_punctuation_exclamation":
                                print("Executing action: punctuation exclamation")
                                keyboard.write("!")
                            elif action == "cmd_punctuation_semicolon":
                                print("Executing action: punctuation semicolon")
                                keyboard.write(";")
                            elif action == "cmd_punctuation_colon":
                                print("Executing action: punctuation colon")
                                keyboard.write(":")
                            elif action == "cmd_punctuation_tilde":
                                print("Executing action: punctuation tilde")
                                keyboard.write("~")
                            elif action == "cmd_punctuation_ampersand":
                                print("Executing action: punctuation ampersand")
                                keyboard.write("&")
                            elif action == "cmd_punctuation_percent":
                                print("Executing action: punctuation percent")
                                keyboard.write("%")
                            elif action == "cmd_punctuation_asterisk":
                                print("Executing action: punctuation asterisk")
                                keyboard.write("*")
                            elif action == "cmd_punctuation_parentheses":
                                print("Executing action: punctuation parentheses")
                                keyboard.write("()")
                                keyboard.press_and_release("left")  # Move cursor between the parentheses
                            elif action == "cmd_punctuation_dash":
                                print("Executing action: punctuation dash")
                                keyboard.write("-")
                            elif action == "cmd_punctuation_underscore":
                                print("Executing action: punctuation underscore")
                                keyboard.write("_")
                            elif action == "cmd_punctuation_plus":
                                print("Executing action: punctuation plus")
                                keyboard.write("+")
                            elif action == "cmd_punctuation_equals":
                                print("Executing action: punctuation equals")
                                keyboard.write("=")
                            elif action == "cmd_press_escape":
                                print("Executing action: press escape")
                                keyboard.press_and_release("escape")
                            elif action == "cmd_screen_shoot":
                                print("Executing action: screen shoot")
                                pyautogui.press("printscreen")
                            elif action == "cmd_screen_shoot_window":
                                print("Executing action: screen shoot window")
                                pyautogui.hotkey("alt", "printscreen")
                            elif action == "cmd_screen_shoot_monitor":
                                print("Executing action: screen shoot monitor")
                                try:
                                    # Try to open the modern Snipping Tool (Windows 11) or fallback to SnippingTool.exe
                                    subprocess.Popen("ms-screenclip:", shell=True)  # Windows 11 Snipping Tool URI
                                    time.sleep(1)  # Wait for the tool to open
                                    pyautogui.hotkey("ctrl", "n")  # Start a new snip
                                except Exception as e:
                                    print(f"Error opening Snipping Tool: {e}")
                                    try:
                                        # Fallback to SnippingTool.exe for older Windows versions
                                        subprocess.Popen("SnippingTool.exe", shell=True)
                                        time.sleep(1)
                                        pyautogui.hotkey("ctrl", "n")
                                    except Exception as e:
                                        print(f"Error opening fallback Snipping Tool: {e}")
                            elif action == "cmd_task_manager":
                                print("Executing action: task manager")
                                keyboard.press("ctrl")
                                keyboard.press("shift")
                                keyboard.press_and_release("esc")
                                keyboard.release("shift")
                                keyboard.release("ctrl")
                            elif action == "cmd_debug_screen":
                                print("Executing action: debug screen")
                                keyboard.press("ctrl")
                                keyboard.press("alt")
                                keyboard.press_and_release("delete")
                                keyboard.release("alt")
                                keyboard.release("ctrl")
                            elif action == "cmd_force_close":
                                print("Executing action: force close")
                                keyboard.press("alt")
                                keyboard.press_and_release("f4")
                                keyboard.release("alt")
                            else:
                                print(f"Executing action: write '{action}'")
                                keyboard.write(action)
                        except Exception as e:
                            print(f"Error executing command '{command}': {e}")
                        break

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
        default_settings = {
            "bit_depth": "int24", "sample_rate": "48000", "pre_scale_factor": 0.002,
            "unit": "Numbers", "relative_sensitivity": False, "silence_threshold": 10.0,
            "show_peaks": False, "theme": "Dark", "host_api": "MME", "input_device": None,
            "output_device": None
        }
        base_path = get_base_path()
        config_path = os.path.join(base_path, CONFIG_PATH)
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.saved_settings = json.load(f)
        except Exception:
            self.saved_settings = default_settings
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self.saved_settings, f, indent=4)
        self.saved_settings = {**default_settings, **self.saved_settings}
        
        bit_depth = self.saved_settings["bit_depth"]
        if " - " in bit_depth:
            bit_depth = bit_depth.split(" - ")[0]
        self.saved_settings["bit_depth"] = bit_depth
        
        dpg.set_value("bit_depth_combo", self.saved_settings["bit_depth"])
        dpg.set_value("sample_rate_combo", self.saved_settings["sample_rate"])
        dpg.set_value("unit_combo", self.saved_settings["unit"])
        self.set_slider_from_pre_scale(self.saved_settings["pre_scale_factor"])
        dpg.set_value("relative_sensitivity_check", self.saved_settings["relative_sensitivity"])
        dpg.set_value("silence_input", self.saved_settings["silence_threshold"])
        dpg.set_value("show_peaks_check", self.saved_settings["show_peaks"])
        dpg.set_value("theme_combo", self.saved_settings["theme"])
        dpg.set_value("host_api_combo", self.saved_settings["host_api"])
        dpg.set_value("input_device_combo", self.saved_settings["input_device"])
        dpg.set_value("output_device_combo", self.saved_settings.get("output_device", ""))
        if "model_path" in self.saved_settings:
            model_path = self.saved_settings["model_path"]
            if not os.path.isabs(model_path):
                base_path = get_base_path()
                model_path = os.path.abspath(os.path.join(base_path, model_path))
            dpg.set_value("model_path_input", model_path)
        self.update_host_api(None, self.saved_settings["host_api"])
    
    def save_settings(self, sender, app_data):
        base_path = get_base_path()
        config_path = os.path.join(base_path, CONFIG_PATH)
        existing_config = {}
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                existing_config = json.load(f)
        except Exception:
            pass
        
        slider_value = dpg.get_value("sensitivity_slider")
        unit = dpg.get_value("unit_combo")
        if unit == "Numbers":
            pre_scale_factor = self.slider_to_pre_scale(slider_value)
        elif unit == "Percent":
            pre_scale_factor = self.percent_to_pre_scale(slider_value)
        elif unit == "dB":
            db = (slider_value / 100) * 100 - 60
            pre_scale_factor = self.db_to_pre_scale(db)
        pre_scale_factor = min(pre_scale_factor, 10.0)
        
        bit_depth = dpg.get_value("bit_depth_combo")
        if " - " in bit_depth:
            bit_depth = bit_depth.split(" - ")[0]
        
        self.saved_settings = {
            "bit_depth": bit_depth,
            "sample_rate": dpg.get_value("sample_rate_combo"),
            "pre_scale_factor": pre_scale_factor,
            "unit": dpg.get_value("unit_combo"),
            "relative_sensitivity": dpg.get_value("relative_sensitivity_check"),
            "silence_threshold": dpg.get_value("silence_input"),
            "show_peaks": dpg.get_value("show_peaks_check"),
            "theme": dpg.get_value("theme_combo"),
            "host_api": dpg.get_value("host_api_combo"),
            "input_device": dpg.get_value("input_device_combo"),
            "output_device": dpg.get_value("output_device_combo")
        }
        
        model_path = dpg.get_value("model_path_input")
        if model_path and model_path != "Not set":
            self.saved_settings["model_path"] = os.path.abspath(model_path)
        
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self.saved_settings, f, indent=4)
            dpg.set_value("status_text", "Settings saved successfully.")
        except Exception as e:
            dpg.set_value("status_text", f"Failed to save settings: {e}")
    
    def setup_audio_tab(self):
        with dpg.group(horizontal=True):
            dpg.add_text("Host API:")
            dpg.add_combo(self.get_host_apis(), default_value="MME", callback=self.update_host_api, tag="host_api_combo")
        
        with dpg.group(horizontal=True):
            dpg.add_text("Input Device:")
            dpg.add_combo([], tag="input_device_combo", callback=self.update_device, width=300)
        
        with dpg.group(horizontal=True):
            dpg.add_text("Output Device:")
            dpg.add_combo([], tag="output_device_combo", callback=self.update_output_device, width=300)
        
        with dpg.group(horizontal=True):
            dpg.add_text("Model Path:")
            dpg.add_input_text(tag="model_path_input", default_value="Not set", width=300, readonly=True)
            dpg.add_button(label="Set Model Path", callback=self.set_model_path)
        
        with dpg.group(horizontal=True):
            dpg.add_text("Data Type:")
            dpg.add_combo(list(DATA_TYPES.keys()), default_value="int24", callback=self.update_bit_depth, tag="bit_depth_combo")
            with dpg.tooltip("bit_depth_combo"):
                dpg.add_text(DATA_TYPES[dpg.get_value("bit_depth_combo")], tag="bit_depth_tooltip")
        
        with dpg.group(horizontal=True):
            dpg.add_text("Sample Rate:")
            dpg.add_combo(SAMPLE_RATES, default_value="48000", callback=self.update_sample_rate, tag="sample_rate_combo")
        
        with dpg.group(horizontal=True):
            dpg.add_text("Sensitivity Unit:")
            dpg.add_combo(["Numbers", "Percent", "dB"], default_value="Numbers", callback=self.update_unit, tag="unit_combo")
        
        with dpg.group(horizontal=True):
            dpg.add_text("Sensitivity (Volume):")
            dpg.add_slider_float(default_value=0, min_value=0, max_value=100, callback=self.update_pre_scale_label, tag="sensitivity_slider", width=400)
            dpg.add_button(label="0.002", tag="pre_scale_label", callback=self.open_manual_sensitivity_input)
        
        dpg.add_checkbox(label="Keep Same Relative Sensitivity", default_value=False, callback=self.update_relative_sensitivity, tag="relative_sensitivity_check")
        dpg.add_checkbox(label="Show Peaks", default_value=False, tag="show_peaks_check")
        dpg.add_checkbox(label="Enable Debug Recording", default_value=False, tag="debug_recording_check")
        
        with dpg.group(horizontal=True):
            dpg.add_text("Silence Threshold:")
            dpg.add_input_float(default_value=10.0, tag="silence_input")
        
        with dpg.group(horizontal=True):
            dpg.add_text("Input dB Level (optional):")
            dpg.add_input_float(default_value=-18.0, tag="db_level_input")
        
        with dpg.group(horizontal=True):
            dpg.add_button(label="Suggest Settings", callback=self.suggest_settings)
            dpg.add_button(label="Calibrate", callback=self.calibrate)
            dpg.add_button(label="Reset to Defaults", callback=self.reset_settings)
        
        with dpg.group(horizontal=True):
            dpg.add_button(label="Test Audio", tag="test_audio_button", callback=self.toggle_audio_test)
            dpg.add_button(label="Record", tag="record_button", callback=self.toggle_recording)
            dpg.add_button(label="Show Waveform", callback=self.show_waveform)
        
        with dpg.group(horizontal=True):
            dpg.add_text("Audio Level:")
            with dpg.drawlist(width=400, height=20, tag="level_drawlist"):
                dpg.draw_rectangle((0, 0), (400 * 0.6, 20), fill=(0, 255, 0, 255))
                dpg.draw_rectangle((400 * 0.6, 0), (400 * 0.8, 20), fill=(255, 255, 0, 255))
                dpg.draw_rectangle((400 * 0.8, 0), (400, 20), fill=(255, 0, 0, 255))
                dpg.draw_rectangle((0, 0), (0, 20), fill=(0, 0, 255, 255), tag="level_bar")
                dpg.draw_rectangle((0, 0), (0, 20), fill=(255, 255, 255, 50), tag="shadow_bar")
                dpg.draw_circle((380, 10), 10, fill=(255, 0, 0, 128), tag="clipping_indicator")
        
        dpg.add_text("", tag="status_text")

    def setup_json_tab(self, tab_name, json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                self.json_data[tab_name] = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load {tab_name} JSON: {e}")
            self.json_data[tab_name] = {}
        
        with dpg.group():
            dpg.add_text(f"{tab_name} Configuration:")
            if isinstance(self.json_data[tab_name], dict):
                for key, value in self.json_data[tab_name].items():
                    dpg.add_input_text(label=key, default_value=str(value), tag=f"{tab_name}_{key}_input")
            else:
                for idx, item in enumerate(self.json_data[tab_name]):
                    dpg.add_input_text(label=f"Item {idx}", default_value=str(item), tag=f"{tab_name}_{idx}_input")
            dpg.add_button(label=f"Save {tab_name}", callback=lambda s, a: self.save_json_tab(tab_name, json_path))
    
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
        base_path = get_base_path()
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
    
    def open_manual_sensitivity_input(self, sender, app_data):
        unit = dpg.get_value("unit_combo")
        slider_value = dpg.get_value("sensitivity_slider")
        if unit == "Numbers":
            default_value = self.slider_to_pre_scale(slider_value)
        elif unit == "Percent":
            default_value = self.pre_scale_to_percent(self.slider_to_pre_scale(slider_value))
        elif unit == "dB":
            default_value = self.pre_scale_to_db(self.slider_to_pre_scale(slider_value))
        with dpg.window(label=f"Set Sensitivity ({unit})", modal=True, width=300, height=150, tag="manual_sensitivity_window"):
            dpg.add_text(f"Enter sensitivity value in {unit}:")
            dpg.add_input_float(default_value=default_value, tag="manual_sensitivity_input", step=0.0001, format="%.4f")
            with dpg.group(horizontal=True):
                dpg.add_button(label="Save", callback=self.save_manual_sensitivity)
                dpg.add_button(label="Cancel", callback=lambda: dpg.delete_item("manual_sensitivity_window"))
    
    def save_manual_sensitivity(self, sender, app_data):
        unit = dpg.get_value("unit_combo")
        input_value = dpg.get_value("manual_sensitivity_input")
        if unit == "Numbers":
            pre_scale = input_value
        elif unit == "Percent":
            pre_scale = self.percent_to_pre_scale(input_value)
        elif unit == "dB":
            pre_scale = self.db_to_pre_scale(input_value)
        self.set_slider_from_pre_scale(pre_scale)
        dpg.delete_item("manual_sensitivity_window")
    
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
        host_api = app_data
        input_devices = self.get_input_devices(host_api)
        dpg.configure_item("input_device_combo", items=input_devices)
        if input_devices:
            if not dpg.get_value("input_device_combo") or dpg.get_value("input_device_combo") not in input_devices:
                dpg.set_value("input_device_combo", input_devices[0])
        else:
            dpg.set_value("input_device_combo", "")
        self.update_device(None, dpg.get_value("input_device_combo"))

        # Populate output devices
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

    def update_output_device(self, sender, app_data):
        device_name = app_data
        host_api_name = dpg.get_value("host_api_combo")
        device_index = self.get_device_index(device_name, host_api_name, is_input=False)
        if device_index is None:
            return
        if self.is_testing:
            self.stop_audio_test()
            self.toggle_audio_test()

    def update_bit_depth(self, sender, app_data):
        # Update the tooltip when the bit depth changes
        bit_depth = app_data
        if " - " in bit_depth:
            bit_depth = bit_depth.split(" - ")[0]
        dpg.set_value("bit_depth_tooltip", DATA_TYPES[bit_depth])
        if self.is_testing:
            self.stop_audio_test()
            self.toggle_audio_test()

    def update_sample_rate(self, sender, app_data):
        if self.is_testing:
            self.stop_audio_test()
            self.toggle_audio_test()

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

    def update_pre_scale_label(self, sender, app_data):
        unit = dpg.get_value("unit_combo")
        slider_value = app_data if app_data is not None else dpg.get_value("sensitivity_slider")
        if unit == "Numbers":
            pre_scale = self.slider_to_pre_scale(slider_value)
            dpg.set_value("pre_scale_label", f"{pre_scale:.3f}")
        elif unit == "Percent":
            percent = self.percent_to_pre_scale(slider_value)
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
            # Cap the pre_scale to prevent over-amplification
            pre_scale = min(pre_scale, 1.0)  # Adjust this threshold as needed
            adjusted_pre_scale = pre_scale / scale_factor
        else:
            adjusted_pre_scale = pre_scale
        self.set_slider_from_pre_scale(adjusted_pre_scale)

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

    def on_data_type_changed(self, sender, app_data):
        old_dtype_str = self.saved_settings.get("selected_data_type", "int16")
        new_dtype_str = app_data

        old_scale_factor = self.get_bit_depth_scale(old_dtype_str)
        new_scale_factor = self.get_bit_depth_scale(new_dtype_str)

        current_slider_value_normalized = dpg.get_value("sensitivity_slider") / 100.0 # Normalize to 0-1

        # Assume the slider linearly controls a gain factor (simplified)
        current_gain = self.slider_to_pre_scale(current_slider_value_normalized * 100) # Use the "Numbers" conversion

        # Estimate current dB level (very rough approximation)
        # This assumes a baseline input level. A more accurate method would require
        # actual audio analysis.
        baseline_amplitude = 0.0001 # Arbitrary baseline
        current_amplitude = baseline_amplitude * current_gain
        current_db = 20 * np.log10(current_amplitude / (1 / old_scale_factor) if (1 / old_scale_factor) != 0 else 1e-9) # Relative to max amplitude

        # Calculate target pre-scale for the new bit depth to achieve a similar dB level
        target_amplitude_relative_to_new_max = 10**(current_db / 20) * (1 / new_scale_factor)
        target_pre_scale = target_amplitude_relative_to_new_max / baseline_amplitude if baseline_amplitude != 0 else 1.0

        # Convert target pre-scale back to a slider value (using the dB conversion in reverse)
        # This requires inverting your db_to_pre_scale logic.
        def pre_scale_to_slider_db(pre_scale):
            db = 20 * np.log10(pre_scale)
            return (db + 60) * 100 / 100 # Reverse of your db conversion

        target_slider_value_db = pre_scale_to_slider_db(target_pre_scale)
        new_slider_value = np.clip(target_slider_value_db, 0, 100)

        dpg.set_value("sensitivity_slider", new_slider_value)
        dpg.set_value("sensitivity_unit_combo", "dB") # Switch the unit to dB for clarity

        self.saved_settings["selected_data_type"] = new_dtype_str
        self.save_settings()

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

    def update_pre_scale_label(self, sender, app_data):
        slider_value = dpg.get_value("sensitivity_slider")
        unit = dpg.get_value("unit_combo")
        if unit == "Numbers":
            pre_scale = self.slider_to_pre_scale(slider_value)
            label = f"{pre_scale:.4f}"
        elif unit == "Percent":
            label = f"{slider_value:.1f}%"
        elif unit == "dB":
            db = (slider_value / 100) * 100 - 60
            label = f"{db:.1f} dB"
        dpg.configure_item("pre_scale_label", label=label)

    def update_unit(self, sender, app_data):
        slider_value = dpg.get_value("sensitivity_slider")
        unit = app_data
        if unit == "Numbers":
            pre_scale = self.slider_to_pre_scale(slider_value)
        elif unit == "Percent":
            pre_scale = self.percent_to_pre_scale(slider_value)
        elif unit == "dB":
            db = (slider_value / 100) * 100 - 60
            pre_scale = self.db_to_pre_scale(db)
        self.set_slider_from_pre_scale(pre_scale)

    def pre_scale_to_db(self, pre_scale):
        return -float("inf") if pre_scale <= 0 else 20 * math.log10(pre_scale)

    def update_shadow(self):
        # Update the shadow bar position based on noise floor
        shadow_position = (self.noise_floor / 32767) * 400
        if dpg.does_item_exist("shadow_bar"):
            dpg.configure_item("shadow_bar", pmax=(shadow_position, 20))

    def update_gui(self):
        # Periodically update the audio level visualization
        if dpg.is_dearpygui_running():
            self.update_shadow()
            # Schedule the next update (approximately 30 FPS)
            dpg.set_frame_callback(dpg.get_frame_count() + 1, self.update_gui)

    def toggle_audio_test(self, sender, app_data):
        if not self.is_testing:
            self.is_testing = True
            dpg.configure_item("test_audio_button", label="Stop Testing")
            self.start_audio_test()
        else:
            self.is_testing = False
            dpg.configure_item("test_audio_button", label="Test Audio")

    def toggle_recording(self, sender, app_data):
        if not self.is_recording:
            if not self.is_testing:
                dpg.set_value("status_text", "Start audio testing before recording.")
                return
            if not os.path.exists(self.recordings_dir):
                dpg.set_value("status_text", "No valid recordings directory.")
                return
            self.is_recording = True
            dpg.configure_item("record_button", label="Stop Recording")
            self.audio_buffer = []
        else:
            self.is_recording = False
            dpg.configure_item("record_button", label="Record")
            self.save_recording()

    def suggest_settings(self, sender, app_data):
        dpg.set_value("sample_rate_combo", "16000")
        dpg.set_value("bit_depth_combo", "int16")
        dpg.set_value("sensitivity_slider", 0.002)
        dpg.set_value("unit_combo", "Numbers")
        dpg.set_value("silence_input", 10.0)
        dpg.set_value("relative_sensitivity_check", False)
        self.update_pre_scale_label(None, 0.002)

    def save_recording(self):
        if not self.audio_buffer:
            dpg.set_value("status_text", "No audio recorded.")
            return
        audio_data = np.concatenate(self.audio_buffer)
        bit_depth = dpg.get_value("bit_depth_combo")
        if " - " in bit_depth:
            bit_depth = bit_depth.split(" - ")[0]
        dtype, max_value = self.get_dtype_and_max(bit_depth)
        sample_width = 1 if bit_depth == "int8" else 2 if bit_depth == "int16" else 3 if bit_depth == "int24" else 4
        filename = os.path.join(self.recordings_dir, f"recording_{int(time.time())}.wav")
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(sample_width)
            wf.setframerate(int(dpg.get_value("sample_rate_combo")))
            wf.writeframes(audio_data.tobytes())
        dpg.set_value("status_text", f"Recording saved as {filename}")

    def save_debug_recording(self):
        if not self.debug_audio_buffer:
            return
        audio_data = np.concatenate(self.debug_audio_buffer)
        bit_depth = dpg.get_value("bit_depth_combo")
        if " - " in bit_depth:
            bit_depth = bit_depth.split(" - ")[0]
        dtype, max_value = self.get_dtype_and_max(bit_depth)
        sample_width = 1 if bit_depth == "int8" else 2 if bit_depth == "int16" else 3 if bit_depth == "int24" else 4
        filename = os.path.join(self.default_recordings_dir, f"debug_recording_{int(time.time())}.wav")
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(sample_width)
            wf.setframerate(int(dpg.get_value("sample_rate_combo")))
            wf.writeframes(audio_data.tobytes())
        print(f"Debug recording saved as {filename}")

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

    def calibrate(self, sender, app_data):
        if self.is_testing:
            dpg.set_value("status_text", "Stop audio testing before calibrating.")
            return
        dtype, max_value = self.get_dtype_and_max()
        self.audio_buffer = []
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
            return
        
        if not self.audio_buffer:
            dpg.set_value("status_text", "No audio captured.")
            return
        
        audio_data = np.concatenate(self.audio_buffer).reshape(-1, 2)[:, 0].astype(np.float32) / (max_value + 1)
        peak_amplitude = np.max(np.abs(audio_data))
        target_amplitude = 0.5  # Target 50% of max amplitude
        pre_scale = target_amplitude / (peak_amplitude + 1e-10)
        self.set_slider_from_pre_scale(pre_scale)
        dpg.set_value("status_text", f"Sensitivity adjusted to {dpg.get_value('pre_scale_label')}.")

    def perform_calibration(self):
        device_index = self.get_device_index(dpg.get_value("input_device_combo"), dpg.get_value("host_api_combo"))
        if device_index is None:
            dpg.set_value("status_text", "Calibration failed: No valid input device.")
            return
        bit_depth = dpg.get_value("bit_depth_combo")
        if " - " in bit_depth:
            bit_depth = bit_depth.split(" - ")[0]
        sample_rate = int(dpg.get_value("sample_rate_combo"))
        amplitudes = []
        def callback(indata, frames, time_info, status):
            data = np.frombuffer(indata, dtype=bit_depth).astype(np.float32)
            if bit_depth == "int8":
                data = data * (32767 / 127)
            elif bit_depth == "int16":
                data = data
            elif bit_depth == "int24":
                data = data * (32767 / 8388607)
            elif bit_depth == "int32":
                data = data * (32767 / 2147483647)
            elif bit_depth == "float32":
                data = data * 32767
            amplitudes.extend(np.abs(data))
        try:
            with sd.RawInputStream(samplerate=sample_rate, device=device_index, dtype=bit_depth, channels=1, callback=callback):
                time.sleep(5)
            if amplitudes:
                avg_amplitude = np.mean(amplitudes)
                pre_scale = 32767 / (avg_amplitude * 2) if avg_amplitude > 0 else 0.002
                pre_scale = min(pre_scale, 10.0)
                self.set_slider_from_pre_scale(pre_scale)
                dpg.set_value("status_text", f"Calibration complete. Sensitivity set to {pre_scale:.3f}.")
            else:
                dpg.set_value("status_text", "Calibration failed: No audio detected.")
        except Exception as e:
            dpg.set_value("status_text", f"Calibration failed: {e}")



    def reset_settings(self, sender, app_data):
        default_settings = {
            "bit_depth": "int24", "sample_rate": "48000", "pre_scale_factor": 0.002,
            "unit": "Numbers", "relative_sensitivity": False, "silence_threshold": 10.0,
            "show_peaks": False, "theme": "Dark", "host_api": "MME", "input_device": None,
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
        dpg.set_value("theme_combo", default_settings["theme"])
        dpg.set_value("host_api_combo", default_settings["host_api"])
        dpg.set_value("input_device_combo", default_settings["input_device"])
        dpg.set_value("output_device_combo", default_settings.get("output_device", ""))
        self.update_host_api(None, default_settings["host_api"])
        self.apply_theme(default_settings["theme"])
        self.save_settings(None, None)
        dpg.set_value("status_text", "Settings reset to defaults and saved.")


    def start_audio_test(self):
        # If already testing, clean up current state without recursive stop
        if self.is_testing:
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
            # Clear queues
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
            # Reset GUI elements
            dpg.configure_item("level_bar", pmax=(0, 20))
            dpg.configure_item("clipping_indicator", fill=(255, 0, 0, 128))
            time.sleep(0.1)  # Small delay to release resources

        self.is_testing = True
        dtype, max_value = self.get_dtype_and_max()
        # Keep output_buffer initialization if needed
        self.output_buffer = []
        # Do NOT reinitialize queues since they're already in __init__
        # Do NOT reset noise_floor, peak_amplitude, last_noise_update unless required
    
        def input_callback(indata, frames, time_info, status):
            if not self.is_testing:
                return
            indata_array = np.frombuffer(indata, dtype=dtype).reshape(-1, 1)
            expected_frames = frames
            actual_frames = indata_array.shape[0]
            if actual_frames != expected_frames:
                if actual_frames < expected_frames:
                    indata_array = np.pad(indata_array, ((0, expected_frames - actual_frames), (0, 0)), mode='constant')
                else:
                    indata_array = indata_array[:expected_frames, :]
            indata_left = indata_array[:, 0].astype(np.float32) / (max_value + (1 if dtype != "float32" else 0))
            if dtype == "int8":
                indata_normalized = indata_left * (32767 / 127)
            elif dtype == "int16":
                indata_normalized = indata_left
            elif dtype == "int24" or dtype == "int32":
                if max_value == 8388607:
                    indata_normalized = indata_left * (32767 / 8388607)
                else:
                    indata_normalized = indata_left * (32767 / 2147483647)
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
            indata_scaled = indata_left * pre_scale
            indata_normalized = indata_scaled
            indata_left_scaled = np.clip(indata_normalized * 32767, -32768, 32767).astype(np.int16)
            scaled_data = (indata_scaled * 32767).astype(np.int16)
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
            monitoring_scale = max(pre_scale, 1.0)
            indata_monitoring = indata_left * monitoring_scale
            indata_monitoring_scaled = np.clip(indata_monitoring * 32767, -32768, 32767).astype(np.int16)
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
            output_device_info = sd.query_devices()[output_device_index]
            default_sample_rate = int(output_device_info["default_samplerate"])
            supported_sample_rates = []
            for sr in SAMPLE_RATES:
                try:
                    sd.check_input_settings(device=input_device_index, samplerate=int(sr), channels=1, dtype=dtype)
                    supported_sample_rates.append(sr)
                except:
                    continue
            if not supported_sample_rates:
                raise ValueError("No supported sample rates for this device.")
            selected_sample_rate = dpg.get_value("sample_rate_combo")
            if selected_sample_rate not in supported_sample_rates:
                selected_sample_rate = max(supported_sample_rates, key=int)
                dpg.set_value("sample_rate_combo", selected_sample_rate)
            sample_rate = int(selected_sample_rate)
            self.audio_stream = sd.RawInputStream(
                samplerate=sample_rate, blocksize=256,
                dtype=dtype, channels=1, callback=input_callback, device=input_device_index,
                latency='low'
            )
            self.audio_stream.start()
            self.output_stream = sd.RawOutputStream(
                samplerate=sample_rate, blocksize=256, dtype=dtype,
                channels=2, callback=output_callback, device=output_device_index,
                latency='low'
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
        # Stop and close the input stream
        if self.audio_stream:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except Exception as e:
                logging.error(f"Error stopping input stream: {e}")
            self.audio_stream = None
    
        # Stop and close the output stream
        if self.output_stream:
            try:
                self.output_stream.stop()
                self.output_stream.close()
            except Exception as e:
                logging.error(f"Error stopping output stream: {e}")
            self.output_stream = None
    
        # Clear audio and GUI update queues
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
    
        # Reset GUI elements
        dpg.configure_item("level_bar", pmax=(0, 20))
        dpg.configure_item("clipping_indicator", fill=(255, 0, 0, 128))
    


    def show_waveform(self, sender, app_data):
        # Check if the waveform window already exists
        if dpg.does_item_exist("waveform_window"):
            # Bring the existing window to the front
            dpg.focus_item("waveform_window")
            return
        bit_depth = dpg.get_value("bit_depth_combo")
        sample_rate = int(dpg.get_value("sample_rate_combo"))
        device_index = self.get_device_index(dpg.get_value("input_device_combo"), dpg.get_value("host_api_combo"))
        unit = dpg.get_value("unit_combo")
        slider_value = dpg.get_value("sensitivity_slider")
        if unit == "Numbers":
            pre_scale = self.slider_to_pre_scale(slider_value)
        elif unit == "Percent":
            pre_scale = self.percent_to_pre_scale(slider_value)
        elif unit == "dB":
            db = (slider_value / 100) * 100 - 60
            pre_scale = self.db_to_pre_scale(db)
        relative_sensitivity = dpg.get_value("relative_sensitivity_check")
        WaveformDisplay(bit_depth, 2, sample_rate, device_index, pre_scale, relative_sensitivity)


    def start_recording(self):
        if self.is_dictating:
            dpg.set_value("status_text", "Cannot record while dictating.")
            return
        if self.is_testing:
            self.stop_audio_test()
        self.audio_buffer = []
        device_index = self.get_device_index(dpg.get_value("input_device_combo"), dpg.get_value("host_api_combo"))
        if device_index is None:
            dpg.set_value("status_text", "Cannot start recording: No valid input device.")
            return
        bit_depth = dpg.get_value("bit_depth_combo")
        if " - " in bit_depth:
            bit_depth = bit_depth.split(" - ")[0]
        sample_rate = int(dpg.get_value("sample_rate_combo"))
        unit = dpg.get_value("unit_combo")
        slider_value = dpg.get_value("sensitivity_slider")
        if unit == "Numbers":
            pre_scale = self.slider_to_pre_scale(slider_value)
        elif unit == "Percent":
            pre_scale = self.percent_to_pre_scale(slider_value)
        elif unit == "dB":
            db = (slider_value / 100) * 100 - 60
            pre_scale = self.db_to_pre_scale(db)
        relative_sensitivity = dpg.get_value("relative_sensitivity_check")
        if relative_sensitivity:
            dtype, max_value = self.get_dtype_and_max(bit_depth)
            scale_factor = 32767 / (max_value + (1 if dtype != "float32" else 0))
            pre_scale /= scale_factor
        def callback(indata, frames, time_info, status):
            try:
                data = np.frombuffer(indata, dtype=bit_depth).reshape(-1, 1)
                data = data[:, 0].astype(np.float32)
                if bit_depth == "int8":
                    data_normalized = data * (32767 / 127)
                elif bit_depth == "int16":
                    data_normalized = data
                elif bit_depth == "int24":
                    data_normalized = data * (32767 / 8388607)
                elif bit_depth == "int32":
                    data_normalized = data * (32767 / 2147483647)
                elif bit_depth == "float32":
                    data_normalized = data * 32767
                data_scaled = data_normalized * pre_scale
                data_scaled = np.clip(data_scaled, -32768, 32767)
                self.audio_buffer.extend(data_scaled)
                self.update_audio_level(data_scaled)
            except Exception as e:
                logging.error(f"Error in recording callback: {e}")
        try:
            self.audio_stream = sd.RawInputStream(
                samplerate=sample_rate, blocksize=32000, device=device_index,
                dtype=bit_depth, channels=1, callback=callback
            )
            self.audio_stream.start()
            self.is_recording = True
            dpg.configure_item("record_button", label="Stop Recording")
            dpg.set_value("status_text", "Recording started.")
        except Exception as e:
            logging.error(f"Failed to start recording: {e}")
            dpg.set_value("status_text", f"Failed to start recording: {e}")

    def stop_recording(self):
        if self.audio_stream:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except Exception as e:
                logging.error(f"Error stopping recording stream: {e}")
            self.audio_stream = None
        self.is_recording = False
        dpg.configure_item("record_button", label="Record")
        if self.audio_buffer:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.recordings_dir, f"recording_{timestamp}.wav")
            try:
                import wave
                with wave.open(filename, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(int(dpg.get_value("sample_rate_combo")))
                    wf.writeframes(np.array(self.audio_buffer, dtype='int16').tobytes())
                dpg.set_value("status_text", f"Recording saved to {filename}")
            except Exception as e:
                logging.error(f"Failed to save recording: {e}")
                dpg.set_value("status_text", f"Failed to save recording: {e}")
        else:
            dpg.set_value("status_text", "Recording stopped. No audio to save.")
        self.audio_buffer = []
        self.update_audio_level(np.array([0]))

    def show_waveform(self, sender, app_data):
        # Check if the waveform window already exists
        if dpg.does_item_exist("waveform_window"):
            # Bring the existing window to the front
            dpg.focus_item("waveform_window")
            return
        bit_depth = dpg.get_value("bit_depth_combo")
        sample_rate = int(dpg.get_value("sample_rate_combo"))
        device_index = self.get_device_index(dpg.get_value("input_device_combo"), dpg.get_value("host_api_combo"))
        unit = dpg.get_value("unit_combo")
        slider_value = dpg.get_value("sensitivity_slider")
        if unit == "Numbers":
            pre_scale = self.slider_to_pre_scale(slider_value)
        elif unit == "Percent":
            pre_scale = self.percent_to_pre_scale(slider_value)
        elif unit == "dB":
            db = (slider_value / 100) * 100 - 60
            pre_scale = self.db_to_pre_scale(db)
        relative_sensitivity = dpg.get_value("relative_sensitivity_check")
        WaveformDisplay(bit_depth, 2, sample_rate, device_index, pre_scale, relative_sensitivity)

    def start_waveform(self):
        if self.bit_depth == 16:
            dtype = "int16"
            max_value = 32767
        elif self.bit_depth == 24 or self.bit_depth == 32:
            dtype = "int32"
            max_value = 8388607 if self.bit_depth == 24 else 2147483647
        
        def audio_callback(indata, frames, time, status):
            if not self.is_running:
                return
            
            # Convert to NumPy array
            indata_array = np.frombuffer(indata, dtype=dtype)
            
            # Normalize to [-1.0, 1.0]
            indata_normalized = indata_array.astype(np.float32) / (max_value + 1)
            
            # Average channels if stereo
            if self.channels > 1:
                indata_normalized = np.mean(indata_normalized.reshape(-1, self.channels), axis=1)
            
            # Downsample to fit the canvas width
            step = max(1, len(indata_normalized) // self.canvas_width)
            downsampled = indata_normalized[::step][:self.canvas_width]
            
            # Update waveform data
            self.waveform_data = downsampled
            
            # Update the canvas
            coords = []
            for i in range(len(self.waveform_data)):
                x = i
                y = 100 - (self.waveform_data[i] * 90)  # Scale to fit canvas height
                coords.extend([x, y])
            self.canvas.coords(self.waveform_line, *coords)
        
        try:
            self.audio_stream = sd.RawInputStream(
                samplerate=self.samplerate,
                blocksize=32000,
                dtype=dtype,
                channels=self.channels,
                callback=audio_callback,
                device=self.device
            )
            self.audio_stream.start()
        except Exception as e:
            print(f"Failed to start waveform: {e}")
            self.is_running = False
            self.toggle_button.configure(text="Start Waveform")
    
    def stop_waveform(self):
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None
        self.canvas.coords(self.waveform_line, 0, 100, 600, 100)

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

    def update_gui(self):
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
            dpg.render_dearpygui_frame()
        dpg.destroy_context()

    def setup_json_tab(self, tab_name, json_path):
        with dpg.group():
            # Table for JSON data
            with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp, tag=f"{tab_name}_table"):
                dpg.add_table_column(label="Key")
                dpg.add_table_column(label="Value")
                base_path = get_base_path()
                full_path = os.path.join(base_path, json_path)
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        self.json_data[tab_name] = json.load(f)
                except Exception:
                    self.json_data[tab_name] = {}
                
                for key, value in self.json_data[tab_name].items():
                    with dpg.table_row():
                        dpg.add_text(key, tag=f"{tab_name}_{key}_key")
                        dpg.add_text(value, tag=f"{tab_name}_{key}_value")
            
            # Buttons
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
        base_path = get_base_path()
        full_path = os.path.join(base_path, json_path)
        with open(json_path, "w", encoding="utf-8") as f:
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
            base_path = get_base_path()
            full_path = os.path.join(base_path, json_path)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self.json_data[tab_name], f, indent=4)
            dpg.delete_item(f"{tab_name}_table", children_only=True)
            dpg.add_table_column(label="Key", parent=f"{tab_name}_table")
            dpg.add_table_column(label="Value", parent=f"{tab_name}_table")
            for k, v in self.json_data[tab_name].items():
                with dpg.table_row(parent=f"{tab_name}_table"):
                    dpg.add_text(k, tag=f"{tab_name}_{k}_key")
                    dpg.add_text(v, tag=f"{tab_name}_{k}_value")
        dpg.delete_item(f"{tab_name}_add_window" if is_add else f"{tab_name}_edit_window")

    def start_dictation(self, sender, app_data):
        import logging
        logging.basicConfig(filename="dictation_gui.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info("Starting dictation process.")
        
        if hasattr(self, "dictation_process") and self.dictation_process is not None:
            if self.dictation_process.poll() is None:
                self.dictation_process.kill()
                self.dictation_process.wait(timeout=2)
            self.dictation_process = None

        if self.is_dictating:
            logging.warning("Dictation already running.")
            return
        if self.is_testing:
            self.stop_audio_test()
        if self.is_recording:
            self.toggle_recording()
        if self.is_output_enabled:
            self.toggle_output()

        # Gather settings
        bit_depth = dpg.get_value("bit_depth_combo")
        if " - " in bit_depth:
            bit_depth = bit_depth.split(" - ")[0]
        bit_depth_value = int(bit_depth.replace("int", ""))
        sample_rate = dpg.get_value("sample_rate_combo")
        device_index = self.get_device_index(dpg.get_value("input_device_combo"), dpg.get_value("host_api_combo"))
        if device_index is None:
            dpg.set_value("status_text", "No valid input device selected.")
            logging.error("No valid input device selected.")
            return

        unit = dpg.get_value("unit_combo")
        slider_value = dpg.get_value("sensitivity_slider")
        if unit == "Numbers":
            pre_scale = self.slider_to_pre_scale(slider_value)
        elif unit == "Percent":
            pre_scale = self.percent_to_pre_scale(slider_value)
        elif unit == "dB":
            db = (slider_value / 100) * 100 - 60
            pre_scale = self.db_to_pre_scale(db)

        relative_sensitivity = 1 if dpg.get_value("relative_sensitivity_check") else 0
        if relative_sensitivity:
            dtype, max_value = self.get_dtype_and_max()
            scale_factor = 32767 / (max_value + (1 if dtype != "float32" else 0))
            pre_scale /= scale_factor

        silence_threshold = dpg.get_value("silence_input")

        model_path = r"C:\Users\MenaBeshai\Downloads\Speech to Text\vosk-model-en-us-0.42-gigaspeech"
        if not os.path.exists(model_path):
            dpg.set_value("status_text", f"Vosk model not found at {model_path}.")
            logging.error(f"Vosk model not found at {model_path}.")
            return

        def launch_dictation_process():
            script_path = os.path.abspath("live_gigaspeech_dictation_v20.py")
            if not os.path.exists(script_path):
                dpg.set_value("status_text", f"Dictation script not found at {script_path}.")
                logging.error(f"Dictation script not found at {script_path}.")
                return None

            cmd = [
                sys.executable,
                script_path,
                "--model", model_path,
                "--bit-depth", str(bit_depth_value),
                "--sample-rate", str(sample_rate),
                "--pre-scale-factor", str(pre_scale),
                "--silence-threshold", str(silence_threshold),
                "--relative-sensitivity", str(relative_sensitivity)
            ]

            cmd_str = " ".join(cmd)
            dpg.set_value("status_text", f"Launching dictation with command: {cmd_str}")
            logging.info(f"Launching dictation with command: {cmd_str}")

            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,  # Fixed typo: 'ferr' to 'stderr'
                    bufsize=1,  # Line-buffered
                    universal_newlines=True,  # Text mode
                    cwd=os.path.dirname(script_path)
                )
                logging.info("Dictation process started.")
                return process
            except Exception as e:
                logging.error(f"Failed to launch dictation process: {e}")
                dpg.set_value("status_text", f"Failed to launch dictation: {e}")
                return None

        try:
            self.monitor_stop_event = threading.Event()
            self.dictation_process = launch_dictation_process()
            if self.dictation_process is None:
                self.is_dictating = False
                dpg.configure_item("start_dictation_button", enabled=True)
                dpg.configure_item("stop_dictation_button", enabled=False)
                return

            self.is_dictating = True
            dpg.configure_item("start_dictation_button", enabled=False)
            dpg.configure_item("stop_dictation_button", enabled=True)

            def monitor_process():
                last_status = "Dictation started."
                retry_count = 0
                max_retries = 3

                while not self.monitor_stop_event.is_set():
                    if not self.is_dictating:
                        break

                    # Read output and error streams
                    output = self.dictation_process.stdout.readline().strip()
                    if output:
                        last_status = f"Dictation output: {output}"
                        dpg.set_value("status_text", last_status)
                        logging.info(last_status)
                        retry_count = 0  # Reset retry count on successful output

                    error = self.dictation_process.stderr.readline().strip()
                    if error:
                        if "LOG (VoskAPI:" in error:
                            logging.debug(f"Vosk log: {error}")
                            continue
                        last_status = f"Dictation error: {error}"
                        dpg.set_value("status_text", last_status)
                        logging.error(last_status)

                    # Check if the process has exited
                    if self.dictation_process.poll() is not None:
                        return_code = self.dictation_process.poll()
                        if return_code == 0:
                            last_status = "Dictation process ended successfully."
                            logging.info(last_status)
                            dpg.set_value("status_text", last_status)
                            break
                        else:
                            last_status = f"Dictation process exited with return code: {return_code}"
                            logging.error(last_status)
                            dpg.set_value("status_text", last_status)

                            # Attempt to restart the process
                            if retry_count < max_retries:
                                retry_count += 1
                                logging.info(f"Attempting to restart dictation (retry {retry_count}/{max_retries})...")
                                dpg.set_value("status_text", f"Restarting dictation (attempt {retry_count}/{max_retries})...")
                                self.dictation_process = launch_dictation_process()
                                if self.dictation_process is None:
                                    break
                                time.sleep(1)  # Give the process time to start
                                continue
                            else:
                                last_status = f"Max retries reached. Stopping dictation."
                                logging.error(last_status)
                                dpg.set_value("status_text", last_status)
                                break

                    time.sleep(0.1)

                self.stop_dictation()

            monitor_thread = threading.Thread(target=monitor_process, daemon=True)
            monitor_thread.start()

        except Exception as e:
            dpg.set_value("status_text", f"Failed to start dictation: {e}")
            logging.error(f"Failed to start dictation: {e}")
            self.is_dictating = False
            dpg.configure_item("start_dictation_button", enabled=True)
            dpg.configure_item("stop_dictation_button", enabled=False)
            if hasattr(self, "dictation_process") and self.dictation_process is not None:
                self.dictation_process.kill()
                self.dictation_process = None
				
    def stop_dictation(self, sender=None, app_data=None):
            if self.is_dictating:
                self.is_dictating = False
                # Signal the monitor thread to stop (if using subprocess)
                if hasattr(self, "monitor_stop_event"):
                    self.monitor_stop_event.set()
                if hasattr(self, "dictation_process") and self.dictation_process is not None:
                    try:
                        self.dictation_process.terminate()
                        self.dictation_process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        self.dictation_process.kill()
                        self.dictation_process.wait(timeout=2)
                    except Exception as e:
                        dpg.set_value("status_text", f"Error terminating dictation process: {e}")
                        self.dictation_process.kill()
                        self.dictation_process.wait(timeout=2)
                    finally:
                        self.dictation_process = None
                dpg.configure_item("start_dictation_button", enabled=True)
                dpg.configure_item("stop_dictation_button", enabled=False)
                dpg.set_value("status_text", "Dictation stopped.")
                # Save debug recording if enabled
                if self.is_debug_recording:
                    self.save_debug_recording()




class CustomDictationGUI(DictationGUI):
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





def main():
    gui = CustomDictationGUI()
    try:
        gui.update_gui()
    except KeyboardInterrupt:
        logging.info("Application terminated by user.")
        gui.stop_dictation()
        if gui.is_recording:
            gui.stop_recording()
        if gui.is_testing:
            gui.stop_audio_test()
        sys.exit(0)
    except Exception as e:
        logging.error(f"Unexpected error in GUI loop: {e}")
        gui.stop_dictation()
        if gui.is_recording:
            gui.stop_recording()
        if gui.is_testing:
            gui.stop_audio_test()
        sys.exit(1)

if __name__ == "__main__":
    main()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    number_lock_on = False
    last_dictated_length = 0
    gui = DictationGUI()
    gui.run()
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s", filename="dictation_script.log")
    if len(sys.argv) > 1 and sys.argv[1] == "waveform":
        bit_depth = sys.argv[2]
        channels = int(sys.argv[3])
        samplerate = int(sys.argv[4])
        device = int(sys.argv[5])
        pre_scale_factor = float(sys.argv[6])
        relative_sensitivity = sys.argv[7] == "True"
        WaveformDisplay(bit_depth, channels, samplerate, device, pre_scale_factor, relative_sensitivity)
    else:
        DictationGUI()
try:
    main()
except Exception as e:
    logging.error(f"Application crashed: {e}", exc_info=True)
    print(f"Error: {e}. Check dictation_script.log for details.")
    input("Press Enter to exit...")