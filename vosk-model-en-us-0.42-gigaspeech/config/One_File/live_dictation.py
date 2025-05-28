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


import dearpygui.dearpygui as dpg # For GUI handling
import debug_utils as debug # For Debugging
import json # For JSON handling
import os # For file handling
import subprocess # For subprocess handling
import sys # For subprocess handling
import time # For time handling
import math # For audio processing
import numpy as np # For audio processing
import sounddevice as sd # For audio input/output
import threading # For threading
import queue # For command queue
import logging # For logging
import keyboard # For keyboard input
import re # For regex
from spellchecker import SpellChecker # For spell checking
from collections import deque # For audio buffer
from vosk import Model, KaldiRecognizer # For speech recognition

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s", filename="dictation_gui.log")

# Placeholder debug_utils class (replace with actual module if available)
class DebugUtils:
    @staticmethod
    def log_debug(message):
        logging.debug(message)
    @staticmethod
    def log_info(message):
        logging.info(message)
    @staticmethod
    def log_error(message):
        logging.error(message)

debug_utils = DebugUtils()

# Helper function to get the base path for bundled files
def get_base_path():
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

# Paths to JSON files
CONFIG_DIR = "config"
COMMANDS_JSON_PATH = os.path.join(CONFIG_DIR, "commands.json")
FRACTIONS_MAP_PATH = os.path.join(CONFIG_DIR, "fractions_map.json")
F_KEYS_MAP_PATH = os.path.join(CONFIG_DIR, "f_keys_map.json")
FUNCTIONS_MAP_PATH = os.path.join(CONFIG_DIR, "functions_map.json")
SYMBOLS_MAP_PATH = os.path.join(CONFIG_DIR, "symbols_map.json")
GOOGLE_NUMBERS_PATH = os.path.join(CONFIG_DIR, "google_numbers.json")
LARGE_NUMBERS_MAP_PATH = os.path.join(CONFIG_DIR, "large_numbers_map.json")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")

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

# Number parsing functions
def load_json_map(file_path):
    try:
        base_path = get_base_path()
        full_path = os.path.join(base_path, file_path)
        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            debug_utils.log_info(f"Loaded JSON map from {full_path}")
            return data
    except FileNotFoundError:
        debug_utils.log_error(f"{file_path} not found!")
        raise FileNotFoundError(f"{file_path} not found!")
    except json.JSONDecodeError:
        debug_utils.log_error(f"Invalid JSON format in {file_path}!")
        raise ValueError(f"Invalid JSON format in {file_path}!")
    except UnicodeDecodeError as e:
        debug_utils.log_error(f"Encoding error in {file_path}! {e}")
        raise ValueError(f"Encoding error in {file_path}! {e}")

SMALL_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000
}

def words_to_number_less_than_thousand(phrase):
    if not phrase:
        debug_utils.log_debug("Empty phrase provided to words_to_number_less_than_thousand")
        return 0

    words = phrase.replace("-", " ").split()
    if not words:
        debug_utils.log_debug("No words after splitting phrase")
        return 0

    total = 0
    current = 0

    for word in words:
        if word in SMALL_NUMBERS:
            value = SMALL_NUMBERS[word]
            if value == 100:
                current *= 100
            elif value >= 1000:
                current *= value
                total += current
                current = 0
            else:
                current += value
        else:
            debug_utils.log_warning(f"Unrecognized word in number phrase: {word}")
            return None

    result = total + current
    debug_utils.log_debug(f"Parsed '{phrase}' to {result}")
    return result

def convert_numbers(phrase, fraction_map=None, symbs_map=None, google_numbers=None, large_numbers_map=None):
    if large_numbers_map is None:
        try:
            large_numbers_map = load_json_map(LARGE_NUMBERS_MAP_PATH)
        except Exception as e:
            debug_utils.log_error(f"Failed to load large_numbers_map: {e}")
            return None

    PRACTICAL_LIMIT = 10**303
    
    if not isinstance(phrase, str):
        debug_utils.log_warning(f"Invalid input type for convert_numbers: {type(phrase)}")
        return None
    
    phrase = phrase.lower().strip()
    if not phrase:
        debug_utils.log_debug("Empty phrase provided to convert_numbers")
        return None

    phrase = re.sub(r'\s+', ' ', phrase)
    phrase = phrase.replace(" and ", " ")
    phrase = phrase.replace(",", "")

    is_negative = False
    if phrase.startswith("negative"):
        is_negative = True
        phrase = phrase[len("negative"):].strip()
        if not phrase:
            debug_utils.log_debug("No number after 'negative'")
            return None

    large_scales = sorted(
        large_numbers_map.items(),
        key=lambda x: int(x[1]),
        reverse=True
    )

    large_scale_words = "|".join(re.escape(scale) for scale, _ in large_scales)
    pattern = f"\\b({large_scale_words})\\b"
    sections = re.split(pattern, phrase)

    total = 0
    current_section_value = 0
    current_scale = 1

    for section in sections:
        section = section.strip()
        if not section:
            continue

        if section in large_numbers_map:
            scale_value = int(large_numbers_map[section])
            if current_section_value == 0:
                current_section_value = 1
            total += current_section_value * scale_value
            current_section_value = 0
            current_scale = 1
        else:
            section_value = words_to_number_less_than_thousand(section)
            if section_value is None:
                debug_utils.log_warning(f"Failed to parse number section: {section}")
                return None
            current_section_value += section_value

    total += current_section_value * current_scale

    if total > PRACTICAL_LIMIT:
        debug_utils.log_warning(f"Number exceeds practical limit: {total}")
        return None

    result = -total if is_negative else total
    debug_utils.log_info(f"Converted '{phrase}' to {result}")
    return result

def parse_number_sequence(words, fraction_map=None, symbs_map=None, google_numbers=None, large_numbers_map=None):
    if not words:
        debug_utils.log_debug("Empty word list provided to parse_number_sequence")
        return None, 0

    phrase = " ".join(words)
    number = convert_numbers(phrase, fraction_map, symbs_map, google_numbers, large_numbers_map)
    if number is not None:
        debug_utils.log_debug(f"Parsed full sequence '{phrase}' to {number}")
        return str(number), len(words)

    for i in range(len(words), 0, -1):
        sub_phrase = " ".join(words[:i])
        number = convert_numbers(sub_phrase, fraction_map, symbs_map, google_numbers, large_numbers_map)
        if number is not None:
            debug_utils.log_debug(f"Parsed subsequence '{sub_phrase}' to {number}")
            return str(number), i
    debug_utils.log_debug("No valid number sequence found")
    return None, 0

class WaveformDisplay:
    def __init__(self, bit_depth, channels, samplerate, device, pre_scale_factor, relative_sensitivity):
        self.bit_depth = bit_depth
        self.channels = channels
        self.samplerate = samplerate
        self.device = device
        self.pre_scale_factor = pre_scale_factor
        self.relative_sensitivity = relative_sensitivity
        self.is_running = False
        self.audio_buffer = deque(maxlen=int(self.samplerate * 1.0))
        self.update_interval = 1.0 / 30
        self.last_update = 0
        self.dtype, self.max_value = self.get_dtype_and_max()
        
        try:
            with dpg.window(label="Waveform Display", tag="waveform_window", width=800, height=300, on_close=self.close):
                with dpg.plot(label="Waveform (16-bit PCM, as Vosk hears)", height=200, width=-1):
                    dpg.add_plot_axis(dpg.Axis.X, label="Time (s)", tag="waveform_x_axis")
                    dpg.add_plot_axis(dpg.Axis.Y, label="Amplitude", tag="waveform_y_axis")
                    if dpg.does_item_exist("waveform_y_axis"):
                        dpg.set_axis_limits(item="waveform_y_axis", ymin=-32768, ymax=32768)
                    else:
                        debug_utils.log_error("GUI ERROR: Waveform y-axis not found.")
                    self.time_axis = np.linspace(0, 1, int(self.samplerate * 1.0))
                    self.audio_data = np.zeros(int(self.samplerate * 1.0))
                    dpg.add_line_series(self.time_axis, self.audio_data, label="Waveform", parent="waveform_y_axis", tag="waveform_series")
        
            self.start_stream()
            dpg.set_frame_callback(dpg.get_frame_count() + 1, self.update_waveform)
        except Exception as e:
            debug_utils.log_error(f"Failed to initialize waveform display: {e}")
            raise

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
            debug_utils.log_error(f"Invalid bit depth: {bit_depth}")
            raise ValueError(f"Invalid bit depth: {bit_depth}")
    
    def start_stream(self):
        self.is_running = True
        def audio_callback(indata, frames, time_info, status):
            if not self.is_running:
                return
            try:
                data = np.frombuffer(indata, dtype=self.dtype).reshape(-1, self.channels)
                data = data[:, 0].astype(np.float32)
                if self.dtype == "int8":
                    data_normalized = data * (32767 / 127)
                elif self.dtype == "int16":
                    data_normalized = data
                elif self.dtype == "int24" or self.dtype == "int32":
                    data_normalized = data * (32767 / self.max_value)
                elif self.dtype == "float32":
                    data_normalized = data * 32767
                else:
                    raise ValueError(f"Unsupported dtype: {self.dtype}")
                pre_scale = self.pre_scale_factor
                data_scaled = data_normalized * pre_scale
                data_scaled = np.clip(data_scaled, -32768, 32767)
                self.audio_buffer.extend(data_scaled)
            except Exception as e:
                debug_utils.log_error(f"Error in waveform audio callback: {e}")
        try:
            self.stream = sd.RawInputStream(
                samplerate=self.samplerate, blocksize=32000, device=self.device,
                dtype=self.dtype, channels=self.channels, callback=audio_callback
            )
            self.stream.start()
            debug_utils.log_info("Waveform stream started successfully.")
        except Exception as e:
            debug_utils.log_error(f"Failed to start waveform stream: {e}")
            self.is_running = False
            raise
    
    def update_waveform(self):
        if not self.is_running or not dpg.does_item_exist("waveform_window"):
            return
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
                debug_utils.log_error(f"Error updating waveform: {e}")
        dpg.set_frame_callback(dpg.get_frame_count() + 1, self.update_waveform)

    def stop_stream(self):
        self.is_running = False
        if hasattr(self, "stream"):
            try:
                self.stream.stop()
                self.stream.close()
                debug_utils.log_info("Waveform stream stopped successfully.")
            except Exception as e:
                debug_utils.log_error(f"Error stopping waveform stream: {e}")

    def close(self, sender, app_data):
        self.is_running = False
        self.stop_stream()
        dpg.delete_item("waveform_window")
        debug_utils.log_info("Waveform window closed.")

class DictationGUI:
    def __init__(self):
        # Audio variables
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
        self.command_queue = queue.Queue()
        self.transcribed_text = []
        self.last_command = ""
        self.last_command_time = 0
        self.COMMAND_DEBOUNCE_TIME = 0.5
        self.spell_checker = SpellChecker()
        self.caps_lock_on = False
        self.number_lock_on = False
        
        # Number parsing resources
        self.fraction_map = None
        self.symbols_map = None
        self.google_numbers = None
        self.large_numbers_map = None
        self.load_number_maps()
        
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
            os.makedirs(self.default_recordings_dir, exist_ok=True)
            debug_utils.log_info(f"Recordings directory ensured at {self.default_recordings_dir}")
        except Exception as e:
            debug_utils.log_error(f"Failed to create recordings directory: {e}")
            self.default_recordings_dir = os.getcwd()
            self.recordings_dir = self.default_recordings_dir
        
        # Initialize saved settings with defaults
        self.saved_settings = {}
        
        # JSON data
        self.json_data = {}
        self.commands = {}
        self.load_commands()
        
        # DPG setup
        try:
            dpg.create_context()
            dpg.create_viewport(title="Speech-to-Text Dictation Configuration", width=800, height=600)
        except Exception as e:
            debug_utils.log_message("debug_status_updates", f"Failed to initialize DearPyGui: {e}", self)
            raise

        # Theme setup
        try:
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
            debug_utils.log_message("debug_status_updates", "Themes initialized successfully", self)
        except Exception as e:
            debug_utils.log_message("debug_status_updates", f"Failed to setup themes: {e}", self)
            raise

        # Main window setup
        try:
            with dpg.window(label="Speech-to-Text Dictation Configuration", tag="primary_window"):
                # Create status_text first to ensure it's available
                dpg.add_text("Status: Ready", tag="status_text")
                
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
            
                # Status bar for CLI-like feedback
                dpg.add_text("Status: Ready", tag="status_text")
        
            self.apply_theme("Dark")
            self.load_settings()
            dpg.setup_dearpygui()
            dpg.show_viewport()
            dpg.set_primary_window("primary_window", True)
            self.create_file_dialog()
            self.update_gui()
            debug_utils.log_message("debug_status_updates", "GUI initialized successfully", self)
        except Exception as e:
            debug_utils.log_message("debug_status_updates", f"Failed to initialize GUI: {e}", self)
            raise

    def setup_audio_tab(self):
        try:
            with dpg.group(horizontal=True):
                dpg.add_text("Host API:")
                dpg.add_combo(self.get_host_apis(), default_value="MME", callback=self.update_host_api, tag="host_api_combo")
        except Exception as e:
                dpg.add_combo(self.get_host_apis(), default_value="WDM", callback=self.update_host_api, tag="host_api_combo")
                raise

    def load_number_maps(self):
        try:
            self.fraction_map = load_json_map(FRACTIONS_MAP_PATH)
            debug_utils.log_info("Loaded fractions map")
        except Exception as e:
            debug_utils.log_error(f"Failed to load fractions map: {e}")
            self.fraction_map = {}
        
        try:
            self.symbols_map = load_json_map(SYMBOLS_MAP_PATH)
            debug_utils.log_info("Loaded symbols map")
        except Exception as e:
            debug_utils.log_error(f"Failed to load symbols map: {e}")
            self.symbols_map = {}
        
        try:
            self.google_numbers = load_json_map(GOOGLE_NUMBERS_PATH)
            debug_utils.log_info("Loaded google numbers map")
        except Exception as e:
            debug_utils.log_error(f"Failed to load google numbers map: {e}")
            self.google_numbers = {}
        
        try:
            self.large_numbers_map = load_json_map(LARGE_NUMBERS_MAP_PATH)
            debug_utils.log_info("Loaded large numbers map")
        except Exception as e:
            debug_utils.log_error(f"Failed to load large numbers map: {e}")
            self.large_numbers_map = {}

    def load_commands(self):
        try:
            base_path = get_base_path()
            full_path = os.path.join(base_path, COMMANDS_JSON_PATH)
            if not os.path.exists(full_path):
                debug_utils.log_warning(f"commands.json not found at {full_path}. Using default commands.")
                self.commands = {"simple_commands": {}, "parameterized_commands": []}
                return
            with open(full_path, "r", encoding="utf-8") as f:
                self.commands = json.load(f)
            debug_utils.log_info("Commands loaded successfully.")
        except Exception as e:
            debug_utils.log_error(f"Failed to load commands.json: {e}")
            self.commands = {"simple_commands": {}, "parameterized_commands": []}

    def type_text(self, text):
        try:
            if text == "\n\n":
                keyboard.press_and_release("enter")
                keyboard.press_and_release("enter")
            else:
                keyboard.write(text)
            debug_utils.log_debug(f"GUI typed: {text}")
            dpg.set_value("status_text", f"Status: Typed '{text}'")
        except Exception as e:
            debug_utils.log_error(f"GUI error typing text: {e}")
            dpg.set_value("status_text", f"Status: Error typing text: {e}")

    def update_command_progress(self, value, status):
        self.command_progress = value
        self.command_status = status
        try:
            dpg.set_value("command_progress_bar", value / self.command_progress_max)
            dpg.set_value("command_status_text", status)
            dpg.set_value("status_text", f"Status: {status}")
        except Exception as e:
            debug_utils.log_error(f"Error updating command progress: {e}")

    def process_transcription(self, text):
        """Process transcribed text, handling numbers and commands."""
        words = text.lower().split()
        processed_text = []
        i = 0
        while i < len(words):
            # Check for number sequences
            number, words_used = parse_number_sequence(
                words[i:],
                self.fraction_map,
                self.symbols_map,
                self.google_numbers,
                self.large_numbers_map
            )
            if number is not None and words_used > 0:
                if self.number_lock_on:
                    processed_text.append(number)
                else:
                    processed_text.append(" ".join(words[i:i+words_used]))
                i += words_used
            else:
                # Check for commands
                word = words[i]
                if word in self.commands.get("simple_commands", {}):
                    self.send_command(self.commands["simple_commands"][word])
                    i += 1
                elif any(word.startswith(cmd) for cmd in self.commands.get("parameterized_commands", [])):
                    # Handle parameterized commands (simplified)
                    self.send_command(word)
                    i += 1
                else:
                    processed_text.append(word)
                    i += 1
        
        final_text = " ".join(processed_text)
        if final_text:
            self.transcribed_text.append(final_text)
            full_text = "".join(self.transcribed_text).rstrip()
            dpg.set_value("output_text", full_text)
            with open("dictation_output.txt", "a", encoding="utf-8") as f:
                f.write(final_text + " ")
            self.type_text(final_text)
        return final_text

    def dictation_loop(self):
        """Main dictation loop for processing audio and transcription."""
        try:
            model_path = dpg.get_value("model_path_input")
            if not model_path or model_path == "Not set" or not os.path.exists(model_path):
                dpg.set_value("status_text", "Status: Invalid or missing model path")
                debug_utils.log_error("Invalid or missing model path")
                return

            self.recognizer = KaldiRecognizer(Model(model_path), int(dpg.get_value("sample_rate_combo")))
            debug_utils.log_info("Vosk recognizer initialized")

            def audio_callback(indata, frames, time_info, status):
                try:
                    if self.is_dictating:
                        data = np.frombuffer(indata, dtype=self.dtype).reshape(-1, self.channels)
                        data = data[:, 0].astype(np.int16)
                        if self.recognizer.AcceptWaveform(data.tobytes()):
                            result = json.loads(self.recognizer.Result())
                            text = result.get("text", "")
                            if text:
                                debug_utils.log_debug(f"Transcribed: {text}")
                                self.process_transcription(text)
                        if self.is_debug_recording:
                            self.debug_audio_buffer.extend(data)
                except Exception as e:
                    debug_utils.log_error(f"Error in dictation audio callback: {e}")

            bit_depth = dpg.get_value("bit_depth_combo")
            self.dtype, _ = self.get_dtype_and_max(bit_depth)
            self.channels = 1
            self.samplerate = int(dpg.get_value("sample_rate_combo"))
            device = self.get_device_index(
                dpg.get_value("input_device_combo"),
                dpg.get_value("host_api_combo"),
                is_input=True
            )

            self.dictation_stream = sd.RawInputStream(
                samplerate=self.samplerate,
                blocksize=8000,
                device=device,
                dtype=self.dtype,
                channels=self.channels,
                callback=audio_callback
            )
            self.dictation_stream.start()
            debug_utils.log_info("Dictation stream started")
            dpg.set_value("status_text", "Status: Dictation running")

            while self.is_dictating:
                time.sleep(0.1)
                while not self.command_queue.empty():
                    command = self.command_queue.get()
                    self.handle_command(command)

        except Exception as e:
            debug_utils.log_error(f"Error in dictation loop: {e}")
            dpg.set_value("status_text", f"Status: Dictation error: {e}")
        finally:
            self.stop_dictation()

    def start_dictation(self, sender=None, app_data=None):
        if self.is_dictating:
            return
        self.is_dictating = True
        dpg.configure_item("start_dictation_button", enabled=False)
        dpg.configure_item("stop_dictation_button", enabled=True)
        dpg.set_value("status_text", "Status: Starting dictation...")
        debug_utils.log_info("Starting dictation")
        threading.Thread(target=self.dictation_loop, daemon=True).start()

    def stop_dictation(self, sender=None, app_data=None):
        self.is_dictating = False
        if self.dictation_stream:
            try:
                self.dictation_stream.stop()
                self.dictation_stream.close()
                debug_utils.log_info("Dictation stream stopped")
            except Exception as e:
                debug_utils.log_error(f"Error stopping dictation stream: {e}")
        self.dictation_stream = None
        self.recognizer = None
        dpg.configure_item("start_dictation_button", enabled=True)
        dpg.configure_item("stop_dictation_button", enabled=False)
        dpg.set_value("status_text", "Status: Dictation stopped")

    def handle_command(self, action):
        debug_utils.log_info(f"GUI handling command action: {action}")
        self.update_command_progress(10.0, f"Executing command: {action}")

        try:
            if action == "cmd_select_all":
                keyboard.press_and_release("ctrl+a")
                self.update_command_progress(100.0, "Selected all")
            elif action == "cmd_select_down":
                keyboard.press("shift")
                keyboard.press_and_release("down")
                keyboard.release("shift")
                self.update_command_progress(100.0, "Selected down")
            elif action == "cmd_select_up":
                keyboard.press("shift")
                keyboard.press_and_release("up")
                keyboard.release("shift")
                self.update_command_progress(100.0, "Selected up")
            elif action == "cmd_copy":
                keyboard.press_and_release("ctrl+c")
                self.update_command_progress(100.0, "Copied selection")
            elif action == "cmd_paste":
                keyboard.press_and_release("ctrl+v")
                self.update_command_progress(100.0, "Pasted content")
            elif action == "cmd_delete":
                keyboard.press_and_release("backspace")
                self.update_command_progress(100.0, "Deleted character")
            elif action == "cmd_undo":
                keyboard.press_and_release("ctrl+z")
                self.update_command_progress(100.0, "Undo performed")
            elif action == "cmd_redo":
                keyboard.press_and_release("ctrl+y")
                self.update_command_progress(100.0, "Redo performed")
            elif action == "cmd_save_document":
                keyboard.press_and_release("ctrl+s")
                self.update_command_progress(100.0, "Saved document")
            elif action == "cmd_move_up":
                keyboard.press_and_release("up")
                self.update_command_progress(100.0, "Moved cursor up")
            elif action == "cmd_move_down":
                keyboard.press_and_release("down")
                self.update_command_progress(100.0, "Moved cursor down")
            elif action == "cmd_move_left":
                keyboard.press_and_release("left")
                self.update_command_progress(100.0, "Moved cursor left")
            elif action == "cmd_move_right":
                keyboard.press_and_release("right")
                self.update_command_progress(100.0, "Moved cursor right")
            elif action == "cmd_enter":
                keyboard.press_and_release("enter")
                self.transcribed_text.append("\n")
                full_text = "".join(self.transcribed_text).rstrip()
                dpg.set_value("output_text", full_text)
                with open("dictation_output.txt", "a", encoding="utf-8") as f:
                    f.write("\n")
                self.update_command_progress(100.0, "Pressed Enter")
            elif action == "cmd_number_lock":
                self.number_lock_on = not self.number_lock_on
                self.update_command_progress(100.0, f"Number lock {'on' if self.number_lock_on else 'off'}")
            elif action == "cmd_caps_lock_on":
                if not self.caps_lock_on:
                    keyboard.press_and_release("caps lock")
                    self.caps_lock_on = True
                self.update_command_progress(100.0, "Caps lock on")
            elif action == "cmd_caps_lock_off":
                if self.caps_lock_on:
                    keyboard.press_and_release("caps lock")
                    self.caps_lock_on = False
                self.update_command_progress(100.0, "Caps lock off")
            elif action == "cmd_bold":
                keyboard.press_and_release("ctrl+b")
                self.update_command_progress(100.0, "Applied bold")
            elif action == "cmd_italicize":
                keyboard.press_and_release("ctrl+i")
                self.update_command_progress(100.0, "Applied italic")
            elif action == "cmd_underline":
                keyboard.press_and_release("ctrl+u")
                self.update_command_progress(100.0, "Applied underline")
            elif action == "cmd_scratch_that":
                for _ in range(len(self.transcribed_text[-1]) if self.transcribed_text else 0):
                    keyboard.press_and_release("backspace")
                if self.transcribed_text:
                    self.transcribed_text.pop()
                full_text = "".join(self.transcribed_text).rstrip()
                dpg.set_value("output_text", full_text)
                with open("dictation_output.txt", "w", encoding="utf-8") as f:
                    f.write(full_text)
                self.update_command_progress(100.0, "Last dictation removed")
            elif action == "cmd_punctuation_period":
                self.type_text(".")
                self.update_command_progress(100.0, "Typed period")
            elif action == "cmd_punctuation_comma":
                self.type_text(",")
                self.update_command_progress(100.0, "Typed comma")
            else:
                debug_utils.log_warning(f"Unknown command action: {action}")
                self.update_command_progress(100.0, f"Unknown command: {action}")
        except Exception as e:
            debug_utils.log_error(f"Error handling command {action}: {e}")
            self.update_command_progress(100.0, f"Error handling command: {e}")

    def send_command(self, action):
        self.command_queue.put(action)
        debug_utils.log_debug(f"GUI sent command: {action}")
        self.update_command_progress(0.0, f"Queued command: {action}")

    def apply_theme(self, theme, app_data=None):
        selected_theme = app_data if app_data is not None else theme
        self.theme = selected_theme
        try:
            if selected_theme == "Dark":
                dpg.bind_theme("dark_theme")
            else:
                dpg.bind_theme("light_theme")
            debug_utils.log_info(f"Applied {selected_theme} theme")
            dpg.set_value("status_text", f"Status: Applied {selected_theme} theme")
        except Exception as e:
            debug_utils.log_error(f"Failed to apply theme: {e}")

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
        
        try:
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
            debug_utils.log_info("Settings loaded successfully")
            dpg.set_value("status_text", "Status: Settings loaded")
        except Exception as e:
            debug_utils.log_error(f"Failed to apply settings to GUI: {e}")

    def save_settings(self, sender, app_data):
        base_path = get_base_path()
        config_path = os.path.join(base_path, CONFIG_PATH)
        existing_config = {}
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                existing_config = json.load(f)
        except Exception:
            pass
        
        try:
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
            
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self.saved_settings, f, indent=4)
            dpg.set_value("status_text", "Status: Settings saved successfully")
            debug_utils.log_info("Settings saved successfully")
        except Exception as e:
            dpg.set_value("status_text", f"Status: Failed to save settings: {e}")
            debug_utils.log_error(f"Failed to save settings: {e}")

    def setup_audio_tab(self):
        try:
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
            
            debug_utils.log_message("debug_status_updates", "Audio tab configured", self)
        except Exception as e:
            debug_utils.log_message("debug_status_updates", f"Failed to setup audio tab: {e}", self)
            raise

    def create_file_dialog(self):
        try:
            with dpg.file_dialog(
                directory_selector=True,
                show=False,
                callback=self.on_model_path_selected,
                tag="file_dialog",
                width=700,
                height=400
            ):
                dpg.add_file_extension(".*")
            debug_utils.log_info("File dialog created")
        except Exception as e:
            debug_utils.log_error(f"Failed to create file dialog: {e}")

    def set_model_path(self, sender, app_data):
        if not dpg.does_item_exist("file_dialog"):
            self.create_file_dialog()
        try:
            dpg.show_item("file_dialog")
            debug_utils.log_info("File dialog shown for model path selection")
        except Exception as e:
            debug_utils.log_error(f"Failed to show file dialog: {e}")

    def on_model_path_selected(self, sender, app_data):
        selected_path = app_data.get("file_path_name", "")
        if not selected_path:
            dpg.set_value("status_text", "Status: No directory selected")
            debug_utils.log_warning("No directory selected for model path")
            return
        absolute_path = os.path.abspath(selected_path)
        try:
            dpg.set_value("model_path_input", absolute_path)
            base_path = get_base_path()
            config_path = os.path.join(base_path, CONFIG_PATH)
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except Exception:
                config = {}
            config["model_path"] = absolute_path
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
            dpg.set_value("status_text", f"Status: Model path set to {absolute_path}")
            debug_utils.log_info(f"Model path set to {absolute_path}")
        except Exception as e:
            dpg.set_value("status_text", f"Status: Failed to save model path: {e}")
            debug_utils.log_error(f"Failed to save model path: {e}")

    def open_manual_sensitivity_input(self, sender, app_data):
        unit = dpg.get_value("unit_combo")
        slider_value = dpg.get_value("sensitivity_slider")
        if unit == "Numbers":
            default_value = self.slider_to_pre_scale(slider_value)
        elif unit == "Percent":
            default_value = self.pre_scale_to_percent(self.slider_to_pre_scale(slider_value))
        elif unit == "dB":
            default_value = self.pre_scale_to_db(self.slider_to_pre_scale(slider_value))
        try:
            with dpg.window(label=f"Set Sensitivity ({unit})", modal=True, width=300, height=150, tag="manual_sensitivity_window"):
                dpg.add_text(f"Enter sensitivity value in {unit}:")
                dpg.add_input_float(default_value=default_value, tag="manual_sensitivity_input", step=0.0001, format="%.4f")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Save", callback=self.save_manual_sensitivity)
                    dpg.add_button(label="Cancel", callback=lambda: dpg.delete_item("manual_sensitivity_window"))
            debug_utils.log_info(f"Opened manual sensitivity input window for unit: {unit}")
        except Exception as e:
            debug_utils.log_error(f"Failed to open manual sensitivity input: {e}")

    def save_manual_sensitivity(self, sender, app_data):
        try:
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
            debug_utils.log_info(f"Saved manual sensitivity: {input_value} ({unit})")
            dpg.set_value("status_text", f"Status: Sensitivity set to {input_value} ({unit})")
        except Exception as e:
            debug_utils.log_error(f"Failed to save manual sensitivity: {e}")

    def get_host_apis(self):
        try:
            return [host_api['name'] for host_api in sd.query_hostapis()]
        except Exception as e:
            debug_utils.log_error(f"Failed to query host APIs: {e}")
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

    def get_dtype_and_max(self, bit_depth=None):
        if bit_depth is None:
            bit_depth = dpg.get_value("bit_depth_combo")
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
            debug_utils.log_error(f"Invalid bit depth: {bit_depth}")
            raise ValueError(f"Invalid bit depth: {bit_depth}")

    def update_host_api(self, sender, app_data):
        try:
            host_api = app_data
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
            debug_utils.log_info(f"Host API updated to {host_api}")
            dpg.set_value("status_text", f"Status: Host API set to {host_api}")
        except Exception as e:
            debug_utils.log_error(f"Failed to update host API: {e}")

    def update_device(self, sender, app_data):
        try:
            device_name = app_data
            host_api_name = dpg.get_value("host_api_combo")
            device_index = self.get_device_index(device_name, host_api_name)
            if device_index is None:
                debug_utils.log_warning(f"No device index found for {device_name}")
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
            dpg.configure_item("sample_rate_combo", items=SAMPLE_RATES)
            if dpg.get_value("sample_rate_combo") not in supported_sample_rates:
                dpg.set_value("sample_rate_combo", max(supported_sample_rates, key=int))
            if self.is_testing:
                self.stop_audio_test()
                self.start_audio_test()
            debug_utils.log_info(f"Input device updated to {device_name}")
            dpg.set_value("status_text", f"Status: Input device set to {device_name}")
        except Exception as e:
            debug_utils.log_error(f"Failed to update input device: {e}")

    def update_output_device(self, sender, app_data):
        try:
            debug_utils.log_info(f"Output device updated to {app_data}")
            dpg.set_value("status_text", f"Status: Output device set to {app_data}")
        except Exception as e:
            debug_utils.log_error(f"Failed to update output device: {e}")

    def update_bit_depth(self, sender, app_data):
        try:
            dpg.set_value("bit_depth_tooltip", DATA_TYPES[app_data])
            if self.is_testing:
                self.stop_audio_test()
                self.start_audio_test()
            debug_utils.log_info(f"Bit depth updated to {app_data}")
            dpg.set_value("status_text", f"Status: Bit depth set to {app_data}")
        except Exception as e:
            debug_utils.log_error(f"Failed to update bit depth: {e}")

    def update_sample_rate(self, sender, app_data):
        try:
            if self.is_testing:
                self.stop_audio_test()
                self.start_audio_test()
            debug_utils.log_info(f"Sample rate updated to {app_data}")
            dpg.set_value("status_text", f"Status: Sample rate set to {app_data}")
        except Exception as e:
            debug_utils.log_error(f"Failed to update sample rate: {e}")

    def update_relative_sensitivity(self, sender, app_data):
        try:
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
            debug_utils.log_info(f"Relative sensitivity updated: {app_data}")
            dpg.set_value("status_text", f"Status: Relative sensitivity {'enabled' if app_data else 'disabled'}")
        except Exception as e:
            debug_utils.log_error(f"Failed to update relative sensitivity: {e}")

    def pre_scale_to_slider(self, pre_scale):
        if pre_scale <= 0:
            return 0
        log_pre_scale = math.log10(pre_scale)
        log_min = math.log10(0.0001)
        log_max = math.log10(99.999)
        return max(0, min(100, (log_pre_scale - log_min) / (log_max - log_min) * 100))

    def slider_to_pre_scale(self, slider_value):
        if slider_value <= 0:
            return 0.0001
        log_min = math.log10(0.0001)
        log_max = math.log10(99.999)
        log_pre_scale = (slider_value / 100) * (log_max - log_min) + log_min
        return 10 ** log_pre_scale

    def pre_scale_to_db(self, pre_scale):
        return -float("inf") if pre_scale <= 0 else 20 * math.log10(pre_scale)

    def db_to_pre_scale(self, db):
        return 10 ** (db / 20)

    def pre_scale_to_percent(self, pre_scale):
        if pre_scale <= 0.0001:
            return 0.0
        if pre_scale >= 10.0:
            return 100.0
        log_pre_scale = math.log10(pre_scale)
        log_min = math.log10(0.0001)
        log_max = math.log10(99.999)
        return (log_pre_scale - log_min) / (log_max - log_min) * 100

    def percent_to_pre_scale(self, percent):
        if percent <= 0:
            return 0.0001
        log_min = math.log10(0.0001)
        log_max = math.log10(99.999)
        log_pre_scale = (percent / 100) * (log_max - log_min) + log_min
        return 10 ** log_pre_scale

    def set_slider_from_pre_scale(self, pre_scale):
        try:
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
            debug_utils.log_info(f"Slider set from pre_scale: {pre_scale} ({unit})")
        except Exception as e:
            debug_utils.log_error(f"Failed to set slider from pre_scale: {e}")

    def update_pre_scale_label(self, sender, app_data):
        try:
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
            debug_utils.log_info(f"Pre-scale label updated: {label}")
        except Exception as e:
            debug_utils.log_error(f"Failed to update pre-scale label: {e}")

    def update_unit(self, sender, app_data):
        try:
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
            debug_utils.log_info(f"Unit updated to {unit}")
            dpg.set_value("status_text", f"Status: Sensitivity unit set to {unit}")
        except Exception as e:
            debug_utils.log_error(f"Failed to update unit: {e}")

    def suggest_settings(self, sender, app_data):
        try:
            # Placeholder for suggesting optimal settings
            dpg.set_value("status_text", "Status: Suggested settings not implemented")
            debug_utils.log_info("Suggest settings called (not implemented)")
        except Exception as e:
            debug_utils.log_error(f"Failed to suggest settings: {e}")

    def calibrate(self, sender, app_data):
        try:
            # Placeholder for calibration
            dpg.set_value("status_text", "Status: Calibration not implemented")
            debug_utils.log_info("Calibrate called (not implemented)")
        except Exception as e:
            debug_utils.log_error(f"Failed to calibrate: {e}")

    def reset_settings(self, sender, app_data):
        try:
            self.saved_settings = {
                "bit_depth": "int24",
                "sample_rate": "48000",
                "pre_scale_factor": 0.002,
                "unit": "Numbers",
                "relative_sensitivity": False,
                "silence_threshold": 10.0,
                "show_peaks": False,
                "theme": "Dark",
                "host_api": "MME",
                "input_device": None,
                "output_device": None
            }
            self.load_settings()
            dpg.set_value("status_text", "Status: Settings reset to defaults")
            debug_utils.log_info("Settings reset to defaults")
        except Exception as e:
            debug_utils.log_error(f"Failed to reset settings: {e}")

    def toggle_audio_test(self, sender, app_data):
        if self.is_testing:
            self.stop_audio_test()
        else:
            self.start_audio_test()

    def start_audio_test(self):
        try:
            bit_depth = dpg.get_value("bit_depth_combo")
            self.dtype, _ = self.get_dtype_and_max(bit_depth)
            self.channels = 1
            self.samplerate = int(dpg.get_value("sample_rate_combo"))
            device = self.get_device_index(
                dpg.get_value("input_device_combo"),
                dpg.get_value("host_api_combo"),
                is_input=True
            )

            def audio_callback(indata, frames, time_info, status):
                try:
                    data = np.frombuffer(indata, dtype=self.dtype).reshape(-1, self.channels)
                    amplitude = np.max(np.abs(data))
                    self.peak_amplitude = max(self.peak_amplitude, amplitude)
                    level = (amplitude / 32767) * 400
                    dpg.configure_item("level_bar", pmax=(level, 20))
                except Exception as e:
                    debug_utils.log_error(f"Error in audio test callback: {e}")

            self.audio_stream = sd.RawInputStream(
                samplerate=self.samplerate,
                blocksize=8000,
                device=device,
                dtype=self.dtype,
                channels=self.channels,
                callback=audio_callback
            )
            self.audio_stream.start()
            self.is_testing = True
            dpg.configure_item("test_audio_button", label="Stop Test")
            dpg.set_value("status_text", "Status: Audio test started")
            debug_utils.log_info("Audio test started")
        except Exception as e:
            debug_utils.log_error(f"Failed to start audio test: {e}")
            dpg.set_value("status_text", f"Status: Failed to start audio test: {e}")

    def stop_audio_test(self):
        try:
            if self.audio_stream:
                self.audio_stream.stop()
                self.audio_stream.close()
            self.is_testing = False
            dpg.configure_item("test_audio_button", label="Test Audio")
            dpg.set_value("status_text", "Status: Audio test stopped")
            debug_utils.log_info("Audio test stopped")
        except Exception as e:
            debug_utils.log_error(f"Failed to stop audio test: {e}")

    def toggle_recording(self, sender, app_data):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        try:
            self.is_recording = True
            dpg.configure_item("record_button", label="Stop Recording")
            dpg.set_value("status_text", "Status: Recording started")
            debug_utils.log_info("Recording started")
        except Exception as e:
            debug_utils.log_error(f"Failed to start recording: {e}")

    def stop_recording(self):
        try:
            self.is_recording = False
            dpg.configure_item("record_button", label="Record")
            dpg.set_value("status_text", "Status: Recording stopped")
            debug_utils.log_info("Recording stopped")
        except Exception as e:
            debug_utils.log_error(f"Failed to stop recording: {e}")

    def show_waveform(self, sender, app_data):
        try:
            bit_depth = dpg.get_value("bit_depth_combo")
            channels = 1
            samplerate = int(dpg.get_value("sample_rate_combo"))
            device = self.get_device_index(
                dpg.get_value("input_device_combo"),
                dpg.get_value("host_api_combo"),
                is_input=True
            )
            pre_scale_factor = self.slider_to_pre_scale(dpg.get_value("sensitivity_slider"))
            relative_sensitivity = dpg.get_value("relative_sensitivity_check")
            WaveformDisplay(bit_depth, channels, samplerate, device, pre_scale_factor, relative_sensitivity)
            dpg.set_value("status_text", "Status: Waveform display opened")
            debug_utils.log_info("Waveform display opened")
        except Exception as e:
            debug_utils.log_error(f"Failed to show waveform: {e}")
            dpg.set_value("status_text", f"Status: Failed to show waveform: {e}")

    def setup_json_tab(self, tab_name, json_path):
        try:
            with dpg.group():
                dpg.add_text(f"Editing {tab_name} JSON:")
                dpg.add_input_text(tag=f"{tab_name}_json_input", multiline=True, height=200, width=-1)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Load", callback=lambda: self.load_json_tab(tab_name, json_path))
                    dpg.add_button(label="Save", callback=lambda: self.save_json_tab(tab_name, json_path))
            self.load_json_tab(tab_name, json_path)
        except Exception as e:
            debug_utils.log_error(f"Failed to setup JSON tab {tab_name}: {e}")

    def load_json_tab(self, tab_name, json_path):
        try:
            data = load_json_map(json_path)
            dpg.set_value(f"{tab_name}_json_input", json.dumps(data, indent=4))
            debug_utils.log_info(f"Loaded {tab_name} JSON")
            dpg.set_value("status_text", f"Status: Loaded {tab_name} JSON")
        except Exception as e:
            debug_utils.log_error(f"Failed to load {tab_name} JSON: {e}")

    def save_json_tab(self, tab_name, json_path):
        try:
            data = json.loads(dpg.get_value(f"{tab_name}_json_input"))
            base_path = get_base_path()
            full_path = os.path.join(base_path, json_path)
            with open(full_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            debug_utils.log_info(f"Saved {tab_name} JSON")
            dpg.set_value("status_text", f"Status: Saved {tab_name} JSON")
        except Exception as e:
            debug_utils.log_error(f"Failed to save {tab_name} JSON: {e}")
            dpg.set_value("status_text", f"Status: Failed to save {tab_name} JSON: {e}")

    def update_gui(self):
        try:
            dpg.set_frame_callback(dpg.get_frame_count() + 1, self.update_gui)
        except Exception as e:
            debug_utils.log_error(f"Failed to update GUI: {e}")

if __name__ == "__main__":
    gui = DictationGUI()
    dpg.start_dearpygui()
    dpg.destroy_context()