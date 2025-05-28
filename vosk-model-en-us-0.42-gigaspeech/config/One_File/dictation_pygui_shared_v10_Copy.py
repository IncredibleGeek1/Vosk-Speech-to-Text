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


import dearpygui.dearpygui as dpg
import json
import os
import subprocess
import sys
import math
import numpy as np
import sounddevice as sd
import threading
import selectors
import msvcrt
import time
import wave
import queue
from collections import deque
from vosk import Model, KaldiRecognizer
import logging
import keyboard
from spellchecker import SpellChecker

# Helper function to get the base path for bundled files
def get_base_path():
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

# Paths to JSON files
COMMANDS_JSON_PATH = "commands.json"
FRACTIONS_MAP_PATH = "fractions_map.json"
F_KEYS_MAP_PATH = "f_keys_map.json"
FUNCTIONS_MAP_PATH = "functions_map.json"
SYMBOLS_MAP_PATH = "symbols_map.json"
GOOGLE_NUMBERS_PATH = "google_numbers.json"
LARGE_NUMBERS_MAP_PATH = "large_numbers_map.json"
CONFIG_PATH = "config.json"

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
        
        with dpg.window(label="Waveform Display", tag="waveform_window", width=800, height=300, on_close=self.close):
            with dpg.plot(label="Waveform (16-bit PCM, as Vosk hears)", height=200, width=-1):
                dpg.add_plot_axis(dpg.Axis.X, label="Time (s)", tag="waveform_x_axis")
                dpg.add_plot_axis(dpg.Axis.Y, label="Amplitude", tag="waveform_y_axis")
                if dpg.does_item_exist("waveform_y_axis"):
                    dpg.set_axis_limits(item="waveform_y_axis", ymin=-32768, ymax=32768)
                else:
                    print("GUI ERROR: Waveform y-axis not found.")
                self.time_axis = np.linspace(0, 1, int(self.samplerate * 1.0))
                self.audio_data = np.zeros(int(self.samplerate * 1.0))
                dpg.add_line_series(self.time_axis, self.audio_data, label="Waveform", parent="waveform_y_axis", tag="waveform_series")
        
        self.start_stream()
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
                data = np.frombuffer(indata, dtype=self.dtype).reshape(-1, self.channels)
                data = data[:, 0].astype(np.float32)
                if self.dtype == "int8":
                    data_normalized = data * (32767 / 127)
                elif self.dtype == "int16":
                    data_normalized = data
                elif self.dtype == "int24" or self.dtype == "int32":
                    if self.max_value == 8388607:
                        data_normalized = data * (32767 / 8388607)
                    else:
                        data_normalized = data * (32767 / 2147483647)
                elif self.dtype == "float32":
                    data_normalized = data * 32767
                else:
                    raise ValueError(f"Unsupported dtype: {self.dtype}")
                pre_scale = self.pre_scale_factor
                data_scaled = data_normalized * pre_scale
                data_scaled = np.clip(data_scaled, -32768, 32767)
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
        self.is_running = False
        self.stop_stream()
        dpg.delete_item("waveform_window")

import os
import json
import logging
import threading
import time
import subprocess
import keyboard
from spellchecker import SpellChecker
import dearpygui.dearpygui as dpg
import sounddevice as sd
import vosk
import numpy as np
from scipy.signal import resample
import nltk
import pyautogui

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    logging.error(f"Error downloading NLTK data: {e}")
    raise SystemExit(1)

class DictationGUI:
    def __init__(self):
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
        self.command_queue = []
        self.transcribed_text = []
        self.last_command = ""
        self.last_command_time = 0
        self.COMMAND_DEBOUNCE_TIME = 1.0
        self.spell_checker = SpellChecker()
        self.caps_lock_on = False
        self.number_lock_on = False
        self.command_progress = 0.0
        self.command_progress_max = 100.0
        self.command_status = ""
        self.saved_settings = {
            "model_path": r"C:\Users\MenaBeshai\Downloads\Speech to Text\vosk-model-en-us-0.42-gigaspeech",
            "bit_depth": 24,
            "sample_rate": 96000,
            "pre_scale_factor": 0.002,
            "silence_threshold": 10.0,
            "relative_sensitivity": 0,
            "device_index": 6
        }
        self.json_data = {}
        self.commands = {}
        self.theme = "Dark"
        self.audio_buffer = []
        self.last_word_end_time = 0.0
        self.last_dictated_text = ""
        self.last_dictated_length = 0
        self.last_processed_command = None
        self.vosk_sample_rate = 16000
        self.wav_file = "output_gigaspeech.wav"
        self.transcription_file = "dictation_output_gigaspeech.txt"
        self.blocksize = 32000
        self.startup_delay = 5
        self.silence_threshold = 1.0

        # Load Vosk model
        vosk.SetLogLevel(-1)
        if not os.path.exists(self.saved_settings["model_path"]):
            logging.error(f"Vosk model not found at {self.saved_settings['model_path']}")
            raise SystemExit(1)
        self.model = vosk.Model(self.saved_settings["model_path"])
        self.recognizer = vosk.KaldiRecognizer(self.model, self.vosk_sample_rate)

        # Configuration paths
        self.config_dir = "config"
        self.commands_json_path = os.path.join(self.config_dir, "commands.json")
        self.fractions_map_path = os.path.join(self.config_dir, "fractions_map.json")
        self.f_keys_map_path = os.path.join(self.config_dir, "f_keys_map.json")
        self.functions_map_path = os.path.join(self.config_dir, "functions_map.json")
        self.symbols_map_path = os.path.join(self.config_dir, "symbols_map.json")
        self.numbers_map_path = os.path.join(self.config_dir, "numbers_map.json")
        self.google_numbers_path = os.path.join(self.config_dir, "google_numbers_map.json")
        self.large_numbers_map_path = os.path.join(self.config_dir, "large_numbers_map.json")

        # Load JSON maps
        self.fraction_map = self.load_json_map(self.fractions_map_path, "fractions map")
        self.f_keys_map = self.load_json_map(self.f_keys_map_path, "f-keys map")
        self.functions_map = self.load_json_map(self.functions_map_path, "functions map")
        self.symbols_map = self.load_json_map(self.symbols_map_path, "symbols map")
        self.numbers_map = self.load_json_map(self.numbers_map_path, "numbers map")
        self.google_numbers = self.load_json_map(self.google_numbers_path, "google numbers map")
        self.large_numbers_map = self.load_json_map(self.large_numbers_map_path, "large numbers map")

        self.numbers_map = {
            "negative": "-", "minus": "-", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
            "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15", "sixteen": "16",
            "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
            "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
            "hundred": "100", "thousand": "1000", "million": "1000000"
        }

        self.load_commands()

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
                    self.setup_audio_tab()
                for tab_name, json_path in {
                    "Fractions": self.fractions_map_path,
                    "F-Keys": self.f_keys_map_path,
                    "Functions": self.functions_map_path,
                    "Symbols": self.symbols_map_path,
                    "Numbers": self.numbers_map_path,
                    "Google Numbers": self.google_numbers_path,
                    "Large Numbers": self.large_numbers_map_path
                }.items():
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

    def load_json_map(self, file_path, map_name):
        if not os.path.exists(file_path):
            logging.error(f"{map_name} file not found at {file_path}")
            return {}
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            if not isinstance(mapping, dict):
                logging.error(f"{map_name} at {file_path} must be a JSON object (dictionary)")
                return {}
            return mapping
        except Exception as e:
            logging.error(f"Failed to load {map_name} from {file_path}: {e}")
            return {}

    def load_commands(self):
        try:
            if not os.path.exists(self.commands_json_path):
                logging.warning(f"commands.json not found at {self.commands_json_path}. Using default commands.")
                self.commands = {"simple_commands": {}, "parameterized_commands": []}
                self.tokenized_simple_commands = {}
                self.tokenized_parameterized_partial = []
                self.tokenized_parameterized_final = []
                return
            with open(self.commands_json_path, "r", encoding="utf-8") as f:
                self.commands = json.load(f)
            
            # Tokenize simple commands
            self.tokenized_simple_commands = {}
            for cmd, action in self.commands["simple_commands"].items():
                normalized_cmd = self.normalize_text(cmd.lower())
                tokens = tuple(nltk.word_tokenize(normalized_cmd))
                self.tokenized_simple_commands[tokens] = (cmd, action)
            
            # Tokenize parameterized commands
            final_only_commands = ["quote unquote "]
            self.tokenized_parameterized_partial = []
            self.tokenized_parameterized_final = []
            for cmd in self.commands["parameterized_commands"]:
                normalized_cmd = self.normalize_text(cmd.lower())
                tokens = tuple(nltk.word_tokenize(normalized_cmd))
                if cmd in final_only_commands:
                    self.tokenized_parameterized_final.append((tokens, cmd))
                else:
                    self.tokenized_parameterized_partial.append((tokens, cmd))
        except Exception as e:
            logging.error(f"Failed to load commands.json: {e}")
            self.commands = {"simple_commands": {}, "parameterized_commands": []}
            self.tokenized_simple_commands = {}
            self.tokenized_parameterized_partial = []
            self.tokenized_parameterized_final = []

    def normalize_text(self, text):
        text = text.replace("-", " ")
        return " ".join(text.split())

    def convert_numbers(self, text):
        if not self.number_lock_on:
            return text
        words = text.split()
        converted_words = []
        for word in words:
            word_lower = word.lower()
            if word_lower in self.numbers_map:
                converted_words.append(self.numbers_map[word_lower])
            else:
                converted_words.append(word)
        return " ".join(converted_words)

    def parse_number_sequence(self, words):
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
        try:
            number = int(text)
            if is_negative:
                number = -number
            return str(number), len(words)
        except ValueError:
            return "", len(words)

    def handle_special_phrases(self, text):
        text_lower = text.lower()
        words = text_lower.split()
        i = 0
        result = []
        while i < len(words):
            if words[i] == "number":
                i += 1
                number_sequence, num_words = self.parse_number_sequence(words[i:])
                if number_sequence:
                    actual_idx = i + num_words
                    if actual_idx < len(words) and words[actual_idx] == "percent":
                        result.append(number_sequence + "%")
                        i = actual_idx + 1
                    else:
                        result.append(number_sequence)
                        i = actual_idx
                continue
            is_potential_number = words[i] in ["negative", "minus"] or (words[i] in self.numbers_map and self.numbers_map[words[i]].isdigit())
            if is_potential_number:
                number_sequence, num_words = self.parse_number_sequence(words[i:])
                if number_sequence:
                    if i + num_words < len(words) and words[i + num_words] == "percent":
                        result.append(number_sequence + "%")
                        i += num_words + 1
                    else:
                        result.append(number_sequence)
                        i += num_words
                continue
            found_fraction = False
            for phrase, replacement in self.fraction_map.items():
                phrase_lower = phrase.lower()
                if " ".join(words[i:i + len(phrase_lower.split())]).lower() == phrase_lower:
                    result.append(replacement)
                    i += len(phrase_lower.split())
                    found_fraction = True
                    break
            if found_fraction:
                continue
            found_symbol = False
            for phrase, replacement in self.symbols_map.items():
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

    def process_text(self, text):
        if not text:
            return text
        text = self.convert_numbers(text)
        words = text.split()
        if not words:
            return text
        if self.caps_lock_on:
            words = [word.upper() for word in words]
        else:
            words[0] = words[0][0].upper() + words[0][1:] if len(words[0]) > 1 else words[0].upper()
        return " ".join(words)

    def audio_callback(self, indata, frames, time, status):
        if status:
            logging.error(f"Audio callback status: {status}")
        bit_depth = self.saved_settings.get("bit_depth", 24)
        if bit_depth == 16:
            dtype = np.int16
            max_value = 32767
        else:
            dtype = np.int32
            max_value = 8388607 if bit_depth == 24 else 2147483647
        indata_array = np.frombuffer(indata, dtype=dtype)
        indata_normalized = indata_array.astype(np.float32) / (max_value + 1)
        pre_scale = self.saved_settings.get("pre_scale_factor", 0.002)
        relative_sensitivity = self.saved_settings.get("relative_sensitivity", 0)
        if relative_sensitivity:
            reference_max = 32767
            scale_factor = reference_max / (max_value + 1)
            adjusted_pre_scale = pre_scale * scale_factor
        else:
            adjusted_pre_scale = pre_scale
        indata_normalized = indata_normalized * adjusted_pre_scale
        indata_array = np.clip(indata_normalized * 32767, -32768, 32767).astype(np.int16)
        if self.saved_settings.get("mic_channels", 1) > 1:
            indata_array = indata_array.reshape(-1, self.saved_settings.get("mic_channels", 1)).mean(axis=1).astype(np.int16)
        sample_rate = self.saved_settings.get("sample_rate", 96000)
        if sample_rate != self.vosk_sample_rate:
            num_samples_resampled = int(len(indata_array) * self.vosk_sample_rate / sample_rate)
            indata_array = resample(indata_array, num_samples_resampled)
            indata_array = indata_array.astype(np.int16)
        max_amplitude = np.max(np.abs(indata_array))
        silence_threshold = self.saved_settings.get("silence_threshold", 10.0)
        if max_amplitude < silence_threshold:
            return
        self.audio_buffer.append(indata_array)
        self.command_queue.append(indata_array.tobytes())

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
            if action == "cmd_stop_listening":
                self.stop_dictation()
                self.update_command_progress(100.0, "Stopped dictation.")
            elif action == "cmd_select_all":
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
                keyboard.press_and_release("shift+home")
                self.update_command_progress(100.0, "Selected to start.")
            elif action == "cmd_select_all_down":
                keyboard.press_and_release("shift+end")
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
                with open(self.transcription_file, "a", encoding="utf-8") as f:
                    f.write("\n")
                self.update_command_progress(100.0, "Pressed Enter.")
            elif action == "cmd_number_lock":
                self.number_lock_on = not self.number_lock_on
                self.update_command_progress(100.0, f"Number lock {'on' if self.number_lock_on else 'off'}.")
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
                for _ in range(self.last_dictated_length):
                    keyboard.press_and_release("backspace")
                if self.transcribed_text:
                    self.transcribed_text.pop()
                full_text = "".join(self.transcribed_text).rstrip()
                dpg.set_value("output_text", full_text)
                with open(self.transcription_file, "w", encoding="utf-8") as f:
                    f.write(full_text)
                self.update_command_progress(100.0, "Last dictation removed.")
            elif action == "cmd_click_that":
                pyautogui.click()
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
                pyautogui.press("printscreen")
                self.update_command_progress(100.0, "Captured screenshot.")
            elif action == "cmd_screen_shoot_window":
                pyautogui.hotkey("alt", "printscreen")
                self.update_command_progress(100.0, "Captured window screenshot.")
            elif action == "cmd_screen_shoot_monitor":
                try:
                    subprocess.Popen("ms-screenclip:", shell=True)
                    time.sleep(0.5)
                    pyautogui.hotkey("ctrl", "n")
                except Exception:
                    try:
                        subprocess.Popen("SnippingTool.exe", shell=True)
                        time.sleep(0.5)
                        pyautogui.hotkey("ctrl", "n")
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
            elif action.startswith("quote_unquote_"):
                param = action[len("quote_unquote_"):].replace("_", " ")
                self.type_text(f'"{param}"')
                self.update_command_progress(100.0, f"Quoted '{param}'.")
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
        self.command_queue.append(action)
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
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.saved_settings.update(json.load(f))
        except Exception:
            pass
        dpg.set_value("theme_combo", self.saved_settings.get("theme", "Dark"))
        self.apply_theme(self.saved_settings.get("theme", "Dark"))

    def save_settings(self, sender=None, app_data=None):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self.saved_settings, f, indent=4)
            dpg.set_value("status_text", "Settings saved successfully.")
        except Exception as e:
            dpg.set_value("status_text", f"Failed to save settings: {e}")

    def setup_audio_tab(self):
        with dpg.group():
            dpg.add_text("Model Path:")
            dpg.add_input_text(default_value=self.saved_settings.get("model_path", ""), tag="model_path_input")
            dpg.add_button(label="Browse", callback=self.set_model_path)
            dpg.add_text("Bit Depth:")
            dpg.add_combo([16, 24, 32], default_value=self.saved_settings.get("bit_depth", 24), tag="bit_depth_combo")
            dpg.add_text("Sample Rate:")
            dpg.add_combo([44100, 48000, 88200, 96000, 176400, 192000], default_value=self.saved_settings.get("sample_rate", 96000), tag="sample_rate_combo")
            dpg.add_text("Pre-Scale Factor:")
            dpg.add_input_float(default_value=self.saved_settings.get("pre_scale_factor", 0.002), tag="pre_scale_input")
            dpg.add_text("Silence Threshold:")
            dpg.add_input_float(default_value=self.saved_settings.get("silence_threshold", 10.0), tag="silence_threshold_input")
            dpg.add_text("Relative Sensitivity:")
            dpg.add_checkbox(default_value=self.saved_settings.get("relative_sensitivity", 0), tag="relative_sensitivity_check")
            dpg.add_text("Device Index:")
            dpg.add_input_int(default_value=self.saved_settings.get("device_index", 6), tag="device_index_input")

    def setup_json_tab(self, tab_name, json_path):
        with dpg.group():
            with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp, tag=f"{tab_name}_table"):
                dpg.add_table_column(label="Key")
                dpg.add_table_column(label="Value")
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
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

    def create_file_dialog(self):
        with dpg.file_dialog(
            directory_selector=True,
            show=False,
            callback=self.on_model_path_selection,
            tag="file_dialog",
            width=700,
            height=400
        ):
            dpg.add_file_extension(".*")

    def set_model_path(self, sender, app_data):
        if not dpg.does_item_exist("file_dialog"):
            self.create_file_dialog()
        dpg.show_item("file_dialog")

    def on_model_path_selection(self, sender, app_data):
        selected_path = app_data["file_path_name"]
        if not selected_path:
            dpg.set_value("status_text", "No directory selected.")
            return
        absolute_path = os.path.abspath(selected_path)
        dpg.set_value("model_path_input", absolute_path)
        self.saved_settings["model_path"] = absolute_path
        self.save_settings()
        logging.info(f"Model path set to {absolute_path}")

    def start_dictation(self, sender, app_data):
        logging.basicConfig(filename="dictation_gui.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info("Starting dictation process.")
        
        if self.is_dictating:
            logging.warning("Dictation already running.")
            return

        self.is_dictating = True
        dpg.configure_item("start_dictation_button", enabled=False)
        dpg.configure_item("stop_dictation_button", enabled=True)
        dpg.set_value("status_text", "Starting dictation... Click into a text field to begin typing.")
        
        # Clear transcription file
        with open(self.transcription_file, "w", encoding="utf-8") as f:
            f.write("")
        
        time.sleep(self.startup_delay)
        
        def transcription_loop():
            last_partial = ""
            skip_dictation = False
            try:
                with sd.RawInputStream(
                    samplerate=self.saved_settings.get("sample_rate", 96000),
                    blocksize=self.blocksize,
                    dtype="int32",
                    channels=self.saved_settings.get("mic_channels", 1),
                    callback=self.audio_callback,
                    device=self.saved_settings.get("device_index", 6)
                ):
                    while self.is_dictating:
                        if not self.command_queue:
                            time.sleep(0.1)
                            continue
                        data = self.command_queue.pop(0)
                        if self.recognizer.AcceptWaveform(data):
                            result_dict = json.loads(self.recognizer.Result())
                            text = result_dict.get("text", "")
                            if text:
                                text = self.handle_special_phrases(text)
                                normalized_text = self.normalize_text(text.lower())
                                tokens = tuple(nltk.word_tokenize(normalized_text))
                                
                                # Check parameterized final commands
                                is_final_command = False
                                for cmd_tokens, command in self.tokenized_parameterized_final:
                                    if len(tokens) >= len(cmd_tokens) and tokens[:len(cmd_tokens)] == cmd_tokens:
                                        logging.info(f"Detected command: {text}")
                                        self.last_processed_command = text
                                        skip_dictation = True
                                        param = text[len(command):].strip().lower()
                                        self.handle_command(f"quote_unquote_{param.replace(' ', '_')}")
                                        is_final_command = True
                                        break
                                if is_final_command:
                                    continue
                                
                                # Check if transcription is a command
                                is_command = False
                                if self.last_processed_command:
                                    last_processed_tokens = tuple(nltk.word_tokenize(self.normalize_text(self.last_processed_command.lower())))
                                    if tokens == last_processed_tokens:
                                        logging.info(f"Skipping dictation for command (already processed): {text}")
                                        self.last_processed_command = None
                                        skip_dictation = False
                                        continue
                                
                                for cmd_tokens, (cmd, _) in self.tokenized_simple_commands.items():
                                    if tokens == cmd_tokens:
                                        logging.info(f"Skipping dictation for command: {text}")
                                        skip_dictation = False
                                        is_command = True
                                        break
                                if is_command:
                                    continue
                                
                                processed_text = self.process_text(text)
                                current_time = time.time()
                                if self.last_word_end_time > 0 and self.last_processed_command != "new paragraph":
                                    silence_duration = current_time - self.last_word_end_time
                                    if silence_duration > self.silence_threshold:
                                        processed_text += "\n\n"
                                
                                if "result" in result_dict and result_dict["result"]:
                                    self.last_word_end_time = result_dict["result"][-1]["end"]
                                
                                logging.info(f"Transcription: {processed_text}")
                                with open(self.transcription_file, "a", encoding="utf-8") as f:
                                    f.write(processed_text + " ")
                                
                                if not any(processed_text.startswith(cmd) for cmd in ["\n\n", "\n", " ", "\t"]):
                                    self.type_text(processed_text)
                                    self.type_text(" ")
                                    self.last_dictated_text = processed_text + " "
                                    self.last_dictated_length = len(self.last_dictated_text)
                                    self.transcribed_text.append(processed_text + " ")
                                    dpg.set_value("output_text", "".join(self.transcribed_text).rstrip())
                                
                                skip_dictation = False
                        else:
                            partial_dict = json.loads(self.recognizer.PartialResult())
                            partial = partial_dict.get("partial", "")
                            if partial and partial != last_partial:
                                logging.debug(f"Partial: {partial}")
                                last_partial = partial
                                
                                normalized_partial = self.normalize_text(partial.lower())
                                partial_tokens = tuple(nltk.word_tokenize(normalized_partial))
                                current_time = time.time()
                                
                                for cmd_tokens, command in self.tokenized_parameterized_partial:
                                    if len(partial_tokens) >= len(cmd_tokens) and partial_tokens[:len(cmd_tokens)] == cmd_tokens:
                                        if self.last_command == partial and (current_time - self.last_command_time) < self.COMMAND_DEBOUNCE_TIME:
                                            continue
                                        logging.info(f"Detected command: {partial}")
                                        self.last_command = partial
                                        self.last_command_time = current_time
                                        self.last_processed_command = partial
                                        skip_dictation = True
                                        param = partial[len(command):].strip().lower().replace(" ", "_")
                                        action = f"{command.replace(' ', '_')}{param}"
                                        self.handle_command(action)
                                        break
                                
                                if "select " in normalized_partial and " through " in normalized_partial:
                                    if self.last_command == partial and (current_time - self.last_command_time) < self.COMMAND_DEBOUNCE_TIME:
                                        continue
                                    logging.info(f"Detected command: {partial}")
                                    self.last_command = partial
                                    self.last_command_time = current_time
                                    self.last_processed_command = partial
                                    skip_dictation = True
                                    parts = partial.lower().split(" through ")
                                    word1 = parts[0].replace("select ", "").strip().replace(" ", "_")
                                    word2 = parts[1].strip().replace(" ", "_")
                                    self.handle_command(f"select_through_{word1}_through_{word2}")
                                
                                if "correct " in normalized_partial:
                                    if self.last_command == partial and (current_time - self.last_command_time) < self.COMMAND_DEBOUNCE_TIME:
                                        continue
                                    logging.info(f"Detected command: {partial}")
                                    self.last_command = partial
                                    self.last_command_time = current_time
                                    self.last_processed_command = partial
                                    skip_dictation = True
                                    word_to_correct = partial.lower().replace("correct ", "").strip().replace(" ", "_")
                                    self.handle_command(f"correct_{word_to_correct}")
                                
                                for cmd_tokens, (command, action) in self.tokenized_simple_commands.items():
                                    if partial_tokens == cmd_tokens:
                                        if self.last_command == command and (current_time - self.last_command_time) < self.COMMAND_DEBOUNCE_TIME:
                                            continue
                                        logging.info(f"Detected command: {command}")
                                        self.last_command = command
                                        self.last_command_time = current_time
                                        self.last_processed_command = command
                                        skip_dictation = True
                                        self.handle_command(action)
                                        break
            except Exception as e:
                logging.error(f"Error during transcription: {e}")
                dpg.set_value("status_text", f"Error during transcription: {e}")
            finally:
                self.stop_dictation()

        self.transcription_thread = threading.Thread(target=transcription_loop, daemon=True)
        self.transcription_thread.start()

    def stop_dictation(self, sender=None, app_data=None):
        if self.is_dictating:
            self.is_dictating = False
            dpg.configure_item("start_dictation_button", enabled=True)
            dpg.configure_item("stop_dictation_button", enabled=False)
            dpg.set_value("status_text", "Dictation stopped.")
            logging.info("Saving audio to WAV file...")
            audio_data = np.concatenate(self.audio_buffer) if self.audio_buffer else np.array([], dtype=np.int16)
            with wave.open(self.wav_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.vosk_sample_rate)
                wf.writeframes(audio_data.tobytes())
            logging.info(f"Audio saved to {self.wav_file}")

    def run(self):
        while dpg.is_dearpygui_running():
            jobs = dpg.get_callback_queue()
            if jobs:
                for job in jobs:
                    try:
                        job()
                    except Exception as e:
                        logging.error(f"GUI callback error: {e}")
            dpg.render_dearpygui_frame()
        dpg.destroy_context()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    number_lock_on = False
    last_dictated_length = 0
    gui = DictationGUI()
    gui.run()