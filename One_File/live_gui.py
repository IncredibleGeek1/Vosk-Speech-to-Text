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


import queue
import sys
import sounddevice as sd
import vosk
import numpy as np
from scipy.signal import resample
import os
import traceback
import wave
import json
import time
import keyboard
import subprocess
from collections import deque
import nltk
import pyautogui
import pyperclip
import argparse
import dearpygui.dearpygui as dpg
import threading
import logging
import math
import re
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from spellchecker import SpellChecker
    spell = SpellChecker()
except ImportError:
    spell = None
    print("SpellChecker not available. Continuing without spell checking.")

    def check_spelling(text):
        if spell:
            words = text.split()
            misspelled = spell.unknown(words)
            for word in misspelled:
                correction = spell.correction(word)
                print(f"Misspelled: {word}, Suggestion: {correction}")
        else:
            print("Spell-checking skipped due to missing library.")

    def monitor_clipboard():
        last_text = ""
        while True:
            text = pyperclip.paste()
            if text != last_text and spell:
                words = text.split()
                misspelled = spell.unknown(words)
                if misspelled:
                    print("Misspelled words:", misspelled)
                    for word in misspelled:
                        print(f"Suggestion for {word}: {spell.correction(word)}")
            last_text = text
            time.sleep(1)

    if spell:
        monitor_clipboard()
    else:
        print("Clipboard monitoring skipped.")

def get_base_path():
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK data: {e}")
    sys.exit(1)

# Simplified words_to_numbers function to replace words_to_numbers_v7
def convert_numbers(text, fraction_map, symbols_map, google_numbers, large_numbers_map):
    """Convert spoken number words to digits (simplified version)."""
    try:
        words = text.lower().split()
        if not words:
            return None
        
        # Handle fractions
        for phrase, replacement in fraction_map.items():
            if text.lower() == phrase.lower():
                return replacement
        
        # Handle large numbers and symbols (if applicable)
        for phrase, replacement in large_numbers_map.items():
            if text.lower() == phrase.lower():
                return replacement
        for phrase, replacement in symbols_map.items():
            if text.lower() == phrase.lower():
                return replacement
        
        # Basic number parsing
        number_words = {
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
            "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
            "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
            "seventy": 70, "eighty": 80, "ninety": 90,
            "hundred": 100, "thousand": 1000, "million": 1000000
        }
        
        result = 0
        current = 0
        is_negative = False
        
        if words[0] in ["negative", "minus"]:
            is_negative = True
            words = words[1:]
        
        for word in words:
            if word in number_words:
                num = number_words[word]
                if num >= 100:
                    current *= num
                    result += current
                    current = 0
                else:
                    current += num
            else:
                return None  # If any word isn't recognized, return None
        
        result += current
        return -result if is_negative else result
    except Exception:
        return None

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Live speech-to-text dictation with Vosk.")
parser.add_argument("--model", type=str, default="vosk-model-en-us-0.42-gigaspeech",
                    help="Path to the Vosk model directory.")
parser.add_argument("--bit-depth", type=int, default=24, choices=[16, 24, 32], help="Bit depth for audio input")
parser.add_argument("--sample-rate", type=int, default=96000, choices=[44100, 48000, 88200, 96000, 176400, 192000], help="Sample rate for audio input (Hz)")
parser.add_argument("--pre-scale-factor", type=float, default=0.002, help="Pre-scale factor for audio input")
parser.add_argument("--silence-threshold", type=float, default=10.0, help="Silence threshold for audio detection")
parser.add_argument("--relative-sensitivity", type=int, default=0, choices=[0, 1], help="Whether to use relative sensitivity (0 or 1)")
parser.add_argument("--device-index", type=int, default=6, help="Index of the audio input device")
parser.add_argument("--list-devices", action="store_true", help="List available audio devices and exit")

args = parser.parse_args()

# Configuration
model_path = args.model
bit_depth = args.bit_depth
sample_rate = args.sample_rate
pre_scale = args.pre_scale_factor
silence_threshold = args.silence_threshold
relative_sensitivity = bool(args.relative_sensitivity)
device_index = args.device_index

# Vosk configuration
vosk.SetLogLevel(-1)
q = queue.Queue()
MODEL_PATH = model_path
MIC_SAMPLERATE = sample_rate
MIC_CHANNELS = 1
MIC_BITDEPTH = bit_depth
VOSK_SAMPLERATE = 16000
WAV_FILE = "output_gigaspeech.wav"
TRANSCRIPTION_FILE = "dictation_output_gigaspeech.txt"
DEVICE_INDEX = device_index
SILENCE_THRESHOLD = 1.0
BLOCKSIZE = 32000
PRE_SCALE_FACTOR = pre_scale
SILENCE_AMPLITUDE_THRESHOLD = silence_threshold
RELATIVE_SENSITIVITY = relative_sensitivity
STARTUP_DELAY = 5
COMMAND_DEBOUNCE_TIME = 1.0

# Paths to JSON files (assumed to be in config directory)
CONFIG_PATH = "config.json"
BASE_PATH = get_base_path()
CONFIG_DIR = "config"
CONFIG_DIR = os.path.join(BASE_PATH, "config")
COMMANDS_JSON_PATH = os.path.join(CONFIG_DIR, "commands.json")
FRACTIONS_MAP_PATH = os.path.join(CONFIG_DIR, "fractions_map.json")
F_KEYS_MAP_PATH = os.path.join(CONFIG_DIR, "f_keys_map.json")
FUNCTIONS_MAP_PATH = os.path.join(CONFIG_DIR, "functions_map.json")
SYMBOLS_MAP_PATH = os.path.join(CONFIG_DIR, "symbols_map.json")
NUMBERS_MAP_PATH = os.path.join(CONFIG_DIR, "numbers_map.json")
GOOGLE_NUMBERS_PATH = os.path.join(CONFIG_DIR, "google_numbers_map.json")
LARGE_NUMBERS_MAP_PATH = os.path.join(CONFIG_DIR, "large_numbers_map.json")

# Load JSON maps
def load_json_map(file_path, map_name):
    if not os.path.exists(file_path):
        print(f"Error: {map_name} file not found at {file_path}")
        sys.exit(1)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        if not isinstance(mapping, dict):
            print(f"Error: {map_name} at {file_path} must be a JSON object")
            sys.exit(1)
        return mapping
    except Exception as e:
        print(f"Error loading {map_name}: {e}")
        sys.exit(1)

FRACTION_MAP = load_json_map(FRACTIONS_MAP_PATH, "fractions map")
F_KEYS_MAP = load_json_map(F_KEYS_MAP_PATH, "f-keys map")
FUNCTIONS_MAP = load_json_map(FUNCTIONS_MAP_PATH, "functions map")
SYMBOLS_MAP = load_json_map(SYMBOLS_MAP_PATH, "symbols map")
NUMBERS_MAP = load_json_map(NUMBERS_MAP_PATH, "numbers map")
GOOGLE_NUMBERS = load_json_map(GOOGLE_NUMBERS_PATH, "google numbers map")
LARGE_NUMBERS_MAP = load_json_map(LARGE_NUMBERS_MAP_PATH, "large numbers map")

# Number map for basic digit conversion
NUMBERS_MAP = {
    "negative": "-", "minus": "-", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
    "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15", "sixteen": "16",
    "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
    "hundred": "100", "thousand": "1000", "million": "1000000"
}


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

# Audio buffer
audio_buffer = []

# Global state
last_word_end_time = 0.0
last_command = None
last_command_time = 0.0
skip_dictation = False
last_dictated_text = ""
last_dictated_length = 0
spell = SpellChecker()
caps_lock_on = False
number_lock_on = False
last_processed_command = None
transcription_running = False
rec = None
stream = None

# GUI state
partial_text = ""
final_text = ""
status_text = "Stopped"


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

def parse_number_sequence(words):
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
    result = convert_numbers(text, FRACTION_MAP, SYMBOLS_MAP, GOOGLE_NUMBERS, LARGE_NUMBERS_MAP)
    if result is None:
        return "", len(words)
    number = int(result)
    if is_negative:
        number = -number
    return str(number), len(words)

def normalize_text(text):
    text = text.replace("-", " ")
    return " ".join(text.split())

def convert_numbers_in_text(text):
    if not number_lock_on:
        return text
    words = text.split()
    converted_words = []
    for word in words:
        word_lower = word.lower()
        converted_words.append(NUMBERS_MAP.get(word_lower, word))
    return " ".join(converted_words)

def load_commands():
    if not os.path.exists(COMMANDS_JSON_PATH):
        print(f"Commands JSON file not found at {COMMANDS_JSON_PATH}")
        sys.exit(1)
    with open(COMMANDS_JSON_PATH, "r") as f:
        commands_data = json.load(f)
    simple_commands = commands_data["simple_commands"]
    tokenized_simple_commands = {}
    for cmd, action in simple_commands.items():
        normalized_cmd = normalize_text(cmd.lower())
        tokens = tuple(nltk.word_tokenize(normalized_cmd))
        tokenized_simple_commands[tokens] = (cmd, action)
    parameterized_commands = commands_data["parameterized_commands"]
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

TOKENIZED_SIMPLE_COMMANDS, TOKENIZED_PARAMETERIZED_PARTIAL, TOKENIZED_PARAMETERIZED_FINAL = load_commands()

def process_text(text):
    global caps_lock_on
    if not text:
        return text
    text = convert_numbers_in_text(text)
    words = text.split()
    if not words:
        return text
    if caps_lock_on:
        words = [word.upper() for word in words]
    else:
        words[0] = words[0][0].upper() + words[0][1:] if len(words[0]) > 1 else words[0].upper()
    processed_text = " ".join(words)
    number = convert_numbers(processed_text, FRACTION_MAP, SYMBOLS_MAP, GOOGLE_NUMBERS, LARGE_NUMBERS_MAP)
    return str(number) if number is not None else processed_text

def handle_special_phrases(text):
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
        is_potential_number = False
        first_word = words[i]
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

def callback(indata, frames, time, status):
    global audio_buffer
    if status:
        print(f"Audio callback status: {status}", file=sys.stderr)
    if MIC_BITDEPTH == 16:
        dtype = np.int16
        max_value = 32767
    elif MIC_BITDEPTH == 24 or MIC_BITDEPTH == 32:
        dtype = np.int32
        max_value = 8388607 if MIC_BITDEPTH == 24 else 2147483647
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
    max_amplitude = np.max(np.abs(indata_array))
    if max_amplitude < SILENCE_AMPLITUDE_THRESHOLD:
        return
    audio_buffer.append(indata_array)
    indata_bytes = indata_array.tobytes()
    q.put(indata_bytes)

def transcription_thread():
    global rec, stream, transcription_running, partial_text, final_text, status_text, last_word_end_time, last_command, last_command_time, skip_dictation, last_dictated_text, last_dictated_length, last_processed_command
    try:
        print("Loading Vosk model...")
        if not os.path.exists(MODEL_PATH):
            status_text = f"Error: Model path {MODEL_PATH} does not exist."
            dpg.set_value("status_text", status_text)
            transcription_running = False
            return
        model = vosk.Model(MODEL_PATH)
        rec = vosk.KaldiRecognizer(model, VOSK_SAMPLERATE)
        last_partial = ""
        with sd.RawInputStream(samplerate=MIC_SAMPLERATE, blocksize=BLOCKSIZE, dtype="int32", channels=MIC_CHANNELS, callback=callback, device=DEVICE_INDEX) as stream:
            status_text = "Running"
            dpg.set_value("status_text", status_text)
            stop_listening = False
            while transcription_running and not stop_listening:
                data = q.get()
                if rec.AcceptWaveform(data):
                    result_dict = json.loads(rec.Result())
                    text = result_dict.get("text", "")
                    if text:
                        text = handle_special_phrases(text)
                        normalized_text = normalize_text(text.lower())
                        tokens = tuple(nltk.word_tokenize(normalized_text))
                        is_final_command = False
                        for cmd_tokens, command in TOKENIZED_PARAMETERIZED_FINAL:
                            if len(tokens) >= len(cmd_tokens) and tokens[:len(cmd_tokens)] == cmd_tokens:
                                print(f"Detected command: {text}")
                                last_processed_command = text
                                skip_dictation = True
                                param = text[len(command):].strip().lower()
                                try:
                                    if command == "quote unquote ":
                                        keyboard.write(f'"{param}"')
                                except Exception as e:
                                    print(f"Error executing command '{command}{param}': {e}")
                                is_final_command = True
                                break
                        if is_final_command:
                            continue
                        is_command = False
                        if last_processed_command:
                            last_processed_tokens = tuple(nltk.word_tokenize(normalize_text(last_processed_command.lower())))
                            if tokens == last_processed_tokens:
                                last_processed_command = None
                                skip_dictation = False
                                continue
                        for cmd_tokens, (cmd, _) in TOKENIZED_SIMPLE_COMMANDS.items():
                            if tokens == cmd_tokens:
                                skip_dictation = False
                                is_command = True
                                break
                        if is_command:
                            continue
                        processed_text = process_text(text)
                        current_time = time.time()
                        if last_word_end_time > 0 and not (last_processed_command == "new paragraph"):
                            silence_duration = current_time - last_word_end_time
                            if silence_duration > SILENCE_THRESHOLD:
                                processed_text += "\n\n"
                        if "result" in result_dict and result_dict["result"]:
                            last_word_end_time = result_dict["result"][-1]["end"]
                        number = convert_numbers(processed_text, FRACTION_MAP, SYMBOLS_MAP, GOOGLE_NUMBERS, LARGE_NUMBERS_MAP)
                        final_output = str(number) if number is not None else processed_text
                        final_text += final_output + " "
                        dpg.set_value("final_text", final_text)
                        with open(TRANSCRIPTION_FILE, "a", encoding="utf-8") as f:
                            f.write(final_output + " ")
                        if not any(final_output.startswith(cmd) for cmd in ["\n\n", "\n", " ", "\t"]):
                            keyboard.write(final_output)
                            keyboard.write(" ")
                            last_dictated_text = final_output + " "
                            last_dictated_length = len(last_dictated_text)
                        skip_dictation = False
                else:
                    partial_dict = json.loads(rec.PartialResult())
                    partial = partial_dict.get("partial", "")
                    if partial and partial != last_partial:
                        partial_text = partial
                        dpg.set_value("partial_text", partial_text)
                        last_partial = partial
                    normalized_partial = normalize_text(partial.lower())
                    partial_tokens = tuple(nltk.word_tokenize(normalized_partial))
                    current_time = time.time()
                    for cmd_tokens, command in TOKENIZED_PARAMETERIZED_PARTIAL:
                        if len(partial_tokens) >= len(cmd_tokens) and partial_tokens[:len(cmd_tokens)] == cmd_tokens:
                            if last_command == partial and (current_time - last_command_time) < COMMAND_DEBOUNCE_TIME:
                                continue
                            print(f"Detected command: {partial}")
                            last_command = partial
                            last_command_time = current_time
                            last_processed_command = partial
                            skip_dictation = True
                            param = partial[len(command):].strip().lower()
                            try:
                                if command == "highlight ":
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
                                    keyboard.press_and_release("ctrl+f")
                                    time.sleep(0.2)
                                    keyboard.write(param)
                                    time.sleep(0.1)
                                    keyboard.press_and_release("enter")
                                    time.sleep(0.1)
                                    keyboard.press_and_release("escape")
                                elif command == "insert after ":
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
                                    keyboard.press_and_release("ctrl+f")
                                    time.sleep(0.2)
                                    keyboard.write(param)
                                    time.sleep(0.1)
                                    keyboard.press_and_release("enter")
                                    time.sleep(0.1)
                                    keyboard.press_and_release("escape")
                                elif command == "copy ":
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
                                    keyboard.press_and_release(param)
                                elif command == "open ":
                                    if not param:
                                        print("Error: No application name provided.")
                                        break
                                    stt_apps = os.environ.get("STT", "")
                                    if not stt_apps:
                                        print("Error: STT environment variable not set.")
                                        break
                                    app_dict = {}
                                    for app in stt_apps.split(";"):
                                        if not app or "=" not in app:
                                            continue
                                        key, value = app.split("=", 1)
                                        app_dict[key] = value
                                    app_name = param.replace(" ", "").lower()
                                    app_path = app_dict.get(app_name)
                                    if app_path:
                                        subprocess.Popen(app_path, shell=True)
                                    else:
                                        print(f"Application '{param}' not found.")
                                elif command == "go to address ":
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
                                    subprocess.Popen(['start', url], shell=True)
                                elif command in ["move up ", "move down ", "move left ", "move right "]:
                                    try:
                                        num = int(param.split()[0])
                                    except (ValueError, IndexError):
                                        num = 1
                                    direction = command.split()[1]
                                    for _ in range(num):
                                        keyboard.press_and_release(direction)
                                elif command == "function ":
                                    function_key = param.lower().replace("f", "")
                                    if function_key in [str(i) for i in range(1, 13)]:
                                        keyboard.press_and_release(f"f{function_key}")
                                    else:
                                        print(f"Invalid function key: {param}")
                            except Exception as e:
                                print(f"Error executing command '{command}{param}': {e}")
                            break
                    if "select " in normalized_partial and " through " in normalized_partial:
                        partial_tokens_list = list(partial_tokens)
                        if "through" in partial_tokens_list:
                            if last_command == partial and (current_time - last_command_time) < COMMAND_DEBOUNCE_TIME:
                                continue
                            print(f"Detected command: {partial}")
                            last_command = partial
                            last_command_time = current_time
                            last_processed_command = partial
                            skip_dictation = True
                            try:
                                parts = partial.lower().split(" through ")
                                word1 = parts[0].replace("select ", "").strip()
                                word2 = parts[1].strip()
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
                    if "correct " in normalized_partial:
                        if partial_tokens[:1] == ("correct",):
                            if last_command == partial and (current_time - last_command_time) < COMMAND_DEBOUNCE_TIME:
                                continue
                            print(f"Detected command: {partial}")
                            last_command = partial
                            last_command_time = current_time
                            last_processed_command = partial
                            skip_dictation = True
                            try:
                                word_to_correct = partial.lower().replace("correct ", "").strip()
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
                    for cmd_tokens, (command, action) in TOKENIZED_SIMPLE_COMMANDS.items():
                        if partial_tokens == cmd_tokens:
                            if last_command == command and (current_time - last_command_time) < COMMAND_DEBOUNCE_TIME:
                                continue
                            print(f"Detected command: {command}")
                            last_command = command
                            last_command_time = current_time
                            last_processed_command = command
                            skip_dictation = True
                            try:
                                if action == "cmd_stop_listening":
                                    stop_listening = True
                                elif action == "cmd_select_all":
                                    keyboard.press("ctrl")
                                    keyboard.press_and_release("a")
                                    keyboard.release("ctrl")
                                elif action == "cmd_select_down":
                                    keyboard.press("shift")
                                    keyboard.press_and_release("down")
                                    keyboard.release("shift")
                                elif action == "cmd_select_up":
                                    keyboard.press("shift")
                                    keyboard.press_and_release("up")
                                    keyboard.release("shift")
                                elif action == "cmd_select_all_up":
                                    keyboard.press("shift")
                                    keyboard.press_and_release("home")
                                    keyboard.release("shift")
                                elif action == "cmd_select_all_down":
                                    keyboard.press("shift")
                                    keyboard.press_and_release("end")
                                    keyboard.release("shift")
                                elif action == "cmd_copy":
                                    keyboard.press("ctrl")
                                    keyboard.press_and_release("c")
                                    keyboard.release("ctrl")
                                elif action == "cmd_paste":
                                    keyboard.press("ctrl")
                                    keyboard.press_and_release("v")
                                    keyboard.release("ctrl")
                                elif action == "cmd_delete":
                                    keyboard.press_and_release("backspace")
                                elif action == "cmd_undo":
                                    keyboard.press("ctrl")
                                    keyboard.press_and_release("z")
                                    keyboard.release("ctrl")
                                elif action == "cmd_redo":
                                    keyboard.press("ctrl")
                                    keyboard.press_and_release("y")
                                    keyboard.release("ctrl")
                                elif action == "cmd_file_properties":
                                    keyboard.press_and_release("menu")
                                elif action == "cmd_save_document":
                                    keyboard.press("ctrl")
                                    keyboard.press_and_release("s")
                                    keyboard.release("ctrl")
                                elif action == "cmd_open_file":
                                    print("Executing action: open file (placeholder)")
                                elif action == "cmd_move_up":
                                    keyboard.press_and_release("up")
                                elif action == "cmd_move_down":
                                    keyboard.press_and_release("down")
                                elif action == "cmd_move_left":
                                    keyboard.press_and_release("left")
                                elif action == "cmd_move_right":
                                    keyboard.press_and_release("right")
                                elif action == "cmd_move_up_paragraph":
                                    keyboard.press("ctrl")
                                    keyboard.press_and_release("up")
                                    keyboard.release("ctrl")
                                elif action == "cmd_move_down_paragraph":
                                    keyboard.press("ctrl")
                                    keyboard.press_and_release("down")
                                    keyboard.release("ctrl")
                                elif action == "cmd_enter":
                                    keyboard.press_and_release("enter")
                                elif action == "cmd_number_lock":
                                    global number_lock_on
                                    number_lock_on = not number_lock_on
                                    print(f"Number lock is now {'on' if number_lock_on else 'off'}")
                                elif action == "cmd_caps_lock":
                                    global caps_lock_on
                                    caps_lock_on = not caps_lock_on
                                    print(f"Caps lock is now {'on' if caps_lock_on else 'off'}")                                        
                                elif action == "cmd_bold":
                                    keyboard.press("ctrl")
                                    keyboard.press_and_release("b")
                                    keyboard.release("ctrl")
                                elif action == "cmd_italicize":
                                    keyboard.press("ctrl")
                                    keyboard.press_and_release("i")
                                    keyboard.release("ctrl")
                                elif action == "cmd_underline":
                                    keyboard.press("ctrl")
                                    keyboard.press_and_release("u")
                                    keyboard.release("ctrl")
                                elif action == "cmd_center":
                                    keyboard.press("ctrl")
                                    keyboard.press_and_release("e")
                                    keyboard.release("ctrl")
                                elif action == "cmd_left_align":
                                    keyboard.press("ctrl")
                                    keyboard.press_and_release("l")
                                    keyboard.release("ctrl")
                                elif action == "cmd_right_align":
                                    keyboard.press("ctrl")
                                    keyboard.press_and_release("r")
                                    keyboard.release("ctrl")
                                elif action == "cmd_cut":
                                    keyboard.press("ctrl")
                                    keyboard.press_and_release("x")
                                    keyboard.release("ctrl")
                                elif action == "cmd_go_to_beginning":
                                    keyboard.press("ctrl")
                                    keyboard.press_and_release("home")
                                    keyboard.release("ctrl")
                                elif action == "cmd_go_to_end":
                                    keyboard.press("ctrl")
                                    keyboard.press_and_release("end")
                                    keyboard.release("ctrl")
                                elif action == "cmd_go_to_beginning_of_line":
                                    keyboard.press_and_release("home")
                                elif action == "cmd_go_to_end_of_line":
                                    keyboard.press_and_release("end")
                                elif action == "cmd_go_to_address":
                                    keyboard.press("ctrl")
                                    keyboard.press_and_release("l")
                                    keyboard.release("ctrl")
                                elif action == "cmd_refresh_page":
                                    keyboard.press_and_release("f5")
                                elif action == "cmd_go_back":
                                    keyboard.press("alt")
                                    keyboard.press_and_release("left")
                                    keyboard.release("alt")
                                elif action == "cmd_go_forward":
                                    keyboard.press("alt")
                                    keyboard.press_and_release("right")
                                    keyboard.release("alt")
                                elif action == "cmd_open_new_tab":
                                    keyboard.press("ctrl")
                                    keyboard.press_and_release("t")
                                    keyboard.release("ctrl")
                                elif action == "cmd_close_tab":
                                    keyboard.press("ctrl")
                                    keyboard.press_and_release("w")
                                    keyboard.release("ctrl")
                                elif action == "cmd_next_tab":
                                    keyboard.press("ctrl")
                                    keyboard.press_and_release("tab")
                                    keyboard.release("ctrl")
                                elif action == "cmd_previous_tab":
                                    keyboard.press("ctrl")
                                    keyboard.press("shift")
                                    keyboard.press_and_release("tab")
                                    keyboard.release("shift")
                                    keyboard.release("ctrl")
                                elif action == "cmd_shift_tab":
                                    keyboard.press("shift")
                                    keyboard.press_and_release("tab")
                                    keyboard.release("shift")
                                elif action == "cmd_scratch_that":
                                    for _ in range(last_dictated_length):
                                        keyboard.press_and_release("backspace")
                                elif action == "cmd_click_that":
                                    pyautogui.click()
                                elif action == "cmd_punctuation_period":
                                    keyboard.write(".")
                                elif action == "cmd_punctuation_comma":
                                    keyboard.write(",")
                                elif action == "cmd_punctuation_question_mark":
                                    keyboard.write("?")
                                elif action == "cmd_punctuation_exclamation":
                                    keyboard.write("!")
                                elif action == "cmd_punctuation_semicolon":
                                    keyboard.write(";")
                                elif action == "cmd_punctuation_colon":
                                    keyboard.write(":")
                                elif action == "cmd_punctuation_tilde":
                                    keyboard.write("~")
                                elif action == "cmd_punctuation_ampersand":
                                    keyboard.write("&")
                                elif action == "cmd_punctuation_percent":
                                    keyboard.write("%")
                                elif action == "cmd_punctuation_asterisk":
                                    keyboard.write("*")
                                elif action == "cmd_punctuation_parentheses":
                                    keyboard.write("()")
                                    keyboard.press_and_release("left")
                                elif action == "cmd_punctuation_dash":
                                    keyboard.write("-")
                                elif action == "cmd_punctuation_underscore":
                                    keyboard.write("_")
                                elif action == "cmd_punctuation_plus":
                                    keyboard.write("+")
                                elif action == "cmd_punctuation_equals":
                                    keyboard.write("=")
                                elif action == "cmd_press_escape":
                                    keyboard.press_and_release("escape")
                                elif action == "cmd_screen_shoot":
                                    pyautogui.press("printscreen")
                                elif action == "cmd_screen_shoot_window":
                                    pyautogui.hotkey("alt", "printscreen")
                                elif action == "cmd_screen_shoot_monitor":
                                    try:
                                        subprocess.Popen("ms-screenclip:", shell=True)
                                        time.sleep(1)
                                        pyautogui.hotkey("ctrl", "n")
                                    except Exception as e:
                                        try:
                                            subprocess.Popen("SnippingTool.exe", shell=True)
                                            time.sleep(1)
                                            pyautogui.hotkey("ctrl", "n")
                                        except Exception as e:
                                            print(f"Error opening Snipping Tool: {e}")
                                elif action == "cmd_task_manager":
                                    keyboard.press("ctrl")
                                    keyboard.press("shift")
                                    keyboard.press_and_release("esc")
                                    keyboard.release("shift")
                                    keyboard.release("ctrl")
                                elif action == "cmd_debug_screen":
                                    keyboard.press("ctrl")
                                    keyboard.press("alt")
                                    keyboard.press_and_release("delete")
                                    keyboard.release("alt")
                                    keyboard.release("ctrl")
                                elif action == "cmd_force_close":
                                    keyboard.press("alt")
                                    keyboard.press_and_release("f4")
                                    keyboard.release("alt")
                                else:
                                    keyboard.write(action)
                            except Exception as e:
                                print(f"Error executing command '{command}': {e}")
                            break
    except Exception as e:
        status_text = f"Error: {str(e)}"
        dpg.set_value("status_text", status_text)
        print(f"Error during transcription: {e}")
        traceback.print_exc()
    finally:
        transcription_running = False
        status_text = "Stopped"
        dpg.set_value("status_text", status_text)
        print("Saving audio to WAV file...")
        audio_data = np.concatenate(audio_buffer) if audio_buffer else np.array([], dtype=np.int16)
        with wave.open(WAV_FILE, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(VOSK_SAMPLERATE)
            wf.writeframes(audio_data.tobytes())
        print(f"Audio saved to {WAV_FILE}")

def start_transcription(sender, app_data, user_data):
    global transcription_running, MODEL_PATH, MIC_SAMPLERATE, MIC_BITDEPTH, DEVICE_INDEX, PRE_SCALE_FACTOR, SILENCE_AMPLITUDE_THRESHOLD, RELATIVE_SENSITIVITY
    if not transcription_running:
        MODEL_PATH = dpg.get_value("model_path")
        MIC_SAMPLERATE = dpg.get_value("sample_rate")
        MIC_BITDEPTH = dpg.get_value("bit_depth")
        DEVICE_INDEX = dpg.get_value("device_index")
        PRE_SCALE_FACTOR = dpg.get_value("pre_scale_factor")
        SILENCE_AMPLITUDE_THRESHOLD = dpg.get_value("silence_threshold")
        RELATIVE_SENSITIVITY = dpg.get_value("relative_sensitivity")
        transcription_running = True
        threading.Thread(target=transcription_thread, daemon=True).start()

def stop_transcription(sender, app_data, user_data):
    global transcription_running
    if transcription_running:
        transcription_running = False

def browse_model_path(sender, app_data, user_data):
    dpg.configure_item("file_dialog", show=True)

def file_dialog_callback(sender, app_data, user_data):
    if app_data.get("file_path_name"):
        dpg.set_value("model_path", app_data["file_path_name"])


class create_gui:
    def __init__(self):
        self.model_path = MODEL_PATH
        self.device_index = DEVICE_INDEX
        self.sample_rate = MIC_SAMPLERATE
        self.bit_depth = MIC_BITDEPTH
        self.pre_scale_factor = PRE_SCALE_FACTOR
        self.silence_threshold = SILENCE_AMPLITUDE_THRESHOLD
        self.relative_sensitivity = RELATIVE_SENSITIVITY
        self.theme = "Dark"
        self.saved_settings = {}
        self.json_data = {}
        self.commands = {}
        self.is_testing = False
        self.is_recording = False
        self.is_debug_recording = False
        self.is_dictating = False
        self.audio_queue = queue.Queue()
        self.gui_update_queue = queue.Queue()
        self.command_queue = queue.Queue()
        self.noise_floor = 0.0
        self.peak_amplitude = 0.0
        self.last_noise_update = time.time()
        self.audio_stream = None
        self.output_stream = None
        self.audio_buffer = []
        self.debug_audio_buffer = []
        self.recordings_dir = "recordings"
        self.dictation_process = None
        self.monitor_stop_event = None
        os.makedirs(self.recordings_dir, exist_ok=True)
        self.load_settings()
        self.create_file_dialog()
        self.update_gui()
        self.load_commands()



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

            def load_commands(self):
                base_path = get_base_path()
                config_path = os.path.join(base_path, CONFIG_PATH)
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        self.commands = json.load(f)
                except Exception:
                    self.commands = {}
                    with open(config_path, "w", encoding="utf-8") as f:
                        json.dump(self.commands, f, indent=4)

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
        except Exception as e:
            logging.error(f"Error in save_manual_sensitivity: {e}", exc_info=True)

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
        logging.error(f"Error in on_data_type_changed: {e}", exc_info=True)

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
        logging.debug("Starting calibration process.")
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
            logging.error(f"Calibration failed: {e}", exc_info=True)
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
        logging.debug("Starting audio test.")
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
            np.clip(indata_normalized * 32767, -32768, 32767).astype(np.int16)== indata_scaled
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
                logging.error(f"Error in audio processing within input_callback: {e}", exc_info=True)

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
            except Exception as e:
                logging.error(f"Error getting device index: {e}", exc_info=True)
                dpg.set_value("status_text", f"Failed to get device index: {e}")
                self.is_testing = False
                dpg.configure_item("test_audio_button", label="Test Audio")
                return
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
                try:
                    while not self.gui_update_queue.empty():
                        update_type, value = self.gui_update_queue.get()
                        if update_type == "update_level":
                            max_amplitude = value
                            dpg.configure_item("clipping_indicator", fill=(255, 0, 0, 255 if max_amplitude >= 32767 else 128))
                            level_position = (max_amplitude / 32767) * 400
                            dpg.configure_item("level_bar", pmax=(level_position, 20))
                        elif update_type == "update_shadow":
                            self.update_shadow()
                except Exception as e:
                    logging.error(f"Error in GUI update: {e}", exc_info=True)
                    dpg.set_value("status_text", f"GUI Update Error: {e}")
                if self.is_testing:
                        dpg.set_frame_callback(dpg.get_frame_count() + 1, update_gui)
                dpg.set_frame_callback(dpg.get_frame_count() + 1, update_gui)
            def start_audio_test(self):
                dpg.set_value("status_text", f"Failed to start audio test: {e}")
                logging.error(f"Failed to start audio test: {e}", exc_info=True)
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
        
        logging.error(f"Error showing waveform: {e}", exc_info=True)


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
            logging;bit_depth = dpg.get_value("bit_depth_combo")
        if " - " in bit_depth:
            bit_depth = bit_depth.split(" - ")[0]
        try:
            sample_rate = int(dpg.get_value("sample_rate_combo"))
        except ValueError as e:
            logging.error(f"Invalid sample rate: {e}", exc_info=True)
            dpg.set_value("status_text", f"Invalid sample rate: {e}")
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
                logging.error(f"Failed to save recording: {e}", exc_info=True)
                dpg.set_value("status_text", f"Failed to save recording: {e}")
        
            logging.error(f"Error in stop_recording: {e}", exc_info=True)
            dpg.set_value("status_text", f"An unexpected error occurred: {e}")
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

    def create_gui():        
        dpg.create_context()
        dpg.create_viewport(title="Speech-to-Text Dictation Configuration", width=800, height=600)

        # Theme setup
        with dpg.theme(tag="dark_theme"):
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (46, 46, 46))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Button, (74, 74, 74))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (74, 74, 74))

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
        dpg.start_dearpygui()
        dpg.destroy_context()

if __name__ == "__main__":
    if args.list_devices:
        print("Available input audio devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"{i}: {device['name']}, {device['max_input_channels']} in")
        sys.exit(0)
    with open(TRANSCRIPTION_FILE, "w", encoding="utf-8") as f:
        f.write("")
    create_gui()