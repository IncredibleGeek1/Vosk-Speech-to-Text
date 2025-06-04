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
import re
import traceback
import wave
import time
import threading
import signal
import keyboard
import subprocess
from spellchecker import SpellChecker
import nltk
import pyautogui
import argparse
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QVBoxLayout, QWidget, QHBoxLayout, QLabel, QComboBox, QPushButton, QSlider, QCheckBox, QLineEdit, QTextEdit)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
import logging
import dearpygui.dearpygui as dpg
from vosk import Model, KaldiRecognizer
import json
import math
import tempfile
import webbrowser

def update_status(message):
    print(message)
    try:
        dpg.set_value("status_text", message)
    except Exception:
        pass  # In case GUI is not initialized yet
    logging.info(message)

# Helper function to get the base path for bundled files
def get_base_path():
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

def uses_large_numbers(tokens, large_numbers_map):
    return any(t in large_numbers_map for t in tokens)


# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK data: {e}")
    sys.exit(1)


# Path to your settings file
SETTINGS_FILE = "settings.json"
# Fallback default model path
FALLBACK_MODEL_PATH = os.path.expanduser("~/Downloads/vosk-model-en-us-0.42-gigaspeech")
# Try to load model path from settings.json
def get_default_model_path():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                settings = json.load(f)
            model_path = settings.get("model_path")
            if model_path:
                return model_path
        except Exception:
            pass
    return FALLBACK_MODEL_PATH

default_model_path = get_default_model_path()

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Live speech-to-text dictation with Vosk.")
parser.add_argument("--model", type=str, default=default_model_path)
parser.add_argument("--bit-depth", type=int, default=24, choices=[16, 24, 32])
parser.add_argument("--sample-rate", type=int, default=96000, choices=[44100, 48000, 88200, 96000, 176400, 192000])
parser.add_argument("--pre-scale-factor", type=float, default=0.002)
parser.add_argument("--silence-threshold", type=float, default=10.0)
parser.add_argument("--relative-sensitivity", type=int, default=0, choices=[0, 1])
parser.add_argument("--device-index", type=int, default=1)
parser.add_argument("--list-devices", action="store_true")
args = parser.parse_args()

# Constants
VOSK_SAMPLERATE = 16000
#WAV_FILE = "output_gigaspeech.wav"
TRANSCRIPTION_FILE = "dictation_output_gigaspeech.txt"
BLOCKSIZE = 32000
STARTUP_DELAY = 5
COMMAND_DEBOUNCE_TIME = 1.0
SILENCE_THRESHOLD = 1.0

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

BIT_DEPTHS = {
    "int8": "8-bit Integer",
    "int16": "16-bit Integer",
    "int24": "24-bit Integer",
    "int32": "32-bit Integer",
    "float32": "32-bit Float"
}

PRE_SCALE_FACTORS = {
    "int8": 0.0001,
    "int16": 0.001,
    "int24": 0.002,
    "int32": 0.004,
    "float32": 0.01
}

SILENCE_AMPLITUDE_THRESHOLDS = {
    "int8": 10.0,
    "int16": 20.0,
    "int24": 30.0,
    "int32": 40.0,
    "float32": 50.0
}

# Suppress Vosk logging
vosk.SetLogLevel(-1)


# Example: Group commands by category for tab display
COMMAND_TABS = [
    {
        "label": "Editing",
        "commands": [
            ("new paragraph", "cmd_new_paragraph"),
            ("new line", "cmd_new_line"),
            ("undo", "cmd_undo"),
            ("redo", "cmd_redo"),
            ("copy", "cmd_copy"),
            ("cut", "cmd_cut"),
            ("paste", "cmd_paste"),
            ("delete", "cmd_delete"),
            ("select all", "cmd_select_all"),
            ("scratch that", "cmd_scratch_that"),
            ("click that", "cmd_click_that"),
            ("clear", "cmd_clear"),
            ("space", "cmd_space"),
            ("tab", "cmd_tab"),
            ("enter", "cmd_enter"),
            ("bold", "cmd_bold"),
            ("italicize", "cmd_italicize"),
            ("underline", "cmd_underline"),
            ("caps lock on", "cmd_caps_lock_on"),
            ("caps lock off", "cmd_caps_lock_off"),
            ("number lock", "cmd_number_lock"),
        ]
    },
    {
        "label": "Navigation",
        "commands": [
            ("move up", "cmd_move_up"),
            ("move down", "cmd_move_down"),
            ("move left", "cmd_move_left"),
            ("move right", "cmd_move_right"),
            ("move up paragraph", "cmd_move_up_paragraph"),
            ("move down paragraph", "cmd_move_down_paragraph"),
            ("go to beginning", "cmd_go_to_beginning"),
            ("go to end", "cmd_go_to_end"),
            ("go to beginning of line", "cmd_go_to_beginning_of_line"),
            ("go to end of line", "cmd_go_to_end_of_line"),
            ("go to address", "cmd_go_to_address"),
            ("refresh page", "cmd_refresh_page"),
            ("go back", "cmd_go_back"),
            ("go forward", "cmd_go_forward"),
            ("open new tab", "cmd_open_new_tab"),
            ("close tab", "cmd_close_tab"),
            ("next tab", "cmd_next_tab"),
            ("previous tab", "cmd_previous_tab"),
            ("shift tab", "cmd_shift_tab"),
            ("switch windows", "cmd_switch_windows"),
            ("next window", "cmd_next_window"),
            ("previous window", "cmd_previous_window"),
            ("switch application", "cmd_switch_application"),
            ("next application", "cmd_next_application"),
            ("previous application", "cmd_previous_application"),
            ("switch window", "cmd_switch_window"),
            ("next window", "cmd_next_window"),
            ("go to address", "cmd_go_to_address"),
            ("screen shoot", "cmd_screen_shoot"),
            ("screen shoot window", "cmd_screen_shoot_window"),
            ("screen shoot monitor", "cmd_screen_shoot_monitor"),
            ("task manager", "cmd_task_manager"),
            ("debug screen", "cmd_debug_screen"),
            ("force close", "cmd_force_close"),
            ("file properties", "cmd_file_properties"),
        ]
    },
    {
        "label": "Formatting",
        "commands": [
            ("bold", "cmd_bold"),
            ("italicize", "cmd_italicize"),
            ("underline", "cmd_underline"),
            ("punctuation period", "cmd_punctuation_period"),
            ("punctuation comma", "cmd_punctuation_comma"),
            ("punctuation question mark", "cmd_punctuation_question_mark"),
            ("punctuation exclamation", "cmd_punctuation_exclamation"),
            ("punctuation semicolon", "cmd_punctuation_semicolon"),
            ("punctuation colon", "cmd_punctuation_colon"),
            ("punctuation tilde", "cmd_punctuation_tilde"),
            ("punctuation ampersand", "cmd_punctuation_ampersand"),
            ("punctuation percent", "cmd_punctuation_percent"),
            ("punctuation asterisk", "cmd_punctuation_asterisk"),
            ("punctuation parentheses", "cmd_punctuation_parentheses"),
            ("punctuation dash", "cmd_punctuation_dash"),
            ("punctuation underscore", "cmd_punctuation_underscore"),
            ("punctuation plus", "cmd_punctuation_plus"),
            ("punctuation equals", "cmd_punctuation_equals"),
            ("punctuation slash", "cmd_punctuation_slash"),
            ("punctuation backslash", "cmd_punctuation_backslash"),
            ("punctuation curly braces", "cmd_punctuation_curly_braces"),
            ("punctuation square brackets", "cmd_punctuation_square_brackets"),
            ("punctuation angle brackets", "cmd_punctuation_angle_brackets"),
            ("punctuation quotes", "cmd_punctuation_quotes"),
            ("punctuation apostrophe", "cmd_punctuation_apostrophe"),
            ("punctuation backtick", "cmd_punctuation_backtick"),
            ("punctuation hash", "cmd_punctuation_hash"),
            ("punctuation at sign", "cmd_punctuation_at_sign"),
            ("punctuation dollar sign", "cmd_punctuation_dollar_sign"),
            ("punctuation exclamation mark", "cmd_punctuation_exclamation_mark"),
            ("punctuation question mark", "cmd_punctuation_question_mark"),
            ("punctuation pipe", "cmd_punctuation_pipe"),
            ("punctuation caret", "cmd_punctuation_caret"),
            ("punctuation tilde", "cmd_punctuation_tilde"),
            ("punctuation tilda", "cmd_punctuation_tilda"),
            ("punctuation underscore", "cmd_punctuation_underscore"),
            ("punctuation ampersand", "cmd_punctuation_ampersand"),
            ("punctuation percent", "cmd_punctuation_percent"),
            ("punctuation asterisk", "cmd_punctuation_asterisk"),
            ("punctuation parentheses", "cmd_punctuation_parentheses"),
            ("punctuation dash", "cmd_punctuation_dash"),
            ("punctuation underscore", "cmd_punctuation_underscore"),
            ("punctuation plus", "cmd_punctuation_plus"),
            ("punctuation equals", "cmd_punctuation_equals"),
            ("punctuation slash", "cmd_punctuation_slash"),
            ("punctuation backslash", "cmd_punctuation_backslash"),
            ("punctuation curly braces", "cmd_punctuation_curly_braces"),
            ("punctuation square brackets", "cmd_punctuation_square_brackets"),
            ("punctuation angle brackets", "cmd_punctuation_angle_brackets"),
        ]
    },
    {
        "label": "Parameterized",
        "commands": [
            ("highlight (word)", "highlight"),
            ("insert after (word)", "insert after"),
            ("insert before (word)", "insert before"),
            ("function f1-f12", "function"),
            ("quote unquote (phrase)", "quote unquote"),
        ]
    },
    {
        "label": "Media",
        "commands": [
            ("play/pause media", "cmd_play_pause_media"),
            ("next track media", "cmd_next_track_media"),
            ("previous track media", "cmd_previous_track_media"),
            ("stop media", "cmd_stop_media"),
            ("volume up media", "cmd_volume_up_media"),
            ("volume down media", "cmd_volume_down_media"),
            ("mute media", "cmd_mute_media"),
            ("unmute media", "cmd_unmute_media"),
        ]
    },
    {
        "label": "Applications",
        "commands": [
            ("open (application name)", "cmd_open_application"),
            ("spotify", "cmd_spotify"),
            ("chrome", "cmd_chrome"),
            ("firefox", "cmd_firefox"),
            ("notepad", "cmd_notepad"),
            ("calculator", "cmd_calculator"),
            ("task manager", "cmd_task_manager"),
            ("settings", "cmd_settings"),
            ("file explorer", "cmd_file_explorer"),
            ("command prompt", "cmd_command_prompt"),
            ("terminal", "cmd_terminal"),
            ("visual studio code", "cmd_vscode"),
            ("word", "cmd_word"),
            ("excel", "cmd_excel"),
            ("powerpoint", "cmd_powerpoint"),
        ]
    },
            # Add more categories as needed
]



# Thresholds (tune as needed)
TOO_QUIET = 2000
TOO_LOUD = 28000
DISTORTION = 32000
NOISE_FLOOR = 1500  # Example: if noise floor is high when not speaking

def send_media_key(key):
    if sys.platform.startswith("win"):
        # Windows: keyboard library supports media keys
        import keyboard
        keyboard.press_and_release(key)
    elif sys.platform.startswith("darwin"):
        # macOS: use AppleScript for media keys
        script_map = {
            "play/pause media": 'tell application "System Events" to key code 16 using {command down}',
            "next track media": 'tell application "System Events" to key code 17 using {command down}',
            "previous track media": 'tell application "System Events" to key code 18 using {command down}',
            "volume up media": 'set volume output volume ((output volume of (get volume settings)) + 10)',
            "volume down media": 'set volume output volume ((output volume of (get volume settings)) - 10)',
            "mute media": 'set volume with output muted',
            "unmute media": 'set volume without output muted',
            "stop media": 'tell application "System Events" to key code 16 using {command down}',  # Play/Pause as fallback
        }
        script = script_map.get(key)
        if script:
            subprocess.call(['osascript', '-e', script])
    elif sys.platform.startswith("linux"):
        # Linux: try playerctl, then xdotool as fallback
        playerctl_map = {
            "play/pause media": "play-pause",
            "next track media": "next",
            "previous track media": "previous",
            "stop media": "stop",
        }
        if key in playerctl_map:
            try:
                subprocess.call(["playerctl", playerctl_map[key]])
                return
            except Exception:
                pass
        # Fallback: try xdotool for XF86 media keys
        xdotool_map = {
            "play/pause media": "XF86AudioPlay",
            "next track media": "XF86AudioNext",
            "previous track media": "XF86AudioPrev",
            "stop media": "XF86AudioStop",
            "volume up media": "XF86AudioRaiseVolume",
            "volume down media": "XF86AudioLowerVolume",
            "mute media": "XF86AudioMute",
        }
        xkey = xdotool_map.get(key)
        if xkey:
            try:
                subprocess.call(["xdotool", "key", xkey])
                return
            except Exception:
                pass
    # Fallback: try spacebar (not global, but works in many players if focused)
    import keyboard
    if key == "play/pause media":
        keyboard.press_and_release("space")


def expand_user_path(path_template):
    username = os.getlogin()
    return path_template.replace("<YourUser>", username)


def is_digit_by_digit_phrase(tokens, digit_words):
    # True if all tokens are digit words and there are at least two tokens
    return len(tokens) > 1 and all(t in digit_words for t in tokens)

def format_number(number_str):
    """Format a number string with commas (e.g., '1247' → '1,247')."""
    try:
        # Only format if not already formatted
        if "," in number_str:
            return number_str
        number = int(number_str)
        return f"{number:,}"
    except ValueError:
        return number_str


def words_to_number(phrase, fractions_map, symbols_map, google_numbers, large_numbers_map, numbers_map):
    """Convert a phrase of number words to a number, handling any size supported by numbers_map and large_numbers_map."""
    if not phrase:
        return 0

    words = phrase.replace("-", " ").split()
    if not words:
        return 0

    # Check if all words are single digits (zero to nine)
    digit_words = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}
    if all(word in digit_words for word in words):
        digit_str = "".join(str(numbers_map[word]) for word in words)
        return int(digit_str) if digit_str else 0

    total = 0
    current = 0
    for word in words:
        if word in numbers_map:
            value = int(numbers_map[word])
        elif large_numbers_map and word in large_numbers_map:
            value = int(large_numbers_map[word])
        else:
            return None
        if value == 100:
            if current == 0:
                current = 1
            current *= value
        elif value >= 1000:
            if current == 0:
                current = 1
            total += current * value
            current = 0
        else:
            current += value
    return total + current


def convert_numbers(phrase, fraction_map, symbols_map, google_numbers, large_numbers_map, numbers_map):
    """Convert a number phrase to a number, handling digit sequences and standard phrases."""
    if large_numbers_map is None:
        raise ValueError("large_numbers_map must be provided!")

    if not isinstance(phrase, str):
        return None
    
    phrase = phrase.lower().strip()
    if not phrase:
        return None

    phrase = re.sub(r'\s+', ' ', phrase).replace(" and ", " ").replace(",", "")

    is_negative = False
    if phrase.startswith("negative") or phrase.startswith("minus"):
        is_negative = True
        phrase = phrase.lstrip("negative").lstrip("minus").strip()
        if not phrase:
            return None

    # Directly handle digit strings (e.g., "100", "1234567")
    if phrase.isdigit():
        number = int(phrase)
        return -number if is_negative else number

    # Check if phrase is a digit-by-digit sequence
    words = phrase.split()
    digit_words = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}
    if all(word in digit_words for word in words):
        number = words_to_number(phrase, fraction_map, symbols_map, google_numbers, large_numbers_map, numbers_map)
        return -number if is_negative else number

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
            section_value = words_to_number(section, fraction_map, symbols_map, google_numbers, large_numbers_map, numbers_map)
            if section_value is None:
                return None
            current_section_value += section_value

    total += current_section_value * current_scale

    return -total if is_negative else total

def parse_number_sequence(words, fraction_map, symbs_map, google_numbers, large_numbers_map, numbers_map):
    """
    Parse a sequence of number words into a formatted number string.
    Returns (number_as_str, words_consumed).
    """
    if not words:
        return None, 0

    is_negative = words[0].lower() in ["negative", "minus"]
    start_idx = 1 if is_negative else 0
    phrase = " ".join(words[start_idx:]).replace("-", " ")

    # Try the whole phrase first
    number = convert_numbers(phrase, fraction_map, symbs_map, google_numbers, large_numbers_map)
    if number is not None:
        try:
            # Try float first, fallback to int if possible
            number_val = float(str(number).replace(",", ""))
            if number_val.is_integer():
                number_val = int(number_val)
        except Exception:
            number_val = str(number)
        if is_negative:
            try:
                number_val = -number_val
            except Exception:
                number_val = f"-{number_val}"
        return str(number_val), len(words)

    # Try shorter sub-phrases (greedy match)
    for i in range(len(words) - start_idx, 0, -1):
        sub_phrase = " ".join(words[start_idx:start_idx + i]).replace("-", " ")
        number = convert_numbers(sub_phrase, fraction_map, symbs_map, google_numbers, large_numbers_map, numbers_map)
        if number is not None:
            try:
                number_val = float(str(number).replace(",", ""))
                if number_val.is_integer():
                    number_val = int(number_val)
            except Exception:
                number_val = str(number)
            if is_negative:
                try:
                    number_val = -number_val
                except Exception:
                    number_val = f"-{number_val}"
            return str(number_val), start_idx + i

    return None, 0


def convert_numbers_chained(text, fraction_map, symbs_map, google_numbers, large_numbers_map, numbers_map):
    """Process chained input, converting number word and percent sequences to formatted numbers."""
    text = re.sub(r'\s+', ' ', text.lower().strip()).replace(" and ", " ").replace("-", " ")
    words = text.split()
    result = []
    i = 0

    while i < len(words):
        # Handle percent phrases
        if (
            i + 1 < len(words)
            and (words[i] in numbers_map or (large_numbers_map and words[i] in large_numbers_map) or words[i].isdigit())
            and words[i + 1] in ("percent", "%")
        ):
            number = convert_numbers(words[i], fraction_map, symbs_map, google_numbers, large_numbers_map)
            if number is not None:
                result.append(f"{number}%")
                i += 2
                continue

        if words[i] in numbers_map or (large_numbers_map and words[i] in large_numbers_map):
            number_words = []
            while i < len(words) and (words[i] in numbers_map or (large_numbers_map and words[i] in large_numbers_map)):
                number_words.append(words[i])
                i += 1
            number_str, words_consumed = parse_number_sequence(number_words, fraction_map, symbs_map, google_numbers, large_numbers_map)
            if number_str:
                result.append(number_str)
            else:
                result.extend(number_words)
        else:
            result.append(words[i])
            i += 1

    return " ".join(result)

def convert_percent_phrase(phrase, *args, **kwargs):
    """
    Convert phrases like 'one percent', '100 percent', 'ten %' to '1%', '100%', '10%' etc.
    """
    phrase = phrase.strip()
    match = re.match(r"(.+?)\s*(percent|%)$", phrase, re.IGNORECASE)
    if match:
        number_part = match.group(1).strip()
        number = convert_numbers(number_part, *args, **kwargs)
        if number is not None:
            return f"{number}%"
    return None

class AudioProcessor:
    def __init__(self, sample_rate, bit_depth, channels, device_index, pre_scale_factor, silence_amplitude_threshold, relative_sensitivity):
        self.sample_rate = sample_rate
        self.bit_depth = bit_depth
        self.channels = channels
        self.device_index = device_index
        self.pre_scale_factor = pre_scale_factor
        self.silence_amplitude_threshold = silence_amplitude_threshold
        self.relative_sensitivity = relative_sensitivity
        self.q = queue.Queue()
        self.audio_buffer = []
        self.is_running = False
        self.stream = None

    def get_dtype_and_max(self):
        if self.bit_depth == 16:
            return np.int16, 32767
        elif self.bit_depth == 24 or self.bit_depth == 32:
            return np.int32, 8388607 if self.bit_depth == 24 else 2147483647

    def callback(self, indata, frames, time, status):
        if status:
            print(f"Audio callback status: {status}", file=sys.stderr)

        dtype, max_value = self.get_dtype_and_max()
        indata_array = np.frombuffer(indata, dtype=dtype)
        indata_normalized = indata_array.astype(np.float32) / (max_value + 1)

        if self.relative_sensitivity:
            reference_max = 32767
            scale_factor = reference_max / (max_value + 1)
            adjusted_pre_scale = self.pre_scale_factor * scale_factor
        else:
            adjusted_pre_scale = self.pre_scale_factor

        indata_normalized = indata_normalized * adjusted_pre_scale
        indata_array = np.clip(indata_normalized * 32767, -32768, 32767).astype(np.int16)

        if self.channels > 1:
            indata_array = indata_array.reshape(-1, self.channels).mean(axis=1).astype(np.int16)

        if self.sample_rate != VOSK_SAMPLERATE:
            num_samples_resampled = int(len(indata_array) * VOSK_SAMPLERATE / self.sample_rate)
            indata_array = resample(indata_array, num_samples_resampled)
            indata_array = indata_array.astype(np.int16)

        max_amplitude = np.max(np.abs(indata_array))
        if max_amplitude < self.silence_amplitude_threshold:
            print(f"Block below silence threshold (max amplitude: {max_amplitude})")
            return

        self.audio_buffer.append(indata_array)
        indata_bytes = indata_array.tobytes()
        self.q.put(indata_bytes)

    def start_stream(self):
        self.is_running = True
        dtype, _ = self.get_dtype_and_max()
        self.stream = sd.RawInputStream(
            samplerate=self.sample_rate, blocksize=BLOCKSIZE, dtype="int32",
            channels=self.channels, callback=self.callback, device=self.device_index
        )
        self.stream.start()

    def stop_stream(self):
        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()

    def get_audio_data(self):
        try:
            return self.q.get(timeout=1.0)
        except queue.Empty:
            print("Warning: Audio queue empty", file=sys.stderr)
            return None

    def list_devices(self):
        print("Available input audio devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"{i}: {device['name']}, {device['max_input_channels']} in")

class WaveformDisplay:
    def __init__(self, target_fps=30):
        self.is_running = False
        self.buffer_size = 22050
        self.buffer_index = 0
        self.current_value = 0.0

        self.display_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.downsample_factor = 10
        self.plot_samples = self.buffer_size // self.downsample_factor
        self.time_axis = np.linspace(0, 0.5, self.plot_samples)
        self.plot_data = np.zeros(self.plot_samples)

        self.win = pg.GraphicsLayoutWidget(title="Waveform Display")
        self.win.resize(800, 300)
        self.plot = self.win.addPlot(title="Waveform (Simulated)", labels={'left': 'Value', 'bottom': 'Time (s)'})
        self.plot.setYRange(-10000, 10000)
        self.plot.setXRange(0, 0.5)
        self.curve = self.plot.plot(self.time_axis, self.plot_data, pen=pg.mkPen('g'))
        self.win.show()

        self.is_running = True
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(int(1000 / target_fps))

    def update_plot(self):
        if not self.is_running:
            return
        try:
            t = np.linspace(0, 2 * np.pi, self.buffer_size)
            simulated_data = 10000 * np.sin(t + self.buffer_index * 0.1)
            self.display_buffer = simulated_data.astype(np.float32)
            self.buffer_index = (self.buffer_index + 1) % self.buffer_size

            shifted_buffer = np.roll(self.display_buffer, -self.buffer_index)
            self.plot_data = shifted_buffer[::self.downsample_factor]
            self.curve.setData(self.time_axis, self.plot_data)

            self.current_value = 10000 * abs(np.sin(self.buffer_index * 0.05))
        except Exception as e:
            print(f"Error updating waveform: {e}")

    def get_display_value(self):
        return self.current_value

    def close(self):
        self.is_running = False
        self.win.close()

class VoskModel:
    def __init__(self, audio_processor, model_path, sample_rate, bit_depth, pre_scale_factor, silence_threshold, relative_sensitivity):
        self.model = None
        self.audio_processor = audio_processor
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.vosk_sample_rate = VOSK_SAMPLERATE
        self.bit_depth = bit_depth
        self.pre_scale_factor = pre_scale_factor
        self.silence_threshold = silence_threshold
        self.relative_sensitivity = relative_sensitivity
        #self.wav_file = WAV_FILE
        self.transcription_file = TRANSCRIPTION_FILE
        self.last_word_end_time = 0.0
        self.last_command = None
        self.last_command_time = 0.0
        self.skip_dictation = False
        self.last_dictated_text = ""
        self.last_dictated_length = 0
        self.caps_lock_on = False
        self.number_lock_on = False
        self.last_processed_command = None
        self.spell = SpellChecker()
        

        # Hardcoded maps
        self.fraction_map = {
            "one half": "½", "half": "½", "one third": "⅓", "one fifth": "⅕",
            "one sixth": "⅙", "one eighth": "⅛", "two thirds": "⅔", "two fifths": "⅖",
            "five sixths": "⅚", "three eighths": "⅜", "three fourths": "¾",
            "three quarters": "¾", "three fifths": "⅗", "five eighths": "⅝",
            "seven eighths": "⅞", "four fifths": "⅘", "one fourth": "¼",
            "one quarter": "¼", "quarter": "¼", "quarter inch": "1/4 inch",
            "one seventh": "1/7", "one ninth": "1/9", "one tenth": "1/10",
            "zero thirds": "0/3", "zero over zero": "0/0", "percent zero": "%0",
            "percent zero zero": "%00", "percent zero zero zero": "%000",
            "three over five": "3/5", "five over eight": "5/8",
            "seven over eight": "7/8", "one over one": "1/1"
        }
        self.symbols_map = {
            "dollar sign": "$", "percent": "%", "ampersand": "&", "at sign": "@",
            "hash": "#", "exclamation mark": "!", "question mark": "?",
            "asterisk": "*", "plus sign": "+", "minus sign": "-", "equals sign": "=",
            "slash": "/", "backslash": "\\", "pipe": "|", "tilde": "~",
            "caret": "^", "underscore": "_", "left parenthesis": "(",
            "right parenthesis": ")", "left bracket": "[", "right bracket": "]",
            "left brace": "{", "right brace": "}", "less than": "<",
            "greater than": ">", "colon": ":", "semicolon": ";",
            "single quote": "'", "double quote": "\"", "backtick": "`"
        }
        self.numbers_map = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
            "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
            "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
            "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
            "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
            "eighty": "80", "ninety": "90", "hundred": "100"
        }
        self.google_numbers = {
            "googol": "10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
            "googolplex": "10^10^100"
        }
        self.large_numbers_map = {
            "thousand": "1000", "million": "1000000",
            "billion": "1000000000", "trillion": "1000000000000",
            "quadrillion": "1000000000000000", "quintillion": "1000000000000000000",
            "sextillion": "1000000000000000000000", "septillion": "1000000000000000000000000",
            "octillion": "1000000000000000000000000000",
            "nonillion": "1000000000000000000000000000000",
            "decillion": "1000000000000000000000000000000000",
            "undecillion": "1000000000000000000000000000000000000",
            "duodecillion": "1000000000000000000000000000000000000000",
            "tredecillion": "1000000000000000000000000000000000000000000",
            "quattuordecillion": "1000000000000000000000000000000000000000000000",
            "quindecillion": "1000000000000000000000000000000000000000000000000",
            "sexdecillion": "1000000000000000000000000000000000000000000000000000",
            "septendecillion": "1000000000000000000000000000000000000000000000000000000",
            "octodecillion": "1000000000000000000000000000000000000000000000000000000000",
            "novemdecillion": "1000000000000000000000000000000000000000000000000000000000000",
            "vigintillion": "1000000000000000000000000000000000000000000000000000000000000000",
            "centillion": "1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
        }
        self.simple_commands = {
            "new paragraph": "cmd_new_paragraph", "new line": "cmd_new_line",
            "space": "cmd_space", "tab": "cmd_tab", "undo": "cmd_undo",
            "redo": "cmd_redo", "caps lock on": "cmd_caps_lock_on",
            "caps lock off": "cmd_caps_lock_off", "number lock": "cmd_number_lock",
            "stop listening": "cmd_stop_listening", "clear": "cmd_clear",
            "select all": "cmd_select_all", "select down": "cmd_select_down",
            "select up": "cmd_select_up", "select all up": "cmd_select_all_up",
            "select all down": "cmd_select_all_down", "copy": "cmd_copy",
            "paste": "cmd_paste", "delete": "cmd_delete",
            "file properties": "cmd_file_properties", "save document": "cmd_save_document",
            "open file": "cmd_open_file", "move up": "cmd_move_up",
            "move down": "cmd_move_down", "move left": "cmd_move_left",
            "move right": "cmd_move_right", "move up paragraph": "cmd_move_up_paragraph",
            "move down paragraph": "cmd_move_down_paragraph", "enter": "cmd_enter",
            "bold": "cmd_bold", "italicize": "cmd_italicize", "underline": "cmd_underline",
            "center": "cmd_center", "left align": "cmd_left_align",
            "right align": "cmd_right_align", "cut": "cmd_cut",
            "go to beginning": "cmd_go_to_beginning", "go to end": "cmd_go_to_end",
            "go to beginning of line": "cmd_go_to_beginning_of_line",
            "go to end of line": "cmd_go_to_end_of_line", "go to address": "cmd_go_to_address",
            "refresh page": "cmd_refresh_page", "go back": "cmd_go_back",
            "go forward": "cmd_go_forward", "open new tab": "cmd_open_new_tab",
            "close tab": "cmd_close_tab", "next tab": "cmd_next_tab",
            "previous tab": "cmd_previous_tab", "shift tab": "cmd_shift_tab",
            "switch windows": "cmd_switch_windows", "next window": "cmd_next_window",
            "previous window": "cmd_previous_window", "switch application": "cmd_switch_application",
            "next application": "cmd_next_application", "previous application": "cmd_previous_application",
            "switch window": "cmd_switch_window", "next window": "cmd_next_window",
            "scratch that": "cmd_scratch_that", "click that": "cmd_click_that",
            "screen shoot": "cmd_screen_shoot", "screen shoot window": "cmd_screen_shoot_window",
            "screen shoot monitor": "cmd_screen_shoot_monitor", "task manager": "cmd_task_manager",
            "debug screen": "cmd_debug_screen", "force close": "cmd_force_close",
            "punctuation period": "cmd_punctuation_period",
            "punctuation comma": "cmd_punctuation_comma",
            "punctuation question mark": "cmd_punctuation_question_mark",
            "punctuation exclamation": "cmd_punctuation_exclamation",
            "punctuation semicolon": "cmd_punctuation_semicolon",
            "punctuation colon": "cmd_punctuation_colon",
            "punctuation tilde": "cmd_punctuation_tilde",
            "punctuation ampersand": "cmd_punctuation_ampersand",
            "punctuation percent": "cmd_punctuation_percent",
            "punctuation asterisk": "cmd_punctuation_asterisk",
            "punctuation parentheses": "cmd_punctuation_parentheses",
            "punctuation dash": "cmd_punctuation_dash",
            "punctuation underscore": "cmd_punctuation_underscore",
            "punctuation plus": "cmd_punctuation_plus",
            "punctuation equals": "cmd_punctuation_equals",
            "press escape": "cmd_press_escape",
            #application specific commands
            "open spotify": "cmd_open_spotify",
            "play spotify": "cmd_play_spotify",
            "pause spotify": "cmd_pause_spotify",
            "next spotify": "cmd_next_spotify",
            "previous spotify": "cmd_previous_spotify",
            "volume up spotify": "cmd_volume_up_spotify",
            "volume down spotify": "cmd_volume_down_spotify",
            "mute spotify": "cmd_mute_spotify",
            "unmute spotify": "cmd_unmute_spotify",
            "search spotify": "cmd_search_spotify",
            "open youtube": "cmd_open_youtube",
            "play youtube": "cmd_play_youtube",
            "pause youtube": "cmd_pause_youtube",
            "next youtube": "cmd_next_youtube",
            "previous youtube": "cmd_previous_youtube",
            "volume up youtube": "cmd_volume_up_youtube",
            "volume down youtube": "cmd_volume_down_youtube",
            "mute youtube": "cmd_mute_youtube",
            "unmute youtube": "cmd_unmute_youtube",
            "search youtube": "cmd_search_youtube",
            "open google": "cmd_open_google",
            "search google": "cmd_search_google",
            "open notepad": "cmd_open_notepad",
            "open calculator": "cmd_open_calculator",
            "open word": "cmd_open_word",
            "open excel": "cmd_open_excel",
            "open powerpoint": "cmd_open_powerpoint",
            "open browser": "cmd_open_browser",
            "open notepad": "cmd_open_notepad",
            "open calculator": "cmd_open_calculator",
            "open notepad plus plus": "cmd_open_notepad_plus_plus",
            "open visual studio code": "cmd_open_visual_studio_code",
            "open firefox": "cmd_open_firefox",
            "open chrome": "cmd_open_chrome", "open edge": "cmd_open_edge",
            "open opera": "cmd_open_opera", "open safari": "cmd_open_safari",
            "open internet explorer": "cmd_open_internet_explorer",
            "open file explorer": "cmd_open_file_explorer",
            "open control panel": "cmd_open_control_panel",
            "open command prompt": "cmd_open_command_prompt",
            "open powershell": "cmd_open_powershell",
            "open terminal": "cmd_open_terminal",
            "open task manager": "cmd_open_task_manager",
            "open settings": "cmd_open_settings", "open run": "cmd_open_run",
            "open start menu": "cmd_open_start_menu", "open taskbar": "cmd_open_taskbar",
            "open steam": "cmd_open_steam",
        }
        self.parameterized_partial = [
        ("highlight", "highlight"), ("find", "find"), ("insert after", "insert after"),
        ("insert before", "insert before"), ("copy", "copy"), ("cut", "cut"),
        ("all caps", "all caps"), ("press", "press"), ("open", "open"),
        ("go to address", "go to address"), ("move up", "move up"),
        ("move down", "move down"), ("move left", "move left"),
        ("move right", "move right"), ("function", "function")
        ]

        self.parameterized_final = [
            ("quote unquote ", "quote unquote ")
        ]
        self.large_number_values = set(str(int(v)) for v in self.large_numbers_map.values())
        # Tokenized commands
        self.tokenized_simple_commands = {}
        for cmd, action in self.simple_commands.items():
            normalized_cmd = self.normalize_text(cmd.lower())
            tokens = tuple(nltk.word_tokenize(normalized_cmd))
            self.tokenized_simple_commands[tokens] = (cmd, action)

        self.tokenized_parameterized_partial = []
        for cmd, cmd_str in self.parameterized_partial:
            normalized_cmd = self.normalize_text(cmd.lower())
            tokens = tuple(nltk.word_tokenize(normalized_cmd))
            self.tokenized_parameterized_partial.append((tokens, cmd_str))

        self.tokenized_parameterized_final = []
        for cmd, cmd_str in self.parameterized_final:
            normalized_cmd = self.normalize_text(cmd.lower())
            tokens = tuple(nltk.word_tokenize(normalized_cmd))
            self.tokenized_parameterized_final.append((tokens, cmd_str))

        # Load model
        #self.model = self.load_model()

    def normalize_text(self, text):
        text = text.replace("-", " ")
        return " ".join(text.split())

    def normalize_and_tokenize(self, text):
        return tuple(nltk.word_tokenize(self.normalize_text(text.lower())))




    def process_text(self, text):
        if not text:
            return ""
        # Replace only number words/phrases, not the whole text
        text = self.replace_number_words(text)
        words = text.split()
        if not words:
            return ""
        if self.caps_lock_on:
            words = [word.upper() for word in words]
        else:
            words[0] = words[0][0].upper() + words[0][1:] if len(words[0]) > 1 else words[0].upper()
        processed_text = " ".join(words)
        return processed_text


    def replace_number_words(self, text):
        if not text:
            return ""
        # Join numbers split by commas (e.g., "12 , 345" -> "12,345")
        text = re.sub(r'(\d+)\s*,\s*(\d+)', r'\1,\2', text)
        tokens = nltk.word_tokenize(text)
        i = 0
        result = []
        # Regex to match formatted numbers (e.g., "1,000", "-12,345,678")
        formatted_number_pattern = r'^-?\d{1,3}(,\d{3})*$'

        while i < len(tokens):
            # Check if the token is an already formatted number
            if re.match(formatted_number_pattern, tokens[i]):
                print(f"Preserving formatted number: {tokens[i]}")  # Debug
                # Check for percent sign immediately after
                if i + 1 < len(tokens) and tokens[i + 1].lower() in ("percent", "%"):
                    result.append(tokens[i] + "%")
                    i += 2
                else:
                    result.append(tokens[i])
                    i += 1
                continue

            # Try to parse the longest possible number phrase
            max_len = len(tokens) - i
            found = False
            for l in range(max_len, 0, -1):
                # Stop at "and" to segment phrases
                if l > 1 and tokens[i + l - 1].lower() == "and":
                    continue
                phrase = " ".join(tokens[i:i+l])
                print(f"Processing phrase: '{phrase}'")  # Debug
                number = convert_numbers(
                    phrase, self.fraction_map, self.symbols_map,
                    self.google_numbers, self.large_numbers_map, self.numbers_map
                )
                if number is not None:
                    formatted = str(number)
                    # Log map usage
                    used_numbers_map = any(t in self.numbers_map for t in tokens[i:i+l])
                    used_large_numbers_map = uses_large_numbers(tokens[i:i+l], self.large_numbers_map)
                    print(f"Used numbers_map: {used_numbers_map}, Used large_numbers_map: {used_large_numbers_map}, Tokens: {tokens[i:i+l]}")  # Debug
                    # Format with commas only if large number words are used
                    if used_large_numbers_map:
                        formatted = format_number(formatted)
                        print(f"Formatted with commas: {formatted}")  # Debug
                    else:
                        print(f"Formatted without commas: {formatted}")  # Debug
                    # Check for percent sign immediately after
                    if i + l < len(tokens) and tokens[i + l].lower() in ("percent", "%"):
                        result.append(formatted + "%")
                        i += l + 1
                    else:
                        result.append(formatted)
                        i += l
                    # If next token is "and", append it
                    if i < len(tokens) and tokens[i].lower() == "and":
                        result.append("and")
                        i += 1
                    found = True
                    break
            if found:
                continue

            # Handle just percent or %
            if tokens[i].lower() in ("percent", "%"):
                result.append("%")
                i += 1
                continue

            # Default: append the token
            result.append(tokens[i])
            i += 1

        # Join and fix contractions
        text = " ".join(result)
        text = text.replace(" n't", "n't").replace(" 's", "'s").replace(" 'm", "'m") \
                   .replace(" 're", "'re").replace(" 've", "'ve").replace(" 'll", "'ll").replace(" 'd", "'d")
        print(f"Final output: {text}")  # Debug
        return text

   









    def handle_special_phrases(self, text):
        return self.replace_number_words(text)

    def load_model(self):
        update_status("Loading Vosk model...")
        if not os.path.exists(self.model_path):
            update_status(f"Vosk model not found at {self.model_path}")
            sys.exit(1)
        model = vosk.Model(self.model_path)
        update_status("Vosk model loaded successfully.")
        return model


class GUI:
    def __init__(self, audio_processor, model_path, sample_rate, bit_depth, pre_scale_factor, silence_threshold, relative_sensitivity, vosk_model):
        self.waveform_display = None
        self.waveform_process = None
        self.current_value = 0.0
        self.dpg_running = False
        self.is_testing = False
        self.is_recording = False
        self.is_debug_recording = False
        self.audio_buffer = []
        self.debug_audio_buffer = []
        self.recordings_dir = "recordings"
        self.default_recordings_dir = "debug_recordings"
        self.noise_floor = 0
        self.peak_amplitude = 0
        self.last_noise_update = 0
        self.audio_stream = None
        self.output_stream = None
        self.audio_queue = queue.Queue()
        self.gui_update_queue = queue.Queue()
        self.is_dictating = False
        self.dictation_stream = None
        self.command_queue = queue.Queue()
        self.transcribed_text = []
        self.last_partial = ""
        self.vosk_model = vosk_model
        self.audio_processor = audio_processor
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.bit_depth = bit_depth
        self.pre_scale_factor = pre_scale_factor
        self.silence_threshold = silence_threshold
        self.relative_sensitivity = relative_sensitivity
        self.stop_transcription = False
        self.custom_commands = []  # List of dicts: {"phrase": ..., "action": ...}
        self.custom_commands_file = "custom_commands.json"
        self.load_custom_commands()
        self.settings_file = "settings.json"


        dpg.create_context()
        dpg.create_viewport(title="Configuration", width=800, height=600)

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

        with dpg.window(label="Configuration", tag="primary_window"):
            with dpg.tab_bar(tag="main_tab_bar"):
                with dpg.tab(label="Settings", tag="settings_tab"):
                    self.setup_settings_tab()
                # Add command tabs
                for tab in COMMAND_TABS:
                    with dpg.tab(label=tab["label"]):
                        dpg.add_text(f"{tab['label']} Commands:")
                        for cmd, action in tab["commands"]:
                            dpg.add_text(f"_ {cmd}")

                with dpg.tab(label="Custom Commands", tag="custom_commands_tab"):
                    dpg.add_text("Add or edit your custom voice commands below.")
                    dpg.add_input_text(label="Phrase", tag="custom_phrase_input", width=250)
                    dpg.add_input_text(label="Hotkey/Program", tag="custom_action_input", width=250)
                    dpg.add_button(label="Add/Update Command", callback=self.add_or_update_custom_command)
                    dpg.add_button(label="Remove Selected", callback=self.remove_selected_custom_command)
                    dpg.add_listbox([], tag="custom_commands_listbox", width=520, num_items=8, callback=self.on_custom_command_select)

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
            "output_device": None, "model_path": self.model_path
        }
        # Try to load from file
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r", encoding="utf-8") as f:
                    self.saved_settings = json.load(f)
            except Exception:
                self.saved_settings = default_settings
        else:
            self.saved_settings = default_settings

        dpg.set_value("bit_depth_combo", self.saved_settings["bit_depth"])
        dpg.set_value("sample_rate_combo", self.saved_settings["sample_rate"])
        dpg.set_value("unit_combo", self.saved_settings["unit"])
        self.set_slider_from_pre_scale(self.saved_settings["pre_scale_factor"])
        dpg.set_value("relative_sensitivity_check", self.saved_settings["relative_sensitivity"])
        dpg.set_value("silence_input", self.saved_settings["silence_threshold"])
        dpg.set_value("show_peaks_check", self.saved_settings["show_peaks"])
        self.update_show_peaks(None, None)
        dpg.set_value("theme_combo", self.saved_settings["theme"])
        dpg.set_value("host_api_combo", self.saved_settings["host_api"])
        dpg.set_value("input_device_combo", self.saved_settings["input_device"])
        dpg.set_value("output_device_combo", self.saved_settings.get("output_device", ""))
        dpg.set_value("model_path_input", self.saved_settings.get("model_path", self.model_path))
        self.update_host_api(None, self.saved_settings["host_api"])


    def save_settings(self, sender, app_data):
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
            "output_device": dpg.get_value("output_device_combo"),
            "model_path": dpg.get_value("model_path_input")
        }
        # Save to file
        try:
            with open(self.settings_file, "w", encoding="utf-8") as f:
                json.dump(self.saved_settings, f, indent=2)
            dpg.set_value("status_text", "Settings saved successfully.")
        except Exception as e:
            dpg.set_value("status_text", f"Failed to save settings: {e}")


    def load_custom_commands(self):
        if os.path.exists(self.custom_commands_file):
            try:
                with open(self.custom_commands_file, "r", encoding="utf-8") as f:
                    self.custom_commands = json.load(f)
            except Exception:
                self.custom_commands = []
        else:
            self.custom_commands = []

    def save_custom_commands(self):
        try:
            with open(self.custom_commands_file, "w", encoding="utf-8") as f:
                json.dump(self.custom_commands, f, indent=2)
        except Exception as e:
            dpg.set_value("status_text", f"Failed to save custom commands: {e}")


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
            host_apis = [host_api['name'] for host_api in sd.query_hostapis()]
            return host_apis if host_apis else ["No Host APIs"]
        except Exception as e:
            logging.error(f"Failed to query host APIs: {e}")
            return ["MME"]

    def get_input_devices(self, host_api_name):
        try:
            host_api_index = next((i for i, h in enumerate(sd.query_hostapis()) if h['name'] == host_api_name), None)
            if host_api_index is None:
                return ["No Input Devices"]
            devices = [d['name'] for d in sd.query_devices() if d['hostapi'] == host_api_index and d['max_input_channels'] > 0]
            return devices if devices else ["No Input Devices"]
        except Exception as e:
            logging.error(f"Failed to get input devices: {e}")
            return ["No Input Devices"]

    def get_output_devices(self, host_api_name):
        try:
            host_api_index = next((i for i, h in enumerate(sd.query_hostapis()) if h['name'] == host_api_name), None)
            if host_api_index is None:
                return ["No Output Devices"]
            devices = [d['name'] for d in sd.query_devices() if d['hostapi'] == host_api_index and d['max_output_channels'] > 0]
            return devices if devices else ["No Output Devices"]
        except Exception as e:
            logging.error(f"Failed to get output devices: {e}")
            return ["No Output Devices"]

    def get_device_index(self, device_name, host_api_name, is_input=True):
        if not device_name or device_name in ["No Input Devices", "No Output Devices"]:
            return None
        try:
            host_api_index = next((i for i, h in enumerate(sd.query_hostapis()) if h['name'] == host_api_name), None)
            if host_api_index is None:
                return None
            for i, d in enumerate(sd.query_devices()):
                if d['name'] == device_name and d['hostapi'] == host_api_index:
                    if (is_input and d['max_input_channels'] > 0) or (not is_input and d['max_output_channels'] > 0):
                        return i
            return None
        except Exception as e:
            logging.error(f"Failed to get device index: {e}")
            return None

    def update_host_api(self, sender, app_data):
        host_api = app_data
        input_devices = self.get_input_devices(host_api)
        dpg.configure_item("input_device_combo", items=input_devices)
        if input_devices and input_devices[0] != "No Input Devices":
            if not dpg.get_value("input_device_combo") or dpg.get_value("input_device_combo") not in input_devices:
                dpg.set_value("input_device_combo", input_devices[0])
        else:
            dpg.set_value("input_device_combo", input_devices[0])
        self.update_device(None, dpg.get_value("input_device_combo"))
        output_devices = self.get_output_devices(host_api)
        dpg.configure_item("output_device_combo", items=output_devices)
        if output_devices and output_devices[0] != "No Output Devices":
            if not dpg.get_value("output_device_combo") or dpg.get_value("output_device_combo") not in output_devices:
                dpg.set_value("output_device_combo", output_devices[0])
        else:
            dpg.set_value("output_device_combo", output_devices[0])
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
        dpg.configure_item("sample_rate_combo", items=SAMPLE_RATES)
        if dpg.get_value("sample_rate_combo") not in supported_sample_rates:
            dpg.set_value("sample_rate_combo", max(supported_sample_rates, key=int))
        if self.is_testing:
            self.stop_audio_test()
            self.start_audio_test()

    def update_output_device(self, sender, app_data):
        pass

    def update_bit_depth(self, sender, app_data):
        dpg.set_value("bit_depth_tooltip", DATA_TYPES[app_data])
        if self.is_testing:
            self.stop_audio_test()
            self.start_audio_test()

    def update_sample_rate(self, sender, app_data):
        if self.is_testing:
            self.stop_audio_test()
            self.start_audio_test()

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

    def update_shadow(self):
        shadow_position = (self.noise_floor / 32767) * 400
        if dpg.does_item_exist("shadow_bar"):
            dpg.configure_item("shadow_bar", pmax=(shadow_position, 20))

    def update_show_peaks(self, sender, app_data):
        show = dpg.get_value("show_peaks_check")
        dpg.configure_item("shadow_bar", show=show)

    def update_gui(self):
        if dpg.is_dearpygui_running():
            self.update_shadow()
            dpg.set_frame_callback(dpg.get_frame_count() + 1, self.update_gui)

    def toggle_audio_test(self, sender, app_data):
        if not self.is_testing:
            self.is_testing = True
            dpg.configure_item("test_audio_button", label="Stop Testing")
            self.start_audio_test()
        else:
            self.is_testing = False
            dpg.configure_item("test_audio_button", label="Test Audio")
            self.stop_audio_test()

    def toggle_recording(self, sender, app_data):
        if not self.is_recording:
            if not self.is_testing:
                dpg.set_value("status_text", "Start audio testing before recording.")
                return
            if not os.path.exists(self.recordings_dir):
                os.makedirs(self.recordings_dir)
            self.is_recording = True
            dpg.configure_item("record_button", label="Stop Recording")
            self.audio_buffer = []
        else:
            self.is_recording = False
            dpg.configure_item("record_button", label="Record")
            self.save_recording()

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
        if not os.path.exists(self.default_recordings_dir):
            os.makedirs(self.default_recordings_dir)
        filename = os.path.join(self.default_recordings_dir, f"debug_recording_{int(time.time())}.wav")
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(sample_width)
            wf.setframerate(int(dpg.get_value("sample_rate_combo")))
            wf.writeframes(audio_data.tobytes())
        logging.info(f"Debug recording saved as {filename}")

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
        target_amplitude = 0.5
        pre_scale = target_amplitude / (peak_amplitude + 1e-10)
        self.set_slider_from_pre_scale(pre_scale)
        dpg.set_value("status_text", f"Sensitivity adjusted to {dpg.get_value('pre_scale_label')}.")

    def suggest_settings(self, sender, app_data):
        db_level = dpg.get_value("db_level_input")
        pre_scale = self.db_to_pre_scale(db_level)
        self.set_slider_from_pre_scale(pre_scale)
        dpg.set_value("status_text", f"Sensitivity set based on {db_level} dB input level.")

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
        dtype, max_value = self.get_dtype_and_max()
        self.noise_floor = 0
        self.peak_amplitude = 0
        self.last_noise_update = time.time()
        self.output_buffer = []
        self.gui_update_queue = queue.Queue()
        self.audio_queue = queue.Queue()

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
                pre_scale = self.db_to_pre_scale(db)
            indata_scaled = indata_left * pre_scale
            indata_normalized = indata_scaled
            indata_left_scaled = np.clip(indata_normalized * 32767, -32768, 32767).astype(np.int16)
            scaled_data = (indata_scaled * 32767).astype(np.int16)
            # Detect likely distortion (clipping)
            clip_threshold = 0.99  # 99% of max value
            clip_count = np.sum(np.abs(scaled_data) >= int(32767 * clip_threshold))
            clip_ratio = clip_count / len(scaled_data)

            if clip_ratio > 0.01:  # More than 1% of samples are clipped
                dpg.set_value("audio_warning_text", "audio distortion due to clipping")

            max_amplitude = np.max(np.abs(scaled_data))
            self.gui_update_queue.put(("update_level", max_amplitude))
            #Audio monitoring warnings
            if dpg.get_value("debug_recording_check"):
                if max_amplitude < 2000:
                    dpg.set_value("audio_warning_text", "too quiet")
                elif max_amplitude > 32000:
                    dpg.set_value("audio_warning_text", "audio distortion due to too loud")
                elif max_amplitude > 28000:
                    dpg.set_value("audio_warning_text", "too loud")
                elif not self.is_recording and self.noise_floor > 1500:
                    dpg.set_value("audio_warning_text", "find a quieter place if background noise. else program is ready")
                else:
                    dpg.set_value("audio_warning_text", "")
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
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None
        if self.output_stream:
            self.output_stream.stop()
            self.output_stream.close()
            self.output_stream = None
        dpg.configure_item("level_bar", pmax=(0, 20))
        dpg.configure_item("clipping_indicator", fill=(255, 0, 0, 128))
        dpg.set_value("audio_warning_text", "")

    def setup_settings_tab(self):
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
            dpg.add_input_text(tag="model_path_input", default_value=self.model_path, width=300, readonly=True)
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
        dpg.add_checkbox(label="Show Peaks", default_value=False, tag="show_peaks_check", callback=self.update_show_peaks)
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
        dpg.add_text("", tag="audio_warning_text")



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

    def add_or_update_custom_command(self, sender, app_data):
        phrase = dpg.get_value("custom_phrase_input").strip()
        action = dpg.get_value("custom_action_input").strip()
        if not phrase or not action:
            dpg.set_value("status_text", "Both phrase and action are required.")
            return
        # Update if exists, else add
        for cmd in self.custom_commands:
            if cmd["phrase"].lower() == phrase.lower():
                cmd["action"] = action
                break
        else:
            self.custom_commands.append({"phrase": phrase, "action": action})
        self.save_custom_commands()
        self.refresh_custom_commands_listbox()
        dpg.set_value("status_text", f"Custom command '{phrase}' saved.")

    def remove_selected_custom_command(self, sender, app_data):
        selected = dpg.get_value("custom_commands_listbox")
        if not selected:
            dpg.set_value("status_text", "No command selected.")
            return
        phrase = selected[0].split(" => ")[0]
        self.custom_commands = [cmd for cmd in self.custom_commands if cmd["phrase"] != phrase]
        self.save_custom_commands()
        self.refresh_custom_commands_listbox()
        dpg.set_value("status_text", f"Custom command '{phrase}' removed.")

    def refresh_custom_commands_listbox(self):
        items = [f"{cmd['phrase']} => {cmd['action']}" for cmd in self.custom_commands]
        dpg.configure_item("custom_commands_listbox", items=items)

    def on_custom_command_select(self, sender, app_data):
        selected = dpg.get_value("custom_commands_listbox")
        if selected:
            phrase, action = selected[0].split(" => ", 1)
            dpg.set_value("custom_phrase_input", phrase)
            dpg.set_value("custom_action_input", action)


    def set_model_path(self, sender, app_data):
        # Launch the dialog as a separate process
        import subprocess
        subprocess.Popen(['python', 'model_path_dialog_v2.py'])
        dpg.set_value("status_text", "Model path dialog opened in a new window. Select a folder and close the dialog.")

        # Optionally, poll for the result in a background thread or on a timer
        def poll_for_model_path():
            import time
            for _ in range(100):  # Wait up to 10 seconds
                if os.path.exists("selected_model_path.txt"):
                    with open("selected_model_path.txt", "r", encoding="utf-8") as f:
                        path = f.read().strip()
                    dpg.set_value("model_path_input", path)
                    self.saved_settings["model_path"] = path
                    dpg.set_value("status_text", f"Model path set to: {path}")
                    os.remove("selected_model_path.txt")
                    return
                time.sleep(0.1)
        threading.Thread(target=poll_for_model_path, daemon=True).start()


    def on_model_path_selected(self, sender, app_data):
        selected_path = app_data["file_path_name"]
        if not selected_path:
            dpg.set_value("status_text", "No directory selected.")
            return
        absolute_path = os.path.abspath(selected_path)
        dpg.set_value("model_path_input", absolute_path)
        self.saved_settings["model_path"] = absolute_path
        dpg.set_value("status_text", f"Model path set to: {absolute_path}")
        logging.info(f"Model path set to {absolute_path}")

    def show_waveform(self, sender, app_data):
        if hasattr(self, 'waveform_process') and self.waveform_process is not None and self.waveform_process.poll() is None:
            self.waveform_process.terminate()
            self.waveform_process.wait()
            self.waveform_process = None
            dpg.set_value("status_text", "Waveform display closed.")
        else:
            waveform_script = "waveform_test_v3.py"
            if not os.path.exists(waveform_script):
                dpg.set_value("status_text", f"Error: {waveform_script} not found.")
                return
            try:
                self.save_settings(sender, app_data)
                self.waveform_process = subprocess.Popen(['python', waveform_script])
                dpg.set_value("status_text", "Waveform display started.")
            except Exception as e:
                self.waveform_process = None
                dpg.set_value("status_text", f"Failed to start waveform: {str(e)}")

    def start_dictation(self, sender, app_data):
        if self.is_dictating:
            dpg.set_value("status_text", "Dictation already running.")
            return
        model_path = dpg.get_value("model_path_input")
        if not model_path or model_path == "Not set" or not os.path.exists(model_path):
            dpg.set_value("status_text", "Please set a valid model path.")
            return
        try:
            self.is_dictating = True
            dpg.configure_item("start_dictation_button", enabled=False)
            dpg.configure_item("stop_dictation_button", enabled=True)
            update_status("Initializing Vosk recognizer...")

            # Initialize Vosk recognizer
            if self.vosk_model.model is None:
                self.vosk_model.model = self.vosk_model.load_model()
            self.recognizer = KaldiRecognizer(self.vosk_model.model, self.vosk_model.vosk_sample_rate)
            update_status("Vosk recognizer initialized successfully.")

            # Start audio stream
            update_status(f"Starting audio stream with sample rate {self.sample_rate}, device index {self.audio_processor.device_index}...")
            self.audio_processor.start_stream()
            self.dictation_stream = self.audio_processor.stream
            update_status("Audio stream started successfully.")
            update_status("Feeding audio to Vosk recognizer...")

            # Enable debug recording if checked
            self.is_debug_recording = dpg.get_value("debug_recording_check")
            self.debug_audio_buffer = []

            # Start transcription thread
            self.stop_transcription = False
            self.transcription_thread = threading.Thread(target=self.transcribe_audio, daemon=True)
            self.transcription_thread.start()

            dpg.set_value("status_text", "Dictation started. Speak now.")
        except Exception as e:
            self.is_dictating = False
            dpg.configure_item("start_dictation_button", enabled=True)
            dpg.configure_item("stop_dictation_button", enabled=False)
            dpg.set_value("status_text", f"Failed to start dictation: {e}")
            logging.error(f"Dictation start error: {e}")


    def stop_dictation(self, sender, app_data):
        if not self.is_dictating:
            return
        self.is_dictating = False
        self.is_debug_recording = False
        self.stop_transcription = True
        if self.dictation_stream:
            self.dictation_stream.stop()
            self.dictation_stream.close()
            self.dictation_stream = None
        self.audio_processor.stop_stream()
        dpg.configure_item("start_dictation_button", enabled=True)
        dpg.configure_item("stop_dictation_button", enabled=False)
        dpg.set_value("status_text", "Dictation stopped.")
        if self.debug_audio_buffer:
            self.save_debug_recording()
        # Only save WAV file if recording checkbox is checked
        if dpg.get_value("debug_recording_check"):
            audio_data = np.concatenate(self.audio_processor.audio_buffer) if self.audio_processor.audio_buffer else np.array([], dtype=np.int16)
            with wave.open("output_gigaspeech.wav", 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.vosk_model.vosk_sample_rate)
                wf.writeframes(audio_data.tobytes())
            print("Audio saved to output_gigaspeech.wav")

    def transcribe_audio(self):
        while not self.stop_transcription and self.is_dictating:
            try:
                audio_data = self.audio_processor.get_audio_data()
                if audio_data is None:
                    continue

                if self.recognizer.AcceptWaveform(audio_data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "")
                    if text:
                        processed_text = self.vosk_model.handle_special_phrases(text)
                        processed_text = self.vosk_model.process_text(processed_text)
                        self.handle_transcription(processed_text)
                else:
                    partial = json.loads(self.recognizer.PartialResult())
                    partial_text = partial.get("partial", "")
                    if partial_text:
                        dpg.set_value("output_text", f"Partial: {partial_text}")
                        self.last_partial = partial_text

            except Exception as e:
                logging.error(f"Transcription error: {e}")
                dpg.set_value("status_text", f"Transcription error: {e}")
                time.sleep(0.1)

    def handle_transcription(self, text):
        # Check for commands
        tokens = self.vosk_model.normalize_and_tokenize(text)

        # Simple commands
        for cmd_tokens, (cmd, action) in self.vosk_model.tokenized_simple_commands.items():
            if tokens == cmd_tokens:
                #print(f"Recognized command: {cmd}")
                self.execute_command(action)
                self.vosk_model.last_processed_command = action
                return

        # Parameterized commands
        for cmd_tokens, cmd_str in self.vosk_model.tokenized_parameterized_partial:
            if tokens[:len(cmd_tokens)] == cmd_tokens:
                param = " ".join(tokens[len(cmd_tokens):])
                self.execute_command(cmd_str, param)
                self.vosk_model.last_processed_command = cmd_str
                return


        # Dictate text if no command
        if not self.vosk_model.skip_dictation:
            self.vosk_model.last_dictated_text = text
            self.vosk_model.last_dictated_length = len(text)
            keyboard.write(text + " ")
            dpg.set_value("output_text", f"Dictated: {text}")

        # Custom commands
        for cmd in self.custom_commands:
            if text.strip().lower() == cmd["phrase"].strip().lower():
                self.execute_custom_command(cmd["action"])
                dpg.set_value("command_status_text", f"Custom: {cmd['phrase']} -> {cmd['action']}")
                return



    def execute_command(self, command, param=None):
        print(f"Executing command: {command!r} param: {param!r}")
        try:
            if command == "cmd_new_paragraph":
                keyboard.press_and_release("enter")
                keyboard.press_and_release("enter")
            elif command == "cmd_new_line":
                keyboard.press_and_release("enter")
            elif command == "cmd_space":
                keyboard.press_and_release("space")
            elif command == "cmd_tab":
                keyboard.press_and_release("tab")
            elif command == "cmd_undo":
                keyboard.press_and_release("ctrl+z")
            elif command == "cmd_redo":
                keyboard.press_and_release("ctrl+y")
            elif command == "cmd_caps_lock_on":
                self.vosk_model.caps_lock_on = True
            elif command == "cmd_caps_lock_off":
                self.vosk_model.caps_lock_on = False
            elif command == "cmd_number_lock":
                self.vosk_model.number_lock_on = not self.vosk_model.number_lock_on
            elif command == "cmd_stop_listening":
                self.stop_dictation(None, None)
            elif command == "cmd_clear":
                keyboard.press_and_release("ctrl+a")
                keyboard.press_and_release("delete")
            elif command == "cmd_select_all":
                keyboard.press_and_release("ctrl+a")
            elif command == "cmd_copy":
                keyboard.press_and_release("ctrl+c")
            elif command == "cmd_paste":
                keyboard.press_and_release("ctrl+v")
            elif command == "cmd_delete":
                keyboard.press_and_release("delete")
            elif command == "cmd_save_document":
                keyboard.press_and_release("ctrl+s")
            elif command == "cmd_open_file":
                keyboard.press_and_release("ctrl+o")
            elif command == "cmd_move_up":
                keyboard.press_and_release("up")
            elif command == "cmd_move_down":
                keyboard.press_and_release("down")
            elif command == "cmd_move_left":
                keyboard.press_and_release("left")
            elif command == "cmd_move_right":
                keyboard.press_and_release("right")
            elif command == "cmd_enter":
                keyboard.press_and_release("enter")
            elif command == "cmd_bold":
                keyboard.press_and_release("ctrl+b")
            elif command == "cmd_italicize":
                keyboard.press_and_release("ctrl+i")
            elif command == "cmd_underline":
                keyboard.press_and_release("ctrl+u")
            elif command == "cmd_center":
                keyboard.press_and_release("ctrl+e")
            elif command == "cmd_left_align":
                keyboard.press_and_release("ctrl+l")
            elif command == "cmd_right_align":
                keyboard.press_and_release("ctrl+r")
            elif command == "cmd_cut":
                keyboard.press_and_release("ctrl+x")
            elif command == "cmd_go_to_beginning":
                keyboard.press_and_release("ctrl+home")
            elif command == "cmd_go_to_end":
                keyboard.press_and_release("ctrl+end")
            elif command == "cmd_refresh_page":
                keyboard.press_and_release("ctrl+r")
            elif command == "cmd_go_back":
                keyboard.press_and_release("alt+left")
            elif command == "cmd_go_forward":
                keyboard.press_and_release("alt+right")
            elif command == "cmd_open_new_tab":
                keyboard.press_and_release("ctrl+t")
            elif command == "cmd_close_tab":
                keyboard.press_and_release("ctrl+w")
            elif command == "cmd_next_tab":
                keyboard.press_and_release("ctrl+tab")
            elif command == "cmd_previous_tab":
                keyboard.press_and_release("ctrl+shift+tab")
            elif command == "cmd_shift_tab":
                keyboard.press_and_release("shift+tab")
            elif command == "cmd_scratch_that":
                if self.vosk_model.last_dictated_text:
                    for _ in range(self.vosk_model.last_dictated_length):
                        keyboard.press_and_release("backspace")
                    self.vosk_model.last_dictated_text = ""
                    self.vosk_model.last_dictated_length = 0
            elif command in ("cmd_switch_windows", "cmd_switch_application", "cmd_switch_window"):
                keyboard.press("alt")
                keyboard.press_and_release("tab")
                dpg.set_value("command_status_text", "Alt held for window switching. Say 'select window' to release Alt.")
            elif command in ("cmd_next_window", "cmd_next_application"):
                keyboard.press_and_release("tab")
            elif command in ("cmd_previous_window", "cmd_previous_application"):
                keyboard.press_and_release("shift+tab")
            elif command == "cmd_select_window":
                keyboard.release("alt")
                dpg.set_value("command_status_text", "Alt released, window selected.")
            elif command == "cmd_punctuation_period":
                keyboard.write(".")
            elif command == "cmd_punctuation_comma":
                keyboard.write(",")
            elif command == "cmd_punctuation_question_mark":
                keyboard.write("?")
            elif command == "cmd_punctuation_asterix":
                keyboard.write("*")
            elif command == "cmd_punctuation_asterisk":
                keyboard.write("*")
            elif command == "cmd_punctuation_dash":
                keyboard.write("-")
            elif command == "cmd_punctuation_underscore":
                keyboard.write("_")
            elif command == "cmd_punctuation_plus":
                keyboard.write("+")
            elif command == "cmd_punctuation_slash":
                keyboard.write("/")
            elif command == "cmd_punctuation_backslash":
                keyboard.write("\\")
            elif command == "cmd_punctuation_parenthesis":
                keyboard.write("()")
            elif command == "cmd_punctuation_bracket":
                keyboard.write("[]")
            elif command == "cmd_punctuation_braces":
                keyboard.write("{}")
            elif command == "cmd_punctuation_angle_brackets":
                keyboard.write("<>")
            elif command == "cmd_punctuation_quotation_mark":
                keyboard.write('"')
            elif command == "cmd_punctuation_apostrophe":
                keyboard.write("'")
            elif command == "cmd_punctuation_at":
                keyboard.write("@")
            elif command == "cmd_punctuation_hashtag":
                keyboard.write("#")
            elif command == "cmd_punctuation_dollar_sign":
                keyboard.write("$")
            elif command == "cmd_punctuation_percent_sign":
                keyboard.write("%")
            elif command == "cmd_punctuation_ampersand":
                keyboard.write("&")
            elif command == "cmd_punctuation_pound":
                keyboard.write("#")
            elif command == "cmd_punctuation_caret":
                keyboard.write("^")
            elif command == "cmd_punctuation_tilde":
                keyboard.write("~")
            elif command == "cmd_punctuation_tilda":
                keyboard.write("~")
            elif command == "cmd_punctuation_equal":
                keyboard.write("=")
            elif command == "cmd_punctuation_pipe":
                keyboard.write("|")
            elif command == "cmd_punctuation_backtick":
                keyboard.write("`")
            elif command == "cmd_punctuation_less_than":
                keyboard.write("<")
            elif command == "cmd_punctuation_greater_than":
                keyboard.write(">")
            elif command == "cmd_punctuation_comma":
                keyboard.write(",")
            elif command == "cmd_punctuation_exclamation":
                keyboard.write("!")
            elif command == "cmd_punctuation_semicolon":
                keyboard.write(";")
            elif command == "cmd_punctuation_colon":
                keyboard.write(":")
            elif command == "cmd_press_escape":
                keyboard.press_and_release("esc")
            elif command.startswith("cmd_punctuation_"):
                symbol = command.replace("cmd_punctuation_", "")
                keyboard.write(symbol)
            elif command == "find" and param:
                keyboard.press_and_release("ctrl+f")
                time.sleep(0.2)
                keyboard.write(param)
            elif command == "open" and param:
                keyboard.press_and_release("ctrl+t")
                time.sleep(0.5)
                keyboard.write(param)
                keyboard.press_and_release("enter")
            elif command == "press" and param:
                keyboard.press_and_release(param)
            elif command == "cmd_select_up":
                keyboard.press_and_release("shift+up")
            elif command == "cmd_select_down":
                keyboard.press_and_release("shift+down")
            elif command == "cmd_select_all_up":
                keyboard.press_and_release("ctrl+shift+up")
            elif command == "cmd_select_all_down":
                keyboard.press_and_release("ctrl+shift+down")
            elif command == "cmd_file_properties":
                keyboard.press_and_release("alt+enter")
            elif command == "cmd_move_up_paragraph":
                keyboard.press_and_release("ctrl+up")
            elif command == "cmd_move_down_paragraph":
                keyboard.press_and_release("ctrl+down")
            elif command == "cmd_go_to_beginning_of_line":
                keyboard.press_and_release("home")
            elif command == "cmd_go_to_end_of_line":
                keyboard.press_and_release("end")
            elif command == "cmd_go_to_address":
                keyboard.press_and_release("alt+d")
            elif command == "cmd_click_that":
                pyautogui.click()
            elif command == "cmd_screen_shoot":
                keyboard.press_and_release("print_screen")
            elif command == "cmd_screen_shoot_window":
                keyboard.press_and_release("alt+print_screen")
            elif command == "cmd_screen_shoot_monitor":
                keyboard.press_and_release("win+shift+s")
            elif command == "cmd_task_manager":
                keyboard.press_and_release("ctrl+shift+esc")
            elif command == "cmd_debug_screen":
                keyboard.press_and_release("ctrl+alt+del")
                pass
            elif command == "cmd_force_close":
                keyboard.press_and_release("alt+f4")

            # Highlight: Find and select the word
            elif command == "highlight" and param:
                # Ctrl+F to find, type the word, press Enter, then Esc to close find bar
                keyboard.press_and_release("ctrl+f")
                time.sleep(0.2)
                keyboard.write(param)
                keyboard.press_and_release("enter")
                time.sleep(0.2)
                keyboard.press_and_release("esc")
                # Most editors will have the word selected after this

            # Insert after: Find the word, move cursor after, type or execute further command
            elif command == "insert after" and param:
                keyboard.press_and_release("ctrl+f")
                time.sleep(0.2)
                keyboard.write(param)
                keyboard.press_and_release("enter")
                time.sleep(0.2)
                keyboard.press_and_release("esc")
                # Move cursor to end of found word
                for _ in range(len(param)):
                    keyboard.press_and_release("right")
                # Now you can type or execute further actions here
                # Example: keyboard.write("your text here")

            # Insert before: Find the word, move cursor before
            elif command == "insert before" and param:
                keyboard.press_and_release("ctrl+f")
                time.sleep(0.2)
                keyboard.write(param)
                keyboard.press_and_release("enter")
                time.sleep(0.2)
                keyboard.press_and_release("esc")
                # Cursor is at start of found word

            elif command == "function" and param:
                fn_key = param.strip().lower().replace(" ", "")
                # Accept "f1"..."f12"
                if fn_key.startswith("f") and fn_key[1:].isdigit():
                    num = fn_key[1:]
                    if num.isdigit() and 1 <= int(num) <= 12:
                        keyboard.press_and_release(f"f{int(num)}")
            
            # Quote: Output the param in double quotes            
            elif command == "quote unquote" and param:
                # Output the param in double quotes
                keyboard.write(f'"{param}"')


            #Application Commands
            elif command == "cmd_open_spotify":
                spotify_path = expand_user_path(r"C:\Users\<YourUser>\AppData\Roaming\Spotify\Spotify.exe")
                subprocess.Popen(spotify_path)
            elif command == "cmd_open_notepad":
                notepad_path = expand_user_path(r"C:\WINDOWS\system32\notepad.exe")
                subprocess.Popen(notepad_path)
            elif command == "cmd_open_word":
                # Create a temporary blank Excel file
                with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
                    word_file = tmp.name
                # Open it with the user's default spreadsheet application
                os.startfile(word_file)
            elif command == "cmd_open_excel":
                # Create a temporary blank Excel file
                with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
                    excel_file = tmp.name
                # Open it with the user's default spreadsheet application
                os.startfile(excel_file)
            elif command == "cmd_open_powerpoint":
                with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as tmp:
                    powerpoint_file = tmp.name
                # Open it with the user's default spreadsheet application
                os.startfile(powerpoint_file)
            elif command == "cmd_open_browser":
                webbrowser.open("about:blank")
            elif command == "cmd_open_chrome":
                chrome_path = expand_user_path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe")
                subprocess.Popen(chrome_path)
            elif command == "cmd_open_firefox":
                firefox_path = expand_user_path(r"C:\Program Files\Mozilla Firefox\firefox.exe")
                subprocess.Popen(firefox_path)
            elif command == "cmd_open_edge":
                edge_path = expand_user_path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
                subprocess.Popen(edge_path)
            elif command == "cmd_open_calculator":
                calc_path = expand_user_path(r"C:\WINDOWS\system32\calc.exe")
                subprocess.Popen(calc_path)
            elif command == "cmd_open_calendar":
                calendar_path = expand_user_path(r"C:\WINDOWS\system32\calendar.exe")
                subprocess.Popen(calendar_path)
            elif command == "cmd_open_file_explorer":
                file_explorer_path = expand_user_path(r"C:\WINDOWS\explorer.exe")
                subprocess.Popen(file_explorer_path)
            elif command == "cmd_open_settings":
                settings_path = expand_user_path(r"C:\Windows\ImmersiveControlPanel\SystemSettings.exe")
                subprocess.Popen(settings_path)
            elif command == "cmd_open_task_manager":
                task_manager_path = expand_user_path(r"C:\Windows\System32\Taskmgr.exe")
                subprocess.Popen(task_manager_path)
            elif command == "cmd_open_command_prompt":
                command_prompt_path = expand_user_path(r"C:\Windows\System32\cmd.exe")
                subprocess.Popen(command_prompt_path)
            elif command == "cmd_open_powershell":
                powershell_path = expand_user_path(r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe")
                subprocess.Popen(powershell_path)
            elif command == "cmd_open_terminal":
                if sys.platform.startswith("win"):
                    # Windows Terminal (Windows 10+), fallback to cmd
                    try:
                        subprocess.Popen("wt.exe")
                    except FileNotFoundError:
                        subprocess.Popen("cmd.exe")
                elif sys.platform.startswith("darwin"):
                    # macOS: open the default Terminal.app
                    subprocess.Popen(["open", "-a", "Terminal"])
                elif sys.platform.startswith("linux"):
                    # Linux: try common terminals in order
                    terminals = ["x-terminal-emulator", "gnome-terminal", "konsole", "xfce4-terminal", "lxterminal", "xterm", "mate-terminal", "tilix", "alacritty", "urxvt", "terminator"]
                    for term in terminals:
                        if subprocess.call(["which", term], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
                            subprocess.Popen([term])
                            break
            elif command == "cmd_open_control_panel":
                control_panel_path = expand_user_path(r"C:\Windows\System32\control.exe")
                subprocess.Popen(control_panel_path)
            elif command == "cmd_open_microsoft_edge":
                microsoft_edge_path = expand_user_path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
                subprocess.Popen(microsoft_edge_path)
            elif command == "cmd_open_discord":
                discord_path = expand_user_path(r"C:\Users\<YourUser>\AppData\Local\Discord\Update.exe")
                subprocess.Popen(discord_path)
            elif command == "cmd_open_steam":
                steam_path = expand_user_path(r"C:\Program Files (x86)\Steam\steam.exe")
                subprocess.Popen(steam_path)
            elif command == "cmd_open_start_menu":
                if sys.platform.startswith("win"):
                    # Windows: use the Win key to open Start Menu
                    keyboard.press_and_release("win")
                elif sys.platform.startswith("darwin"):
                    # macOS: use Command+Space to open Spotlight (similar to Start Menu)
                    keyboard.press_and_release("cmd+space")
                elif sys.platform.startswith("linux"):
                    # Linux: use Super key (often mapped to Win key)
                    keyboard.press_and_release("super")
                else:
                    dpg.set_value("command_status_text", "Unsupported platform for opening Start Menu.")
                return



            # Media Commands (universal)
            elif command == "cmd_play_pause":
                send_media_key("play/pause media")
            elif command == "cmd_next_track":
                send_media_key("next track media")
            elif command == "cmd_previous_track":
                send_media_key("previous track media")
            elif command == "cmd_volume_up":
                send_media_key("volume up media")
            elif command == "cmd_volume_down":
                send_media_key("volume down media")
            elif command == "cmd_mute":
                send_media_key("mute media")
            elif command == "cmd_unmute":
                send_media_key("unmute media")
            elif command == "cmd_stop":
                send_media_key("stop media")



            dpg.set_value("command_status_text", f"Executed: {command} {param or ''}")
        except Exception as e:
            dpg.set_value("command_status_text", f"Command error: {e}")
            logging.error(f"Command execution error: {e}")


    def execute_custom_command(self, action):
        # If action looks like a hotkey, send it; else, try to run as a program
        if "+" in action or action.isalpha():
            try:
                keyboard.press_and_release(action)
            except Exception:
                pass
        else:
            try:
                subprocess.Popen(action, shell=True)
            except Exception:
                pass



    def run(self):
        try:
            self.dpg_running = True
            while dpg.is_dearpygui_running():
                dpg.render_dearpygui_frame()
            self.dpg_running = False
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_dictation(None, None)
            if self.waveform_display:
                self.waveform_display.close()
            dpg.destroy_context()

def main():
    try:
        audio_processor = AudioProcessor(
            sample_rate=args.sample_rate,
            bit_depth=args.bit_depth,
            channels=1,
            device_index=args.device_index,
            pre_scale_factor=args.pre_scale_factor,
            silence_amplitude_threshold=args.silence_threshold,
            relative_sensitivity=args.relative_sensitivity
        )
        if args.list_devices:
            audio_processor.list_devices()
            sys.exit(0)

        vosk_model = VoskModel(
            audio_processor=audio_processor,
            model_path=args.model,
            sample_rate=args.sample_rate,
            bit_depth=args.bit_depth,
            pre_scale_factor=args.pre_scale_factor,
            silence_threshold=args.silence_threshold,
            relative_sensitivity=args.relative_sensitivity
        )

        gui = GUI(
            audio_processor=audio_processor,
            model_path=args.model,
            sample_rate=args.sample_rate,
            bit_depth=args.bit_depth,
            pre_scale_factor=args.pre_scale_factor,
            silence_threshold=args.silence_threshold,
            relative_sensitivity=args.relative_sensitivity,
            vosk_model=vosk_model
        )

          

        gui.run()

    except KeyboardInterrupt:
        update_status("Received Ctrl+C. Shutting down gracefully...")
        try:
            dpg.set_value("status_text", "Dictation stopped by user.")
        except Exception:
            pass
        update_status("GUI closed successfully.")
        update_status("Application terminated.")
        sys.exit(0)
    except Exception as e:
        update_status(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        try:
            dpg.destroy_context()
        except Exception:
            pass


if __name__ == "__main__":
    main()