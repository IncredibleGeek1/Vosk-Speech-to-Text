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
import logging
import sounddevice as sd
import threading
import queue
import sys
import json
import re
import sounddevice as sd
import vosk
import numpy as np
from scipy.signal import resample
import os
import traceback
import math
import wave
import json
import time
import words_to_numbers_v7 as words_to_numbers
import keyboard  # For typing and commands
import subprocess  # For launching applications
from spellchecker import SpellChecker  # For autocorrect
from collections import deque # For managing command history
import nltk  # For tokenization to separate commands from dictation
import pyautogui  # For mouse clicks and screenshots
import argparse  # For command-line argument parsing
import logging

# Download NLTK data (run once, automatically on first run)
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK data: {e}")
    print("Please ensure you have an internet connection and try again.")
    sys.exit(1)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Live speech-to-text dictation with Vosk.")
parser.add_argument("--model", type=str, default="C:/Users/MenaBeshai/Downloads/Speech to Text/vosk-model-en-us-0.42-gigaspeech",
                    help="Path to the Vosk model directory.")
parser.add_argument("--bit-depth", type=int, default=24, choices=[16, 24, 32], help="Bit depth for audio input (16, 24, or 32)")
parser.add_argument("--sample-rate", type=int, default=96000, choices=[44100, 48000, 88200, 96000, 176400, 192000], help="Sample rate for audio input (Hz)")
parser.add_argument("--pre-scale-factor", type=float, default=0.002, help="Pre-scale factor for audio input")
parser.add_argument("--silence-threshold", type=float, default=10.0, help="Silence threshold for audio detection")
parser.add_argument("--relative-sensitivity", type=int, default=0, choices=[0, 1], help="Whether to use relative sensitivity (0 or 1)")
parser.add_argument("--device-index", type=int, default=6, help="Index of the audio input device (run with --list-devices to see options)")
parser.add_argument("--list-devices", action="store_true", help="List available audio devices and exit")

args = parser.parse_args()

# Use the arguments, with fallbacks if not provided
model_path = args.model if args.model else os.path.join(os.path.dirname(__file__), "vosk-model-en-us-0.42-gigaspeech")
bit_depth = args.bit_depth if args.bit_depth is not None else 16  # Default to 16-bit
sample_rate = args.sample_rate if args.sample_rate is not None else 16000  # Default to 16000 Hz
pre_scale = args.pre_scale_factor if args.pre_scale_factor is not None else 1.0  # Default to 1.0
silence_threshold = args.silence_threshold if args.silence_threshold is not None else 0.1  # Default to 0.1
relative_sensitivity = bool(args.relative_sensitivity) if args.relative_sensitivity is not None else False

# Function to load JSON maps with error handling
def load_json_map(file_path, map_name):
    """Load a mapping from a JSON file, with error handling."""
    if not os.path.exists(file_path):
        print(f"Error: {map_name} file not found at {file_path}")
        sys.exit(1)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        if not isinstance(mapping, dict):
            print(f"Error: {map_name} at {file_path} must be a JSON object (dictionary)")
            sys.exit(1)
        return mapping
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse {map_name} at {file_path}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to load {map_name} from {file_path}: {e}")
        sys.exit(1)

# Suppress Vosk logging for cleaner output
vosk.SetLogLevel(-1)

# Queue for audio data
q = queue.Queue()

# Configuration (using arguments)
MODEL_PATH = args.model  # Use the model path from command-line argument
MIC_SAMPLERATE = args.sample_rate  # Use the sample rate from arguments
MIC_CHANNELS = 1  # Stereo, but identical signals from mono mic
MIC_BITDEPTH = args.bit_depth  # Use the bit depth from arguments
VOSK_SAMPLERATE = 16000  # Vosk's optimal sample rate
WAV_FILE = "output_gigaspeech.wav"  # File to save the audio
TRANSCRIPTION_FILE = "dictation_output_gigaspeech.txt"  # File to save transcriptions
DEVICE_INDEX = 6  # MY MIC (Realtek USB Audio)
SILENCE_THRESHOLD = 1.0  # Seconds of silence to detect a paragraph WAV_FILE = "output_gigaspeech.wav"  # File to save the audio
TRANSCRIPTION_FILE = "dictation_output_gigaspeech.txt"  # File to save transcriptions
DEVICE_INDEX = 1  # MY MIC (Realtek USB Audio)
SILENCE_THRESHOLD = 1.0  # Seconds of silence to detect a paragraph break
BLOCKSIZE = 32000  # Block size from giga script
PRE_SCALE_FACTOR = args.pre_scale_factor  # Use the pre-scale factor from arguments
SILENCE_AMPLITUDE_THRESHOLD = args.silence_threshold  # Use the silence threshold from arguments
RELATIVE_SENSITIVITY = bool(args.relative_sensitivity)  # Use the relative sensitivity from arguments
STARTUP_DELAY = 5  # Seconds to give user time to select a text field
COMMAND_DEBOUNCE_TIME = 1.0  # Seconds to debounce commands

# Path to commands JSON file
COMMANDS_JSON_PATH = os.path.join("config", "commands.json")

# Paths to map JSON files (in the config folder)
CONFIG_DIR = "config"
CONFIG_PATH = "config.json"
FRACTIONS_MAP_PATH = os.path.join(CONFIG_DIR, "fractions_map.json")
F_KEYS_MAP_PATH = os.path.join(CONFIG_DIR, "f_keys_map.json")
FUNCTIONS_MAP_PATH = os.path.join(CONFIG_DIR, "functions_map.json")
SYMBOLS_MAP_PATH = os.path.join(CONFIG_DIR, "symbols_map.json")
NUMBERS_MAP_PATH = os.path.join(CONFIG_DIR, "numbers_map.json")
GOOGLE_NUMBERS_PATH = os.path.join(CONFIG_DIR, "google_numbers.json")  # Updated to numbers_map.json
LARGE_NUMBERS_MAP_PATH = os.path.join(CONFIG_DIR, "large_numbers_map.json")

# Dictionary to convert spoken numbers to digits (0 to 1 million)
NUMBERS_MAP = {
    "negative": "-", "minus": "-",  # Support for negative numbers
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

# Load maps from JSON files
FRACTION_MAP = load_json_map(FRACTIONS_MAP_PATH, "fractions map")
F_KEYS_MAP = load_json_map(F_KEYS_MAP_PATH, "f-keys map")
FUNCTIONS_MAP = load_json_map(FUNCTIONS_MAP_PATH, "functions map")
SYMBOLS_MAP = load_json_map(SYMBOLS_MAP_PATH, "symbols map")
NUMBERS_MAP = load_json_map(NUMBERS_MAP_PATH, "numbers map")  # From numbers_map.json
GOOGLE_NUMBERS = load_json_map(GOOGLE_NUMBERS_PATH, "google numbers map")  # From numbers_map.json
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

def parse_number_sequence(words):
    """Parse a sequence of number words into a single string of digits, handling negatives."""
    is_negative = False
    start_idx = 0
    i = 0
    
    # Check for "negative" or "minus" at the start
    if words and words[0].lower() in ["negative", "minus"]:
        is_negative = True
        start_idx = 1
        i = 1
    
    # Join the remaining words into a single string
    text = " ".join(words[start_idx:])
    if not text:  # If only "negative" or "minus" was provided, return empty
        return "", i
    
    # Use words_to_numbers.convert() to process the text
    result = words_to_numbers.convert_numbers(text, FRACTION_MAP, SYMBOLS_MAP, GOOGLE_NUMBERS, LARGE_NUMBERS_MAP)
    
    # Check if the result is a valid number
    if result is None:
        # If conversion failed, return empty string and number of words processed
        return "", len(words)
    
    number = int(result)
    if is_negative:
        number = -number
    # Return the number and the total number of words processed
    return str(number), len(words)

# Normalize transcription text before tokenization
def normalize_text(text):
    """Normalize text by replacing hyphens with spaces and ensuring consistent spacing."""
    text = text.replace("-", " ")  # Replace hyphens with spaces (e.g., "select-all" -> "select all")
    text = " ".join(text.split())  # Normalize spacing (e.g., "select   all" -> "select all")
    return text

# Convert spoken numbers to digits if number lock is on
def convert_numbers(text):
    global number_lock_on  # Declare global at the start of the function
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

##VOSK MODEL WAS HERE


    # Load a JSON file (generic function for any JSON mappings)
    def load_json_map(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"{file_path} not found!")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in {file_path}!")
        except UnicodeDecodeError as e:
            raise ValueError(f"Encoding error in {file_path}! {e}")

    # Define mappings for small numbers (0-999)
    SMALL_NUMBERS = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
        "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
        "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
        "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
        "hundred": 100, "thousand": 1000
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
                return None  # Return None instead of raising an exception

        return total + current

    # Main function to convert a word phrase to a number
    def convert_numbers(phrase, fraction_map=None, symbs_map=None, google_numbers=None, large_numbers_map=None):
        # Load large_numbers_map if not provided
        if large_numbers_map is None:
            large_numbers_map = load_json_map("large_numbers_map.json")

        # Ignore fraction_map, symbs_map, and google_numbers for now
        # These can be used in future implementations if needed

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

        return -total if is_negative else total

    # Helper function to parse a sequence of number words from a list
    def parse_number_sequence(words, fraction_map=None, symbs_map=None, google_numbers=None, large_numbers_map=None):
        """Parse a sequence of number words into a single number, returning the number and the number of words consumed."""
        if not words:
            return None, 0

        # Join words into a phrase and try to parse it
        phrase = " ".join(words)
        number = convert_numbers(phrase, fraction_map, symbs_map, google_numbers, large_numbers_map)
        if number is not None:
            return str(number), len(words)

        # Try parsing smaller sequences until we find a valid number
        for i in range(len(words), 0, -1):
            sub_phrase = " ".join(words[:i])
            number = convert_numbers(sub_phrase, fraction_map, symbs_map, google_numbers, large_numbers_map)
            if number is not None:
                return str(number), i
        return None, 0

    # Example usage (for testing purposes)
    if __name__ == "__main__":
        # Load the large numbers map
        large_numbers_map = load_json_map("large_numbers_map.json")

        # Test cases
        test_phrases = [
            "one hundred and twenty-three",
            "nine hundred and ninety-nine thousand",
            "one billion",
            "one trillion two hundred and thirty-four billion five hundred and sixty-seven million",
            "negative one hundred",
            "one vigintillion",
            "billion",  # Edge case: just the scale word
            "hello"  # Should return None
        ]

        for phrase in test_phrases:
            result = convert_numbers(phrase, None, None, None, large_numbers_map)
            print(f"{phrase} -> {result}")




class AudioProcessor:
    def __init__(self, recordings_dir, gui=None):
        self.gui = gui
        self.recordings_dir = recordings_dir
        self.audio_queue = queue.Queue()
        self.gui_update_queue = queue.Queue()
        self.audio_stream = None
        self.output_stream = None
        self.is_testing = False
        self.is_recording = False
        self.is_transcribing = False
        self.model = None
        self.recognizer = None
        self.last_partial = ""
        self.last_word_end_time = 0.0
        self.last_processed_command = None
        self.skip_dictation = False
        self.last_dictated_text = ""
        self.last_dictated_length = 0
        self.audio_buffer = deque(maxlen=48000)  # For WaveformDisplay
        self.output_buffer = []
        self.debug_audio_buffer = []
        self.noise_floor = 0
        self.last_noise_update = 0
        self.peak_amplitude = 0
        self.is_debug_recording = False
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

    def save_settings(self):
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
            dpg.add_combo(["Numbers", "Percent", "dB"], default_value="Numbers", callback=self.update_unit, tag="unit_combo")
        
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

    def audio_callback(self, indata, frames, time, status):
        if status:
            logging.error(f"Audio callback status: {status}")
        audio_data = np.frombuffer(indata, dtype=np.int32)
        scaled_data = audio_data.astype(np.int16)  # Convert to 16-bit for Vosk
        self.audio_buffer.extend(scaled_data)  # For WaveformDisplay
        self.audio_queue.put(scaled_data.tobytes())
        if self.is_recording:
            self.recording_buffer.append(scaled_data.copy())

    def start_transcription(self):
        if self.is_transcribing:
            return
        model_path = self.saved_settings["model_path"]
        if not os.path.exists(model_path):
            if self.gui:
                self.gui.update_status(f"Invalid model path: {model_path}")
            return
        self.is_transcribing = True
        self.model = vosk.Model(model_path)
        self.recognizer = vosk.KaldiRecognizer(self.model, int(self.saved_settings["sample_rate"]))
        self.start_audio_stream()
        threading.Thread(target=self.transcription_loop, daemon=True).start()
        if self.gui:
            self.gui.update_status("Transcription started.")

    def stop_transcription(self):
        if not self.is_transcribing:
            return
        self.is_transcribing = False
        self.stop_audio_stream()
        self.recognizer = None
        self.model = None
        if self.gui:
            self.gui.update_status("Transcription stopped.")



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
            scaled_data = (indata_scaled).astype(np.int16)
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
    def __init__(self, AudioProcessor, gui, plot_width=400, plot_height=100):
        self.audio_processor = AudioProcessor
        self.gui = gui
        self.plot_width = plot_width
        self.plot_height = plot_height
        self.samplerate = int(self.audio_processor.saved_settings.get("sample_rate", 16000))
        self.bit_depth = self.audio_processor.saved_settings.get("bit_depth", "int16")
        self.pre_scale_factor = self.audio_processor.saved_settings.get("pre_scale_factor", 0.002)
        self.relative_sensitivity = self.audio_processor.saved_settings.get("relative_sensitivity", False)
        self.channels = self.audio_processor.saved_settings.get("channels", 2)  # Assume stereo from AudioProcessor
        self.is_running = True
        self.audio_buffer = deque(maxlen=int(self.samplerate * 1.0))  # 1 second of data
        self.update_interval = 1.0 / 30  # Update at 30 FPS
        self.last_update = time.time()
        self.setup_plot()


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

    def setup_plot(self):
        with dpg.window(label="Waveform Display", tag="waveform_window", width=self.plot_width + 20, height=self.plot_height + 50, on_close=self.close):
            with dpg.plot(label="Waveform", height=self.plot_height, width=self.plot_width, tag="waveform_plot"):
                dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag="waveform_x_axis")
                with dpg.plot_axis(dpg.mvYAxis, label="Amplitude", tag="waveform_y_axis"):
                    dpg.add_line_series([], [], label="Waveform", tag="waveform_series")
        dpg.set_axis_limits("waveform_y_axis", -32768, 32768)  # 16-bit range
        dpg.set_axis_limits("waveform_x_axis", 0, 1.0)

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
                if self.audio_processor.is_testing and self.audio_processor.audio_buffer:
                    # Get recent audio data from AudioProcessor (last 1 second)
                    audio_data = np.concatenate(self.audio_processor.audio_buffer)[-int(self.samplerate * 1.0):]
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
        if dpg.does_item_exist("waveform_window"):
            dpg.delete_item("waveform_window")








# Constants
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

JSON_FILES = {
    "Commands": "commands.json",
    "Fractions": "fractions_map.json",
    "F-Keys": "f_keys_map.json",
    "Functions": "functions_map.json",
    "Symbols": "symbols_map.json",
    "Google Numbers": "google_numbers.json",
    "Large Numbers": "large_numbers_map.json"
}



class DictationGUI:
    def __init__(self):
        self.audio_processor = AudioProcessor(recordings_dir="Recordings", gui=self)
        self.waveform_display = None
        
        self.is_dictating = False
        self.command_queue = queue.Queue()
        self.transcribed_text = []
        self.command_progress = 0.0
        self.command_progress_max = 100.0
        self.command_status = ""
        self.last_partial = ""
        self.last_word_end_time = 0.0
        self.last_command = None
        self.last_command_time = 0.0
        self.skip_dictation = False
        self.last_dictated_text = ""
        self.last_dictated_length = 0
        self.last_processed_command = None
        self.audio_buffer = []
        self.stop_transcription = False
        
        self.default_recordings_dir = os.path.join(os.getcwd(), "Recordings")
        self.recordings_dir = self.default_recordings_dir
        if not os.path.exists(self.default_recordings_dir):
            os.makedirs(self.default_recordings_dir)
        
        self.saved_settings = {"model_path": "Not set",
            "bit_depth": "int16",
            "sample_rate": "16000",
            "pre_scale_factor": 0.002,
            "silence_threshold": 10.0,
            "relative_sensitivity": False,
            "device_index": 1,}
        self.json_data = {}

        self.transcribed_text = deque(maxlen=100) # Store recent transcriptions
        
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
        
        self.theme = "Dark"
        
        # Setup GUI
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
        with dpg.group():
            with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp, tag=f"{tab_name}_table"):
                dpg.add_table_column(label="Key")
                dpg.add_table_column(label="Value")
                base_path = os.path.dirname(os.path.abspath(__file__))
                full_path = os.path.join(base_path, json_path)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        self.json_data[tab_name] = json.load(f)
                except Exception as e:
                    logging.error(f"Failed to load {json_path}: {e}")
                    self.json_data[tab_name] = {}
                
                for key, value in self.json_data[tab_name].items():
                    with dpg.table_row():
                        dpg.add_text(key, tag=f"{tab_name}_{key}_key")
                        dpg.add_text(str(value), tag=f"{tab_name}_{key}_value")
            
            with dpg.group(horizontal=True):
                dpg.add_button(label="Add", callback=lambda: self.add_json_entry(tab_name, json_path))
                dpg.add_button(label="Edit", callback=lambda: self.edit_json_entry(tab_name, json_path))
                dpg.add_button(label="Delete", callback=lambda: self.delete_json_entry(tab_name, json_path))

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
        try:
            with open("config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            config = {}
        config["model_path"] = absolute_path
        try:
            with open("config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
            dpg.set_value("status_text", f"Model path set to: {absolute_path}")
            logging.info(f"Model path set to {absolute_path}")
        except Exception as e:
            dpg.set_value("status_text", f"Failed to save model path: {e}")
            logging.error(f"Failed to save model path: {e}")
    
    def load_settings(self):
        default_settings = {"model_path": "Not set"}
        try:
            with open("config.json", "r", encoding="utf-8") as f:
                self.saved_settings = json.load(f)
        except Exception:
            self.saved_settings = default_settings
            with open("config.json", "w", encoding="utf-8") as f:
                json.dump(self.saved_settings, f, indent=4)
        self.saved_settings = {**default_settings, **self.saved_settings}
        dpg.set_value("model_path_input", self.saved_settings["model_path"])
    
    def start_dictation(self, sender, app_data):
        if self.is_dictating:
            dpg.set_value("status_text", "Dictation already running.")
            return
        model_path = dpg.get_value("model_path_input")
        if not model_path or model_path == "Not set" or not os.path.exists(model_path):
            dpg.set_value("status_text", "Please set a valid model path.")
            return
        dpg.set_value("status_text", "Dictation starting...")
        dpg.configure_item("start_dictation_button", enabled=False)
        dpg.configure_item("stop_dictation_button", enabled=True)
        self.is_dictating = True
    
    def stop_dictation(self, sender, app_data):
        if not self.is_dictating:
            return
        self.is_dictating = False
        dpg.configure_item("start_dictation_button", enabled=True)
        dpg.configure_item("stop_dictation_button", enabled=False)
        dpg.set_value("status_text", "Dictation stopped.")
    
    def run(self):
        try:
            dpg.start_dearpygui()
        finally:
            dpg.destroy_context()






# Load Vosk model
print("Loading Vosk model...")
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
with sd.RawInputStream(samplerate=MIC_SAMPLERATE, blocksize=BLOCKSIZE, dtype="int32",
                        channels=MIC_CHANNELS, callback=callback, device=DEVICE_INDEX):
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



try:
    KeyboardInterrupt
    print("\nTranscription stopped.")
except Exception as e:
    print(f"Error during transcription: {e}")
    print("Stack trace:")
    traceback.print_exc()
finally:
    print("Saving audio to WAV file...")
    audio_data = np.concatenate(audio_buffer) if audio_buffer else np.array([], dtype=np.int16)
    with wave.open(WAV_FILE, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(VOSK_SAMPLERATE)
        wf.writeframes(audio_data.tobytes())
    print(f"Audio saved to {WAV_FILE}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    gui = DictationGUI()
    gui.run()