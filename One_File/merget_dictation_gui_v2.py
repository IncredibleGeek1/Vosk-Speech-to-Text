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
import math
import numpy as np
import sounddevice as sd
import vosk
import numpy as np
from scipy.signal import resample
import os
import traceback
import wave
import json
import time
import words_to_numbers_v7 as words_to_numbers
import keyboard  # For typing and commands
import subprocess  # For launching applications
from spellchecker import SpellChecker  # For autocorrect
import nltk  # For tokenization to separate commands from dictation
import pyautogui  # For mouse clicks and screenshots
import argparse  # For command-line argument parsing

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

DATA_TYPES = {
    "int8": "8-bit Integer (Fastest, lowest quality)",
    "int16": "16-bit Integer (Standard quality)",
    "int24": "24-bit Integer (High quality)",
    "int32": "32-bit Integer (Highest integer quality)",
    "float32": "32-bit Float (Best for GPU acceleration)"
}

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
FRACTIONS_MAP_PATH = os.path.join(CONFIG_DIR, "fractions_map.json")
F_KEYS_MAP_PATH = os.path.join(CONFIG_DIR, "f_keys_map.json")
FUNCTIONS_MAP_PATH = os.path.join(CONFIG_DIR, "functions_map.json")
SYMBOLS_MAP_PATH = os.path.join(CONFIG_DIR, "symbols_map.json")
NUMBERS_MAP_PATH = os.path.join(CONFIG_DIR, "numbers_map.json")
GOOGLE_NUMBERS_PATH = os.path.join(CONFIG_DIR, "google_numbers_map.json")  # Updated to numbers_map.json
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
try:
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

except KeyboardInterrupt:
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