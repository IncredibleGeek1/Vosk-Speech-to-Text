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

import argparse
import queue
import sys
import sounddevice as sd
import vosk
import keyboard
import os
import numpy as np
from scipy.signal import resample
import wave
import json
import time

# Suppress Vosk logging for cleaner output
vosk.SetLogLevel(-1)

# Queue for audio data
q = queue.Queue()

# Argument parser
parser = argparse.ArgumentParser(description="Live speech-to-text transcription with commands using Vosk")
parser.add_argument("--model", required=True, type=str, help="Path to the Vosk model directory")
parser.add_argument("--samplerate", default=16000, type=int, help="Sampling rate expected by the model")
parser.add_argument("--mic-samplerate", default=96000, type=int, help="Sampling rate of the microphone (default: 96000 Hz)")
parser.add_argument("--mic-channels", default=2, type=int, help="Number of microphone channels (default: 2)")
parser.add_argument("--mic-bitdepth", default=24, type=int, choices=[16, 24], help="Bit depth of the microphone (16 or 24, default: 24)")
parser.add_argument("--device", default=1, type=int, help="Device index for the microphone (default: 1)")
args = parser.parse_args()

# Configuration
MODEL_PATH = args.model
WAV_FILE = "output_combined.wav"
TRANSCRIPTION_FILE = "dictation_output_combined.txt"
SILENCE_THRESHOLD = 1.0  # Seconds of silence to detect a paragraph break
PRE_SCALE_FACTOR = 0.002  # To prevent clipping
SILENCE_AMPLITUDE_THRESHOLD = 10  # Adjusted for lower amplitude

# List to store audio data for WAV file
audio_buffer = []

# Store the last word's end time for silence detection
last_word_end_time = 0.0

# Command set (expanded)
COMMANDS = {
    "new paragraph": "\n\n",
    "new line": "\n",
    "space": " ",
    "tab": "\t",
    "open file": "cmd_open_file",
    "save document": "cmd_save_document",
    "stop listening": "cmd_stop_listening",
    "select all": "cmd_select_all",
    "copy that": "cmd_copy",
    "paste that": "cmd_paste",
    "delete that": "cmd_delete",
}

def add_basic_punctuation(text):
    """Add enhanced punctuation and capitalization to the transcribed text."""
    if not text:
        return text

    # Split the text into words
    words = text.split()
    if not words:
        return text

    # Capitalize the first letter
    words[0] = words[0][0].upper() + words[0][1:] if len(words[0]) > 1 else words[0].upper()

    # Add punctuation based on rules
    final_text = ""
    for i, word in enumerate(words):
        final_text += word
        if i < len(words) - 1 and word.lower() in ["and", "but", "or"]:
            final_text += ","
        if i == len(words) - 1 and words[0].lower() in ["what", "where", "when", "why", "how", "who"]:
            final_text += "?"
        elif i == len(words) - 1 and any(w.lower() in ["wow", "great"] for w in words):
            final_text += "!"
        elif i == len(words) - 1 and not final_text.endswith((".", "!", "?")):
            final_text += "."
        if i < len(words) - 1:
            final_text += " "

    return final_text

def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(f"Audio callback status: {status}", file=sys.stderr)
    
    if args.mic_bitdepth == 16:
        indata_array = np.frombuffer(indata, dtype=np.int16)
        num_samples = len(indata_array) // args.mic_channels
        indata_array = indata_array.reshape((num_samples, args.mic_channels))
    else:
        indata_bytes = bytes(indata)
        num_samples = len(indata_bytes) // (4 * args.mic_channels)
        indata_array = np.zeros((num_samples, args.mic_channels), dtype=np.int32)
        for i in range(num_samples):
            for ch in range(args.mic_channels):
                start_idx = (i * args.mic_channels + ch) * 4
                sample = int.from_bytes(indata_bytes[start_idx:start_idx + 4], byteorder='little', signed=True)
                indata_array[i, ch] = sample
        
        # Debug: Check raw audio amplitude
        max_amplitude_raw = np.max(np.abs(indata_array))
        print(f"Raw audio max amplitude: {max_amplitude_raw}")
        
        # Pre-scale to handle extremely loud audio
        indata_array = (indata_array * PRE_SCALE_FACTOR).astype(np.int32)
        
        # Convert 24-bit (stored as 32-bit) to 16-bit
        indata_array = np.clip(indata_array // 256, -32768, 32767).astype(np.int16)
    
    # Convert stereo to mono by averaging the channels
    if args.mic_channels > 1:
        indata_array = np.mean(indata_array, axis=1, dtype=np.int16)
    
    # Resample the audio to the model's expected sample rate (16000 Hz)
    if args.mic_samplerate != args.samplerate:
        num_samples_resampled = int(len(indata_array) * args.samplerate / args.mic_samplerate)
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

# Start transcription
print("Starting live speech-to-text with Vosk. Speak to type or use commands in the active text field! Press Ctrl+C to stop.")
try:
    rec = vosk.KaldiRecognizer(model, args.samplerate)
    last_partial = ""  # To track the last partial result and reduce spam
    with sd.RawInputStream(samplerate=args.mic_samplerate, blocksize=32000, dtype="int32",
                           channels=args.mic_channels, callback=callback, device=args.device):
        stop_listening = False
        while True:
            if stop_listening:
                break
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())["text"]
                if result:
                    # Add punctuation
                    final_text = add_basic_punctuation(result)
                    
                    # Add paragraph breaks based on silence
                    current_time = time.time()
                    if last_word_end_time > 0:
                        silence_duration = current_time - last_word_end_time
                        if silence_duration > SILENCE_THRESHOLD:
                            final_text += "\n\n"
                    
                    # Update the last word's end time for silence detection
                    result_dict = json.loads(rec.Result())
                    if "result" in result_dict and result_dict["result"]:
                        last_word_end_time = result_dict["result"][-1]["end"]
                    
                    print(f"Transcription: {final_text}")
                    
                    # Save to file
                    with open(TRANSCRIPTION_FILE, "a", encoding="utf-8") as f:
                        f.write(final_text + " ")
                    
                    # Type the transcribed text (excluding commands)
                    if not any(final_text.startswith(cmd) for cmd in ["\n\n", "\n", " ", "\t"]):
                        keyboard.write(final_text)
                        keyboard.write(" ")
            else:
                partial = json.loads(rec.PartialResult())["partial"]
                if partial:
                    # Only print partial if it has changed
                    if partial != last_partial:
                        print(f"Partial: {partial}", end="\r")
                        last_partial = partial
                    
                    # Check for commands in partial results (substring match)
                    for command, action in COMMANDS.items():
                        if command in partial.lower():
                            print(f"\nCommand: {command}")
                            if action == "cmd_stop_listening":
                                stop_listening = True
                            elif action == "cmd_select_all":
                                keyboard.press_and_release("ctrl+a")
                            elif action == "cmd_copy":
                                keyboard.press_and_release("ctrl+c")
                            elif action == "cmd_paste":
                                keyboard.press_and_release("ctrl+v")
                            elif action == "cmd_delete":
                                keyboard.press_and_release("backspace")
                            elif action == "cmd_open_file":
                                print("Opening file... (placeholder)")
                            elif action == "cmd_save_document":
                                print("Saving document... (placeholder)")
                            else:
                                keyboard.write(action)
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
        wf.setframerate(args.samplerate)
        wf.writeframes(audio_data.tobytes())
    print(f"Audio saved to {WAV_FILE}")
