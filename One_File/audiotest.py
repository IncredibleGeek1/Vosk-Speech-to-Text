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


from collections import deque
import json
import queue
import sys
import time
import math
import logging
from tkinter import UNITS
import numpy as np
import sounddevice as sd
import dearpygui.dearpygui as dpg
import wave
import os


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


class AudioProcessor:
    def __init__(self, recordings_dir):
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
        self.recordings_dir = recordings_dir
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
    def __init__(self, AudioProcessor):
        self.AudioProcessor = AudioProcessor
        self.samplerate = int(self.audio_processor.saved_settings.get("sample_rate", 48000))
        self.bit_depth = self.audio_processor.saved_settings.get("bit_depth", "int24")
        self.pre_scale_factor = self.audio_processor.saved_settings.get("pre_scale_factor", 0.002)
        self.relative_sensitivity = self.audio_processor.saved_settings.get("relative_sensitivity", False)
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
        dpg.delete_item("waveform_window")





if __name__ == "__main__":

    