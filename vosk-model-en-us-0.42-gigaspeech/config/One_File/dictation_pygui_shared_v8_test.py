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
import math
import time
import numpy as np
import sounddevice as sd
import threading
import queue
from collections import deque
from vosk import Model, KaldiRecognizer
import logging
import keyboard
import sys
import wave

# Helper function to get the base path for bundled files
def get_base_path():
    if hasattr(sys, '_MEIPASS'):
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
        self.audio_buffer = deque(maxlen=int(self.samplerate * 1.0))  # 1 second of data
        self.update_interval = 1.0 / 30  # Update at 30 FPS
        self.last_update = 0

        # Determine dtype and max_value
        self.dtype, self.max_value = self.get_dtype_and_max()
        
        # DPG setup
        with dpg.window(label="Waveform Display", tag="waveform_window", width=800, height=300, on_close=self.close):
            with dpg.plot(label="Waveform (16-bit PCM, as Vosk hears)", height=200, width=-1):
                dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag="waveform_x_axis")
                dpg.add_plot_axis(dpg.mvYAxis, label="Amplitude", tag="waveform_y_axis")
                dpg.set_axis_limits("waveform_y_axis", -32768, 32767)
                dpg.set_axis_limits("waveform_x_axis", 0, 1.0)
                self.time_axis = np.linspace(0, 1.0, int(self.samplerate * 1.0))
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
                    data_normalized = data * (32767 / self.max_value)
                elif self.dtype == "float32":
                    data_normalized = data * 32767
                else:
                    raise ValueError(f"Unsupported dtype: {self.dtype}")

                data_scaled = data_normalized * self.pre_scale_factor
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
        self.stop_stream()
        dpg.delete_item("waveform_window")

class DictationGUI:
    def __init__(self, command_callback=None):
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
        self.transcribed_text = []
        self.caps_lock_on = False
        self.command_callback = command_callback  # Allow main script to override
        
        # Debug recording
        self.is_debug_recording = False
        self.debug_audio_buffer = []
        
        # Recordings directory
        self.default_recordings_dir = os.path.join(os.getcwd(), "Recordings")
        self.recordings_dir = self.default_recordings_dir
        if not os.path.exists(self.default_recordings_dir):
            os.makedirs(self.default_recordings_dir)
        
        # Initialize saved settings with defaults
        self.saved_settings = {}
        
        # JSON data
        self.json_data = {}
        self.number_mappings = self.load_number_mappings()
        
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
        
        self.apply_theme("Dark")
        self.load_settings()
        self.load_commands()
        self.create_file_dialog()
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("primary_window", True)
        
        # Start GUI update loop
        self.update_gui()

    def load_number_mappings(self):
        mappings = {}
        for json_file in [LARGE_NUMBERS_MAP_PATH, GOOGLE_NUMBERS_PATH]:
            try:
                base_path = get_base_path()
                full_path = os.path.join(base_path, json_file)
                if os.path.exists(full_path):
                    with open(full_path, "r", encoding="utf-8") as f:
                        mappings.update(json.load(f))
                else:
                    logging.warning(f"Number mapping file not found: {full_path}")
            except Exception as e:
                logging.error(f"Error loading number mappings from {json_file}: {e}")
        return mappings

    def words_to_numbers(self, text):
        if not text:
            return text
        words = text.lower().split()
        result = []
        i = 0
        while i < len(words):
            word = words[i]
            if word in self.number_mappings:
                result.append(self.number_mappings[word])
                i += 1
                continue
            matched = False
            for j in range(len(words) - i, 0, -1):
                phrase = " ".join(words[i:i+j])
                if phrase in self.number_mappings:
                    result.append(self.number_mappings[phrase])
                    i += j
                    matched = True
                    break
            if not matched:
                result.append(word)
                i += 1
        return " ".join(str(x) for x in result)

    def process_recognized_text(self, text):
        if not text:
            return

        text_clean = text.strip().lower()
        logging.debug(f"Processing recognized text: '{text_clean}'")

        # Load commands if not already loaded
        if not hasattr(self, "commands"):
            self.load_commands()

        # Check for commands
        command_key = None
        for key, value in self.commands.get("simple_commands", {}).items():
            if text_clean == key.lower():
                command_key = value
                break

        if command_key:
            logging.info(f"Executing command: {command_key}")
            self.handle_command(command_key)
            if command_key == "cmd_stop_dictation":
                self.stop_dictation(None, None)
        else:
            try:
                formatted_text = self.words_to_numbers(text)
                if self.caps_lock_on:
                    formatted_text = formatted_text.upper()
                self.type_text(formatted_text + " ")
                self.transcribed_text.append(formatted_text + " ")
                full_text = "".join(self.transcribed_text).rstrip()
                dpg.set_value("output_text", full_text)
                with open("dictation_output_gigaspeech.txt", "a", encoding="utf-8") as f:
                    f.write(formatted_text + " ")
                global last_dictated_length
                last_dictated_length = len(formatted_text)
                dpg.set_value("status_text", f"Typed: {formatted_text}")
            except Exception as e:
                logging.error(f"Error typing text: {e}")
                dpg.set_value("status_text", f"Error typing text: {e}")

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
        except Exception as e:
            dpg.set_value("status_text", f"Failed to save model path: {e}")

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
        return [host_api['name'] for host_api in sd.query_hostapis()]

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
                dpg.set_value("status_text", "No valid recordings directory.")
                return
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
                indata_normalized = indata_left * (32767 / max_value)
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

    def show_waveform(self, sender, app_data):
        if dpg.does_item_exist("waveform_window"):
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

    def load_commands(self):
        try:
            base_path = get_base_path()
            full_path = os.path.join(base_path, COMMANDS_JSON_PATH)
            if not os.path.exists(full_path):
                logging.warning(f"commands.json not found at {full_path}. Creating default.")
                self.commands = {"simple_commands": {}, "parameterized_commands": []}
                with open(full_path, "w", encoding="utf-8") as f:
                    json.dump(self.commands, f, indent=4)
                return
            with open(full_path, "r", encoding="utf-8") as f:
                self.commands = json.load(f)
            logging.info("Commands loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load commands.json: {e}")
            self.commands = {"simple_commands": {}, "parameterized_commands": []}
            base_path = get_base_path()
            full_path = os.path.join(base_path, COMMANDS_JSON_PATH)
            with open(full_path, "w", encoding="utf-8") as f:
                json.dump(self.commands, f, indent=4)

    def type_text(self, text):
        try:
            if text == "\n\n":
                keyboard.press_and_release("enter")
                keyboard.press_and_release("enter")
            else:
                keyboard.write(text)
            logging.debug(f"Typed: {text}")
            dpg.set_value("status_text", f"Typed: {text}")
        except Exception as e:
            logging.error(f"Error typing text: {e}")
            dpg.set_value("status_text", f"Error typing text: {e}")

    def handle_command(self, action):
        logging.info(f"Handling command action: {action}")
        dpg.set_value("status_text", f"Executing command: {action}")

        try:
            if action == "cmd_select_all":
                keyboard.press_and_release("ctrl+a")
                dpg.set_value("status_text", "Selected all.")
            elif action == "cmd_copy":
                keyboard.press_and_release("ctrl+c")
                dpg.set_value("status_text", "Copied selection.")
            elif action == "cmd_paste":
                keyboard.press_and_release("ctrl+v")
                dpg.set_value("status_text", "Pasted content.")
            elif action == "cmd_delete":
                keyboard.press_and_release("backspace")
                dpg.set_value("status_text", "Deleted character.")
            elif action == "cmd_undo":
                keyboard.press_and_release("ctrl+z")
                dpg.set_value("status_text", "Undo performed.")
            elif action == "cmd_redo":
                keyboard.press_and_release("ctrl+y")
                dpg.set_value("status_text", "Redo performed.")
            elif action == "cmd_save_document":
                keyboard.press_and_release("ctrl+s")
                dpg.set_value("status_text", "Saved document.")
            elif action == "cmd_move_up":
                keyboard.press_and_release("up")
                dpg.set_value("status_text", "Moved cursor up.")
            elif action == "cmd_move_down":
                keyboard.press_and_release("down")
                dpg.set_value("status_text", "Moved cursor down.")
            elif action == "cmd_move_left":
                keyboard.press_and_release("left")
                dpg.set_value("status_text", "Moved cursor left.")
            elif action == "cmd_move_right":
                keyboard.press_and_release("right")
                dpg.set_value("status_text", "Moved cursor right.")
            elif action == "cmd_enter":
                keyboard.press_and_release("enter")
                self.transcribed_text.append("\n")
                full_text = "".join(self.transcribed_text).rstrip()
                dpg.set_value("output_text", full_text)
                with open("dictation_output_gigaspeech.txt", "a", encoding="utf-8") as f:
                    f.write("\n")
                dpg.set_value("status_text", "Pressed Enter.")
            elif action == "cmd_caps_lock_on":
                if not self.caps_lock_on:
                    keyboard.press_and_release("caps lock")
                    self.caps_lock_on = True
                dpg.set_value("status_text", "Caps lock on.")
            elif action == "cmd_caps_lock_off":
                if self.caps_lock_on:
                    keyboard.press_and_release("caps lock")
                    self.caps_lock_on = False
                dpg.set_value("status_text", "Caps lock off.")
            elif action == "cmd_punctuation_period":
                self.type_text(".")
                dpg.set_value("status_text", "Typed period.")
            elif action == "cmd_punctuation_comma":
                self.type_text(",")
                dpg.set_value("status_text", "Typed comma.")
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
                dpg.set_value("status_text", "Last dictation removed.")
            elif action == "cmd_stop_dictation":
                self.stop_dictation(None, None)
            else:
                logging.warning(f"Unknown command action: {action}")
                dpg.set_value("status_text", f"Unknown command: {action}")
        except Exception as e:
            logging.error(f"Error handling command {action}: {e}")
            dpg.set_value("status_text", f"Error handling command: {e}")

    def setup_json_tab(self, tab_name, json_path):
        with dpg.group():
            with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp, tag=f"{tab_name}_table"):
                dpg.add_table_column(label="Key")
                dpg.add_table_column(label="Value")
                base_path = get_base_path()
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
        base_path = get_base_path()
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
            base_path = get_base_path()
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

        # Gather settings
        bit_depth = dpg.get_value("bit_depth_combo") if dpg.does_item_exist("bit_depth_combo") else "int16"
        if " - " in bit_depth:
            bit_depth = bit_depth.split(" - ")[0]
        sample_rate = dpg.get_value("sample_rate_combo") if dpg.does_item_exist("sample_rate_combo") else "16000"
        device_index = self.get_device_index(
            dpg.get_value("input_device_combo") if dpg.does_item_exist("input_device_combo") else None,
            dpg.get_value("host_api_combo") if dpg.does_item_exist("host_api_combo") else "MME",
            is_input=True
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
        else:
            pre_scale = 0.002
        relative_sensitivity = dpg.get_value("relative_sensitivity_check") if dpg.does_item_exist("relative_sensitivity_check") else False
        model_path = dpg.get_value("model_path_input") if dpg.does_item_exist("model_path_input") else "Not set"

        if model_path == "Not set" or not os.path.exists(model_path):
            dpg.set_value("status_text", "Please set a valid model path.")
            logging.error("Invalid or unset model path.")
            return

        try:
            # Initialize Vosk model
            model = Model(model_path)
            self.recognizer = KaldiRecognizer(model, int(sample_rate))
            dtype, max_value = self.get_dtype_and_max(bit_depth)
            self.is_dictating = True
            dpg.configure_item("start_dictation_button", enabled=False)
            dpg.configure_item("stop_dictation_button", enabled=True)
            dpg.set_value("status_text", "Dictation started.")

            # Initialize transcription storage
            self.transcribed_text = []
            self.caps_lock_on = False
            global number_lock_on
            number_lock_on = False
            global last_dictated_length
            last_dictated_length = 0

            # Clear output file
            with open("dictation_output_gigaspeech.txt", "w", encoding="utf-8") as f:
                f.write("")

            def dictation_callback(indata, frames, time_info, status):
                if not self.is_dictating:
                    return

                try:
                    data = np.frombuffer(indata, dtype=dtype).reshape(-1, 2)
                    data = data[:, 0].astype(np.float32)

                    if dtype == "int8":
                        data_normalized = data * (32767 / 127)
                    elif dtype == "int16":
                        data_normalized = data
                    elif dtype == "int24" or dtype == "int32":
                        if max_value == 8388607:
                            data_normalized = data * (32767 / 8388607)
                        else:
                            data_normalized = data * (32767 / 2147483647)
                    elif dtype == "float32":
                        data_normalized = data * 32767
                    else:
                        raise ValueError(f"Unsupported dtype: {dtype}")

                    data_scaled = data_normalized * pre_scale
                    data_scaled = np.clip(data_scaled, -32768, 32767).astype(np.int16)

                    if self.recognizer.AcceptWaveform(data_scaled.tobytes()):
                        result = json.loads(self.recognizer.Result())
                        text = result.get("text", "")
                        if text:
                            logging.debug(f"Recognized text: {text}")
                            self.process_recognized_text(text)
                    else:
                        partial_result = json.loads(self.recognizer.PartialResult())
                        partial_text = partial_result.get("partial", "")
                        if partial_text:
                            logging.debug(f"Partial text: {partial_text}")

                    if dpg.get_value("debug_recording_check"):
                        self.is_debug_recording = True
                        self.debug_audio_buffer.append(data.copy())
                    else:
                        if self.is_debug_recording:
                            self.is_debug_recording = False
                            self.save_debug_recording()
                            self.debug_audio_buffer = []

                except Exception as e:
                    logging.error(f"Error in dictation callback: {e}")
                    dpg.set_value("status_text", f"Dictation error: {e}")

            # Start audio stream
            self.dictation_stream = sd.RawInputStream(
                samplerate=int(sample_rate),
                blocksize=32000,
                device=device_index,
                dtype=dtype,
                channels=2,
                callback=dictation_callback
            )
            self.dictation_stream.start()
            logging.info("Dictation stream started.")

        except Exception as e:
            self.is_dictating = False
            dpg.configure_item("start_dictation_button", enabled=True)
            dpg.configure_item("stop_dictation_button", enabled=False)
            dpg.set_value("status_text", f"Failed to start dictation: {e}")
            logging.error(f"Failed to start dictation: {e}")
            
    def stop_dictation(self, sender, app_data):
        logging.debug("Stop dictation button clicked.")
        if not self.is_dictating:
            dpg.set_value("status_text", "Dictation not running.")
            return

        self.is_dictating = False
        if self.dictation_stream:
            try:
                self.dictation_stream.stop()
                self.dictation_stream.close()
                self.dictation_stream = None
            except Exception as e:
                logging.error(f"Error stopping dictation stream: {e}")
        if self.recognizer:
            self.recognizer = None
        if self.is_debug_recording:
            self.is_debug_recording = False
            self.save_debug_recording()
            self.debug_audio_buffer = []

        dpg.configure_item("start_dictation_button", enabled=True)
        dpg.configure_item("stop_dictation_button", enabled=False)
        dpg.set_value("status_text", "Dictation stopped.")
        logging.info("Dictation stopped.")