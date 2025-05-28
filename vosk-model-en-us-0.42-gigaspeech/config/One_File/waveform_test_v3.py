
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


import pyqtgraph as pg  # For legacy support if needed, but prefer Matplotlib

# Replace pyqtgraph with Matplotlib for better integration
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure
import numpy as np
import sounddevice as sd
import logging
import sys
import threading
import queue
from PyQt5.QtWidgets import QApplication, QMainWindow

class WaveformDisplay(QMainWindow):
    def __init__(self, bit_depth, channels, samplerate, device, pre_scale_factor, relative_sensitivity, target_fps=30):
        super().__init__()
        self.setWindowModifications(True)
        self.setWindowTitle("Waveform Display")
        
        # Initialize Matplotlib
        self.fig = Figure(figsize=(8, 4))
        self.canvas = self (width=8, height=4)
        self.toolbar = NavToolbar(self.canvas, self)

        # Create a canvas subplot
        self.ax = self.fig.add_subplot(111)
        
        # Initialize data buffers and visualization parameters
        self.bit_depth = bit_depth
        self.channels = channels
        self.samplerate = samplerate
        self.device = device
        self.pre_scale_factor = pre_scale_factor
        self.relative_sensitivity = relative_sensitivity
        
        # Create audio stream
        self.stream = None
        self.is_running = True

        # Create Matplotlib animation and update function
        self.ani = None
        self.target_fps = target_fps
        self.data = []
        self.x = []

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

                data_scaled = data_normalized * self.pre_scale_factor
                data_scaled = np.clip(data_scaled, -32768, 32767)

                for sample in data_scaled:
                    if not hasattr(self, "audio_buffer") or getattr(self, "buffer_index", 0) < len(self.audio_buffer):
                        self.audio_buffer[self.buffer_index] = sample
                        self.buffer_index = (self.buffer_index + 1) % len(self.audio_buffer)
            except Exception as e:
                logging.error(f"Error in waveform audio callback: {e}")

        try:
            blocksize = int(self.samplerate / 30)
            self.stream = sd.RawInputStream(
                samplerate=self.samplerate,
                blocksize=blocksize,
                device=self.device,
                dtype=self.dtype,
                channels=self.channels,
                callback=audio_callback
            )
            self.stream.start()
        except Exception as e:
            logging.error(f"Failed to start waveform stream: {e}")
            self.is_running = False

    def update_plot(self):
        if not self.is_running:
            return
        
        # Roll the buffer so the latest data is at the end
        try:
            shifted_buffer = np.roll(self.audio_buffer, -self.buffer_index)
            self.data = shifted_buffer[::self.downsample_factor]
            self.x = np.arange(0, len(self.data)) * (1 / self.target_fps)

            # Update plot
            self.ax.clear()
            self.ax.plot(self.x, self.data)
            self.ax.set_xlim(0, 0.5)
            self.ax.set_ylim(-32768, 32768)
        except Exception as e:
            logging.error(f"Error updating waveform: {e}")

    def stop_stream(self):
        if hasattr(self, "stream"):
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logging.error(f"Error stopping waveform stream: {e}")

    def closeEvent(self, event):
        self.is_running = False
        self.stop_stream()
        self.close()

# Example usage within the main application
if __name__ == "__main__":
    # Initialize PyQt5 app
    app = QApplication(sys.argv)
    
    # Create and show waveform display window
    window = WaveformDisplay(bit_depth="int16", channels=1, samplerate=44100, device=None, pre_scale_factor=1.0, relative_sensitivity=1.0, target_fps=30)
    window.show()
    
    # Run the application event loop
    sys.exit(app.exec_())
