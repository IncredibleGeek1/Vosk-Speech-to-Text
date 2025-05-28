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


import numpy as np
import sounddevice as sd
import logging
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication
import sys

class WaveformDisplay:
    def __init__(self, bit_depth, channels, samplerate, device, pre_scale_factor, relative_sensitivity, target_fps=30):
        self.bit_depth = bit_depth
        self.channels = channels
        self.samplerate = samplerate
        self.device = device
        self.pre_scale_factor = pre_scale_factor
        self.relative_sensitivity = relative_sensitivity
        self.is_running = False
        self.buffer_size = int(self.samplerate * 0.5)  # 0.5 seconds buffer
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)  # Pre-allocated array
        self.buffer_index = 0

        # Determine dtype and max_value
        self.dtype, self.max_value = self.get_dtype_and_max()
        
        # Downsample for plotting (e.g., every 10th sample)
        self.downsample_factor = 10
        self.plot_samples = self.buffer_size // self.downsample_factor
        self.time_axis = np.linspace(0, 0.5, self.plot_samples)
        self.audio_data = np.zeros(self.plot_samples)

        # Set up the pyqtgraph plot
        self.app = QApplication(sys.argv)
        self.win = pg.GraphicsLayoutWidget(title="Waveform Display")
        self.win.resize(800, 300)
        self.plot = self.win.addPlot(title="Waveform (16-bit PCM)", labels={'left': 'Amplitude', 'bottom': 'Time (s)'})
        self.plot.setYRange(-32768, 32768)
        self.plot.setXRange(0, 0.5)
        self.curve = self.plot.plot(self.time_axis, self.audio_data, pen=pg.mkPen('g'))
        self.win.show()

        self.start_stream()
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(int(1000 / target_fps))  # Convert to milliseconds (e.g., 33 ms for 30 FPS)

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

                # Append to pre-allocated buffer
                for sample in data_scaled:
                    self.audio_buffer[self.buffer_index] = sample
                    self.buffer_index = (self.buffer_index + 1) % self.buffer_size
            except Exception as e:
                logging.error(f"Error in waveform audio callback: {e}")

        try:
            # Reduce blocksize to get more frequent updates
            self.stream = sd.RawInputStream(
                samplerate=self.samplerate, blocksize=int(self.samplerate // 30), device=self.device,
                dtype=self.dtype, channels=self.channels, callback=audio_callback
            )
            self.stream.start()
        except Exception as e:
            logging.error(f"Failed to start waveform stream: {e}")
            self.is_running = False
    
    def update_plot(self):
        if not self.is_running:
            return
        try:
            # Roll the buffer so the latest data is at the end
            shifted_buffer = np.roll(self.audio_buffer, -self.buffer_index)
            self.audio_data = shifted_buffer[::self.downsample_factor]
            self.curve.setData(self.time_axis, self.audio_data)
        except Exception as e:
            logging.error(f"Error updating waveform: {e}")

    def stop_stream(self):
        self.is_running = False
        if hasattr(self, "stream"):
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logging.error(f"Error stopping waveform stream: {e}")

    def close(self):
        self.is_running = False
        self.stop_stream()
        self.win.close()
        self.app.quit()

# Example usage
if __name__ == "__main__":
    waveform = WaveformDisplay(bit_depth="int16", channels=1, samplerate=44100, device=None, pre_scale_factor=1.0, relative_sensitivity=1.0, target_fps=30)
    sys.exit(waveform.app.exec_())