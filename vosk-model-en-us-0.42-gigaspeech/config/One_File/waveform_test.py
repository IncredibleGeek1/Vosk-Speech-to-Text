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


import sounddevice as sd
import dearpygui as dpg
import os

# Initialize DearPyGUI
dpg.init()

def main():
    # Create the main window
    window = dpg.window("Spectrum Analyser", width=1200, height=800)

    # Window properties (position and size on creation)
    if dpg.get_wndtitle() == "Spectrum Analyser":
        pass  # Remove epitome handling as it's not part of the current script
    else:
        dpg.set_position((500, 500))
        dpg.set_minimizable(False)
    dpg.set_close_button(True)

    # Create dropdown menu for selecting audio files
    with dpg.group("Audio Source", margin_top=10):
        default_files = ["Use default"] + [os.path.join(os.getcwd(), f) for f in os.listdir('.') if f.endswith('.wav')]
        selected_file = dpg.addselectbox(
            "Select Audio File...",
            items=default_files,
            default="Use default"
        )

    # Create play/stop button
    with dpg.button("Play") as play_button:
        def handle_play(sender, user_data):
            if not selected_file:  # Check if a file was selected
                print("No audio file selected")
                return
            
            try:
                # Load the audio file
                audio_path = selected_file[0]  # Get full path from selectbox
                data, rate = sd.read(audio_path)
                
                # Play the audio and capture frequency spectrum
                sd.play(data, rate)
                frequencies = sd.frequencies(rate, len(data))
                
                dpg.show_plot(
                    "Waveform",
                    xtitle="Time (samples)",
                    ytitle="Amplitude",
                    default_data=data,
                    items=(list(range(len(data))) + [0.5*len(data)])
                )
                
                dpg.show_plot(
                    "Spectrum",
                    xtitle="Frequency (Hz)",
                    ytitle="Magnitude",
                    default_data=sd.fft_magnitude(data),
                    items=(list(range(len(frequencies))) + [0.5*len(frequencies)])
                )
            except Exception as e:
                print(f"Error during playback: {e}")

    # Window close handler
    window.on_close = save_settings

    # Show the window
    dpg.show(window)

def get_audio_data(path):
    if not path:
        print("No audio file selected")
        return None
    
    try:
        data, rate = sd.InputStream(samplerate=44100).read =512
        return (data, rate)
    
    except Exception as e:
        print(f"Error reading audio: {e}")
        return None

def save_settings():
    global settings  # Add this line if you want to save user preferences
    with open("spectrum_settings.txt", "w") as f:
        f.write(str(settings))

# Run the main function
if __name__ == "__main__":
    dpg.create_windows()  # Initialize DearPyGUI windowing system
    main()
    
    # Handle window close event (user clicking red X)
    save_settings()

dpg.destroy_window()
