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


class DictationGUI:
    def __init__(self):
        self.audio_processor = AudioProcessor(os.path.join(os.getcwd(), "Recordings"))
        self.recordings_dir = os.path.join(os.getcwd(), "Recordings")
        if not os.path.exists(self.recordings_dir):
            os.makedirs(self.recordings_dir)
        # Add other initialization code (e.g., for dictation settings, GUI setup)
        self.setup_gui()

    def setup_gui(self):
        dpg.create_context()
        dpg.create_viewport(title="Dictation GUI", width=800, height=600)
        
        with dpg.window(label="Dictation GUI", tag="primary_window"):
            with dpg.tab_bar():
                with dpg.tab(label="Audio Settings"):
                    self.audio_processor.setup_audio_tab()
                with dpg.tab(label="Dictation"):
                    self.setup_dictation_tab()
        
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("primary_window", True)

    def setup_dictation_tab(self):
        # Placeholder for dictation tab setup
        with dpg.group(horizontal=True):
            dpg.add_button(label="Start Dictation", callback=self.start_dictation)
        dpg.add_text("", tag="status_text")
        # Replace with your actual dictation tab setup code

    def start_dictation(self, sender, app_data):
        # Placeholder for dictation logic
        dpg.set_value("status_text", "Dictation started...")
        # Replace with your actual dictation logic
        # Example: Access audio_processor.audio_buffer or audio_processor.save_recording() if needed
        pass

    def load_settings(self):
        self.audio_processor.load_settings()
        # Add non-audio settings loading if needed
        pass

    def save_settings(self):
        self.audio_processor.save_settings()
        # Add non-audio settings saving if needed
        pass

    def run(self):
        dpg.start_dearpygui()
        dpg.destroy_context()

class CustomDictationGUI(DictationGUI):
    def __init__(self):
        super().__init__()
        # Add custom initialization code if needed

    def setup_dictation_tab(self):
        # Override dictation tab setup if needed
        super().setup_dictation_tab()
        # Add custom dictation tab elements
        pass

    def start_dictation(self, sender, app_data):
        # Override dictation logic if needed
        super().start_dictation(sender, app_data)
        # Add custom dictation logic
        pass

if __name__ == "__main__":
    app = CustomDictationGUI()
    app.run()