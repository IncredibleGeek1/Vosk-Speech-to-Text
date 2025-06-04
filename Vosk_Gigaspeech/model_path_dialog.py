# MIT License
#
# Copyright (c) 2025 IncredibleGeek
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import dearpygui.dearpygui as dpg
import sys
import os
import json

def on_model_path_selected(sender, app_data):
    selected_path = app_data["file_path_name"]
    if selected_path:
        # Save to a temp file or print to stdout
        with open("selected_model_path.txt", "w", encoding="utf-8") as f:
            f.write(os.path.abspath(selected_path))
    dpg.stop_dearpygui()

dpg.create_context()
with dpg.file_dialog(
    directory_selector=True,
    show=True,
    callback=on_model_path_selected,
    tag="file_dialog",
    width=700,
    height=400
):
    dpg.add_file_extension(".*")

dpg.create_viewport(title="Select Model Path", width=720, height=440)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
