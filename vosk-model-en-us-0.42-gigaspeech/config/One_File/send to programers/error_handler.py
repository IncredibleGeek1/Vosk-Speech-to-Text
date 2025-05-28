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




import logging
import traceback
import dearpygui.dearpygui as dpg
from datetime import datetime

class ErrorHandler:
    def __init__(self, log_file="error.log"):
        # Configure logging to write to a file
        logging.basicConfig(
            filename=log_file,
            level=logging.ERROR,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.logger = logging.getLogger(__name__)

    def handle(self, operation, exception, status_field="status_text", verbose=True):
        # Extract error details
        error_type = type(exception).__name__
        error_message = str(exception)
        full_traceback = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))

        # Create a concise message for the GUI
        gui_message = f"Error in {operation}: {error_type} - {error_message}"

        # Create a verbose message for CLI and log
        verbose_message = (
            f"Operation: {operation}\n"
            f"Error Type: {error_type}\n"
            f"Error Message: {error_message}\n"
            f"Traceback:\n{full_traceback}"
        )

        # Log the full error to the file
        self.logger.error(verbose_message)

        # Print verbose details to CLI if requested
        if verbose:
            print(verbose_message)

        # Update the GUI status field with the concise message
        if dpg.does_item_exist(status_field):
            dpg.set_value(status_field, gui_message)

        # Return the concise message in case the caller needs it
        return gui_message