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


import sys
import os
from PyQt5.QtWidgets import QApplication, QFileDialog

def main():
    app = QApplication(sys.argv)
    # Show the native directory selection dialog
    folder = QFileDialog.getExistingDirectory(
        None,
        "Select Model Directory",
        os.getcwd(),
        QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
    )
    if folder:
        with open("selected_model_path.txt", "w", encoding="utf-8") as f:
            f.write(os.path.abspath(folder))
    # Exit the app
    sys.exit(0)

if __name__ == "__main__":
    main()