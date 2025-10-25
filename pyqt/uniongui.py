# notebooksAuthor: Daniel Lomholt Christensen Fall 2025 
# Using pyqt 5 because this is what mcgui version 3.5.32 uses
import sys
import watchdog
import nbformat
import nbconvert
from nbconvert.preprocessors import ExecutePreprocessor
import argparse
import os
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QMainWindow



def parse_args():
    parser = argparse.ArgumentParser(description="Program to facilitate rapid development of McStas Union samples."
        + "Provides a GUI for this purpose, that updates when the given instrument is updated."
        + " For now, this program only supports python based instruments.")
    parser.add_argument("-name", required=True, help="Name of the file")
    return parser.parse_args()



def run_notebook_and_capture_variables(notebook_path):
    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Create a shared namespace
    namespace = {}

    # Execute each code cell manually
    for cell in nb.cells:
        if cell.cell_type == 'code':
            # Check if the code contains a backengine, and then don't execute the backengine().
            try:
                exec(cell.source, namespace)
            except Exception as e:
                print(f"Error executing cell:\n{cell.source}\n{e}")

    return namespace



app = QApplication(sys.argv)

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Union GUI")

window = MainWindow()
window.setGeometry(100, 100, 280, 80)
window.show()

if __name__ == '__main__':
    args = parse_args()
    # filepath = os.path.abspath(args.name)
    # print(f"Resolved path: {filepath}")
    #
    # namespace = run_notebook_and_capture_variables(filepath)
    # print(namespace.keys())  # See what variables were defined
    #print(namespace['some_variable'])  # Access a specific variable


    #with open(filepath) as f:
    #    nb = nbformat.read(f, as_version=4)
    #ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    #ep.preprocess(nb, {'metadata': {'path': './'}})
    #print(H_inc)
    # TODO: if the file is jupyter, make it executable 
    sys.exit(app.exec())



