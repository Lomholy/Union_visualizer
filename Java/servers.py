import subprocess
import sys
import os
import threading
import signal
import webbrowser

WEB_DIR = os.path.dirname(os.path.abspath(__file__))

# Flag to open the browser only once
browser_opened = False

def stream_vite_output(pipe):
    global browser_opened
    for line in iter(pipe.readline, b''):
        decoded = line.decode().rstrip()
        print(f"[VITE] {decoded}")
        if not browser_opened and "Local:" in decoded:
            # Extract the Vite dev server URL
            vite_url = decoded.split("Local:")[1].strip()
            print(f"🚀 Vite dev server available at {vite_url}")
            # Open default browser
            webbrowser.open(vite_url)
            browser_opened = True

# Start FastAPI server
py_server = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "mcstas_parser:app", "--host", "127.0.0.1", "--port", "5000"],
    stdout=sys.stdout,
    stderr=sys.stderr
)
print("⚡ FastAPI server started at http://127.0.0.1:5000")

# Start Vite dev server
vite_server = subprocess.Popen(
    ["npx vite"],
    cwd=WEB_DIR,
    shell=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT
)
print("over vite now")
# Start a thread to watch Vite stdout and open browser
threading.Thread(target=stream_vite_output, args=(vite_server.stdout,), daemon=True).start()

try:
    py_server.wait()
    vite_server.wait()
except KeyboardInterrupt:
    print("\nShutting down servers...")
    py_server.send_signal(signal.SIGINT)
    vite_server.send_signal(signal.SIGINT)
