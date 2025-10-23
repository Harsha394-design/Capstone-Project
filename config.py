import threading

# Global flag to control proctoring threads
RUNNING = threading.Event()
RUNNING.set()  # Initially True (proctoring runs)