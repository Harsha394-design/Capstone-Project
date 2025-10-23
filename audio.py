import sounddevice as sd
import numpy as np
import config  # Import from config.py

# Placeholders and global variables
SOUND_AMPLITUDE = 0
AUDIO_CHEAT = 0

# Sound variables
CALLBACKS_PER_SECOND = 38
SUS_FINDING_FREQUENCY = 2
SOUND_AMPLITUDE_THRESHOLD = 20

FRAMES_COUNT = int(CALLBACKS_PER_SECOND / SUS_FINDING_FREQUENCY)
AMPLITUDE_LIST = [0] * FRAMES_COUNT
SUS_COUNT = 0
count = 0

def calculate_rms(indata):
    return np.sqrt(np.mean(indata**2)) * 1000

def print_sound(indata, outdata, frames, time, status):
    global SOUND_AMPLITUDE, SUS_COUNT, count, SOUND_AMPLITUDE_THRESHOLD, AUDIO_CHEAT
    rms_amplitude = calculate_rms(indata)
    AMPLITUDE_LIST.append(rms_amplitude)
    count += 1
    AMPLITUDE_LIST.pop(0)
    
    if count == FRAMES_COUNT:
        avg_amp = sum(AMPLITUDE_LIST) / FRAMES_COUNT
        SOUND_AMPLITUDE = avg_amp
        
        if SUS_COUNT >= 2:
            AUDIO_CHEAT = 1
            SUS_COUNT = 0
        
        if avg_amp > SOUND_AMPLITUDE_THRESHOLD:
            SUS_COUNT += 1
        else:
            SUS_COUNT = 0
            AUDIO_CHEAT = 0
        
        count = 0

def sound():
    with sd.Stream(callback=print_sound):
        while config.RUNNING.is_set():  # Use config.RUNNING
            sd.sleep(100)
        print("Audio proctoring stopped")

if __name__ == "__main__":
    sound()