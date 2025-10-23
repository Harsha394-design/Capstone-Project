import time
import audio
import head_pose
import matplotlib.pyplot as plt
import numpy as np
import config

PLOT_LENGTH = 200
GLOBAL_CHEAT = 0
PERCENTAGE_CHEAT = 0
CHEAT_THRESH = 0.6
XDATA = list(range(200))
YDATA = [0] * 200

def avg(current, previous):
    if previous > 1:
        return 0.65
    if current == 0:
        if previous < 0.01:
            return 0.01
        return previous / 1.01
    if previous == 0:
        return current
    return previous + 0.1 * current

def is_any_cheating(values):
    return any(v == 1 for v in values)

def process():
    global GLOBAL_CHEAT, PERCENTAGE_CHEAT
    
    x_cheat = is_any_cheating(head_pose.X_AXIS_CHEAT)
    y_cheat = is_any_cheating(head_pose.Y_AXIS_CHEAT)
    audio_cheat = audio.AUDIO_CHEAT
    
    if x_cheat or y_cheat or audio_cheat:
        PERCENTAGE_CHEAT = avg(0.6, PERCENTAGE_CHEAT)
    else:
        PERCENTAGE_CHEAT = avg(0, PERCENTAGE_CHEAT)

    if PERCENTAGE_CHEAT > CHEAT_THRESH:
        GLOBAL_CHEAT = 1
        print("CHEATING")
    else:
        GLOBAL_CHEAT = 0
    
    print("Cheat percent:", PERCENTAGE_CHEAT, GLOBAL_CHEAT)

def run_detection():
    global XDATA, YDATA
    
    plt.ion()
    fig, axes = plt.subplots()
    axes.set_xlim(0, PLOT_LENGTH)
    axes.set_ylim(0, 1)
    line, = axes.plot(XDATA, YDATA, 'r-')
    
    while config.RUNNING.is_set():
        process()
        
        YDATA.append(PERCENTAGE_CHEAT)
        YDATA.pop(0)

        line.set_ydata(YDATA)
        
        if GLOBAL_CHEAT == 1:
            line.set_color('red')
        else:
            line.set_color('green')
            
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        time.sleep(0.01)

    plt.close('all')