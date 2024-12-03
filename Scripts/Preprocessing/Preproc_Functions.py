import re
import mne 
import os 
import numpy as np
from pathlib import Path

# %% find the correct annotation and it's duration
def find_anot_onset(annotation,target): 
    first_time = annotation.onset[0]
    for i, annot in enumerate(annotation):
        if annot['description'] == target:
            onset = annot['onset']-first_time  #start time of the annotation
            dur = annot['duration'] # duration
           
            if i+1<len(annotation):
                next_onset = annotation[i+1]['onset'] - first_time
            else:
                next_onset = None 
            
            return onset, dur, first_time
      
# %% use to extract the correct order when concatenating the files later 
def extract_sort_key(file_path):
    # Extract the task type (SRT or WL or others)
    task_match = re.search(r'task-([A-Za-z]+)', file_path.name)
    task_type = task_match.group(1) if task_match else ''
    
    # Extract the run number
    run_match = re.search(r'run-(\d+)', file_path.name)
    run_number = int(run_match.group(1)) if run_match else 0
    
    # Return a tuple (task_type, run_number)
    # This will sort first by task_type alphabetically, and then by run_number numerically
    return (task_type, run_number)
 
# %% annotate the start and ends of each block so we can keep track of them when they are concatenated 
def block_annotate(data, path):
    start_block = data.times[0] # mark start of the file
    dur_block = data.times[-1] - data.times[0] # duration of the file
    
    pattern = r'(SRT|WL)_run-\d+'
    data_name = re.search(pattern, path.name)
    
    annot_file = mne.Annotations(onset = [start_block],
                                 duration = [dur_block],
                                 description = [data_name.group()],
                                 orig_time = None) # no need to sync annotations across files, so stick to relative annotation times
    
    return annot_file
# %% Read the events as well as the audio track and save it separately for later use 
def read_events(data, data_name): 
    sup_chan    = ['STI101','STI102','MISC007']
    out_path    = data_name.parent  / 'events'
    audio_path  = data_name.parent / 'audio'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if not os.path.exists(audio_path):
        os.makedirs(audio_path)    
    
    
    if 'SRT' in data_name.name:
        # keep track of onscreen events
        event_dict_STI101 = {'Baseline' :0,  #   Baseline always same?
                             'Visual'   :11,
                             'Response' :37}
        
        block_num = re.findall(r'\d+', data_name.name)
       
        if block_num[-1] == '1': # if the SRT is 1 then base-line rest will also be contained here 
            event_dict_STI101['Rest Onset']     = 200
            event_dict_STI101['Rest Retrieval'] = 200
        
        # keep track of button press events. Any other values are probably too close together and have to be disentangled. Thumb press probably means restart
        event_dict_STI102 = {'Right Pinky'  :   64,
                             'Right Ring'   :  128,
                             'Right Middle' :  256,
                             'Right Index'  :  512,
                             'Right_Thumb'  : 1024}
    elif 'WL' in data_name.name:
        # keep track of onscreen events
        event_dict_STI101 = {'Baseline'     : 0,
                             'Visual'       : 11,
                             'Recall On/Off': 23,
                             'Start/End'    : 200}
        # keep track of button press events. Any other values are probably too close together and have to be disentangled. Thumb press probably means restart
        event_dict_STI102 = {}
        
    
    events_STI101 = mne.find_events(data, stim_channel=[sup_chan[0]], min_duration=0.0011)    
    events_STI102 = mne.find_events(data, stim_channel=[sup_chan[1]], min_duration=0.0011)  
    
    events_dict = {sup_chan[0]:events_STI101 , sup_chan[1]: events_STI102}
    np.save(out_path / Path(data_name.name[:-13] + '_events.npy'), events_dict)
    
    if sup_chan[2] in data.info.ch_names:
        data_dum = data.copy().pick(sup_chan[2]) # copy file to make sure data is not changed further down the line
        full_audio_path = audio_path / Path(data_name.name[:-13] + '_meg-audio.fif')
        if not os.path.exists(full_audio_path):
            data_dum.save(full_audio_path)
        