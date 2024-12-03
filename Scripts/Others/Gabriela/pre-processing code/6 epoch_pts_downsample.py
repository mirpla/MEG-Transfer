# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 14:52:51 2023

@author: gabriela
"""

import mne
import os
import re
from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'qt')
import numpy as np

from preprocessor.epoching_decim import Epoch

#%%
# data_dir =   r'/analyse/Project0349/ANALYSIS'
# epoch_dirs = [r'/analyse/Project0372/ANALYSIS',r'/analyse/Project0375/ANALYSIS',r'/analyse/Project0376/ANALYSIS',r'/analyse/Project0377/ANALYSIS',r'/analyse/Project0378/ANALYSIS',r'/analyse/Project0379/ANALYSIS']

data_dir =   r'Y:\ANALYSIS'
epoch_dirs = [r'S:\ANALYSIS',r'R:\ANALYSIS',r'Q:\ANALYSIS',r'O:\ANALYSIS',r'N:\ANALYSIS',r'M:\ANALYSIS']

input_dir = 'pts'
output_dir = 'epochs'


subjects_list = [ f.name for f in os.scandir(data_dir) if f.is_dir() and f.name[0] == 'S' ]
subjects_list.sort()
print(subjects_list) 

#%%
redo_list  = {
  "bsr27":  {"session1": False, "session2": False, "session3": False},
  "dsa23":  {"session1": True, "session2": True, "session3": True},
  "mtr13":  {"session1": False, "session2": False, "session3": True},
  "gto28":  {"session1": False, "session2": False, "session3": False},
  "ami28":  {"session1": False, "session2": False, "session3":False},
  "lka10":  {"session1": True, "session2": False, "session3": True},
  "qqn19":  {"session1": True, "session2": True, "session3": False},
  "mtr19":  {"session1": True, "session2": True, "session3": True},
  "fha01":  {"session1": False, "session2": False, "session3": True},
  "hwh21":  {"session1": True, "session2": True, "session3": True},
  "rsh17":  {"session1": True, "session2": False, "session3": True},
  "zwi25":  {"session1": True, "session2": False, "session3": True},
  "tdn02":  {"session1": True, "session2": False, "session3": True},
  "uka11":  {"session1": True, "session2": False, "session3": False},
  "csi07":  {"session1": True, "session2": False, "session3": True},
  "rsg06":  {"session1": True, "session2": False, "session3": False},
  "mwa29":  {"session1": True, "session2": False, "session3": False},
  "ade02":  {"session1": False, "session2": False, "session3": False},
  "dtl05":  {"session1": True, "session2": True, "session3": True},
  "rbe04":  {"session1": False, "session2": False, "session3": False},
  "jmn22":  {"session1": False, "session2": False, "session3": False},
  "dss19":  {"session1": False, "session2": False, "session3": False},
  "fte25":  {"session1": False, "session2": False, "session3": False},
  "hyr24":  {"session1": True, "session2": False, "session3": False},
  "ank24":  {"session1": True, "session2": False, "session3": False},
  "dja01":  {"session1": False, "session2": False, "session3": True},
  "fmn28":  {"session1": True, "session2": True, "session3": False},
  "omr03":  {"session1": True, "session2": True, "session3": True},
  "amy20":  {"session1": True, "session2": False, "session3": False},
  "ski23":  {"session1": False, "session2": True, "session3": False},
  "jry29":  {"session1": False, "session2": False, "session3": False},
  "jpa10":  {"session1": True, "session2": True, "session3": True},
  "epa14":  {"session1": False, "session2": False, "session3": False},
  "crr22":  {"session1": False, "session2": False, "session3": False},
  "jyg27":  {"session1": False, "session2": False, "session3": False},
  "bbi29":  {"session1": False, "session2": False, "session3": False},
  "ece24":  {"session1": False, "session2": False, "session3": False},
  "hay06":  {"session1": False, "session2": False, "session3": False},
  "awa19":  {"session1": False, "session2": False, "session3": False},
  "mca10":  {"session1": False, "session2": False, "session3": False},
  "hky23":  {"session1": False, "session2": False, "session3": False},
  } 
#overwrite = True
#%%

# Define some interesting epochs and their conditions
sfreq = 1000
n_jobs = 'cuda' # needs cupy. Also be vary of finite GPU resources
n_cycles = 7    # wavelet width
freqs = np.geomspace(1, 100, 50)
epoch_win = [-1500,1500]
length = int(epoch_win[1]-epoch_win[0]) # Set length explicitly in samples, just for robustness



#%%

for subject_id in subjects_list:
    # subject_id = 'S002_dsa23'
    print('Doing subject ' + subject_id + '\File')
    subject_path = os.path.join(data_dir,subject_id)
    data_path =  os.path.join(subject_path,input_dir)
    event_path = os.path.join(subject_path,'events','attention')
    
    # make a directory for epochs
    check_other_dirs = True
    idx = 0
    while check_other_dirs:
        
        use_dir = epoch_dirs[idx] 
         
        if not os.path.exists(use_dir):
            os.makedirs(use_dir)
                
        if len(os.listdir(use_dir)) < 12:
            epoch_dir =  os.path.join(use_dir,subject_id, output_dir)
            check_other_dirs = False
        elif len(os.listdir(use_dir)) >= 12:
                epoch_dir =  os.path.join(use_dir,subject_id, output_dir)
                if os.path.exists(epoch_dir):
                    check_other_dirs = False
                else:
                    idx += 1
                
    #epoch_dir_old = os.path.join(data_dir,subject_id, output_dir)
    #epoch_dir = os.path.join(data2_dir,subject_id, output_dir)
    if not os.path.exists(epoch_dir):
        os.makedirs(epoch_dir)
        
    pts_files = os.listdir(data_path)
    event_files = os.listdir(event_path)
    
    if pts_files == []:
        continue
    pts_files.sort()
    
    for pts_file in pts_files: # loop through block files
        print(pts_file)
        
        saving_fname = '_'.join([pts_file.split('_')[0],pts_file.split('_')[1],'epts'] )
        if any(saving_fname in s for s in os.listdir(epoch_dir) ): # or os.listdir(epoch_dir_old)
            if redo_list[subject_id[5:]]['session' + str(idx+1)] == False:
                continue
            else:
                print('but it will be done again!!!!')
                
            
        
        event_file = [i for i in event_files if re.findall('[0-9]+',pts_file.split('_')[1][1:])[0] + '.' in i]
        event_file.sort()
        # load parcel timeseries and events
        pts = np.load(os.path.join(data_path,pts_file))
        events = np.load(os.path.join(event_path,event_file[0]))
        events_tags = np.load(os.path.join(event_path,event_file[1]),allow_pickle=True)
        events_tags = events_tags.item()
        
        
        # Now working outside MNE, therefore we need to get first sample to correct event latencies
        ica_dir = os.path.join(data_dir,subject_id,'ica',pts_file.split('_')[0])
        ica_file = [i for i in os.listdir(ica_dir) if re.findall('[0-9]+',pts_file.split('_')[1][1:])[0] in i] # os.path.join(data_dir,subject_id,'ica')
        ica = mne.io.read_raw_fif(os.path.join(ica_dir,ica_file[0]),preload = True)
        first_samp = ica.first_samp
        # ica.plot(events=events,duration=20.0)
        # exclude muscle and eye blinks using annotation
        annot, annot_id = mne.events_from_annotations(ica)
        incl = [True] * len(events)
        for e in range(len(events)):
            if str(events[e,2]).startswith('4'):
                for a in range(len(annot)):
                    if events[e,0]  <= annot[a,0] <= events[e+1,0]:
                        print(e)
                        incl[e] = False
        
        events = events[incl]

        del ica
        
        # correct event latencies
        events[:,0] = events[:,0] - first_samp
        
        '''
        I have already coded the events, details of every event type is in the event_tags file
        I'm not including correct catch trials for the moment, having a behavioural correlate of attention (correct_hit) reassure us participant was putting attention on the trial'
        
        17 May 2024: include catch trials

        '''
        cor_seqs = {'double_noattc'         :  [tuple([int(events_tags[x]) for x in ['double/noAtt/noCue/correct_miss'] if x in events_tags]),(16,)],
                    'double_neutralc'       :  [tuple([int(events_tags[x]) for x in ['double/neutral/neutral/correct_hit'] if x in events_tags]),(8,),(16,)], #,'double/neutral/catch/correct_miss' 
                    'double_neutralcatchc'  :  [tuple([int(events_tags[x]) for x in ['double/neutral/catch/correct_miss' ] if x in events_tags]),(16,)],
                    'double_leftc'          :  [tuple([int(events_tags[x]) for x in ['double/left/valid/correct_hit','double/left/invalid/correct_hit'] if x in events_tags]),(8,),(16,)], #,'double/left/catch/correct_miss'
                    'double_leftcatchc'    :  [tuple([int(events_tags[x]) for x in ['double/left/catch/correct_miss'] if x in events_tags]),(16,)],
                    'double_rightc'  :  [tuple([int(events_tags[x]) for x in ['double/right/valid/correct_hit','double/right/invalid/correct_hit'] if x in events_tags]),(8,),(16,)], #,'double/right/catch/correct_miss'
                    'double_rightcatchc'  :  [tuple([int(events_tags[x]) for x in ['double/right/valid/correct_hit','double/right/invalid/correct_hit'] if x in events_tags]),(8,),(16,)],
                    'single_noattc'  :  [tuple([int(events_tags[x]) for x in ['single/noAtt/noCue/correct_miss'] if x in events_tags]),(16,)],
                    'single_neutralc':  [tuple([int(events_tags[x]) for x in ['single/neutral/neutral/correct_hit'] if x in events_tags]),(8,),(16,)], #,'single/neutral/catch/correct_miss'
                    'single_leftc'   :  [tuple([int(events_tags[x]) for x in ['single/left/valid/correct_hit','single/left/invalid/correct_hit'] if x in events_tags]),(8,),(16,)], #,'single/left/catch/correct_miss'
                    'single_rightc'  :  [tuple([int(events_tags[x]) for x in ['single/right/valid/correct_hit','single/right/invalid/correct_hit'] if x in events_tags]),(8,),(16,)], #,'single/right/catch/correct_miss'
                    'nostim_noattc'  :  [tuple([int(events_tags[x]) for x in ['noStim/noAtt/noCue/correct_miss'] if x in events_tags]),(16,)],
                    'nostim_neutralc':  [tuple([int(events_tags[x]) for x in ['noStim/neutral/neutral/correct_hit'] if x in events_tags]),(8,),(16,)], # ,'noStim/neutral/catch/correct_miss'
                    'nostim_leftc'   :  [tuple([int(events_tags[x]) for x in ['noStim/left/valid/correct_hit','noStim/left/invalid/correct_hit'] if x in events_tags]),(8,),(16,)], #,'noStim/left/catch/correct_miss'
                    'nostim_rightc'  :  [tuple([int(events_tags[x]) for x in ['noStim/right/valid/correct_hit','noStim/right/invalid/correct_hit'] if x in events_tags]),(8,),(16,)], #,'noStim/right/catch/correct_miss'
                    'double_noatti'  :  [tuple([int(events_tags[x]) for x in ['double/noAtt/noCue/incorrect_hit'] if x in events_tags]),(16,)],
                    'double_neutrali':  [tuple([int(events_tags[x]) for x in ['double/neutral/neutral/incorrect_miss'] if x in events_tags]),(8,),(16,)], 
                    'double_lefti'   :  [tuple([int(events_tags[x]) for x in ['double/left/valid/incorrect_miss','double/left/invalid/incorrect_miss'] if x in events_tags]),(8,),(16,)], 
                    'double_righti'  :  [tuple([int(events_tags[x]) for x in ['double/right/valid/incorrect_miss','double/right/invalid/incorrect_miss'] if x in events_tags]),(8,),(16,)], 
                    'single_noatti'  :  [tuple([int(events_tags[x]) for x in ['single/noAtt/noCue/incorrect_hit'] if x in events_tags]),(16,)],
                    'single_neutrali':  [tuple([int(events_tags[x]) for x in ['single/neutral/neutral/incorrect_miss'] if x in events_tags]),(8,),(16,)],
                    'single_lefti'   :  [tuple([int(events_tags[x]) for x in ['single/left/valid/incorrect_miss','single/left/invalid/incorrect_miss'] if x in events_tags]),(8,),(16,)],
                    'single_righti'  :  [tuple([int(events_tags[x]) for x in ['single/right/valid/incorrect_miss','single/right/invalid/incorrect_miss'] if x in events_tags]),(8,),(16,)],
                    'nostim_noatti'  :  [tuple([int(events_tags[x]) for x in ['noStim/noAtt/noCue/incorrect_hit'] if x in events_tags]),(16,)],
                    'nostim_neutrali':  [tuple([int(events_tags[x]) for x in ['noStim/neutral/neutral/incorrect_miss'] if x in events_tags]),(8,),(16,)], 
                    'nostim_lefti'   :  [tuple([int(events_tags[x]) for x in ['noStim/left/valid/incorrect_miss','noStim/left/invalid/incorrect_miss'] if x in events_tags]),(8,),(16,)],
                    'nostim_righti'  :  [tuple([int(events_tags[x]) for x in ['noStim/right/valid/incorrect_miss','noStim/right/invalid/incorrect_miss'] if x in events_tags]),(8,),(16,)], 
                    }
        
        # check all conditions contain epochs, remove conditions with zero epochs.
        # Empty values may happen when a participant did not have an incorrect response for one of the conditions
        # or when invalid trials were not detected
        cor_seqs = {k: v for k, v in cor_seqs.items() if np.array(v[0]).size > 0} 
        
        
        # Instantiate class
        epoch = Epoch(sfreq, epoch_dir, saving_fname, pts, events)     

        # parse sequences from events that satisfy conditions
        epoch.parse_sequences(cor_seqs)
        
        # remove conditions with zero epochs and parse again
        if any(i == 0 for i in epoch.n_epochs.values()):
            cor_seqs = {k: cor_seqs[k] for k, v in epoch.n_epochs.items() if v > 0 }
            epoch.parse_sequences(cor_seqs)
        
        # Time windows of interest
        times = {}
        keys = cor_seqs.keys()
        for k in keys:
            times[k] = {'att':(epoch_win[0], epoch_win[1])}

         # Codes to stimulus types
        codes = {}
        keys = cor_seqs.keys()
        for k in keys:
            codes[k] = {'att':cor_seqs[k][0]}
    

        # make intervals that are used for epoching
        epoch.make_intervals(times, codes, length, decim = 1)
        
        
        # Sanity check        
        # tmp_nepochs = []
        # for i in epoch_types:
        #     tmp_nepochs.append(epoch.n_epochs[i])
        #print('Nr of epochs in ' + saving_fname + ' is ' + str(epoch.n_epochs[relep]) + '.')
        
        # make the epochs, reject bad epochs based on standard deviation
        epoch_types = list(codes.keys())
        epoch.make_epochs(freqs=freqs, n_jobs=n_jobs, epoch_types=epoch_types, n_cycles=n_cycles, reject=True, sd=2)
        
        # save epochs to disk
        # try:
        #     epoch.save_epochs(epoch_types=epoch_types)
        # except:
        # epoch_dir = os.path.join(data2_dir , subject_path , output_dir)
        # if not os.path.exists(epoch_dir):
        #     os.makedirs(epoch_dir)
        # epoch.epoch_dir = epoch_dir
        epoch.save_epochs(epoch_types=epoch_types)
            
#         'miss':    (-1000, 100)}
        
        
        #         # Codes to stimulus types
        # codes =  {'double_noattc'  : {'notarget': (16,)},
        #           'double_neutralc': {'target':   (8,)},
        #           'double_leftc'   : {'target':   (8,)},
        #           'double_rightc'  : {'target':   (8,)},
        #           'single_noattc'  : {'notarget': (16,)},
        #           'single_neutralc': {'target':   (8,)},
        #           'single_leftc'   : {'target':   (8,)},
        #           'single_rightc'  : {'target':   (8,)},
        #           'nostim_noattc'  : {'notarget': (16,)},
        #           'nostim_neutralc': {'target':   (8,)},
        #           'nostim_leftc'   : {'target':   (8,)},
        #           'nostim_rightc'  : {'target':   (8,)},
        #           'double_noatti'  : {'notarget': (16,)},
        #           'double_neutrali': {'target':   (8,)},
        #           'double_lefti'   : {'target':   (8,)},
        #           'double_righti'  : {'target':   (8,)},
        #           'single_noatti'  : {'notarget': (16,)},
        #           'single_neutrali': {'target':   (8,)},
        #           'single_lefti'   : {'target':   (8,)},
        #           'single_righti'  : {'target':   (8,)},
        #           'nostim_noatti'  : {'notarget': (16,)},
        #           'nostim_neutrali': {'target':   (8,)},
        #           'nostim_lefti'   : {'target':   (8,)},
        #           'nostim_righti'  : {'target':   (8,)},
        #           }


        # codes2 =  {'double_noattc'  : {'att': [tuple([int(events_tags[x]) for x in ['double/noAtt/noCue/correct_miss'] if x in events_tags])]},
        #           'double_neutralc': {'att': [tuple([int(events_tags[x]) for x in ['double/neutral/neutral/correct_hit'] if x in events_tags])]},
        #           'double_leftc'   : {'att': [tuple([int(events_tags[x]) for x in ['double/left/valid/correct_hit','double/left/invalid/correct_hit'] if x in events_tags])]},
        #           'double_rightc'  : {'att': [tuple([int(events_tags[x]) for x in ['double/right/valid/correct_hit','double/right/invalid/correct_hit'] if x in events_tags])]},
        #           'single_noattc'  : {'att': [tuple([int(events_tags[x]) for x in ['single/noAtt/noCue/correct_miss'] if x in events_tags])]},
        #           'single_neutralc': {'att': [tuple([int(events_tags[x]) for x in ['single/neutral/neutral/correct_hit'] if x in events_tags])]},
        #           'single_leftc'   : {'att': [tuple([int(events_tags[x]) for x in ['single/left/valid/correct_hit','single/left/invalid/correct_hit'] if x in events_tags])]},
        #           'single_rightc'  : {'att': [tuple([int(events_tags[x]) for x in ['single/right/valid/correct_hit','single/right/invalid/correct_hit'] if x in events_tags])]},
        #           'nostim_noattc'  : {'att': [tuple([int(events_tags[x]) for x in ['noStim/noAtt/noCue/correct_miss'] if x in events_tags])]},
        #           'nostim_neutralc': {'att': [tuple([int(events_tags[x]) for x in ['noStim/neutral/neutral/correct_hit'] if x in events_tags])]},
        #           'nostim_leftc'   : {'att': [tuple([int(events_tags[x]) for x in ['noStim/left/valid/correct_hit','noStim/left/invalid/correct_hit'] if x in events_tags])]},
        #           'nostim_rightc'  : {'att': [tuple([int(events_tags[x]) for x in ['noStim/right/valid/correct_hit','noStim/right/invalid/correct_hit'] if x in events_tags])]},
        #           'double_noatti'  : {'att': [tuple([int(events_tags[x]) for x in ['double/noAtt/noCue/incorrect_hit'] if x in events_tags])]},
        #           'double_neutrali': {'att': [tuple([int(events_tags[x]) for x in ['double/neutral/neutral/incorrect_miss'] if x in events_tags])]},
        #           'double_lefti'   : {'att': [tuple([int(events_tags[x]) for x in ['double/left/valid/incorrect_miss','double/left/invalid/incorrect_miss'] if x in events_tags])]},
        #           'double_righti'  : {'att': [tuple([int(events_tags[x]) for x in ['double/right/valid/incorrect_miss','double/right/invalid/incorrect_miss'] if x in events_tags])]},
        #           'single_noatti'  : {'att': [tuple([int(events_tags[x]) for x in ['single/noAtt/noCue/incorrect_hit'] if x in events_tags])]},
        #           'single_neutrali': {'att': [tuple([int(events_tags[x]) for x in ['single/neutral/neutral/incorrect_miss'] if x in events_tags])]},
        #           'single_lefti'   : {'att': [tuple([int(events_tags[x]) for x in ['single/left/valid/incorrect_miss','single/left/invalid/incorrect_miss'] if x in events_tags])]},
        #           'single_righti'  : {'att': [tuple([int(events_tags[x]) for x in ['single/right/valid/incorrect_miss','single/right/invalid/incorrect_miss'] if x in events_tags])]},
        #           'nostim_noatti'  : {'att': [tuple([int(events_tags[x]) for x in ['noStim/noAtt/noCue/incorrect_hit'] if x in events_tags])]},
        #           'nostim_neutrali': {'att': [tuple([int(events_tags[x]) for x in ['noStim/neutral/neutral/incorrect_miss'] if x in events_tags])]},
        #           'nostim_lefti'   : {'att': [tuple([int(events_tags[x]) for x in ['noStim/left/valid/incorrect_miss','noStim/left/invalid/incorrect_miss'] if x in events_tags])]},
        #           'nostim_righti'  : {'att': [tuple([int(events_tags[x]) for x in ['noStim/right/valid/incorrect_miss','noStim/right/invalid/incorrect_miss'] if x in events_tags])]},
        #           }
    
    
        # times = {'double_noattc'  : {'att': (epoch_win[0], epoch_win[1])},
        #          'double_neutralc': {'att': (epoch_win[0], epoch_win[1])},
        #          'double_leftc'   : {'att': (epoch_win[0], epoch_win[1])}, 
        #          'double_rightc'  : {'att': (epoch_win[0], epoch_win[1])}, 
        #          'single_noattc'  : {'att': (epoch_win[0], epoch_win[1])},
        #          'single_neutralc': {'att': (epoch_win[0], epoch_win[1])},
        #          'single_leftc'   : {'att': (epoch_win[0], epoch_win[1])},
        #          'single_rightc'  : {'att': (epoch_win[0], epoch_win[1])}, 
        #          'nostim_noattc'  : {'att': (epoch_win[0], epoch_win[1])},
        #          'nostim_neutralc': {'att': (epoch_win[0], epoch_win[1])},
        #          'nostim_leftc'   : {'att': (epoch_win[0], epoch_win[1])}, 
        #          'nostim_rightc'  : {'att': (epoch_win[0], epoch_win[1])}, 
        #          'double_noatti'  : {'att': (epoch_win[0], epoch_win[1])},
        #          'double_neutrali': {'att': (epoch_win[0], epoch_win[1])}, 
        #          'double_lefti'   : {'att': (epoch_win[0], epoch_win[1])}, 
        #          'double_righti'  : {'att': (epoch_win[0], epoch_win[1])}, 
        #          'single_noatti'  : {'att': (epoch_win[0], epoch_win[1])},
        #          'single_neutrali': {'att': (epoch_win[0], epoch_win[1])},
        #          'single_lefti'   : {'att': (epoch_win[0], epoch_win[1])},
        #          'single_righti'  : {'att': (epoch_win[0], epoch_win[1])},
        #          'nostim_noatti'  : {'att': (epoch_win[0], epoch_win[1])},
        #          'nostim_neutrali': {'att': (epoch_win[0], epoch_win[1])}, 
        #          'nostim_lefti'   : {'att': (epoch_win[0], epoch_win[1])},
        #          'nostim_righti'  : {'att': (epoch_win[0], epoch_win[1])}, 
 #                  }