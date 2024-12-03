
#%%

# import os.path as op
# import mne
# import os
# from mne.preprocessing import ICA
import os.path as op
import os
#import sys
import numpy as np
import matplotlib.pyplot as plt


import mne
#import matplotlib.pyplot as plt

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt') # so MNE plots are in a pop-up window, not inline
# %matplotlib qt

#%% 

# To do, add the response!!!
def create_events(main_event,propixx_triggers,events_tags):
    '''
    Ooutput funtion is a new event structure plus an ID dictionary
    This function will assign a new trigger value for each condition of interest
    The new event matrix will mantain the code of the event of interes and it will add a four diggit value
    to identify the stimulus and attention condition, validity and response type as folow
    stimulus(5-7)  + att (0 - 3) + validity (0-2) + response (0 or 1)
    For example, 45101 corresponds to#:
    4: attention window, 5:double stimulation, 1:right att cue,0: invalid cue, 1: correct response
    '''
     
    # stims code
    double_code = '5'
    noStim_code = '6'
    single_code = '7'
    
    #att code
    left =    '0'
    right =   '1'
    neutral = '2'
    noAtt =   '3'
    
    #validity code
    invalid = '0'
    valid =   '1'
    catch =   '2'
    noCue =   '3'
    
    # response code
    correct_hit = '11'
    correct_miss = '10'
    incorrect_hit = '01'
    incorrect_miss = '00'
    
    
    
    
    id_dict = dict()     
    for nb,tag in enumerate(events_tags):
        #tmp_tag = events_tags[ei]
        tmp_code = [str(main_event)]
        tag = tag.split('/')
        
        if 'double' in tag[0]:
            tmp_code.append(double_code)
        elif 'single' in tag:
            tmp_code.append(single_code)
        elif 'noStim' in tag:
            tmp_code.append(noStim_code)
        else:
            tmp_code.append('9') # every event is either double,single or noStim. A 9 should never follow the main event code
            # add error message!
            
        if 'left' in tag[1]:
            tmp_code.append(left)
        elif 'right' in tag[1]:
            tmp_code.append(right)
        elif 'neutral' in tag[1]:
            tmp_code.append(neutral)
        elif 'noAtt' in tag[1]:
            tmp_code.append(noAtt)
        else:
            tmp_code.append('9') # every event must match an attention condition. 9 should never be here
            # add a messare error!
            
        if 'invalid' in tag[2]:
            tmp_code.append(invalid)
        elif 'valid' in tag[2]:
            tmp_code.append(valid)
        elif 'catch' in tag[2]:
            tmp_code.append(catch)        
        elif 'noCue' in tag[2]:
            tmp_code.append(noCue)    
        else:
            tmp_code.append('9')
            
        if 'incorrect_hit' in tag[3]:
            tmp_code.append(incorrect_hit)
        elif 'incorrect_miss' in tag[3]:
            tmp_code.append(incorrect_miss)
        elif 'correct_miss' in tag[3]:
            tmp_code.append(correct_miss)
        elif 'correct_hit' in tag[3]:
            tmp_code.append(correct_hit) 
        else:
            tmp_code.append('9')
    
    
        #for idx,value in enumerate(tmp_code): # Iterate over all the items in dictionary of Events ID
        if not ''.join(tmp_code) in id_dict.values():
            id_dict['/'.join(tag[:-1])] = ''.join(tmp_code)
        
        
    main_event_idx = np.where(propixx_triggers[:,2] == main_event)[0][:]
    new_events = propixx_triggers.copy()
    for idx,tag in enumerate(events_tags):
        tag = tag.split('/')
        tag = '/'.join(tag[:-1])
        new_events[main_event_idx[idx],2] = id_dict[tag]           
         
        
    return id_dict,new_events
        
    

#%%

# data_dir = r'Y:\ANALYSIS' 
# output_plot = r'Y:\PLOTS\events'

data_dir = r'/analyse/Project0349/ANALYSIS' 
output_plot = r'/analyse/Project0349/PLOTS/events'

#subjects_list = ['S001_bsr27', 'S002_dsa23' ]
subjects_list = [ f.name for f in os.scandir(data_dir) if f.is_dir() and f.name[0] == 'S' ]
print(subjects_list)
subjects_list.sort()

overwrite = False 
#%%
'''
Trial structure
1 - Pre-basline (ITI) 1 - 1.5s (when stim on it appears here)  ['Double','Single','None']
2 - BaseLine 1 - 1.5s 
4 - Attention Window 1 - 1.5s onset of the attention cue ['left','right','neutral','noAtt']
8 - Target 0.1s ['valid','invalid','catch'] valid/invalid only applies to spatial cues 
16 - Response ['correct_miss','correct_hit','incorrect_miss','incorrect_hit']
32 - Precision
'''
# Create a new event file depending on the events of interest
lock2 = 'attention' 

evens_dict = { # R, G, B
    'iti':       1, # (255, 1, 0), # 2^0 -> G0 ->  0001 , STI101 , value 1,
    'baseline':  2, #(0, 2, 0), #G1, STI101 2^1, value 2, 0010
    'attention': 4,# (255, 4, 0), #G2, STI101 2^2, value 4, 0100
    'target':    8, # (0, 8, 0),#':(0, 8, 0), #G3, STI101 2^3, value 8, 1000
    'response':  16, #(255, 0, 1), #B0, STI101 2^4, value 16 
    'precision': 32,#(0, 0, 2), #B1, STI101 2^5, value 32
}


    

# use baseline for NCM

#%%
for subject_id in subjects_list:
    # subject_id = 'S020_rbe04'
    print('Doing subject ' + subject_id + '\nSessions')
    subject_path = op.join(data_dir,subject_id,'ica')
    if os.path.exists(subject_path):
        session_folders = os.listdir(subject_path)
    else:
        print(subject_path + ' does not exist yet')
        continue
        
    sessions =  [x for x in session_folders if x.startswith('2')]
    print(sessions)
    sessions.sort()
    save_plot = op.join(output_plot,lock2,subject_id) # set path for output files
    save_dir = op.join(data_dir,subject_id,'events',lock2)
    raw_list = list()
    events_list = list()
    for idx,session in enumerate(sessions):
        print(session)
        # idx = 0, session = sessions[idx]

        # create folder if it doesn't exist
        if not os.path.exists(save_plot):
            os.makedirs(save_plot)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # session_idx = 0 # loop through sessions
        # session = sessions[session_idx]
        file_dir = op.join(subject_path,session)
        blocks =  os.listdir(file_dir)
        blocks = [x for x in blocks if x.startswith('B')]
        blocks.sort()
        #blocks = [x for x in blocks if 'tsss' in x]
        print(blocks)
        
        subject_events_path = op.join(save_dir,session)
        subject_triggers_path = op.join(data_dir,subject_id,'triggers',session)
    
    
        for nb,block in enumerate(blocks):
            # nb = 0, block = blocks[nb]
            print(block)
            # if subject_id == 'S027_fmn28' and block == 'Block_04_tsss_mc_ica.fif':
            #     continue
            # if subject_id == 'S001_bsr27' and block == 'Block_17_tsss_ica.fif':
            #     print('All propixx triggers lost')
            #     continue

            # if subject_id == 'S019_dtl05' and block == 'Block_05_tsss_mc_ica.fif':
            #     print('All propixx triggers lost')
            #     continue
            #-----------------------------------------------------------------------------
            # Load file
            #-----------------------------------------------------------------------------
            
            # check if file has been created, continue if so
            file_name = op.join(save_dir,'events_' + block[:8] + '.npy')
            if os.path.exists(file_name) and not overwrite:
                continue
            
            raw_file = op.join(file_dir,block)
            raw = mne.io.read_raw_fif(raw_file,preload = True) # only MEG
            print(f'original data had {raw.info["nchan"]} channels.')
            
            if raw.info['bads'] == []:
                raw.pick(['meg','eeg','stim'])# select MEG only
                chs = ['MEG0111', 'MEG0121', 'MEG0131', 
                       'EEG001', 'EEG002', 'EEG010',
                        'STI001', 'STI002','STI003','STI004','STI005','STI006']
            else:
                raw.pick(['meg','stim'])
                chs = ['MEG0111', 'MEG0121', 'MEG0131', 
                       'STI001', 'STI002','STI003','STI004','STI005','STI006']
            print(f'after channel selection, data contains {raw.info["nchan"]} channels.')
            
            #Events_id = np.load(op.join(subject_events_path,'ParallelEvents_' + block[:-16] + '.npy'))
            events_tags = np.load(op.join(subject_triggers_path,'EventTags_' + block[:8] + '.npy'))
            propixx_triggers = np.load(op.join(subject_triggers_path,'PROPixxEvents_' + block[:8] + '.npy'))
            
            events_id, events = create_events(evens_dict[lock2],propixx_triggers,events_tags)
            # Save the events in a dedicted FIF-file: 
            # filename_events = op.join(result_path,file_name + 'eve-' + str(subfile) +'.fif')
            #mne.write_events(filename_events,events)
            # Save PROPixxEvents, Parallel port events and trials tags
            
            
            np.save(file_name,events)
            
            file_name = op.join(save_dir,'events_dict_' + block[:8])
            np.save(file_name,np.array(events_id))
            
            # sanity check plots
            # show some frontal channels to clearly illustrate the artifact removal

            chan_idxs = [raw.ch_names.index(ch) for ch in chs]
            events_idx =  [x for x,n in  enumerate(events[:,2]) if str(n).startswith('4')] # [x for x in events[:,2] if re.findall('[4]+',str(x))]
            with mne.viz.use_browser_backend('matplotlib'):
                fig = raw.plot(events=events[events_idx,:],order=chan_idxs,start=100,duration=80) #40e-6
                filename =  op.join(save_plot, block[:-4] + '_events.png')
                fig.savefig(filename)    
                plt.close()

# fig = raw.plot(events=events[events_idx,:],order=chan_idxs,start=100,duration=80) 

        
            
           