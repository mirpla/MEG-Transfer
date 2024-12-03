
#%%
#import sys
import os.path as op
import numpy as np
import mne
import os
import matplotlib.pyplot as plt
import pandas as pd
import re

#source1_directory    = 'L:\\nttk-data3\\palva\\Common repos\\OL2015\\source\\'
#sys.path.append(source1_directory + 'Python37\\Utilities')

#import file_handling as fh

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt') # so MNE plots are in a pop-up window, not inline

'''
Extract propixx and parallel port triggers from fif file
use tsss file that has already been pre-processed
a couple of datasets have been cropped before tsss

''' 
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


#%% Functions
def rgb2triggervalue(rgb):

    green_bin = list(bin(rgb[1]))
    green_bin = green_bin[2:]
    green_bin = list('0'*max(0, 8-len(green_bin))) + green_bin
    blue_bin = list(bin(rgb[2]))
    blue_bin = blue_bin[2:]
    blue_bin = list('0'*max(0, 8-len(blue_bin))) + blue_bin

    #trigger_bin = green_bin[0::2] + blue_bin[0::2]
    trigger_bin =  blue_bin[4:8] + green_bin[4:8]
    trigger_bin = "0b" + "".join(trigger_bin)
    return(int(trigger_bin,2))

def upper_bin_val(val):
    upper_bin =  bin(val)[2:] + '00000000' 
    trigger_val = "0b" + "".join(upper_bin)
    return int(trigger_val,2)

def lower_bin_val(val):
    trigger_val = bin(val)[2:-8]
    return int(trigger_val,2)

#%% Trigger dictionary

parallel_port_dict = { 
  'DL_valid': 10, 
  'DL_invalid':11,
  'DL_catch':12,
  'DR_valid':13,
  'DR_invalid':14,
  'DR_catch':15,
  'Dn':16,
  'DN_TargOn_R':17,
  'DN_catch_R':18,
  'DN_TargOn_L':19,
  'DN_catch_L':20,
  'SL_valid':21,
  'SL_invalid':22,
  'SL_catch':23,
  'SR_valid':24,
  'SR_invalid':25,
  'SR_catch':26,
  'Sn_L':27,
  'Sn_R':28,
  'SN_TargOn_R':29,
  'SN_catch_R':30,
  'SN_TargOn_L':31,
  'SN_catch_L':32,
  'NL_valid':33,
  'NL_invalid':34,
  'NL_catch':35,
  'NR_valid':36,
  'NR_invalid':37,
  'NR_catch':38,
  'Nn':39,
  'NN_TargOn_R':40,
  'NN_catch_R':41,
  'NN_TargOn_L':42,
  'NN_catch_L':43,
  'Block_beggins': 4
}
# The first participants have triggers for correct incorrect. I removed this info further down the line
#  'HIT' : 1,
#  'MISS': 2,

Pix_trigger_color= { # R, G, B
    'fixation_cross': (255, 1, 0), # 2^0 -> G0 ->  0001 , STI101 , value 1,
    'baseline': (0, 2, 0), #G1, STI101 2^1, value 2, 0010
    'Attention_win': (255, 4, 0), #G2, STI101 2^2, value 4, 0100
    'Target_yes':(0, 8, 0),#':(0, 8, 0), #G3, STI101 2^3, value 8, 1000
    'Target_no':(128, 0, 4), # background color (no target)
    'Response':(255, 0, 1), #B0, STI101 2^4, value 16 
    'Precision':(0, 0, 2), #B1, STI101 2^5, value 32
}
#     'Precision_response':(0, 0, 8), some participants may have a trigger for the precision response, this info is not relevant here
#%% Get STI101 contribution values
parallel_port_values = dict()
for (key, value) in parallel_port_dict.items():
    upper_bin =  bin(value)[2:] + '00000000' 
    trigger_val = "0b" + "".join(upper_bin)
    #new_value = 2**8*value
    parallel_port_values[key] = int(trigger_val,2) #new_value

propixx_values = dict()
for (key, value) in Pix_trigger_color.items():
    new_value = rgb2triggervalue(Pix_trigger_color[key])
    propixx_values[key] = new_value

#%%
#data_dir = r'D:\PROJECTS\CAUSAL_NETWORKS\ANALYSIS\ICA_CLEANED'
in_folder = 'ica' # use ica folder to make sure event latency is correct because a couple of datasets have been cropped 
out_folder = 'triggers'

# data_dir = r'Y:\ANALYSIS'
# logfile_dir = r'Y:\LOGFILES'
# plot_dir = r'Y:\PLOTS\triggers'


data_dir = r'/analyse/Project0349/ANALYSIS'
logfile_dir = r'/analyse/Project0349/LOGFILES'
plot_dir = r'/analyse/Project0349/PLOTS/triggers'



subjects_list = [ f.name for f in os.scandir(data_dir) if f.is_dir() and f.name[0] != '_' ]
subjects_list.sort()
print(subjects_list) 
#%%
# loop through participants
event_issues = []
faulty_trigger_line = []

for sub_idx,subject_id in enumerate(subjects_list):
    # sub_idx = 18
    # subject_id =subjects_list[sub_idx]
    
    output_plots = op.join(plot_dir,subject_id)
    print('Doing subject ' + subject_id + '\nSessions')
    subject_path = op.join(data_dir,subject_id,in_folder)
    if not os.path.exists(subject_path):
        continue
    session_folders = os.listdir(subject_path)
    sessions =  [x for x in session_folders if x.startswith('2')]
    sessions =  [x for x in sessions if not x.endswith('f')]
    sessions.sort()
    print(sessions)
    if not os.path.exists(output_plots):
        os.makedirs(output_plots)
    
    #% Loop through sessions
    
    for idx,session in enumerate(sessions):
    #idx = 1
    #session = sessions[idx]
        print(session)
        save_dir = op.join(data_dir,subject_id,out_folder,session) # set path for output files
        # create folder if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # session_idx = 0 # loop through sessions
        # session = sessions[session_idx]
        file_dir = op.join(subject_path,session)
        blocks =  os.listdir(file_dir)
        blocks = [x for x in blocks if x.startswith('B')]
        blocks.sort()
        # print(blocks)
        # loop trhough the blocks annotate bad channels, compute PSD and check trial events
        # max filter
        # concatenate after maxfilter
        for block in blocks:
            #block = blocks[8]
            print(block)
            if subject_id == 'S027_fmn28' and block == 'Block_04_tsss_mc_ica.fif':
                continue
            if os.path.exists(op.join(save_dir,'PROPixxEvents_' + block[:8] + '.npy')):
                    continue
            # if subject_id == 'S001_bsr27' and block == 'Block_17_tsss_ica.fif':
            #     print('All propixx triggers lost')
            #     continue
            
            # # Continue if triggers file for this block has been created 
            # file_name = op.join(save_dir,'PROPixxEvents_' + block[:8] + '.npy')
            # if os.path.exists(file_name):
            #     continue
            
            # block = blocks[0]
            #-----------------------------------------------------------------------------
            # Load file
            #-----------------------------------------------------------------------------
            #file_name = blocks[0] # loop through blocks
            raw_file = op.join(file_dir,block)
            if not os.path.exists(raw_file):
                continue

            raw = mne.io.read_raw_fif(raw_file,preload = True)

            #% Check data info
            print(raw.info)
            print(block)
            
            # Scroll through STIM channels
            # raw.copy().pick_types(meg=False, stim=True).plot(start=3, duration=50)
            
            #==========================================================================
            # fix for bsr27\\220304\\Block07_08.fif only
            #raw1 = raw.copy()
            #raw1.crop(tmin=0, tmax=660)
            #raw.crop(tmin=660, tmax=None)
            
            #raw.copy().pick_types(meg=False, stim=True).plot(start=3, duration=20)
            #raw1.copy().pick_types(meg=False, stim=True).plot(start=3, duration=20)
            #raw1.save(raw_file[:-7]+'.fif', overwrite=True)
            #raw.save(raw_file[:-8]+'8.fif', overwrite=True)
            #==========================================================================
            
            #-----------------------------------------------------------------------------
            # Get task events
            #-----------------------------------------------------------------------------
            '''
             The first column contains the event time in samples
             The third column contains the event id
             For output = ‘onset’ or ‘step’, the second column contains the value of the stim channel immediately before the event/step. 
             For output = ‘offset’, the second column contains the value of the stim channel after the event offset.
             
            '''
            # get PROPIxx events
            PROPixxEvents = mne.find_events(raw, consecutive = True,mask = 65280, mask_type = 'not_and',stim_channel='STI101',min_duration=0.002) #shortest_event=1
            print(PROPixxEvents[:5])  # show the first 5
        
            # Remove irrelevant Propixx trigger values, some appear times at the beginning or at the end
            remove_row = list()
            for idx,i in enumerate(PROPixxEvents[:,2]):
                if i not in propixx_values.values():
                    remove_row.append(idx)
            PROPixxEvents = np.delete(PROPixxEvents,remove_row,axis = 0)
            
            # Create events ID dictionary for PROPixx events
            '''
            Note: For most of the participants PROPixxEvents_dict will be the same as propixx_values (line 104)
                  I do this step because some of the first participants have non-informative Propixx values I decided to remove further down the line
            '''
            # get unique PROPixx events 
            PROPixxEvents_ID = np.unique(PROPixxEvents[:,2])
            
            PROPixxEvents_dict = dict()
            for (key, value) in propixx_values.items(): # Iterate over all the items in dictionary of Events ID
               if value in PROPixxEvents_ID:
                   PROPixxEvents_dict[key] = value
            #print(PROPixxEvents_dict)
                        
            #% Plot events with MNE
            fig = mne.viz.plot_events(PROPixxEvents,  sfreq=raw.info['sfreq'], event_id=PROPixxEvents_dict,
                                      first_samp=raw.first_samp) #
            filename =  op.join(output_plots, block[:-3] + 'png')
            fig.savefig(filename) # save figure
            plt.close()
            
            # Get parallel port events
            ParallelEvents = mne.find_events(raw, consecutive = False, mask = 65280, mask_type = 'and', stim_channel='STI101',min_duration=0.002) # ,mask = 65280, mask_type = 'and',
            print(ParallelEvents[:5])  # show the first 5
        
            # Remove irrelevant Parallel trigger values
            remove_row = list()
            for idx,i in enumerate(ParallelEvents[:,2]):
                if i not in list(parallel_port_values.values())[:-1]:
                    remove_row.append(idx)
            ParallelEvents = np.delete(ParallelEvents,remove_row,axis = 0)
        
            # Create events ID dictionary for Parallel Port events
            '''
            Note: Not every Trial ID is present in every block, therefore ParallelEvents_dict is not exactly the same as parallel_port_values
            '''
            ParallelEvents_ID = np.unique(ParallelEvents[:,2])
            ParallelEvents_dict = dict()
            for (key, value) in parallel_port_values.items(): # Iterate over all the items in dictionary of Events ID
                if value in ParallelEvents_ID:
                    ParallelEvents_dict[key] = value           
            print(ParallelEvents_dict)
            
            #% Plot events with MNE
            #fig = mne.viz.plot_events(ParallelEvents,  sfreq=raw.info['sfreq'], event_id=ParallelEvents_dict,first_samp=raw.first_samp) #
    
            #% Sanity Check
            '''
            There should be the same number of PROPixx triggers as events in the log file
            If this is not the case, the operator may have forgotten to start recording on time
            Each block is 120 trials (some of the first participants only have 108 trials per block)
            Just for reference: Number of trials per block increased from 108 to 120 to reduce total number of blocks
            '''
            
            # load log file
            log_files = os.listdir(op.join(logfile_dir,subject_id[-5:],'log_files'))
            log_file = [i for i in log_files if re.findall('[0-9]+',block)[0] + '.' in i][0] # block[-16:-13]
            log = pd.read_csv(op.join(logfile_dir,subject_id[-5:],'log_files',log_file))
            log = log.values
            
            count_PROPixxTriggers = [np.count_nonzero(PROPixxEvents[:,2]==1),np.count_nonzero(PROPixxEvents[:,2]==2),np.count_nonzero(PROPixxEvents[:,2]==4),np.count_nonzero(PROPixxEvents[:,2]==16),np.count_nonzero(PROPixxEvents[:,2]==32)]
            if all(elem == len(log) for elem in count_PROPixxTriggers):
                print('All PRopixx triggers are correct\nNow checking ID for every trial\n===============================')
            else:
                print('There are less trials than expected in Block\nIt could be that part of the recording is missing\n===============================')    
                # toDo Remove incomplete trials / adjust trials ID
                
                
                nb_trials = np.min(count_PROPixxTriggers)
                
                first_propixx_fixation = np.where(PROPixxEvents[:,2]==1)[0][0] # remove previous propixx trigggers if first is not 0
                if first_propixx_fixation !=0 :
                    PROPixxEvents = PROPixxEvents[np.where(PROPixxEvents[:,2]==1)[0][0]:,:] 
                
                trial_start_lat = PROPixxEvents[PROPixxEvents[:,2] == 1,0]
                parallel_por_ID_for_PROPixx = [] # find parallel port id for each propixx 1, if no parallel port complete with zero
                for i in trial_start_lat:
                    if any(np.abs(ParallelEvents[:,0] - i) < 50):
                        parallel_por_ID_for_PROPixx.append(ParallelEvents[np.where(np.abs(ParallelEvents[:,0] - i) < 50)[0][0],2])
                    else:
                        parallel_por_ID_for_PROPixx.append(0) # zero is added when there is no matching event in the parallel port (if parallel port failed) 
                
                perc = 0
                count = 0
                while perc < 45: # find the best match between the log file and the trial ID of the Propixx. This number is low becuase some times parallel port fails
                    x = abs(parallel_por_ID_for_PROPixx - log[0+count:len(parallel_por_ID_for_PROPixx)+count,2].astype(int) * 2**8 )
                    perc = (len([ele for ele in x if ele == 0]) / len(x)) * 100
                    count+=1
                first_trial_idx = count - 1
                
                if first_trial_idx > 0:
                    log = log[first_trial_idx:,:] # log[first_trial_idx:,2] * 256 - parallel_por_ID_for_PROPixx
                elif first_trial_idx == 0:
                    new_PROPixxlat = []
                    lat = PROPixxEvents[first_trial_idx,0]
                    
                    for ev in range(len(log)):
                        #print(ev)
                        tmp_list = [lat]
                        for n,l in enumerate([16,17,18,20]):
                            #print(l)
                            tmp_list.append(int(np.round(tmp_list[n] + log[ev,l] / 120 * 1000) ) )
                            #tmp_list = np.array(tmp_list)
                        new_PROPixxlat.append(tmp_list)
                        if ev < 119:
                            lat = int ( np.round( ( ( log[ev+1,0] - log[ev,0] )* 1000 ) + lat ) )
                    new_PROPixxlat = np.array(new_PROPixxlat)
                    new_PROPixxlat = np.reshape(new_PROPixxlat, new_PROPixxlat.shape[0] * new_PROPixxlat.shape[1])
                    id_list = np.repeat(np.array([[1,2,4,16,32]]),new_PROPixxlat.shape[0]/5,axis=0)  
                    id_list = np.reshape(id_list, id_list.shape[0] * id_list.shape[1])
                    new_PROPixxEvents = np.transpose(np.array( [new_PROPixxlat , np.insert(id_list[:-1],0,0) ,id_list] ) )
                    # add trget trigger
                    target_list = []
                    for ti,target_on in enumerate(log[:,7]):
                        print(ti)
                        if target_on == 'yes':
                            target_list.append([ int( ( log[ti,0] - log[0,0] ) * 1000) + new_PROPixxlat[0] + int(np.round(np.sum(log[ti,16:19])/120*1000)) ,4 , 8 ] )
                            
                    new_PROPixxEvents = np.append(new_PROPixxEvents,np.array(target_list),axis=0)
                    new_PROPixxEvents = new_PROPixxEvents[new_PROPixxEvents[:, 0].argsort()]
                    # make sure the order of trigger is correct when target is on, and correct latency adding the 100ms 
                    for ie, ev in enumerate(new_PROPixxEvents[:,2]):
                        if ev == 8 and new_PROPixxEvents[ie-1,2] == 16:
                            #print([ev,new_PROPixxEvents[ie-1,2]])
                            new_PROPixxEvents[ie-1,2] = 8
                            new_PROPixxEvents[ie,2] = 16
                            new_PROPixxEvents[ie,1] = 8
                            new_PROPixxEvents[ie,0] = new_PROPixxEvents[ie-1,0] + 100
                        elif ev == 8 and new_PROPixxEvents[ie+1,2] == 16:
                            #print('exception')
                            new_PROPixxEvents[ie+1,0] = new_PROPixxEvents[ie,0] + 100
                            
                            
                    #sanity check
                    delay = PROPixxEvents[:,0] - new_PROPixxEvents[:len(PROPixxEvents),0]
                    fix = delay[PROPixxEvents[:,2] == 1] / 1000
                    bl = delay[PROPixxEvents[:,2] == 2] / 1000
                    att = delay[PROPixxEvents[:,2] == 4] /1000
                    targ = delay[PROPixxEvents[:,2] == 8] /1000
                    resp = delay[PROPixxEvents[:,2] == 16] /1000
                    psn =  delay[PROPixxEvents[:,2] == 32] /1000
                    
                    import matplotlib.ticker as mticker
                    
                    fig, ax = plt.subplots()
                    ax.violinplot([fix, bl, att, targ, resp, psn],showmeans=False, showmedians=False, showextrema=False)
                    ax.boxplot([fix, bl, att, targ, resp, psn])
                    labels = [item.get_text() for item in ax.get_xticklabels()]
                    labels = ['fixation', 'baseline', 'attention', 'target','detection', 'precision']
                    ax.set_ylabel('ms')
                    ax.set_xticks(np.arange(1, 7, 1.0))
                    ticks_loc = ax.get_xticks().tolist()
                    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
                    ax.set_xticklabels(labels)
                    ax.set_title('PROPixx triggers minus Triggers created from log file')
                    filename =  op.join(output_plots, block[:-4] + '_fixed_triggers_delay.png')
                    fig.savefig(filename) # save figure
                    plt.close()
                    
                    fig = mne.viz.plot_events(PROPixxEvents,  sfreq=raw.info['sfreq'], event_id=PROPixxEvents_dict,
                                              first_samp=raw.first_samp) #
                    filename =  op.join(output_plots, block[:-3] + 'png')
                    fig.savefig(filename) # save figure
                    plt.close()
                    
                    if np.mean(att) < 0.03:
                        PROPixxEvents = new_PROPixxEvents
                        count_PROPixxTriggers = [np.count_nonzero(new_PROPixxEvents[:,2]==1),np.count_nonzero(PROPixxEvents[:,2]==2),np.count_nonzero(PROPixxEvents[:,2]==4),np.count_nonzero(PROPixxEvents[:,2]==16),np.count_nonzero(PROPixxEvents[:,2]==32)]
                        nb_trials = np.min(count_PROPixxTriggers)
                    
                if len(log) != nb_trials:
                    print('no solution yet')
                    event_issues.append(subject_id + '_' + block[:8] + '_less trials than expected, no solution yet')
                else:
                    event_issues.append(subject_id + '_' + block[:8] + '_less trials than expected, it has been sorted')
                    # last_propixx_precision = np.where(PROPixxEvents[:,2]==32)[0][-1]
                    # if PROPixxEvents[last_propixx_precision,2] != PROPixxEvents[-1,2]:
                    #     PROPixxEvents = PROPixxEvents[:np.where(PROPixxEvents[:,2]==32)[0][-1]+1,:] # find the last trigger value 32 and clear triggers after that
                    #     last_trial_idx = np.sum(PROPixxEvents[:,2]==32)
                    #     log = log[:last_trial_idx,:]
                    
                count_PROPixxTriggers = [np.count_nonzero(PROPixxEvents[:,2]==1),np.count_nonzero(PROPixxEvents[:,2]==2),np.count_nonzero(PROPixxEvents[:,2]==4),np.count_nonzero(PROPixxEvents[:,2]==16),np.count_nonzero(PROPixxEvents[:,2]==32)]
                    
                if all(count_PROPixxTriggers / nb_trials == 1):
                    print('This block has %d trials\n ' %nb_trials)
                    fig = mne.viz.plot_events(PROPixxEvents,  sfreq=raw.info['sfreq'], event_id=PROPixxEvents_dict,
                                              first_samp=raw.first_samp) #
                    filename =  op.join(output_plots, block[:-4] + '_fixed.png')
                    fig.savefig(filename)
                    plt.close()


            if ParallelEvents.shape[0] == len(log):
                print('Every event has an ID\n All looking good\n\n')
            else:
                print('Some parallel port IDs are missing\nNow fixing missing parallel port ID\n===============================')
                trial_start_lat = PROPixxEvents[PROPixxEvents[:,2] == 1,0]
                trial_id_idx = []
                for i in ParallelEvents[:,0]:
                    trial_id_idx.append(np.abs(trial_start_lat - i).argmin()) #parallel port trigger ID should happen approx 17 samples before Propixx target value 1 (fixation cross)
                missing_trial_id_idx = [item for item in range(0, len(log)) if item not in trial_id_idx]
                # Sanity check
                trigger_lag = np.bincount(abs(trial_start_lat[trial_id_idx]  - ParallelEvents[:,0])).argmax() 
                
                x = abs(ParallelEvents[:,2] - log[trial_id_idx,2].astype(int) * 2**8 )
                perc = (len([ele for ele in x if ele == 0]) / len(x)) * 100
                
                if perc < 70: # if most of the triggers do not match you are likely looking at a wrong log file, unlikely, but it could happen if the fif name of the file is wrong 
                    print('log file and parallel port do not match\nCheck log file names\n===============================')    
                    continue
                
                if len(missing_trial_id_idx) + len(trial_id_idx) == len(log):
                    # compute trigger lag here 
                    missing_trial_id = log[missing_trial_id_idx,2] *2**8 # multiply by 256 to get the STI101 contribution
                    missing_trial_lat = trial_start_lat[missing_trial_id_idx] - trigger_lag #parallel port trigger ID should happen approx 17 samples before Propixx target value 1 (fixation cross)
                    add_trials = np.transpose(np.array([missing_trial_lat, np.zeros(len(missing_trial_lat),dtype=int),missing_trial_id]))
                    #ParallelEvents.resize(len(log),3) # More efficient way to append trials
                    ParallelEvents = np.resize(ParallelEvents,[len(log),3])
                    ParallelEvents[len(log) - len(add_trials):,:] = add_trials
                    ParallelEvents = ParallelEvents[ParallelEvents[:,0].argsort()] # sort according latency
                    print('Fixed!')
                    event_issues.append(subject_id + '_' + block[:8] + '_parallel port triggers missing, parallel port and log file match in ' + str(np.floor(perc)) + str(len(missing_trial_id_idx)) + ' missing triggers,fixed!')
                else:
                    event_issues.append(subject_id + '_' + block[:8] + '_parallel port triggers missing, parallel port and log file match in ' + str(np.floor(perc)) + str(len(missing_trial_id_idx)) + ' missing triggers, no solution yet')

            
            # Sanity check: trigger code from log and fif file should be exactly the same  
            if np.sum(ParallelEvents[:,2] - log[:,2] *2**8) != 0:
                print('Warning: It looks like the parallel port was not working properly!')
                wrong_value = np.abs(np.unique(ParallelEvents[:,2] - log[:,2] *2**8 ))
                faulty_triger = []
                for i in wrong_value[wrong_value!=0]:
                    faulty_triger.append(lower_bin_val(i))
                print('Check parallel port line(s):')
                print(faulty_triger)
                ParallelEvents[:,2] = log[:,2]*2**8
                faulty_trigger_line.append(subject_id + '_' + block[:8] + '_faulty trigger code: ' + ','.join([str(element) for element in faulty_triger]) )
        
            #% Create list with trial tags containing stimulus/attention/validity/response/RT/precision
            
            columns = [3,4,5,9,10]
            trial_info =  list(log[:,columns])
            for idx,i in enumerate(trial_info):
                i[-1] = 'RT'+str(int(i[-1] *1000))
                trial_info[idx] =  '/'.join(i) 
            
            #ToDo add precision info
            
            #%
            # Save PROPixxEvents, Parallel port events and trials tags
            file_name = op.join(save_dir,'PROPixxEvents_' + block[:8])
            np.save(file_name,PROPixxEvents)
            
            file_name = op.join(save_dir,'ParallelEvents_' + block[:8])
            np.save(file_name,ParallelEvents)
            
            file_name = op.join(save_dir,'EventTags_' + block[:8])
            np.save(file_name,np.array(trial_info))
            

                


