
#%%

import os.path as op
import mne
import os
#from mne.preprocessing import ICA
import matplotlib.pyplot as plt

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt') # so MNE plots are in a pop-up window, not inline
# %matplotlib qt

#%%
exclude_ics  = {
  "meg": {
  "bsr27":  {"session1": [1,25,26,27,28], "session2": [4,26,27,28,29],"session3": [1,4,20,25,26,27,28,29]}, # ok
  "dsa23":  {"session1": [1,2,3,15,19,21,23,24], "session2": [0,1,3,17,20,24], "session3": [0,1,2,5,15,19,20,25,28]}, # session1: 24? ok
  "mtr13":  {"session1": [0,12,13,21,27], "session2": [0,8,24,25],"session3": [0,11,19,21]}, # session1:10,18,21?, session2:11,27?, session3:8,28?
  "gto28":  {"session1": [1,6,9,15,16,21,25,29], "session2": [0,12,13,14,16,22],"session3": [0,10,14,15,16,17,19,21,28]}, # session 1 noisy, session2:16?, session3:14?
  "ami28":  {"session1": [0,14,15,24,26,29], "session2": [0,11,14,17,20,24,25,26,29],"session3": [0,10,13,16,19,21,22,24]}, #session3:28?
  "lka10":  {"session1": [0,1,3,8,9,16,18], "session2": [0,3,4,5,6,10], "session3": [0,3,4,9,19,25]}, #session2:1,22,25 (eyes+brain)?session3:24  
  "qqn19":  {"session1": [0,1,14,15,22,23,26], "session2": [0,1,2,9,13],"session3": [0,1,2,8,13,16,19,20,21,24,28]}, #session1:check9, session2:5, session3:4 (brain+blink)
  "mtr19":  {"session1": [0,3,4,12,18,19,27,29], "session2":[0,1,16,19,25,26], "session3": [0,3,5,9,16,17,20]}, #session1:21 (brain+eyes), session2:16 (brain+eyes), session3:20
  "fha01":  {"session1": [0,4,13,14], "session2": [0,10,14,16,20,22,28],"session3": [0,8,14,19,20,29]}, #session1: 1,25?, session2: ok
  "hwh21":  {"session1": [1,3,15,17,18], "session2": [0,7,10,14,19,29],"session3": [0,1,4,5,7,10,25,27,28,29]}, # session1:10?, session2:22, session3:28?  
  "rsh17":  {"session1": [0,1,3,14,15,19], "session2":[0,1,2,4,17,21,22,24], "session3": [0,1,2,3,5]}, #session3:13, 27
  "zwi25":  {"session1": [0,16,17,22,24], "session2": [0,25,27,28],"session3": [0,12,26]}, 
  "tdn02":  {"session1": [4,15,21,22,23,24,25,26,27,28,29], "session2": [5,13,14,15,21,25,26,27,29], "session3": [8,14,15,17,24,25,26,27]}, # session2: 22,24,28,29?, session3:26
  "uka11":  {"session1": [0,1,2,4,5,14], "session2": [0,1,2,21,13],"session3": [0,4,8,15,22,25,28]}, 
  "csi07":  {"session1": [0,1,15,17,18,23], "session2": [0,4,10,11,12,13,16,17,23],"session3": [0,1,14,16]}, # session1:1(eye+brain), session2:26(eye+brain)
  "rsg06":  {"session1": [0,2,16,20,21,22,23,24], "session2": [0,1,23,25],"session3": [0,1,14,25,26]}, # session1: 13,19 (eye+brain), session3:14,21,22
  "mwa29":  {"session1": [0,1,2,12,14,15,19,24,25,28], "session2": [0,1,3,6,14,16], "session3": [0,2,11,17,20,23,24]},
  "ade02":  {"session1": [2,4,11,13,24], "session2": [9,12,23,25,26],"session3": [3,8,13,14,16,22,24,25,28]}, # session1:19 (brain+eye), session3:18,20,24,27
  "dtl05":  {"session1": [0,20,23,24,25], "session2": [0,23,24,25,26,28],"session3": [0,22,23,24,27,28,29]},
  "rbe04":  {"session1": [0,2,11,13,17,19,21,24,28], "session2": [0,2,7,13,14,19,21,22,25,27],"session3": [0,9,12,13,15,20,28,22,24,26,29]},
  "jmn22":  {"session1": [0,9,14,15,18,19,25,27,28,29], "session2": [0,11,15,22,25,27,28,29],"session3": [0,9,13,18,19,21,23,25,27,28,29]},
  "dss19":  {"session1": [1,3,16,19,22,23], "session2": [0,2,8,9,13,19,22,23,26],"session3": [0,5,12,15,20,21,22,25,26,28,29]},
  "fte25":  {"session1": [0,2,11,12,13,14,15,17,18,25,26,27,28,29], "session2": [0,2,12,13,14,15,18,19,22,26],"session3": [0,3,8,9,10,11,16,17,28]},
  "hyr24":  {"session1": [0,7,10,13,14,21,22,23,24,26], "session2": [0,2,3,4,14,12,13,14,18,19,20],"session3": [0,2,3,12,13,14,17,22,24]},
  "dja01":  {"session1": [0,12,13,21,26,29], "session2": [0,9,10,13,17,18,21,24,29],"session3": [0,10,13,14,24,25]},
  "fmn28":  {"session1": [0,3, 4, 9,10,12, 14,15,20,21,24,27,28,29], "session2": [0,1, 2, 7, 10, 12, 14, 15, 23,26],"session3": [0,2, 4, 5, 14]},
  "omr03":  {"session1": [0,1,2,3,4,5,6,7,8,9,10,11, 12, 13, 14,15, 16, 17, 25, 27], "session2": [0,1,2,3,4,5,6,7, 8,9, 10,11,12, 13, 14,20],"session3": [0,1,2,3,4,5,6,7,8,9,10,11,14,20,21,22]},
  "amy20":  {"session1": [0,7,10, 11, 12, 17, 20,22, 27, 28, 29], "session2": [0,3,11,7, 9, 14, 17, 22,23,27,28,29],"session3": [0,3,8, 9, 16, 17,18, 19,20,21]},
  "ski23":  {"session1": [0,2,21,27], "session2": [0,2,10,23,28],"session3": [0,1,6,7,15,21,24]},
  "jry29":  {"session1": [0,3,27], "session2": [0,3,25],"session3": [0,3,12,21,27,28,29]},
  "jpa10":  {"session1": [0,1,10,19,20], "session2": [0,1,11,25,27,29],"session3": [0,1,12,26]},
  "epa14":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "crr22":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "jyg27":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "bbi29":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "ece24":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "hay06":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "awa19":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "mca10":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "hky23":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  }  , 
  "eeg": {
    "bsr27":  {"session1": [0,5,8,12], "session2": [1,12,14],"session3": [1,5,6,15,19]}, # ok
    "dsa23":  {"session1": [0,1,2,3,6,9], "session2": [0,1,2,5,6,7,8,12], "session3": [0,2,3,5]}, # session1: 24? ok
    "mtr13":  {"session1": [0,1,2], "session2": [0,5,14,16,17],"session3": [1,3,11,13,14]}, # session1:10,18,21?, session2:11,27?, session3:8,28?
    "gto28":  {"session1": [0,2,3,4,15,19,25], "session2": [0,1,2,3,4,5,9,17],"session3": [0,1,2,3,4,9,16,17]}, # session 1 noisy, session2:16?, session3:14?
    "ami28":  {"session1": [0,1,3,4,7,13,14,18,20,21], "session2": [0,2,3,4,7,9,10,13,14,16],"session3": [0,1,2,3,4,6,8,9,11,12,14]}, #session3:28?
    "lka10":  {"session1": [0,2,4,5,6,11], "session2": [0,1,4,5,6,8,9], "session3": [0,2,3,4,5,7]}, #session2:1,22,25 (eyes+brain)?session3:24  
    "qqn19":  {"session1": [0,1,2,3,4,7,8,10], "session2": [1,2,3,5,6,7,8,10,17],"session3": [0,1,2,3,4,5,6,7,9,16]}, #session1:check9, session2:5, session3:4 (brain+blink)
    "mtr19":  {"session1": [0,1,5,6,8,9,12,13], "session2":[0,2,3,4,5,18,11], "session3": [0,3,6,11]}, #session1:21 (brain+eyes), session2:16 (brain+eyes), session3:20
    "fha01":  {"session1": [0,4,10,14], "session2": [0,1,2,5,9,10,13],"session3": [0,3,4,7,6,9]}, #session1: 1,25?, session2: ok
    "hwh21":  {"session1": [0,1,3,5,6], "session2": [0,1,7,8,9,15],"session3": [1,2,5,7,8,9]}, # session1:10?, session2:22, session3:28?  
    "rsh17":  {"session1": [0,2,3,4,5,6,8,17], "session2":[0,2,3,4,5,7,13,14], "session3": [0,2,4,6,13]}, #session3:13, 27
    "zwi25":  {"session1": [4,10,11,12,17], "session2": [0,4,6,9,10,12,13],"session3": [0,4,5,6,8,9,11,14]}, 
    "tdn02":  {"session1": [0,4,5,10,13,17], "session2": [1,7,8,17], "session3": [1,5,7,12]}, # session2: 22,24,28,29?, session3:26
    "uka11":  {"session1": [0,1,2,3,4,5,6,7,13], "session2": [0,1,2,3,4,6,7,12,14],"session3": [0,1,2,3,4,5,7,15]}, 
    "csi07":  {"session1": [0,4,7,8,9,12,13,18], "session2": [0,1,6,7,8,10],"session3": [0,1,3,5,9]}, # session1:1(eye+brain), session2:26(eye+brain)
    "rsg06":  {"session1": [0,3,6,10,18], "session2": [0,3,6],"session3": [0,1,2,3,6]}, # session1: 13,19 (eye+brain), session3:14,21,22
    "mwa29":  {"session1": [0,1,3,5,11,14], "session2": [0,1,2,3,8,12], "session3": [0,1,3,4,9]},
    "ade02":  {"session1": [0,2,3,7,10], "session2": [0,2,3,5,7,14],"session3": [0,1,8,9,12,17,19]}, # session1:19 (brain+eye), session3:18,20,24,27
    "dtl05":  {"session1": [0,1,12,13,14], "session2": [0,2,12,13,14,15],"session3": [0,8,10,12,14,18]},
    "rbe04":  {"session1": [0,1,3,5,9,11,14], "session2": [0,1,3,6,7,9,12,14],"session3": [0,3,4,5,7,9,12,14,16]},
    "jmn22":  {"session1": [1,6,10,12,14,15], "session2": [0,1,3,5,6,14],"session3": [0,3,6,7,8,14,16]},
    "dss19":  {"session1": [0,2,4,5,6,8,11,12,19], "session2": [0,3,4,7,10,12,13,17],"session3": [0,2,3,4,5,11,14,16]}  ,
    "fte25":  {"session1": [], "session2": [0,1,3,5,6,16,19,20,22],"session3": [0,1,5,6,7,9,14]},
    "hyr24":  {"session1": [0,2,4,5,6,7,8,9,10], "session2": [0,1,3,4,5,6,7,8,9,10,11,13,14,15,17,19],"session3": [0,1,4,5,7,9,10,12,13,14]},
    "ank24":  {"session1": [0,1,4,12,15,6], "session2": [0,1,2,3,4,5,7,10,12,15,23],"session3": [0,1,2,5,7,8,14,13,15,19,26,27,28,29]},
    "dja01":  {"session1": [0,2,3,4,5,8,14,15,17,18,19,22], "session2": [0,1,2,3,4,5,6,7,13,14,19,20,24,28],"session3": [0,1,2,5,6,11,12,14,15,16,17,23,24,25,26,27,28,29]},
    "fmn28":  {"session1": [0,1,3,4,5, 9,10,12,13,14,17,19,20,23, 26, 29], "session2": [0,3,6, 9, 16, 17, 18,19, 21, 22,23,24, 25, 26,28, 29],"session3": [0,3, 4, 7, 9, 10, 14, 17, 19, 20, 22, 25, 27, 28, 29]},
    "omr03":  {"session1": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,19,20], "session2": [0,1,2,3,4,5,6,7,8,9,10,11],"session3": [0,1,2,3,4,5,6,7]},
    "amy20":  {"session1": [0,1,2,7,9, 10, 13, 16,19,20,21,23, 27, 28,29], "session2": [0,1,3, 4, 5, 6, 8, 9, 11, 19, 20, 21, 23, 24, 25, 26, 27, 28,29],"session3": [0,2,1, 3, 5, 10, 12,16,17,21, 23,24, 26, 28]},  
    "ski23":  {"session1": [0,1,2,3,4,5,6,11,12,13,14,17,25], "session2": [0,2,6,8,13,15,16],"session3": [0,2,4,5,6,8,9,11,12,13,14]},  
    "jry29":  {"session1": [0,2,8,10,14,15,22,27], "session2": [0,2,3,13,23,28],"session3": [0,7,10,13,19,23,28]},
    "jpa10":  {"session1": [0,1,4,5,6,8,13,14,18,23], "session2": [0,1,3,4,5,8,13,16,26],"session3": [0,1,2,4,5,7]},
    "epa14":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
    "crr22":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
    "jyg27":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
    "bbi29":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
    "ece24":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
    "hay06":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
    "awa19":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
    "mca10":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
    "hky23":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
    }  
  }


#%%

# data_dir =  r'Y:\ANALYSIS'
# output_plot = r'Y:\PLOTS\ica' 

data_dir =   r'/analyse/Project0349/ANALYSIS'
output_plot = r'/analyse/Project0349/PLOTS/ica' 

subjects_list = [ f.name for f in os.scandir(data_dir) if f.is_dir() and f.name[0] == 'S' ]
print(subjects_list) 

#%%
# loop through participants
for subject_id in subjects_list:
    # subject_id = 'S001_bsr27'
    print(subject_id)
    if not subject_id[5:] in exclude_ics['eeg']:
        print(subject_id[5:] + ' does not have ICA yet')
        continue
    #subject_nb = folder_mapping[subject_id]
    # sub_idx = 0
    # subject_id =subjects_list[sub_idx]
    print('Doing subject ' + subject_id + '\nSessions')
    subject_path = op.join(data_dir,subject_id,'tsss')
    session_folders = os.listdir(subject_path)
    sessions =  [x for x in session_folders if x.startswith('2')]
    sessions.sort()
    print(sessions)


    for idx,session in enumerate(sessions):
        
        # idx = 0
        # session = sessions[idx]
        print(session)
        save_plot = op.join(output_plot,subject_id,'traces_and_properties') # set path for output files
        load_dir = op.join(data_dir,subject_id,'ica',session)
        
        if not os.path.exists(save_plot): # create folder if it doesn't exist
            os.makedirs(save_plot)
        
        if any(session in s for s in os.listdir(save_plot)):
            print('Data has already been inspected for ' + subject_id + 'session ' + str(idx) + '\n================================' )
            continue

        file_dir = op.join(subject_path,session)
        blocks =  os.listdir(file_dir)
        blocks = [x for x in blocks if x.startswith('B')]
        blocks = [x for x in blocks if 'tsss' in x]
        print(blocks)

        ica_fit = {}  #load ica for the different sensor type. Store in a dictionary
        for sensor_type in ['eeg','meg']:
            file_name = op.join(load_dir,sensor_type +'_ica.fif')
            if os.path.exists(file_name):
                ica = mne.preprocessing.read_ica(file_name, verbose=None)
                ica_fit[sensor_type] = ica
        
        eeg_all = []    
        for nb,block in enumerate(blocks):
            print(block)
            tsss_file = op.join(file_dir,block)
            raw = mne.io.read_raw_fif(tsss_file,preload = True)
            
            if len(raw.info['bads']) == 64:
                print('No EEG data for ' + subject_id + ' ' +  block +'\n ==========================================\n')
                raw.pick_types(meg=True,eeg=False)
            else:
                raw.pick_types(meg=True,eeg=True) 
           

            raw.resample(200)
            raw.filter(1, 40)
            if nb == 0:
                raw_all = mne.io.concatenate_raws([raw],on_mismatch='ignore' )
            else:
                try:
                    
                    raw_all = mne.io.concatenate_raws([raw_all, raw],on_mismatch='ignore')
                except:
                    # this exception was created for participant bsr27. EEG data was lost for some of the blocks in session idx = 1
                    if raw.info['nchan'] > 306:
                        eeg = raw.copy().pick_types(meg=False,eeg=True)
                        if eeg_all == []:
                            eeg_all = mne.io.concatenate_raws([eeg],on_mismatch='ignore' )
                        else:
                            eeg_all = mne.io.concatenate_raws([eeg_all,eeg],on_mismatch='ignore' )
     
                    raw.pick_types(meg=True,eeg=False)
                    raw_all.pick_types(meg=True,eeg=False)
                    raw_all = mne.io.concatenate_raws([raw_all, raw],on_mismatch='ignore')
                    
           
        del raw
        if eeg_all != []:
            del eeg
        
        
        for sensor_type in ['eeg','meg']:
            # if session in no_eeg and sensor_type == 'eeg':
            #     continue
            if raw_all.info['nchan'] == 306 and sensor_type == 'eeg':
                continue

            c = exclude_ics[sensor_type][subject_id[5:]]['session'+ str(idx+1)]
            for i in c:
                print(i)
                fig = ica_fit[sensor_type].plot_properties(raw_all, picks= i)
                filename =  op.join(save_plot,session + '_' + sensor_type + '_ic' + str(i) + '.png')
                plt.savefig(filename)
                plt.close()
          
        if eeg_all != []: # this will run for the case when the eeg and meg data has been split into separate variables. See Try Exception above
            
            c = exclude_ics['eeg'][subject_id[5:]]['session'+ str(idx+1)]
            for i in c:
                print('Plotting properties for IC ' + str(i))
                fig = ica_fit['eeg'].plot_properties(eeg_all, picks= i)
                filename =  op.join(save_plot,session + '_eeg_ic' + str(i) + '.png')
                plt.savefig(filename)
                plt.close()
 
#%%
# Sources are now plotted in the computing ICA code 
# with mne.viz.use_browser_backend('matplotlib'):
#     fig = ica_fit[sensor_type].plot_sources(raw_all,title='ICA',picks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],start = 100);
#     filename =  op.join(save_plot,session + '_' + sensor_type + '_ica_sources_00-14.png')
#     fig.savefig(filename)
#     plt.close()

# with mne.viz.use_browser_backend('matplotlib'):
#     fig = ica_fit[sensor_type].plot_sources(raw_all,title='ICA',picks=[15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],start = 100);
#     filename =  op.join(save_plot,session + '_' + sensor_type + '_ica_sources_15-29.png')
#     fig.savefig(filename)
#     plt.close()       
 
    
 #ica.plot_components();
            # raw_ica.filter(l_freq=1,h_freq=148,h_trans_bandwidth=2)
            # raw_ica.plot()
            # raw = mne.io.read_raw_fif(raw_file,preload = True)
            # raw.filter(l_freq=1,h_freq=148,h_trans_bandwidth=2)
            # raw.plot()
            # raw.plot_psd(tmin=10,fmin=3,fmax=148,xscale='log')            
            
        
        #% Check data info
        #print(raw.info)

        # raw.info['bads'] = []
        # raw.info['bads'] = bad_meg_channels[subject_id]['session'+str(idx+1)]

        # fig = raw.plot_psd(tmin=10,fmin=3,fmax=145,xscale='log')
        # filename =  op.join(save_plot,'PSD_' + block[:-3] + 'png')
        # fig.savefig(filename)
        
        # raw_tsss = mne.preprocessing.maxwell_filter(raw, st_duration=16, st_correlation=.9,
        #                                             head_pos=None,coord_frame='head')
        
        # fig = raw_tsss.plot_psd(tmin=10,fmin=3,fmax=145,xscale='log')
        # filename =  op.join(save_plot,'PSD_tsss_' + block[:-3] + 'png')
        # fig.savefig(filename)

            
        # new_file_name = op.join(save_dir,block[:-4] + '_tsss.fif')
        # raw_tsss.save(new_file_name, overwrite=True)

# with mne.viz.use_browser_backend('matplotlib'):
#     fig = ica_fit['eeg'].plot_sources(eeg_all,title='ICA',picks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],start = 100);
#     filename =  op.join(save_plot,session + '_eeg_ica_sources_00-14.png')
#     fig.savefig(filename)
#     plt.close()

# with mne.viz.use_browser_backend('matplotlib'):
#     fig = ica_fit['eeg'].plot_sources(eeg_all,title='ICA',picks=[15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],start = 100);
#     filename =  op.join(save_plot,session + '_eeg_ica_sources_15-29.png')
#     fig.savefig(filename)
#     plt.close()