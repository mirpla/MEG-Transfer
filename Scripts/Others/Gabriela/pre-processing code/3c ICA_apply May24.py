
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

data_dir =   r'Y:\ANALYSIS'
output_plot = r'Y:\PLOTS\ica' #r'D:\PROJECTS\CAUSAL_NETWORKS\PLOTS\QUALITY_CHECK\psd'
output_plot_psd = r'Y:\PLOTS\QUALITY_CHECK\psd\ica'
output_plot_traces = r'Y:\PLOTS\QUALITY_CHECK\traces\ica'

# data_dir =   r'/analyse/Project0349/ANALYSIS'
# output_plot = r'/analyse/Project0349/PLOTS/ica' 
# output_plot_psd = r'/analyse/Project0349/PLOTS/QUALITY_CHECK/psd/ica'
# output_plot_traces = r'/analyse/Project0349/PLOTS/QUALITY_CHECK/traces/ica'

#subjects_list = ['S005_ami28' ]
subjects_list = [ f.name for f in os.scandir(data_dir) if f.is_dir() and f.name[0] == 'S' ]
#subjects_list =  ['S020_rbe04', 'S021_jmn22']
subjects_list.sort()
print(subjects_list) 
#subjects_list = subjects_list[8:]

#%%
exclude_ics  = {
  "meg": {
  "bsr27":  {"session1": [1,25,26,27,28], "session2": [4,26,28],"session3":[1,25,26]}, # ok
  "dsa23":  {"session1": [0,1,2,3,15], "session2": [0,1,4,17,20], "session3": [0,1,2,5,15]}, # session1: 24? ok
  "mtr13":  {"session1": [0,12,13], "session2":[0,8,24],"session3": [0,12,22,23]}, # session1:10,18,21?, session2:11,27?, session3:8,28?
  "gto28":  {"session1": [1,15,16], "session2":[0,13,14],"session3": [0,10]}, # session 1 noisy, session2:16?, session3:14?
  "ami28":  {"session1": [0,14,15,24], "session2":[0,17,24,25],"session3": [0,10]}, #session3:28?
  "lka10":  {"session1": [0,8,16], "session2": [0,3], "session3": [0,3]}, #session2:1,22,25 (eyes+brain)?session3:24  
  "qqn19":  {"session1": [0,1,10,21], "session2": [0,1,2,9,13,21],"session3": [0,1,2,13]}, #session1:check9, session2:5, session3:4 (brain+blink)
  "mtr19":  {"session1": [0,3,7,21,27], "session2":[0,1,16,18,23,25,25], "session3": [0,3,9,16]}, #session1:21 (brain+eyes), session2:16 (brain+eyes), session3:20
  "fha01":  {"session1": [0,4,13], "session2": [0,10,14],"session3": [0,8,15,21]}, #session1: 1,25?, session2: ok
  "hwh21":  {"session1": [0,1,2,3,11,15,18], "session2": [0,7,10,14,25],"session3": [0,1,2,4,5,10,12,22]}, # session1:10?, session2:22, session3:28?  
  "rsh17":  {"session1": [0,1,3,14,15], "session2":[0,1,2,4,17,21,22], "session3": [0,1,2,3,5,15]}, #session3:13, 27
  "zwi25":  {"session1": [0,22,24,25], "session2": [0,22],"session3": [0,12,21,23]}, 
  "tdn02":  {"session1": [4,15], "session2": [5,15],"session3": [8,14,24,25]}, # session2: 22,24,28,29?, session3:26
  "uka11":  {"session1": [0,1,2], "session2": [0,1,8],"session3": [0,4]}, 
  "csi07":  {"session1": [0,1,11,22,23], "session2": [0,4,10],"session3": [0,1,7,16]}, # session1:1(eye+brain), session2:26(eye+brain)
  "rsg06":  {"session1": [0,2,20,21,22], "session2": [0,1,25],"session3": [0,1,22]}, # session1: 13,19 (eye+brain), session3:14,21,22
  "mwa29":  {"session1": [0,1,12,14,15], "session2": [0,1,14,16], "session3": [0,2,11]},
  "ade02":  {"session1": [2], "session2": [9,23,25],"session3": [8,16]}, # session1:19 (brain+eye), session3:18,20,24,27
  "dtl05":  {"session1": [0,20,24], "session2": [0,24,25,26],"session3": [0,22,27]},
  "rbe04":  {"session1": [0,13], "session2": [0,14,21,27],"session3": [0,9,12,15,28]},
  "jmn22":  {"session1": [0,15], "session2": [0,15,25],"session3": [0,23,25]},
  "dss19":  {"session1": [1,3,22], "session2": [0,8,23],"session3": [0,5,21]},
  "fte25":  {"session1": [0,2,11], "session2": [0,2],"session3": [0,3,16,17]},
  "hyr24":  {"session1": [0,7,13,14,18,21], "session2": [0,3,14,20],"session3": [0,2,12,13]},
  "ank24":  {"session1": [0,2,5,7,11], "session2": [0,1,2,10],"session3": [0,1,7,8]},
  "dja01":  {"session1": [0,12,21,29], "session2": [0,13,21],"session3": [0,10,25]},
  "fmn28":  {"session1": [0,9,20], "session2": [0,10],"session3": [0,5]},
  "omr03":  {"session1": [0,1,2,3,4,5,6,7,8,9,10,11,13], "session2": [0,1,2,3,4,5,6,7,8,9,10,11,12],"session3": [0,1,2,3,4,5,6,7,8,9,10,11,14,20,21,22]},
  "amy20":  {"session1": [0,10,17], "session2": [0,3,11],"session3": [0,3,16]},
  "ski23":  {"session1": [0,2,21,27],  "session2": [0,2,10,14,20],"session3": [0,1,6,7,24]},
  "jry29":  {"session1": [0,3,27], "session2": [0,3,25],"session3": [0,3,27]},
  "jpa10":  {"session1": [0,1,9,10,13,19,20], "session2": [0,1,11,25,27,29],"session3": [0,1,12,19,20,25,26]},
  "epa14":  {"session1": [0,6,18,19,20,21], "session2": [0,11,18,21,24], "session3": [0,11,25,27,29]},
  "crr22":  {"session1": [0,6,14,15,23,25,26], "session2": [0,5,19,23,27,29], "session3": [4,10,17,24]},
  "jyg27":  {"session1": [0,1,2,5,11,13,14,18,19,20,21,23,24], "session2": [0,6,7,10], "session3": [0,3,5,7,11,12,13,14,16,17,20]},
  "bbi29":  {"session1": [0,5,17,22], "session2": [0,1,8,15,16,17,20], "session3": [0,6,16,17,20,21,22]},
  "ece24":  {"session1": [0,7,23,26,28,29], "session2": [0,3,10,16,18,19,21], "session3": [0,5,6,8,9,21,27]},
  "hay06":  {"session1": [0,1,2,3,4,5,10,12,16,17,18,19,20,21], "session2": [0,9,11,12,17,18,19,21,22], "session3": [0,8,10,11,14,17]},
  "awa19":  {"session1": [0,1,21,23,24], "session2": [0,1,23,25,26], "session3": [0,1,21,24,26,27,28,29]},
  "mca10":  {"session1": [0,22,24,28,29], "session2": [1,26,27,28,29]},
  "hky23":  {"session1": [6,10,11,19,20,27,28,29], "session2": [6,17,27,28,29]},
  }  ,
  "eeg": {
    "bsr27":  {"session1": [0,5], "session2": [1],"session3": [1]}, # ok
    "dsa23":  {"session1": [0,1,2,3,6,10,11,12,15], "session2": [0,1,2,3,4,6,7,11,14,15,20,21], "session3": [0,1,2,3,9,12,13,14,15,19]}, # session1: 24? ok
    "mtr13":  {"session1": [0,1,2], "session2": [0,5],"session3": [0,1,7,9,11,16,23]}, # session1:10,18,21?, session2:11,27?, session3:8,28?
    "gto28":  {"session1": [0,2,3,25,29], "session2": [0,3,5,20,24],"session3": [0,2,3]}, # session 1 noisy, session2:16?, session3:14?
    "ami28":  {"session1": [0,4,18,21], "session2": [0,9,13,16],"session3": [0,1,2,11]}, #session3:28?
    "lka10":  {"session1": [0,6,7,9,10,14,16], "session2": [0,8],  "session3": [0,2,9,12,14,15,16,20]}, #session2:1,22,25 (eyes+brain)?session3:24  
    "qqn19":  {"session1": [0,1,2,4,5], "session2": [0,1,2,3,4,10,11,14],"session3": [0,2,3,9]}, #session1:check9, session2:5, session3:4 (brain+blink)
    "mtr19":  {"session1": [0,6,12,15,17], "session2":[0,5,13,15,19,26], "session3": [0,1,2,8,10,11,13,14,15]}, #session1:21 (brain+eyes), session2:16 (brain+eyes), session3:20
    "fha01":  {"session1": [0,4], "session2": [0,2],"session3": [0,2,8,15,18,22]}, #session1: 1,25?, session2: ok
    "hwh21":  {"session1": [0,1,4,5,12,15,19,24], "session2": [0,1,2,5,10,11,12,13,24,25],"session3": [0,1,2,4,7,8,10,13,14,15,16]}, # session1:10?, session2:22, session3:28?  
    "rsh17":  {"session1": [0,1,3,4,9,11,15,16,17,18,24], "session2":[0,2,3,4,6], "session3": [0,1,2,3,6,18,22,23]}, #session3:13, 27
    "zwi25":  {"session1": [0,1,2,3,4,7,9,10,14,18], "session2": [0,1,3,4,10],"session3": [0,6,8,10,11,15,22,26]}, 
    "tdn02":  {"session1": [0,10,12,16], "session2": [1], "session3": [0,1,13,14]}, # session2: 22,24,28,29?, session3:26
    "uka11":  {"session1": [0,1,2,4,11,17,26], "session2": [0,1,2,6],"session3": [0,1,2,3,7]}, 
    "csi07":  {"session1": [0,1,4,9,14,16], "session2": [0,1,10,22,26],"session3": [0,1,8,13,19,21]}, # session1:1(eye+brain), session2:26(eye+brain)
    "rsg06":  {"session1": [0,3,8,11,12,13,15,18,19], "session2": [0,4],"session3": [0,3,6]}, # session1: 13,19 (eye+brain), session3:14,21,22
    "mwa29":  {"session1": [0,1,5,12,19], "session2": [0,1,2,8], "session3": [0,1]},
    "ade02":  {"session1": [0,10], "session2": [0,2,5],"session3": [1,12,13,17,19]}, # session1:19 (brain+eye), session3:18,20,24,27
    "dtl05":  {"session1": [0,1,2,15,23], "session2": [0,2,14,19,21,22,24],"session3": [0,10,12,18,17,22]},
    "rbe04":  {"session1": [0,3,5], "session2": [0,1,4,7],"session3": [0,3,14,16]},
    "jmn22":  {"session1": [0,1,15], "session2": [0,1,14],"session3": [0,3,14,16]},
    "dss19":  {"session1": [0,6,8,28], "session2": [0,5,10],"session3": [0,17]}  ,
    "fte25":  {"session1": [], "session2": [0,1,3,9,17],"session3": [0,5,14]},
    "hyr24":  {"session1": [0,2,5,9,10,11,12,13,16,17,18], "session2": [0,4,7,9,10,11,19],"session3": [0,4,13]},
    "ank24":  {"session1": [0,1,3,4,5,8,9,10,17,18], "session2": [0,1,2,4,5,7,10,15],"session3": [0,1,2,7,8,14,15,19,26]},
    "dja01":  {"session1": [0,2,3,5,14,18,22], "session2": [0,1,2,3,5,6,7,13,19,20,24],"session3": [0,1,2,4,5,13,14,15,16]},
    "fmn28":  {"session1": [0,1,3,10,12,14,17,19,20,23,29], "session2": [0,3,18,21,22,23,24,25,26,28,29],"session3": [0,3,10, 14, 19, 20,22, 25, 27, 28, 29]},
    "omr03":  {"session1": [0,3,4,5,6,7,10,13,15,16], "session2": [0,1,4,5,6,7,8,9,10],"session3": [0,1,2,3,4,6,7]},
    "amy20":  {"session1": [0,2,9,13], "session2": [0,1, 11, 19,21,27,28],"session3": [0,2,3,17]},  
    "ski23":  {"session1": [0,1,2,5], "session2": [0,2,10,14,23,28],"session3": [0,2,6,8]},
    "jry29":  {"session1": [0,2,8,10,14], "session2": [0,2,3,13,23,28],"session3": [0,10,23]},
    "jpa10":  {"session1": [0,1,3,4,5,6,8,11,12,13,15,16,17,20], "session2": [0,1,3,4,5,8,11,13,16,26,29],"session3": [0,2,3,4,5,7,8,9,12,13,14]},
    "epa14":  {"session1": [0,1,2,17,20], "session2": [0,1,2,3,17,18,19], "session3": [0,3,5,14,24]},
    "crr22":  {"session1": [0,2,14,15,17,18,19,20,21], "session2": [0,3,10,14,19], "session3": [0,1,3,10,14,19]},
    "jyg27":  {"session1": [0,2,3,4,5,11,16,17,18,19,20], "session2": [0,2,3,9,11,12,13,16,17,18,19,20,21,22], "session3": [0,1,3,9,10,11,14,15,16,17,18,19,20]},
    "bbi29":  {"session1": [0,3,11,13,15,16,17,18,19,20], "session2": [0,4,8,9,14,15,19,12,16,17,18,25,28], "session3": [0,3,4,9,11,13,14,18,19,20,27]},
    "ece24":  {"session1": [0,4,13,14,16,23], "session2": [0,3,6,10,11,14,15,16,17,19], "session3": [0,1,4,5,6,13,14,15,16,17,18]},
    "hay06":  {"session1": [0,1,2,4,8,11,12,13,14,18,20,21,22,23,27,29], "session2": [0,1,2,3,4,7,11,14,16,17,18,19], "session3": [0,1,2,4,6,8,13,17,18,19,21,26]},
    "awa19":  {"session1": [0,3,8,10,16,18,19,20], "session2": [0,8,9,14,16,17,18,19], "session3": [0,6,14,13,16,17,18,19]},
    "mca10":  {"session1": [0,5,9,11,15,21,23,24,25,26,27], "session2": [0,6,9,12,13,18,19,20,21,22,23,24]},
    "hky23":  {"session1": [6,8,15,16,17,19,20,27], "session2": [1,2,9,11,12,13,14,16,17]},
    }  
  }
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

# Notes:
# fmn28, removed more ICs (I have not redone ICA, just selected more ICs to remove)
# same for omr03 session3
# same for jpa10 session2

#%%

chs = {'meg':['MEG0111', 'MEG0121', 'MEG0131', 'MEG0211', 'MEG0221', 'MEG0231','MEG0311', 'MEG0321', 'MEG0331', 'MEG1511', 'MEG1521', 'MEG1531'],
       'eeg':['EEG001', 'EEG002', 'EEG003', 'EEG004', 'EEG005', 'EEG006','EEG007', 'EEG008']
       }

scalings = {'meg': None,
            'eeg': 40e-6}

overwrite = False
#%%
# loop through participants
for subject_id in subjects_list:
    # subject_id = 'S002_dsa23'
    if not subject_id[5:] in exclude_ics['eeg']:
        print(subject_id[5:] + ' does not have ICA yet')
        continue
    print('Doing subject ' + subject_id + '\nSessions')
    subject_path = op.join(data_dir,subject_id,'tsss')
    session_folders = os.listdir(subject_path)
    sessions =  [x for x in session_folders if x.startswith('2')]
    sessions.sort()

    for idx,session in enumerate(sessions):
        # idx = 0
        # session = sessions[idx]
        save_plot_psd = op.join(output_plot_psd,subject_id,session) # set path for output files
        save_plot_traces = op.join(output_plot_traces,subject_id,session)
        save_plot_ica = op.join(output_plot,subject_id,'ICs_removed',session)
        save_dir = op.join(data_dir,subject_id,'ica',session)

        if not os.path.exists(save_plot_ica):
            os.makedirs(save_plot_ica)
        if not os.path.exists(save_plot_psd):
            os.makedirs(save_plot_psd)
        if not os.path.exists(save_plot_traces):
            os.makedirs(save_plot_traces)
            
        file_dir = op.join(subject_path,session)
        blocks =  os.listdir(file_dir)
        blocks = [x for x in blocks if x.startswith('B')]
        blocks = [x for x in blocks if 'tsss' in x]
        blocks.sort()
        print(blocks)

        tags = subject_id.split("_")
        subject_idx = tags[1]    
        
        for block in blocks:
            # block = 'Block_01_tsss.fif'
            #print(block)
            raw_file = op.join(file_dir,block)
            save_dir = op.join(data_dir,subject_id,'ica',session)
            path_outfile = op.join(save_dir,block[:-4] +'_ica.fif')# os.path.join(result_path,file_name +'ica-' + str(subfile) + '.fif') 
            if os.path.exists(path_outfile) and not overwrite:
                print(subject_id + '_'  + block + ' has been done already!\n')
                if redo_list[subject_id[5:]]['session' + str(idx+1)] == False:
                    continue
                else:
                    print('but it will be done again!!!!')
            else:
                print('Applying ICA weights to ' + subject_id + '_'  + block)
                if overwrite:
                    print('Doing ' + subject_id + '_'  + block + ' again!')

            #load ica if it has not been loaded already
            if 'ica_fit' not in locals():
                print('\n\nLoading ica weigths for session ' + str(session) + '\n====================================================\n')
                ica_fit = {}
                for sensor_type in ['eeg','meg']:
                    file_name = op.join(save_dir,sensor_type +'_ica.fif')
                    # load ica
                    if os.path.exists(file_name):
                        ica = mne.preprocessing.read_ica(file_name, verbose=None)
                        ica_fit[sensor_type] = ica
                    
             
            raw_ica = mne.io.read_raw_fif(raw_file,verbose=True,preload=True)  
            
            for sensor_type in ['eeg','meg']:
                if len(raw_ica.info['bads']) == 64 and sensor_type == 'eeg':
                    continue
                
                components = exclude_ics[sensor_type][subject_idx]['session'+str(idx+1)]
                ica_fit[sensor_type].exclude = components

                if components != []:
                    c = exclude_ics[sensor_type][subject_id[5:]]['session'+ str(idx+1)]
                    for i in c:
                        print(i)
                        fig = ica_fit[sensor_type].plot_properties(raw_ica, picks= i)
                        filename =  op.join(save_plot_ica,session + '_' + sensor_type + '_' + block[:-4]+ '_ic' + str(i) + '.png')
                        plt.savefig(filename)
                        plt.close()                
                  
                    print('Removing ' + sensor_type + ' ICs'  + str(components) )
                    ica_fit[sensor_type].apply(raw_ica)

            if len(raw_ica.info['bads']) != 64:
                raw_ica.set_eeg_reference(ref_channels='average', projection=True) # https://mne.tools/stable/auto_tutorials/preprocessing/55_setting_eeg_reference.html

            raw_ica.save(path_outfile,overwrite=True)
            
          
            fig = raw_ica.plot_psd(tmin=10,fmin=3,fmax=147,xscale='log')
            filename =  op.join(save_plot_psd,'PSD_ica_' + block[:-3] + 'png')
            fig.savefig(filename)    
            plt.close()
            
            raw_ica.filter(l_freq=1,h_freq=148,h_trans_bandwidth=2) #for plottign purporses, doing this after saving
            for sensor_type in ['eeg','meg']:
                chan_idxs = [raw_ica.ch_names.index(ch) for ch in chs[sensor_type]]
                with mne.viz.use_browser_backend('matplotlib'):
                    fig = raw_ica.plot(order=chan_idxs,start=100,duration=5,scalings=scalings[sensor_type]) #40e-6
                    filename =  op.join(save_plot_traces, block[:-4] + '_' + sensor_type + '_ica.png')
                    fig.savefig(filename)    
                    plt.close()
                
        
        if 'ica_fit' in locals():
            del ica_fit
