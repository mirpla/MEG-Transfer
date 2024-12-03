
#%%

import os.path as op
import mne
import os
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt') # so MNE plots are in a pop-up window, not inline
# %matplotlib qt

#%%
# local machine
# data_dir =   r'Y:\ANALYSIS'
# output_plot = r'Y:\PLOTS\ica' #r'D:\PROJECTS\CAUSAL_NETWORKS\PLOTS\QUALITY_CHECK\psd'

data_dir =   r'/analyse/Project0349/ANALYSIS'
output_plot = r'/analyse/Project0349/PLOTS/ica' #r'D:\PROJECTS\CAUSAL_NETWORKS\PLOTS\QUALITY_CHECK\psd'

input_folder = 'tsss'
output_folder = 'ica'


#subjects_list = ['S005' ]
subjects_list = [ f.name for f in os.scandir(data_dir) if f.is_dir() and f.name[0] == 'S' ]
#subjects_list =  ['S022_dss19']
subjects_list.sort()

print(subjects_list)


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
  "fmn28":  {"session1": False, "session2": False, "session3": False},
  "omr03":  {"session1": True, "session2": True, "session3": False},
  "amy20":  {"session1": True, "session2": False, "session3": False},
  "ski23":  {"session1": False, "session2": True, "session3": False},
  "jry29":  {"session1": False, "session2": False, "session3": False},
  "jpa10":  {"session1": True, "session2": False, "session3": True},
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


#%%

# loop through participants
error_list = []
for subject_id in subjects_list:
    #subject_nb = folder_mapping[subject_id]
    # sub_idx = -1
    # subject_id =subjects_list[sub_idx]
    print('Doing subject ' + subject_id + '\nSessions')
    subject_path = op.join(data_dir,subject_id,input_folder)
    session_folders = os.listdir(subject_path)
    sessions =  [x for x in session_folders if x.startswith('2')]
    print(sessions)
    sessions.sort()

    for idx,session in enumerate(sessions):
        # idx = 1
        # session = sessions[idx]
        print(session)
        
        save_plot = op.join(output_plot,subject_id,session) # set path for output files
        save_dir = op.join(data_dir,subject_id,output_folder,session)
        # create folder if it doesn't exist
        if not os.path.exists(save_plot):
            os.makedirs(save_plot)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            print('\n\nICA has been computed for ' + subject_id + ' session ' + session + '\n\n===================================')
            
            if  redo_list[subject_id[5:]]['session' + str(idx+1)] == False:
                continue
            else:
                print('But REDOING pre-processing!!!' )
                
        # session_idx = 0 # loop through sessions
        # session = sessions[session_idx]
        file_dir = op.join(subject_path,session)
        blocks =  os.listdir(file_dir)
        blocks = [x for x in blocks if x.startswith('B')]
        blocks = [x for x in blocks if 'tsss' in x]
        blocks.sort()
        print(blocks)


        eeg_all = []  # this will be used for the case when the eeg and meg data has been split into separate variables. See Try Exception below  
        for nb,block in enumerate(blocks):
            print(block)

            #-----------------------------------------------------------------------------
            # Load file
            #-----------------------------------------------------------------------------
            raw_file = op.join(file_dir,block)
            raw = mne.io.read_raw_fif(raw_file,preload = True)

            try:
                raw.pick_types(meg=True,eeg=True) 
            except:
                print('No EEG data for ' + subject_id + ' ' +  block +'\n ==========================================\n')
                raw.pick_types(meg=True,eeg=False) 
                
            #raw_meg.info['bads'] = ['EEG 007']
            raw.resample(200)
            raw.filter(1, 40)
            if nb == 0:
                raw_all = mne.io.concatenate_raws([raw],on_mismatch='ignore' )
            else:
                try:
                    print('Concatenating file containing meg and eeg data\n')
                    raw_all = mne.io.concatenate_raws([raw_all, raw],on_mismatch='ignore')
                except Exception as ex:
                    # this exception was created for participant bsr27. EEG data was lost for some of the blocks in session idx = 1
                    print('Some blocks in this session do not contain eeg data\nConcatenating data for eeg and meg separately')
                    error_list.append(subject_id + '_' + block + '_' + repr(ex))
                    eeg = raw.copy().pick_types(meg=False,eeg=True)
                    raw.pick_types(meg=True,eeg=False)
                    raw_all.pick_types(meg=True,eeg=False)
                    raw_all = mne.io.concatenate_raws([raw_all, raw],on_mismatch='ignore')
                    
                    if eeg_all == []:
                        eeg_all = mne.io.concatenate_raws([eeg],on_mismatch='ignore' )
                    else:
                        eeg_all = mne.io.concatenate_raws([eeg_all,eeg],on_mismatch='ignore' )
            
        del raw
        if eeg_all != []:
            del eeg
        
        ica = ICA(method='fastica', random_state=97, n_components=30,verbose=True)

        for sensor_type in ['eeg','meg']:
            print(sensor_type)
            
            # if session in no_eeg and sensor_type == 'eeg':
            #     continue

            if raw_all.info['nchan'] == 306 and sensor_type == 'eeg':
                continue
            
            ica.fit(raw_all,picks = sensor_type, verbose=True)
            
            fig = ica.plot_components(picks = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]);
            filename =  op.join(save_plot,sensor_type + '_ica_topo_00-14.png')
            fig.savefig(filename)
            plt.close()
            
            fig = ica.plot_components(picks = [15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]);
            filename =  op.join(save_plot,sensor_type +'_ica_topo_15-29.png')
            fig.savefig(filename)
            plt.close()
            
            with mne.viz.use_browser_backend('matplotlib'):
                fig = ica.plot_sources(raw_all,title='ICA',picks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],start = 100);
                filename =  op.join(save_plot,sensor_type + '_ica_sources_00-14.png')
                fig.savefig(filename)
                plt.close()
                
            with mne.viz.use_browser_backend('matplotlib'):
                fig = ica.plot_sources(raw_all,title='ICA',picks=[15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],start = 100);
                filename =  op.join(save_plot,sensor_type + '_ica_sources_15-29.png')
                fig.savefig(filename)
                plt.close()
                
            file_name = op.join(save_dir,sensor_type + '_ica.fif')
            ica.save(file_name,overwrite=True)
            
        if eeg_all != []: # this will run for the case when the eeg and meg data has been split into separate variables. See Try Exception above  
            ica.fit(eeg_all, verbose=True)
            
            fig = ica.plot_components(picks = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]);
            filename =  op.join(save_plot,'eeg_ica_00-14.png')
            fig.savefig(filename)
            plt.close()
            
            fig = ica.plot_components(picks = [15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]);
            filename =  op.join(save_plot,'eeg_ica_15-29.png')
            fig.savefig(filename)
            plt.close()
            
            with mne.viz.use_browser_backend('matplotlib'):
                fig = ica.plot_sources(raw_all,title='ICA',picks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],start = 100);
                filename =  op.join(save_plot,'eeg_ica_sources_00-14.png')
                fig.savefig(filename)
                plt.close()
                
            with mne.viz.use_browser_backend('matplotlib'):
                fig = ica.plot_sources(raw_all,title='ICA',picks=[15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],start = 100);
                filename =  op.join(save_plot,'eeg_ica_sources_15-29.png')
                fig.savefig(filename)
                plt.close()
                
            file_name = op.join(save_dir,'eeg_ica.fif')
            ica.save(file_name,overwrite=True)
            


        
        #%% do artifact reduction
        # ica.exclude = [0,15, 10, 17, 18, 23, 29]
        # # Loop over the subfiles 
        # for block in blocks:
        #     print(block)
        #     raw_file = op.join(file_dir,block)
        #     path_outfile = op.join(save_dir,block[:-4] +'_ica.fif')# os.path.join(result_path,file_name +'ica-' + str(subfile) + '.fif') 
        #     raw_ica = mne.io.read_raw_fif(raw_file,verbose=True,preload=True)  
        #     #raw_ica.info['bads'] = ['EEG 007']
        #     ica.apply(raw_ica)
        
        #     raw_ica.save(path_outfile,overwrite=True) 
        #     # raw_ica.filter(l_freq=1,h_freq=148,h_trans_bandwidth=2)
        #     # raw_ica.plot()
        #     # raw = mne.io.read_raw_fif(raw_file,preload = True)
        #     # raw.filter(l_freq=1,h_freq=148,h_trans_bandwidth=2)
        #     # raw.plot()
        #     # raw.plot_psd(tmin=10,fmin=3,fmax=148,xscale='log')            
            
        
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

