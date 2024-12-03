
#%%

import numpy as np
import re
#import os.path as op
import mne
import os
#import struct
import matplotlib.pyplot as plt

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt') # so MNE plots are in a pop-up window, not inline
# %matplotlib qt / inline

#%%
def check_block_name(dir_list):
    blocks_list = []
    load_list = []
    for block in dir_list:
        string = re.findall("[a-zA-Z]+", block)
        pattern = "^Block_[0-9]{2,2}.fif$"
        state = bool(re.match(pattern, block))
        
        if state and block not in blocks_list: 
                blocks_list.append(block)
                load_list.append(block)
        else:
            num = re.findall(r'\d+', block) # extract number from string
            newstring = string[0].capitalize()
            if num and newstring.startswith('B'):
                num = "%02d" % (int(num[0]),) # makes sure it is a two digit number
                if string[1] == 'fif':
                    new_name = 'Block_' + num + '.' + string[1]
                    if new_name not in blocks_list: 
                        blocks_list.append(new_name)
                        load_list.append(block)
            else:
                print('File name removed from block list. Not a task file')
    return blocks_list, load_list

#%%
#data_dir = r'D:\PROJECTS\CAUSAL_NETWORKS\DATA'
#output_plot_folder = r'D:\PROJECTS\CAUSAL_NETWORKS\PLOTS\QUALITY_CHECK\head_movements'
#output_dir = r'D:\PROJECTS\CAUSAL_NETWORKS\ANALYSIS\pos' #r'D:\PROJECTS\CAUSAL_NETWORKS\ANALYSIS\HEAD_POS'

#data_dir =  r'Z:' # r'D:\PROJECTS\CAUSAL_NETWORKS\DATA'
#output_plot_folder = r'Y:\PLOTS\QUALITY_CHECK\head_movements' #r'D:\PROJECTS\CAUSAL_NETWORKS\PLOTS\QUALITY_CHECK\psd'
#output_dir = r'Y:\ANALYSIS\pos'#r'D:\PROJECTS\CAUSAL_NETWORKS\ANALYSIS\tsss'

data_dir =  r'/raw/Project0349'  # r'D:\PROJECTS\CAUSAL_NETWORKS\DATA'
output_plot_folder = r'/analyse/Project0349/PLOTS/QUALITY_CHECK/head_movements' #r'D:\PROJECTS\CAUSAL_NETWORKS\PLOTS\QUALITY_CHECK\psd'
output_dir = r'/analyse/Project0349/ANALYSIS'#r'r'/analyse/Project0349/
#subjects_list = ['ami28' ]#'ami28','jbe07','bsr27', 'dsa23', 'gto28',
subjects_list = [ f.name for f in os.scandir(data_dir) if f.is_dir() and f.name[0] != '_' and f.name[0].islower()]
#subjects_list = ['ade02', 'csi07', 'mwa29',  'rsg06', 'tdn02', 'uka11']
#subjects_list =  ['qqn19']
subjects_list.sort()
print(subjects_list) 


#%% folder mapping 
folder_mapping  = {
  "bsr27": "S001",
  "dsa23": "S002",
  "mtr13": "S003",
  "gto28": "S004",
  "ami28": "S005",
  "lka10": "S006",
  "qqn19": "S007",
  "mtr19": "S008",
  "fha01": "S009",
  "hwh21": "S010",
  "rsh17": "S011",
  "zwi25": "S012",
  "tdn02": "S013",
  "uka11": "S014",
  "csi07": "S015",
  "rsg06": "S016",
  "mwa29": "S017",
  "ade02": "S018",
  "dtl05": "S019",
  "rbe04": "S020",
  "jmn22": "S021",
  "dss19": "S022",
  "fte25": "S023",
  "hyr24": "S024",
  "ank24": "S025",
  "dja01": "S026",
  "fmn28": "S027",
  "omr03": "S028",
  "amy20": "S029",
  "ski23": "S030",
  "jry29": "S031",
  "jpa10": "S032",
  "epa14": "S033",
  "crr22": "S034",
  "jyg27": "S035",
  "bbi29": "S036",
  "ece24": "S037",
  "awa19": "S038",
  "hay06": "S039",
  "mca10": "S040",
  "hky23": "S041",
  }


#%%
# loop through participants
for subject_id in subjects_list:
    # subject_id = 'jbe07'
    if subject_id not in folder_mapping: # these participants did not complete the experiment
        print(subject_id + ' did not complete the experiment')
        continue
    
    print(subject_id)
    print('Doing subject ' + subject_id + '\nSessions')
    subject_path = os.path.join(data_dir,subject_id)
    session_folders = os.listdir(subject_path)
    sessions =  [x for x in session_folders if x.startswith('2')]
    print(sessions)

    for idx,session in enumerate(sessions):
        # idx = 2, session = sessions[idx]
        print(session)
        if subject_id in folder_mapping:
            save_dir = os.path.join(output_dir,folder_mapping[subject_id] + '_' + subject_id,'pos',session) # data will be saved as subject code
        else:
            continue
        
        save_plot = os.path.join(output_plot_folder,subject_id,session)
        if not os.path.exists(save_dir): # create folder if it doesn't exist
            os.makedirs(save_dir)
        else:
            print('Participant ' + subject_id + ' already has head position file computed!\n')
            continue
        
        if not os.path.exists(save_plot): # create folder if it doesn't exist
            os.makedirs(save_plot)

        file_dir = os.path.join(subject_path,session)
        blocks =  os.listdir(file_dir)
        #blocks = [x for x in blocks if x.startswith('B')]
        save_blocks, load_blocks = check_block_name(blocks)
        
        for nb, block in enumerate(load_blocks):
            if subject_id == 'dtl05' and idx ==2  or  subject_id == 'uka11' and idx ==2 :
                print('Fixing block name for ' + subject_id + ' ' + block)
                num = int(re.findall(r'\d+', block)[0]) + 1
                block = 'Block_' + "%02d" % (num,) + '.fif'
                save_blocks[nb] = block
                #block = 'Block_25.fif'
            # block = load_blocks[nb] 
            print(block)
            #-----------------------------------------------------------------------------
            # Load file
            #-----------------------------------------------------------------------------
            # if file exists continue
            
            #file_name = blocks[0] # loop through blocks
            # check if movement comepnsation mne.chpi.KIT['CONTINUOUS'] <- result is 1 even if no cHPI
            plot_movement = True
            raw_file = os.path.join(file_dir,block)
            save_file= os.path.join(save_dir,save_blocks[nb])
            
            file_exists = [f for f in os.listdir(save_dir) if (save_blocks[nb][:-4] in f)]
            if file_exists:
                print('head position has been computed')
                continue
            
            raw = mne.io.read_raw_fif(raw_file,preload=True, verbose=False)
            # raw2 = mne.io.read_raw_fif('Z:\\jmn22\\221215\\Block11.fif',preload=True, verbose=False)
            chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw) # this will run even if no cHPI
            check_cHPI = np.all(np.isnan(np.unique(chpi_amplitudes['slopes']))) # if true, all values are nan, therefore no cHPI used  

            if check_cHPI:
                print('cHPI was off')
                # create a dummy head position file that contains origin information 
                pos_file = save_file[:-4] + '_dummy_pos.fif'
                chpi_pos = np.zeros((1,10))
                chpi_pos[:,4:7] = raw.info['dev_head_t']['trans'][:3,3] 
                mne.chpi.write_head_pos(pos_file,chpi_pos)

            else:
                print('cHPI was on! extracting head movements during the recording')
                chpi_locs = mne.chpi.compute_chpi_locs(raw.info,chpi_amplitudes,verbose=True)
                chpi_pos  = mne.chpi.compute_head_pos(raw.info, chpi_locs,verbose=True)
                pos_file = save_file[:-4] + '_pos.fif'
                mne.chpi.write_head_pos(pos_file,chpi_pos)
                if plot_movement:
                    fig = mne.viz.plot_head_positions(chpi_pos, mode='traces')
                    filename =  os.path.join(save_plot,save_blocks[nb][:-4] + '_head_movement.png')
                    fig.savefig(filename)    
                    plt.close()
                    #mne.viz.plot_head_positions(chpi_pos, mode='field')

            
            # raw_tsss.filter(l_freq=1,h_freq=148,h_trans_bandwidth=2)
            # raw_tsss.plot()
            # raw.filter(l_freq=1,h_freq=148,h_trans_bandwidth=2)
            # raw.plot()
            
            #%%
            
# data_path = mne.datasets.sample.data_path()
# sample_raw = mne.io.read_raw_fif(data_path + '/MEG/sample/sample_audvis_raw.fif')

# sample_raw.info['dev_head_t']['trans'] = raw.info['dev_head_t']['trans'] 
    # np.array(
    # [[ 0.71960711, 0.63715118, -0.27605024, -0.07788484],
    #  [-0.67182261, 0.73935592, -0.04479922, -0.01268383],
    #  [ 0.17555553, 0.21769466, 0.96009851, 0.06360817],
    #  [ 0., 0., 0., 1. ]])

# trans = mne.read_trans(data_path + '/MEG/sample/sample_audvis_raw-trans.fif')
        
# mne.viz.plot_alignment(raw.info, trans=trans, subject='sample',
#                        subjects_dir=data_path + '/subjects',
#                        coord_frame='meg', dig=True)

# chpi_pos2 = mne.chpi.read_head_pos('D:\\PROJECTS\\CAUSAL_NETWORKS\\ANALYSIS\\HEAD_POS\\rsh17\\220627\\Block_19_pos.fif')

# HPI_pos = [[],[]]
# HPI_pos[0] = chpi_pos
# HPI_pos[1] = chpi_pos2
           
# Maxfilter Data
# Block_raw_sss = mne.preprocessing.maxwell_filter(
#     Block_raw, st_duration=100, st_correlation=.9, 
#     coord_frame='head', destination=tuple(np.mean(np.concatenate(HPI_pos, axis=0)[:,4:7],axis=0)), 
#     head_pos=HPI_pos[b-1], verbose=True)
    
# dest = tuple(np.mean(chpi_pos[:,4:7],axis=0))
# raw.info['dev_head_t']['trans'][:3,3] 


# example from no error run
# raw.info['dev_head_t'] 
# <Transform | MEG device->head>
# [[ 0.99851245  0.05385358  0.00854943  0.00089119]
#  [-0.05440345  0.99450105  0.08948842  0.00443683]
#  [-0.00368315 -0.08982041  0.99595135  0.05514803]
#  [ 0.          0.          0.          1.        ]]

#raw.info['dev_head_t']['trans'] 
# array([[ 9.98512447e-01,  5.38535789e-02,  8.54943134e-03,
#          8.91192933e-04],
#        [-5.44034466e-02,  9.94501054e-01,  8.94884244e-02,
#          4.43683285e-03],
#        [-3.68314562e-03, -8.98204073e-02,  9.95951355e-01,
#          5.51480316e-02],
#        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#          1.00000000e+00]])


