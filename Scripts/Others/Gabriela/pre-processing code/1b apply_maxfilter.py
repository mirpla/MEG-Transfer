
#%%

import numpy as np
import os.path as op
import mne
import os
import matplotlib.pyplot as plt
import re


#from meegkit import dss
#from meegkit.utils import create_line_data, unfold
#from scipy import signal

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt') # so MNE plots are in a pop-up window, not inline
# %matplotlib qt / inline
#%% Bad channels
"""
Annotate Bad channles

"""
bad_meg_channels  = {
  "bsr27":  {"session1": ['MEG0922','MEG1042','MEG1341','MEG1243','MEG2332','MEG0243','MEG0332'], "session2": ['MEG0922','MEG1243', 'MEG1042', 'MEG2332','MEG2433','MEG1341'],
             "session3": ['MEG1243', 'MEG1042', 'MEG2332','MEG2433','MEG1341','MEG1511']},
  "dsa23":  {"session1": ['MEG0922', 'MEG1243',  'MEG1042', 'MEG2332'], "session2": ['MEG0922','MEG1511','MEG1341','MEG1243',  'MEG1042', 'MEG2332','MEG2341','MEG2311'], "session3": ['MEG1243',  'MEG1042', 'MEG2332','MEG2341']},
  "mtr13":  {"session1": [ 'MEG1243',  'MEG1042', 'MEG2332', 'MEG1341','MEG1341','MEG0922'], "session2": [ 'MEG1243',  'MEG1042', 'MEG2332','MEG1341','MEG0922'], "session3": ['MEG2341','MEG0922', 'MEG1042', 'MEG1243', 'MEG2332', 'MEG1211']},
  "gto28":  {"session1": ['MEG0922','MEG1243', 'MEG1042', 'MEG2332', 'MEG1341'], "session2":  ['MEG0922','MEG1243', 'MEG1042', 'MEG2332','MEG1341'], "session3": ['MEG0922','MEG1243', 'MEG1042', 'MEG2332']},
  "ami28":  {"session1": [ 'MEG0922','MEG1243',  'MEG1042', 'MEG2332'], "session2": ['MEG1243', 'MEG0922', 'MEG1042', 'MEG2332','MEG2433'], "session3": [ 'MEG2332', 'MEG1042', 'MEG0922', 'MEG1243']},
  "lka10":  {"session1": ['MEG2332', 'MEG1042', 'MEG1243','MEG0922'], "session2": ['MEG2332', 'MEG1042', 'MEG1243','MEG0922'], "session3": ['MEG0922', 'MEG1243',  'MEG1042', 'MEG2332']},
  "qqn19":  {"session1": ['MEG2332', 'MEG1042', 'MEG1243'], "session2":['MEG2332', 'MEG1042', 'MEG1243'], "session3": ['MEG2332', 'MEG1042', 'MEG1243','MEG0221']},
  "mtr19":  {"session1": ['MEG0922', 'MEG1243',  'MEG1042', 'MEG2332','MEG0243'], "session2":['MEG0922', 'MEG1243',  'MEG1042', 'MEG2332','MEG0243','MEG1512','MEG1823'], "session3": ['MEG0922', 'MEG1243',  'MEG1042', 'MEG2332','MEG0243']},
  "fha01":  {"session1": [ 'MEG0922','MEG1243',  'MEG1042', 'MEG2332','MEG1621','MEG2311','MEG2331','MEG2531','MEG0141','MEG2441'], "session2":[ 'MEG0922','MEG1243',  'MEG1042', 'MEG2332','MEG2433'], "session3": [ 'MEG0922','MEG1243',  'MEG1042', 'MEG2332']},
  "hwh21":  {"session1": ['MEG0922', 'MEG1243',  'MEG1042', 'MEG2332','MEG0243','MEG1823'], "session2":['MEG0922', 'MEG1243',  'MEG1042', 'MEG2332','MEG0243','MEG1823'], "session3": ['MEG0922', 'MEG1243',  'MEG1042', 'MEG2332','MEG0243','MEG1823']},
  "rsh17":  {"session1": ['MEG0922', 'MEG1243',  'MEG1042', 'MEG2332','MEG0243','MEG1823'], "session2":['MEG0922', 'MEG1243',  'MEG1042', 'MEG2332','MEG0243','MEG1823'], "session3": ['MEG0922', 'MEG1243',  'MEG1042', 'MEG2332','MEG0243','MEG1823']},
  "zwi25":  {"session1": ['MEG0922', 'MEG1243',  'MEG1042', 'MEG2332','MEG0243','MEG1823'], "session2":['MEG1823','MEG0922', 'MEG1243',  'MEG1042', 'MEG2332','MEG0243','MEG1823'], "session3": ['MEG0922', 'MEG1243',  'MEG1042', 'MEG2332','MEG0243','MEG1823']},
  "tdn02":  {"session1": ['MEG0922', 'MEG1243',  'MEG1042', 'MEG2332','MEG0243','MEG1823'], "session2":['MEG1823','MEG0922', 'MEG1243',  'MEG1042', 'MEG2332','MEG0243','MEG1823'], "session3": ['MEG0922', 'MEG1243',  'MEG1042', 'MEG2332','MEG0243','MEG1823']},
  "uka11":  {"session1": ['MEG0922', 'MEG1243',  'MEG1042', 'MEG2332','MEG0243','MEG1823'], "session2":['MEG1823','MEG0922', 'MEG1243',  'MEG1042', 'MEG2332','MEG0243','MEG1823'], "session3": ['MEG0922', 'MEG1243',  'MEG1042', 'MEG2332','MEG0243','MEG1823']},
  "csi07":  {"session1": ['MEG0922', 'MEG1243',  'MEG1042', 'MEG2332','MEG0243','MEG1823'], "session2":['MEG1823','MEG0922', 'MEG1243',  'MEG1042', 'MEG2332','MEG0243','MEG1823'], "session3": ['MEG0922', 'MEG1243',  'MEG1042', 'MEG2332','MEG0243','MEG1823']},
  "rsg06":  {"session1": ['MEG0922', 'MEG1243',  'MEG1042', 'MEG2332','MEG0243','MEG1823'], "session2":['MEG1823','MEG0922', 'MEG1243',  'MEG1042', 'MEG2332','MEG0243','MEG1823'], "session3": ['MEG0922', 'MEG1243',  'MEG1042', 'MEG2332','MEG0243','MEG1823']},
  "mwa29":  {"session1": ['MEG0922', 'MEG1243',  'MEG1042', 'MEG2332','MEG0243','MEG1823'], "session2":['MEG1823','MEG0922', 'MEG1243',  'MEG1042', 'MEG2332','MEG0243','MEG1823'], "session3": ['MEG0922', 'MEG1243',  'MEG1042', 'MEG2332','MEG0243','MEG1823']},
  "ade02":  {"session1": ['MEG0922', 'MEG1243',  'MEG1042', 'MEG2332','MEG0243','MEG1823'], "session2":['MEG1823','MEG0922', 'MEG1243',  'MEG1042', 'MEG2332','MEG0243','MEG1823'], "session3": ['MEG0922', 'MEG1243',  'MEG1042', 'MEG2332','MEG0243','MEG1823']},
  "dtl05":  {"session1": ['MEG0922', 'MEG1042',  'MEG1432', 'MEG2332','MEG0133','MEG1243'], "session2":['MEG0922', 'MEG1042',  'MEG1432', 'MEG2332','MEG0133','MEG1243'], "session3": ['MEG0922', 'MEG1042',  'MEG1432', 'MEG2332','MEG0133','MEG1243']},
  "rbe04":  {"session1": ['MEG0922', 'MEG1042',  'MEG1432', 'MEG2332','MEG0133','MEG1243'], "session2":['MEG0922', 'MEG1042',  'MEG1432', 'MEG2332','MEG0133','MEG1243'], "session3": ['MEG0922', 'MEG1042',  'MEG1432', 'MEG2332','MEG0133','MEG1243']},
  "jmn22":  {"session1": ['MEG0922', 'MEG1042',  'MEG1432', 'MEG2332','MEG0133','MEG1243'], "session2":['MEG0922', 'MEG1042',  'MEG1432', 'MEG2332','MEG0133','MEG1243'], "session3": ['MEG0922', 'MEG1042',  'MEG1432', 'MEG2332','MEG0133','MEG1243']},
  "dss19":  {"session1": ['MEG0922', 'MEG1243', 'MEG1042',  'MEG1432', 'MEG2332','MEG0133','MEG2433','MEG0243'], "session2":['MEG0922', 'MEG1042',  'MEG1432', 'MEG2332','MEG0133','MEG1243','MEG2433','MEG0243'], "session3": ['MEG0922', 'MEG1042',  'MEG1432', 'MEG2332','MEG0133','MEG1243','MEG2433','MEG0243']},
  "fte25":  {"session1": ['MEG0922', 'MEG1243', 'MEG1042',  'MEG1432', 'MEG2332','MEG0133','MEG2433','MEG0243','MEG2622','MEG2623'], "session2":['MEG0922', 'MEG1042',  'MEG1432', 'MEG2332','MEG0133','MEG1243','MEG2433','MEG0243','MEG0242'], "session3": ['MEG0922', 'MEG1042',  'MEG1432', 'MEG2332','MEG0133','MEG1243','MEG2433','MEG0243']},
  "hyr24":  {"session1": ['MEG0133','MEG0922', 'MEG1243', 'MEG1042',  'MEG1813', 'MEG1432','MEG2332'], "session2":['MEG0133', 'MEG0922', 'MEG1243', 'MEG1042',  'MEG1813', 'MEG1432','MEG2332','MEG2433'], "session3": ['MEG0133', 'MEG0922', 'MEG1243', 'MEG1042',  'MEG1813', 'MEG1432','MEG2332','MEG2433']},
  "ank24":  {"session1": ['MEG0133','MEG0922', 'MEG1243', 'MEG1042',  'MEG1813', 'MEG1432','MEG1633','MEG2332','MEG2433'], "session2":['MEG0133', 'MEG0922', 'MEG1243', 'MEG1042',  'MEG1813', 'MEG1432','MEG1633','MEG2332','MEG2433'], "session3":  ['MEG1243', 'MEG1042',   'MEG2332']},
  "dja01":  {"session1": ['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433'], "session2":['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433'], "session3": ['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433']},
  "fmn28":  {"session1": ['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433'], "session2":['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433'], "session3": ['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433']},
  "omr03":  {"session1": ['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433'], "session2":['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433'], "session3": ['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433']},
  "amy20":  {"session1": ['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433'], "session2":['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433'], "session3": ['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433']},
  "ski23":  {"session1": ['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433'], "session2":['MEG0723','MEG1042', 'MEG1243', 'MEG2332', 'MEG2433'], "session3": ['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433']},
  "jry29":  {"session1": ['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433'], "session2":['MEG0723','MEG1042', 'MEG1243', 'MEG2332', 'MEG2433'], "session3": ['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433']},
  "jpa10":  {"session1": ['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433'], "session2":['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433'], "session3": ['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433']},
  "epa14":  {"session1": ['MEG0323','MEG1042', 'MEG1243', 'MEG2332'], "session2":['MEG1042', 'MEG1243', 'MEG2332'], "session3": ['MEG1042', 'MEG1243', 'MEG2332']},
  "crr22":  {"session1": ['MEG1042', 'MEG1243', 'MEG2332'], "session2":['MEG1042', 'MEG1243', 'MEG2332'], "session3": ['MEG1042', 'MEG1243', 'MEG2332']},
  "jyg27":  {"session1": ['MEG1042', 'MEG1243', 'MEG2332'], "session2":['MEG1042', 'MEG1243', 'MEG2332'], "session3": ['MEG1042', 'MEG1243', 'MEG2332', 'MEG2611']},
  "bbi29":  {"session1": ['MEG1042', 'MEG1243', 'MEG2332', 'MEG2611'], "session2":['MEG1042', 'MEG1243', 'MEG2332', 'MEG2611'], "session3": ['MEG1042', 'MEG1243', 'MEG2332']},
  "ece24":  {"session1": ['MEG1042', 'MEG1243', 'MEG2332'], "session2":['MEG1042', 'MEG1243', 'MEG2332'], "session3": ['MEG1042', 'MEG1243', 'MEG2332']},
  "awa19":  {"session1": ['MEG1042', 'MEG1243', 'MEG2332'], "session2":['MEG1042', 'MEG1243', 'MEG2332'], "session3": ['MEG1042', 'MEG1243', 'MEG2332']},
  "hay06":  {"session1": ['MEG1042', 'MEG1243', 'MEG2332'], "session2":['MEG1042', 'MEG1243', 'MEG2332'], "session3": ['MEG1042', 'MEG1243', 'MEG2332']},
  "mca10":  {"session1": ['MEG1042', 'MEG1243', 'MEG2332'], "session2":['MEG1042', 'MEG1243', 'MEG2332']},
  "hky23":  {"session1": ['MEG1042', 'MEG1243', 'MEG2332'], "session2":['MEG1042', 'MEG1243', 'MEG2332']},
  }

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
# import library

 
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

            
            
#%% Experiment notes
'''
bsr27, block07 and block09 saved in the same file, alarm went off half through block08. That part of the data is removed in this code 
uka11, session 3 blocks names are off by one
dtl05, session 3 blocks names are off by one
fha01, first session cancelled for technical reasons. Task stats in session 2
'''




#%% set-up paths
# Paths used in local computer
# data_dir =  r'Z:'
# root_dir =  r'Y:'
# output_plot_folder = r'Y:\PLOTS\QUALITY_CHECK\psd' 
# output_dir =  r'Y:\ANALYSIS'

# # Use the following if in deepnet
data_dir =  r'/raw/Project0349'
root_dir =  r'/analyse/Project0349/' 
output_plot_folder = r'/analyse/Project0349/PLOTS/QUALITY_CHECK/psd' 
output_dir =  r'/analyse/Project0349/ANALYSIS'

subjects_list = [ f.name for f in os.scandir(data_dir) if f.is_dir() and f.name[0] != '_' and f.name[0].islower()]
print(subjects_list)


 
#%%

#filter seetings
l_freq = 0 # no low filter
h_freq = 270 # below cHPI 


# loop through participants
for subject_id in subjects_list:
    # subject_id = 'jmn22'
    print(subject_id)
    if subject_id not in bad_meg_channels.keys():
        print('Data for this subject has not been checked yet')
        continue

    subject_nb = folder_mapping[subject_id]
    #subject_id =subjects_list[sub_idx]
    print('Doing subject ' + subject_id + '\nSessions')
    subject_path = op.join(data_dir,subject_id)
    session_folders = os.listdir(subject_path)
    sessions =  [x for x in session_folders if x.startswith('2')]
    sessions.sort()
    print(sessions)

    if subject_id == 'fha01':
        sessions = sessions[1:]
        
    for idx,session in enumerate(sessions):
       
        # idx = 0, session = sessions[idx]
        print(session)
        save_plot_raw = op.join(output_plot_folder,'raw',subject_id,session) # set path for output files
        save_plot_tsss = op.join(output_plot_folder,'tsss',subject_id,session)
        save_dir = op.join(output_dir,folder_mapping[subject_id] +'_' + subject_id,'tsss',session) # data will be saved as subject code
        # create folder if it doesn't exist
        if not os.path.exists(save_plot_raw):
            os.makedirs(save_plot_raw)
        if not os.path.exists(save_plot_tsss):
            os.makedirs(save_plot_tsss)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            print('tsss has been done for ' + subject_id + ' session ' + session)
            continue
        
        file_dir = op.join(subject_path,session)
        blocks =  os.listdir(file_dir)
        blocks.sort()
        save_block, blocks = check_block_name(blocks) # make sure all block names conform to conventions

        
        pos_files_path = os.path.join(output_dir,folder_mapping[subject_id] +'_' + subject_id,'pos',session)
        HPI_pos = []
        cHPI_on = []
        print('loading head position files')
       
        for nb,block in enumerate(save_block): # here using save_blocks instead of blocks, becuase pos files already have correct block file name
            print(block)
            if subject_id == 'dtl05' and idx ==2  or  subject_id == 'uka11' and idx ==2 :
                print('Fixing block name for ' + subject_id + ' ' + block)
                num = int(re.findall(r'\d+', block)[0]) + 1
                block = 'Block_' + "%02d" % (num,) + '.fif'
                save_block[nb] = block
                
            
            chpi_file = [f for f in os.listdir(pos_files_path) if (block[:-4] in f)]
            
            # load cHPI file
            chpi_pos = mne.chpi.read_head_pos(os.path.join(pos_files_path,chpi_file[0]))
            HPI_pos.append(chpi_pos)
            
            if chpi_file[0].find('dummy') > 0:
                cHPI_on.append(False)
            else:
                cHPI_on.append(True)
                
       
        for nb, block in enumerate(blocks):
            print(block)
            # nb =0
            # block = blocks[nb]
            save_block_name = save_block[nb]
            #block = 'BLock_27.fif'
            print(block)  
            #-----------------------------------------------------------------------------
            # Load file
            #-----------------------------------------------------------------------------
            
            #file_name = blocks[0] # loop through blocks
            raw_file = op.join(file_dir,block)
            raw = mne.io.read_raw_fif(raw_file,preload = True)
            raw.load_data()
            #raw.plot_psd(tmin=10)
            # fix block name for dtl05 and uka11
            if subject_id == 'dtl05' and idx ==2  or  subject_id == 'uka11' and idx ==2 :
                print('Fixing block name for ' + subject_id)
                num = int(re.findall(r'\d+', block)[0]) + 1
                block = 'Block_' + "%02d" % (num,) + '.fif'
                
            save_file= op.join(save_dir,save_block[nb])
            
            # crop end of file fpr bsr27
            if subject_id == 'bsr27' and save_block[nb] == 'Block_07.fif':
                print('cutting final bad segment')
                raw.crop(tmax=656) #<- remove 23 first seconds  # example 10 seconds form start and end raw.crop(tmin=10, tmax=raw_filt.times[-1] - 10) 
                #raw.plot()
            
            # copy EEG points and turn them into Extra points so no trick is necessary:
            digpts = raw.info['dig']
            
            # first, identify how many extra points there are (it's listed in 'info', but I cannot extract the number from there)
            extracount = []
            [extracount.append(1) for p in digpts if p['kind'] == 4] # 'kind' value for extra pts is 4
            num_extra = sum(extracount)
            
            for pt in digpts:
                if pt['kind'] == 3: # 'kind' value for EEG pts is 3
                    raw.info['dig'].append(mne.io._digitization.DigPoint(
                        {'kind':  4, # change kind value to extra, leave everything else the same
                         'r':     pt['r'],
                         'ident': num_extra+1, # the order number of this new, appended extra point will be the highest previous extra point number plus 1
                         'coord_frame': pt['coord_frame']}))
                    num_extra += 1
            
            
            raw.info['bads'] = []
            raw.info['bads'] = bad_meg_channels[subject_id]['session'+str(idx+1)]
            
            destination = tuple(np.mean(np.concatenate(HPI_pos, axis=0)[:,4:7],axis=0))
            actual_destination = raw.info['dev_head_t']['trans'][:3,3]
            
            if not cHPI_on[nb]:
                print('cHPI was off')
                chpi_pos = None
                outfile = save_file[:-4] + '_tsss.fif'
            else:
                print('cHPI was on! Maxfilter will compensate for head movements')
                chpi_pos = HPI_pos[nb]
                outfile = save_file[:-4] +'_tsss_mc.fif'

          

            fig = raw.plot_psd(tmin=10,fmin=3,fmax=500,xscale='log')
            filename =  op.join(save_plot_raw,'PSD_' + save_block[nb][:-3] + 'png')
            fig.savefig(filename)    
            plt.close()
                
            raw_tsss = mne.preprocessing.maxwell_filter(raw, st_duration=16, st_correlation=.9,
                                                       head_pos=chpi_pos,coord_frame='head',
                                                       destination = destination,
                                                       calibration = os.path.join(root_dir,'CALIBRATION_FILES','sss_cal.dat'),
                                                       cross_talk = os.path.join(root_dir,'CALIBRATION_FILES','ct_sparse.fif'))
            
            # apply notch, after maxfilter!
            raw.filter(l_freq = l_freq,h_freq=h_freq,h_trans_bandwidth=2)             
            raw_tsss.notch_filter(np.arange(50, 251, 50))

            # print PSD for sanity check
            fig = raw_tsss.plot_psd(tmin=10,fmin=3,fmax=147,xscale='log')
            filename =  op.join(save_plot_tsss,'PSD_tsss_' + save_block[nb][:-3] + 'png')
            fig.savefig(filename)    
            plt.close()
            
            # check if the data has a peak close to 60 Hz
            data1, freqs, = mne.time_frequency.psd_array_welch(raw_tsss.get_data(picks='meg'),sfreq=raw.info['sfreq'],n_fft=4000,fmin=20,fmax=70, n_per_seg = 1000)#, n_per_seg = 100)
            
            fi = np.argmin(np.abs(freqs-58))
            fii = np.argmin(np.abs(freqs-60))
            
            peak = []
            chan = []
            fracs = []

            for i in range(306):
                for tp in np.arange(fi,fii,1):
                    frac = data1[i,tp] / (data1[i,tp+1] + data1[i,tp-1])
                    if frac > 0.7 : 
                        print('channel ' + str(i))
                        print(freqs[tp])
                        print(frac)
                        peak.append(freqs[tp])
                        chan.append(i)
                        fracs.append(frac)
                        
            if not peak:
                raw_tsss.save(outfile, overwrite=True)
            else:
                print('Removing 60Hz peaks')
                ch , ch_idx  = np.unique(chan,return_index=True)
                peak = [peak[j] for j in ch_idx] 
                chan =  [chan[j] for j in ch_idx]
                fracs =  [fracs[j] for j in ch_idx]
                            
                f , count = np.unique(peak,return_counts=True)
               
                
                #==============================================================
                # plot peaks to be removed
                mask = np.ones(len(data1), dtype=bool)
                mask[chan] = False
                plt.plot(freqs,data1[mask,:].transpose(),alpha=0.1,linewidth=1)
                plt.plot(freqs,data1[chan,:].transpose(),linewidth=1)
                plt.xlabel('freqs (Hz)')
                for p in f:
                    plt.axvline(x = p, color = 'k', linestyle = '--',alpha=0.5,linewidth=1)
                #plt.plot(freqs,data1.transpose(),alpha=0.1,linewidth=1)
                #plt.show()
                filename =  op.join(save_plot_tsss,'PSD_tsss_' + save_block[nb][:-4] + '_60Hz_to_remove.png')
                plt.savefig(filename)
                plt.close()
                #==============================================================
                
                # apply notch only in channels with noise    
                chan_idx = mne.pick_channels_regexp(raw.info['ch_names'], 'MEG *')
                ch_names = raw.info['ch_names']
                ch_names = [ch_names[j] for j in chan_idx]
                ch_names = [ch_names[j] for j in chan]
                
                if len(f)==1: # asusmes there is only one peak to remove
                    raw_tsss.notch_filter(f[0],picks = ch_names)
                    raw_tsss.save(outfile, overwrite=True)
                    raw_tsss.plot_psd(tmin=10,fmin=3,fmax=147,xscale='log')
                    filename =  op.join(save_plot_tsss,'PSD_tsss_' + save_block[nb][:-4] + '_60Hz_removed.png')
                    fig.savefig(filename)    
                    plt.close()

#%%

