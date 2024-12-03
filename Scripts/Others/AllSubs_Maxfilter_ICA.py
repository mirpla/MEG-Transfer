# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 00:03:36 2022
@author: Christopher Postzich

Load up raw Data, MaxFilter and ICA projections
"""

# Import Libraries 
import os
import mne
import autoreject
import numpy as np
import pandas as pd
import struct
import gc
from mne.preprocessing import ICA, read_ica, find_bad_channels_maxwell


# Directories
raw_files = os.path.join('//raw','Project0320')
analyse_files = os.path.join('//analyse4','Project0320','MNE_MEG_Analysis') if os.name == 'nt' else os.path.join('//analyse','Project0320','MNE_MEG_Analysis')

# Headposition Import Function
def impAddHeadpos(Data, Path2AddHeadposFiles):
    with open(Path2AddHeadposFiles) as f:
        lines = f.readlines()
    for line in lines:
        if(line.replace('\n','').split(',')[0] == '4'): 
            Data.info['dig'].append(mne.io._digitization.DigPoint(
                {'kind':  int(line.replace('\n','').split(',')[0]),
                 'r':     np.array(line.replace('\n','').split(',')[2:], dtype='single'),
                 'ident': int(line.replace('\n','').split(',')[1]),
                 'coord_frame': 4}
            ))
    return Data

# Read Addtional Headpositions
def getAddHeadpos(Data):
    ct = 1
    fid, _, directory = mne.io.fiff_open(Data._filenames[0].replace('\\','/'))
    listofaddHP = [i for i in directory if i.__dict__['kind'] in (213,234)]
    for hp in listofaddHP:
        if(hp.__dict__['kind'] != 213):
            fid.seek(hp.__dict__['pos'] + 16,0)
            (kind,ident,size) = struct.unpack('>i i i',fid.read(12))
            for cd in range(size):
                Data.info['dig'].append(mne.io._digitization.DigPoint(
                    {'kind':  kind,
                     'r':     np.array(struct.unpack('>f f f',fid.read(12)), dtype='single'),
                     'ident': ct,
                     'coord_frame': 4}
                ))
                ct += 1
    return Data


# Import Subject Info
Subject_Info = pd.read_csv(os.path.join(analyse_files,'MEG_Data','Subject_Informations','Subject_Information.csv'), delimiter=';')

# Create Subject Names List
Sub_Names = Subject_Info[Subject_Info.ExcludeSub == False].Participant.to_list()



###############################################################################
# Maxfilter Data


# Loop over Subject Names
for i,sub in enumerate(Sub_Names):
    
    # Retrieve HashCode and Date
    datapaths = [rootdir for rootdir, _, files in os.walk(raw_files) if(files) if(sub in files[0])]
    
    # Get HPIs
    HPI_Names = os.listdir(os.path.join(analyse_files,'MEG_Data','Preprocessing_Data','HeadPos_Files',sub))
    HPI_pos = {}
    for hpf in HPI_Names:
        # Load up Head Position File per Block
        HPI_pos[hpf[8:]] = mne.chpi.read_head_pos(os.path.join(analyse_files,'MEG_Data','Preprocessing_Data','HeadPos_Files',sub,hpf))
    
    for b in range(1,9):
        
        block = f'B0{b}'
        
        # Get Path and Load up Data
        sample_data_raw_file = os.path.join(datapaths[0], f'{sub}_{block}.fif')
        Block_raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False)
        
        # Get Additional Headposition Locations for incompatible fif Files
        existExtra = next((i for i in Block_raw.info['dig'] if i['kind']._name == 'FIFFV_POINT_EXTRA'), None)
        if(not existExtra):
            Block_raw = getAddHeadpos(Block_raw)
        
        # Automatic Bad Channel Detection if Crosstalk and Finetuning is available
        Block_raw.info['bads'] = []
        Block_raw_check = Block_raw.copy()
        auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
            Block_raw_check, return_scores=True, verbose=True)
        print(auto_noisy_chs)  # we should find them!
        print(auto_flat_chs)  # none for this dataset
        if len(auto_noisy_chs) | len(auto_flat_chs):
            pass #break
        
        bads = Block_raw.info['bads'] + auto_noisy_chs + auto_flat_chs
        Block_raw.info['bads'] = bads
    
            
        # Maxfilter Data
        Block_raw_sss = mne.preprocessing.maxwell_filter(
            Block_raw, st_duration=100, st_correlation=.9, 
            coord_frame='head', destination=tuple(np.mean(np.concatenate([HPI_pos[k] for k in HPI_pos], axis=0)[:,4:7],axis=0)), 
            head_pos=HPI_pos[block], verbose=True)
        
        
        if b == 1 and Subject_Info[Subject_Info.Participant == sub].FirstBlockSplit.item() == 'yes':
            
            block0 = 'B00'
            
            # Get Path and Load up Data
            sample_data_raw_file = os.path.join(datapaths[0], f'{sub}_{block0}.fif')
            Block_raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False)
            
            # Get Additional Headposition Locations for incompatible fif Files
            existExtra = next((i for i in Block_raw.info['dig'] if i['kind']._name == 'FIFFV_POINT_EXTRA'), None)
            if(not existExtra):
                Block_raw = getAddHeadpos(Block_raw)
            
            # Automatic Bad Channel Detection if Crosstalk and Finetuning is available
            Block_raw.info['bads'] = []
            Block_raw_check = Block_raw.copy()
            auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
                Block_raw_check, return_scores=True, verbose=True)
            print(auto_noisy_chs)  # we should find them!
            print(auto_flat_chs)  # none for this dataset
            if len(auto_noisy_chs) | len(auto_flat_chs):
                pass #break
            
            bads = Block_raw.info['bads'] + auto_noisy_chs + auto_flat_chs
            Block_raw.info['bads'] = bads
        
                
            # Maxfilter Data
            Block_raw_sss0 = mne.preprocessing.maxwell_filter(
                Block_raw, st_duration=100, st_correlation=.9, 
                coord_frame='head', destination=tuple(np.mean(np.concatenate([HPI_pos[k] for k in HPI_pos], axis=0)[:,4:7],axis=0)), 
                head_pos=HPI_pos[block0], verbose=True)
            
            # Concatenate fif files
            mne.concatenate_raws([Block_raw_sss0, Block_raw_sss])
            Block_raw_sss0._filenames.pop(0)
            Block_raw_sss = Block_raw_sss0
        
        
        # save file
        try:
            os.mkdir(os.path.join(analyse_files,'raw_sss_files',sub))
        except:
            print(f"Directory {os.path.join('raw_sss_files',sub)} already exists!")  
        Block_raw_sss.save(os.path.join(analyse_files,'MEG_Data','Preprocessing_Data','raw_sss_files', f'{sub}',f'{block}_raw_sss.fif'))
        
        del Block_raw
        del Block_raw_sss


###############################################################################
# Compute ICA

# Import Random States
random_state = pd.read_csv(os.path.join(analyse_files,'MEG_Data','Subject_Informations','RandState_Subject.csv'))

for i,sub in enumerate(Sub_Names):
    
    # Retrieve HashCode and Date
    datapaths = [rootdir for rootdir, _, files in os.walk(raw_files) if(files) if(sub in files[0])]
    
    
    event_1_dict = {'StartB_PPix': 1, 'StartB_PPort': 8192, 
                    'Enc_Trial': 1024, 'Enc_Cue': 1026, 'Enc_Img': 1028, 
                    'Dist_Trial': 2048, 'Dist_Num': 2056,
                    'Ret_Trial1': 4096, 'Ret_Trial2': 3840, 
                    'Ret_Cue1': 4112, 'Ret_Cue2': 16, 'Ret_Cue3': 3856,
                    'Ret_Rect1': 4128, 'Ret_Rect2': 32, 'Ret_Rect3': 3872, 
                    'Ret_Catch1': 4160, 'Ret_Catch2': 64, 'Ret_Catch3': 3904}
    event_1_revdict = dict(map(reversed, event_1_dict.items()))

    event_2_dict = {'ButtonPress_Left': 4095, 'ButtonPress_Right': 511,
                    'RightPinky': 63, 'RightRing': 127, 'RightMiddle': 255,
                    'RightThumb': 1023, 'LeftThumb': 2047, 'LeftMiddle': 8191,
                    'LeftRing': 16383, 'LeftPinky': 32767}
    event_2_revdict = dict(map(reversed, event_2_dict.items()))

    Block_raw_sss_list = [[],[],[],[],[],[],[],[]]
    events_list = [[],[],[],[],[],[],[],[]]
    for b in range(1,9):
        # Load up Block
        BlockNo = f'B0{b}'
        # Load up Data
        Block_raw_sss_list[b-1] = mne.io.read_raw_fif(os.path.join(analyse_files,'raw_sss_files', f'{sub}',f'{BlockNo}_raw_sss.fif'),preload=True)
        
        # Create Events List out of STI101 (ParPort and PPixx)
        events_STI101 = mne.find_events(Block_raw_sss_list[b-1], stim_channel=['STI101'], min_duration=0.0011)
        # Kick out random triggers
        random_triggers = list(set(events_STI101[:,2]) - set(event_1_revdict.keys()))
        events_STI101 = np.delete(events_STI101, [e in random_triggers for e in events_STI101[:,2]], axis=0)
        
        # Create Event List out of STI102 (Responses)
        events_STI102 = mne.find_events(Block_raw_sss_list[b-1], stim_channel=['STI102'], min_duration=0.0011)
        
        # Get Difference of 
        events_STI102[:,2] = events_STI102[:,2] - events_STI102[:,1]
        
        # Redefine Values for responses to 511 and 4095 
        events_STI102[:,2] = events_STI102[:,2] - 1
        
        # Kick out random triggers
        random_triggers = list(set(events_STI102[:,2]) - set(event_2_revdict.keys()))
        events_STI102 = np.delete(events_STI102, [e in random_triggers for e in events_STI102[:,2]], axis=0)
        
        # Merge and Sort Event Lists
        events = np.concatenate((events_STI101, events_STI102), axis=0)
        events_list[b-1] = events[events[:,0].argsort()]

    # Concatenate Raw files
    #Block_raw_sss = Block_raw_sss_list[0]
    Block_raw_sss, events = mne.concatenate_raws(Block_raw_sss_list, events_list=events_list)
    del Block_raw_sss_list
    gc.collect(generation=2)

    # Calculate timings of Interblock periods
    Interblock_periods = np.zeros([9,3], dtype=int)
    cur_ind = 0
    next_start = 0
    for ind,x in enumerate(events_list):
        Interblock_periods[ind,0] = next_start
        Interblock_periods[ind,1] = events[cur_ind+np.min(np.concatenate([np.where(events[cur_ind+0:cur_ind+x.shape[0],2] == 511)[0],np.where(events[cur_ind+0:cur_ind+x.shape[0],2] == 4095)[0]])),0]
        Interblock_periods[ind,2] = 3
        next_start = events[np.max(np.concatenate([np.where(events[:cur_ind+x.shape[0],2] == 511)[0], np.where(events[:cur_ind+x.shape[0],2] == 4095)[0]])),0] + 4000
        cur_ind += x.shape[0]
    Interblock_periods[8,0] = events[-1,0] + 4000
    Interblock_periods[8,1] = events[-1,0] + 4000 + 120000
    Interblock_periods[8,2] = 3
    Interblock_periods = np.insert(Interblock_periods, 1, [events_list[0][19+np.min(np.where(events_list[0][20:,2] == 1)[0]),0] + 4000, events_list[0][1+np.max(np.where(events_list[0][:-20,2] == 8192)[0]),0], 3], axis=0)
        

    # Annotate Bad Interblock Periods
    block_annots = mne.Annotations(onset=Interblock_periods[:,0]/1000,
                                   duration=np.ravel(np.diff(Interblock_periods[:,0:2], axis=1))/1000,
                                   description='BAD',
                                   orig_time=Block_raw_sss.info['meas_date'])
    Block_raw_sss.set_annotations(Block_raw_sss.annotations + block_annots) 
    
    # Filter Data with Bandpass between 1 and 40 Hz for ICA
    Block_raw_sss.pick_types(meg=True, eeg=False, stim=False, misc=True).filter(1,40)
    
    # Split whole data into 3 second intervals
    Block_raw_sss_epochs = mne.make_fixed_length_epochs(Block_raw_sss, duration = 3.)
    
    # Estimate global rejection thresholds
    reject = autoreject.get_rejection_threshold(Block_raw_sss_epochs, decim=10)
    
    # Fit ICA
    ica = ICA(n_components=.98, max_iter='auto', method='picard', 
              random_state=int(random_state.loc[random_state['subject'] == sub, 'random_state']), verbose=True)
    ica.fit(Block_raw_sss, reject=reject, decim=10, verbose=True)
    
    del Block_raw_sss
    gc.collect(generation=2)
    
    # Save ICA file
    try:
        os.mkdir(os.path.join(analyse_files,'ica_projections',sub))
    except:
        print(f"Directory {os.path.join('ica_projections',sub)} already exists!")
    ica.save(os.path.join(analyse_files,'ica_projections', f'{sub}','Proj_ica.fif'))


###############################################################################
# Apply ICA

# Import chosen ICA components
ICA_components = pd.read_csv(os.path.join(analyse_files,'Subject_Informations','ICA_components_perSub.csv'), sep=';')

for i,sub in enumerate(Sub_Names[2:]):
    
    # Retrieve HashCode and Date
    datapaths = [rootdir for rootdir, _, files in os.walk(raw_files) if(files) if(sub in files[0])]
    
    
    event_1_dict = {'StartB_PPix': 1, 'StartB_PPort': 8192, 
                    'Enc_Trial': 1024, 'Enc_Cue': 1026, 'Enc_Img': 1028, 
                    'Dist_Trial': 2048, 'Dist_Num': 2056,
                    'Ret_Trial1': 4096, 'Ret_Trial2': 3840, 
                    'Ret_Cue1': 4112, 'Ret_Cue2': 16, 'Ret_Cue3': 3856,
                    'Ret_Rect1': 4128, 'Ret_Rect2': 32, 'Ret_Rect3': 3872, 
                    'Ret_Catch1': 4160, 'Ret_Catch2': 64, 'Ret_Catch3': 3904}
    event_1_revdict = dict(map(reversed, event_1_dict.items()))

    event_2_dict = {'ButtonPress_Left': 4095, 'ButtonPress_Right': 511,
                    'RightPinky': 63, 'RightRing': 127, 'RightMiddle': 255,
                    'RightThumb': 1023, 'LeftThumb': 2047, 'LeftMiddle': 8191,
                    'LeftRing': 16383, 'LeftPinky': 32767}
    event_2_revdict = dict(map(reversed, event_2_dict.items()))

    Block_raw_sss_list = [[],[],[],[],[],[],[],[]]
    events_list = [[],[],[],[],[],[],[],[]]
    for b in range(1,9):
        # Load up Block
        BlockNo = f'B0{b}'
        # Load up Data
        Block_raw_sss_list[b-1] = mne.io.read_raw_fif(os.path.join(analyse_files,'raw_sss_files', f'{sub}',f'{BlockNo}_raw_sss.fif'),preload=True)
        
        # Delete Edge Boundary between Training and First Block
        for ind, an in enumerate(Block_raw_sss_list[b-1].annotations):
            if(an):
                if(an['description'] == 'EDGE boundary'):
                    Block_raw_sss_list[b-1].annotations.delete(ind)
        
        # Create Events List out of STI101 (ParPort and PPixx)
        events_STI101 = mne.find_events(Block_raw_sss_list[b-1], stim_channel=['STI101'], min_duration=0.0011)
        # Kick out random triggers
        random_triggers = list(set(events_STI101[:,2]) - set(event_1_revdict.keys()))
        events_STI101 = np.delete(events_STI101, [e in random_triggers for e in events_STI101[:,2]], axis=0)
        
        # Create Event List out of STI102 (Responses)
        events_STI102 = mne.find_events(Block_raw_sss_list[b-1], stim_channel=['STI102'], min_duration=0.0011)
        
        # Get Difference of 
        events_STI102[:,2] = events_STI102[:,2] - events_STI102[:,1]
        
        # Redefine Values for responses to 511 and 4095 
        events_STI102[:,2] = events_STI102[:,2] - 1
        
        # Kick out random triggers
        random_triggers = list(set(events_STI102[:,2]) - set(event_2_revdict.keys()))
        events_STI102 = np.delete(events_STI102, [e in random_triggers for e in events_STI102[:,2]], axis=0)
        
        # Merge and Sort Event Lists
        events = np.concatenate((events_STI101, events_STI102), axis=0)
        events_list[b-1] = events[events[:,0].argsort()]

    # Concatenate Raw files
    Block_raw_sss, events = mne.concatenate_raws(Block_raw_sss_list, events_list=events_list)
    del Block_raw_sss_list
    gc.collect(generation=2)

    # Calculate timings of Interblock periods
    Interblock_periods = np.zeros([9,3], dtype=int)
    cur_ind = 0
    next_start = 0
    for ind,x in enumerate(events_list):
        Interblock_periods[ind,0] = next_start
        Interblock_periods[ind,1] = events[cur_ind+np.min(np.concatenate([np.where(events[cur_ind+0:cur_ind+x.shape[0],2] == 511)[0],np.where(events[cur_ind+0:cur_ind+x.shape[0],2] == 4095)[0]])),0]
        Interblock_periods[ind,2] = 3
        next_start = events[np.max(np.concatenate([np.where(events[:cur_ind+x.shape[0],2] == 511)[0], np.where(events[:cur_ind+x.shape[0],2] == 4095)[0]])),0] + 4000
        cur_ind += x.shape[0]
    Interblock_periods[8,0] = events[-1,0] + 4000
    Interblock_periods[8,1] = events[-1,0] + 4000 + 120000
    Interblock_periods[8,2] = 3
    Interblock_periods = np.insert(Interblock_periods, 1, [events_list[0][19+np.min(np.where(events_list[0][20:,2] == 1)[0]),0] + 4000, events_list[0][1+np.max(np.where(events_list[0][:-20,2] == 8192)[0]),0], 3], axis=0)
        

    # Annotate Bad Interblock Periods
    block_annots = mne.Annotations(onset=Interblock_periods[:,0]/1000,
                                   duration=np.ravel(np.diff(Interblock_periods[:,0:2], axis=1))/1000,
                                   description='BAD',
                                   orig_time=Block_raw_sss.info['meas_date'])
    Block_raw_sss.set_annotations(Block_raw_sss.annotations + block_annots)
    
    # Import ICA fit
    ica = read_ica(os.path.join(analyse_files,'ica_projections', f'{sub}','Proj_ica.fif'))
    
    # Components to exclude
    # Eye Component
    eye_comp = ICA_components.loc[ICA_components["Participant"] == sub, "Eye Component"].str.replace('[()[\]{}]','',regex=True).str.split(',').item()
    # Heart Component
    hrt_comp = ICA_components.loc[ICA_components["Participant"] == sub, "Heart Component"].str.replace('[()[\]{}]','',regex=True).str.split(',').item()
    # Write into exclude
    ica.exclude = [int(ec) for ec in eye_comp] + [int(hc) for hc in hrt_comp]
    
    # Apply ICA Correction to the Data
    ica.apply(Block_raw_sss, verbose=True)

    # Save ICA corrected Data
    for b in range(1,9):
        # Load up Block
        BlockNo = f'B0{b}'
        if b == 1:
            start_of_block = 0
        if b == 8:
            end_of_block = None
        else:
            end_of_block = Block_raw_sss.annotations[np.char.equal('EDGE boundary', Block_raw_sss.annotations.description)][(b-1)]['onset'] - Block_raw_sss.annotations[0]['onset']
        # Save Block Data
        try:
            os.mkdir(os.path.join(analyse_files,'ica_raw_sss_files',sub))
        except:
            print(f"Directory {os.path.join('ica_raw_sss_files',sub)} already exists!")
        Block_raw_sss.save(os.path.join(analyse_files,'ica_raw_sss_files', f'{sub}',f'{BlockNo}_ica_raw_sss.fif'),
                           tmin=start_of_block, tmax=end_of_block)
        if b < 8:
            start_of_block = Block_raw_sss.annotations[np.char.equal('EDGE boundary', Block_raw_sss.annotations.description)][(b-1)]['onset'] - Block_raw_sss.annotations[0]['onset']

    del Block_raw_sss, ica
    gc.collect(generation=2)
    

