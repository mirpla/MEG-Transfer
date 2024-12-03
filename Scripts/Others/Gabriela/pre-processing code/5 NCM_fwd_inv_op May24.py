# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 14:52:51 2023

@author: gabriela
"""

import mne
import os
from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'qt')
import numpy as np

from preprocessor.sourcemodeling import Sourcemodels


#%%
# data_dir =   r'Y:\ANALYSIS'
# mri_dir = r'Y:\MRI'
# pts_dirs = [r'S:\ANALYSIS',r'R:\ANALYSIS',r'Q:\ANALYSIS',r'O:\ANALYSIS',r'N:\ANALYSIS',r'M:\ANALYSIS']


data_dir =   r'/analyse/Project0349/ANALYSIS'
mri_dir = r'/analyse/Project0349/MRI'
pts_dirs = [r'/analyse/Project0372/ANALYSIS',r'/analyse/Project0375/ANALYSIS',r'/analyse/Project0376/ANALYSIS',r'/analyse/Project0377/ANALYSIS',r'/analyse/Project0378/ANALYSIS',r'/analyse/Project0379/ANALYSIS']


output_dir = 'pts'

# data_dir =   r'/analyse/Project0349/ANALYSIS'
# output_plot = r'/analyse/Project0349/PLOTS/ica' 
# output_plot_psd = r'/analyse/Project0349/PLOTS/QUALITY_CHECK/psd/ica'
# output_plot_traces = r'/analyse/Project0349/PLOTS/QUALITY_CHECK/traces/ica'

#subjects_list = ['S005_ami28' ]
subjects_list = [ f.name for f in os.scandir(data_dir) if f.is_dir() and f.name[0] == 'S' ]
#subjects_list =  ['S020_rbe04', 'S021_jmn22']
subjects_list.sort()
print(subjects_list)
#subjects_list.sort(reverse=True) 

overwrite = False

#%%

n_parc = 200
parc = 'parc2018yeo17_200'

fwdproblems = []
invproblems = []
fidproblems = []
corproblems = []

for subject_id in subjects_list:
    # subject_id = 'S011_rsh17'
    print('Doing subject ' + subject_id + '\nSessions')
    subject_path = os.path.join(data_dir,subject_id)
    sub = subject_id.split("_")[1]
    ica_path =  os.path.join(subject_path,'ica')
    session_folders = os.listdir(ica_path)
    sessions =  [x for x in session_folders if x.startswith('2')]
    sessions.sort()
    print(sessions)
    subj_mridir = os.path.join(mri_dir,sub)
    if not os.path.exists(subj_mridir):
        print('Skipping ' + subject_id + ' because they have no MRI (yet).')
        continue


    # make a directory to save pts
    check_other_dirs = True
    idx = 0
    while check_other_dirs:
        
        use_dir = pts_dirs[idx] 
         
        if not os.path.exists(use_dir):
            os.makedirs(use_dir)
                
        if subject_id in os.listdir(use_dir):
            check_other_dirs = False
            pts_dir =  os.path.join(use_dir,subject_id, output_dir)
        
        if len(os.listdir(use_dir)) < 12:
            pts_dir =  os.path.join(use_dir,subject_id, output_dir)
            check_other_dirs = False
        elif len(os.listdir(use_dir)) >= 12:
                pts_dir =  os.path.join(use_dir,subject_id, output_dir)
                if os.path.exists(pts_dir):
                    check_other_dirs = False
                else:
                    idx += 1

    if not os.path.exists(pts_dir):
        os.makedirs(pts_dir)
    #print(pts_dir)

    bem_folder = os.path.join(data_dir,subject_id,'bem')
    bemfile = os.path.join(bem_folder,sub + '_bem.h5')
    srcfile = os.path.join(bem_folder,sub + '-src.fif')
    
    if not os.path.isfile(bemfile):
        print('Skipping ' + subject_id + ' because they have no BEM file.')
        continue
    
    # bem = mne.read_bem_solution(bemfile)
    # src = mne.read_source_spaces(srcfile)
    
    for session in sessions:
        # idx = 0
        # session = sessions[idx]
        # print(session)

        cor_folder = os.path.join(subject_path,'cor')
        corfile = os.path.join(cor_folder,sub + '_' + session)
        if os.path.exists(corfile):
            cor = mne.read_trans(corfile)
        else:
            corproblems.append('cor file does not exist for ' + subject_id + ' session ' + session)
            continue
            
        file_dir = os.path.join(ica_path,session)
        blocks =  os.listdir(file_dir)
        blocks = [x for x in blocks if x.startswith('B')]
        blocks = [x for x in blocks if 'tsss' in x]
        blocks.sort()
        #print(blocks)
        
        for block in blocks:
            # block = blocks[0]
            # for noise covar matrix
            # make directory for NCM files. This I'm doing block level
            ncm_dir = os.path.join(data_dir, subject_id, 'ncm')
            
            #pts_dir = os.path.join(pts_dir, subject_id, 'pts')
            if not overwrite and os.path.exists(os.path.join(pts_dir, session + '_B' + block.split("_")[1] + '_parc2018yeo17_200_1000_weighted.npy')):
                continue
            
            print('Doing ' + block)
            if 'bem' not in locals():
                bem = mne.read_bem_solution(bemfile)
                src = mne.read_source_spaces(srcfile)
            
            
            fname = os.path.join(ica_path,session,block)
            clnd_orig = mne.io.read_raw_fif(fname,preload = True)
            
            if len(clnd_orig.info['bads']) == 64: # in few datasets eeg data is missing
                domeg = True
                doeeg = False
                set_name = session + '_meg'
                clnd_orig.pick('meg')
            else:
                domeg = True
                doeeg = True
                set_name = session
                
            
            sourcemodels = Sourcemodels(clnd_orig, set_name, sub, mri_dir, corfile, bem, src) #subjects_dir = 'Y:\MRI' # MRI directory, NOT MEG directory
            
            # fwd done at session level!!!            
            #=================================================================
            fwd_dir = os.path.join(data_dir, subject_id, 'fwd') # make directory for fwd files
            if not os.path.exists(fwd_dir):
                os.makedirs(fwd_dir)
            
            sourcemodels.cor = cor # for me it didn't work if .cor was a path and not a file
            if not os.path.exists(os.path.join(fwd_dir,set_name + '_fwd.fif')): # Done at session level!
                try:
                    sourcemodels.make_fwd(fwd_dir, spacing=5, eeg=doeeg, meg=domeg) # make forward solution and save
                except Exception as ex:
                    fwdproblems.append(subject_id + '_' + set_name + '_' + block + '_' + repr(ex))
            else:
                fwd = mne.read_forward_solution(os.path.join(fwd_dir,set_name + '_fwd.fif')) # Load fwd if has been computed already 
                sourcemodels.fwd = fwd 
            

            #=================================================================
            if not os.path.exists(ncm_dir):
                os.makedirs(ncm_dir)
            
            # load events. From propixx events for NCM
            events_dir = os.path.join(data_dir,subject_id,'triggers',session,'PROPixxEvents_Block_' + block.split("_")[1] + '.npy')
            
            if os.path.exists(events_dir):
                events = np.load(events_dir)
            else:
                continue
            sourcemodels.events = events
            sourcemodels.set_name =  session + '_B' + block.split("_")[1]
            # compute ncm around some specified event(s) and save to disk
            
            event_id = {'attention cue': 4}
            tmin = -0.4
            tmax = -0.1


            sourcemodels.make_cov_epochs(ncm_dir, event_id, tmin, tmax)

            
            inv_dir = os.path.join(data_dir, subject_id, 'inv')
            if not os.path.exists(inv_dir):
                os.makedirs(inv_dir)
            
            # inv done at block level!!!            
            #=================================================================
            try:
                sourcemodels.make_inv(inv_dir)
            except Exception as ex:
                invproblems.append(subject_id + '_' + set_name + '_' + block + '_' + repr(ex) )


            if not os.path.exists(pts_dir):
                os.makedirs(pts_dir)
            
            fid_dir =  os.path.join(data_dir, subject_id, 'fid')
            if not os.path.exists(fid_dir):
                os.makedirs(fid_dir)
            
            try:
                sourcemodels.make_fidelity_weighting(pts_dir, fid_dir, parc, eeg=doeeg, meg=domeg)
            except Exception as ex:
                fidproblems.append(subject_id + '_' + set_name + '_' + block + '_' + repr(ex))

        if 'bem' in locals():
            del bem
            del src
                
