# Import necessary Packages
import mne
import pandas as pd
import os
import re
import glob
from pathlib import Path
from ast import literal_eval
from Scripts.Preprocessing.Preproc_Functions import extract_sort_key

def apply_ICA():
    # %% Remove ICA Components:
    # %% Set up Paths 
    script_dir      = Path(__file__).resolve() # Location of current scripts
    base_path       = script_dir.parent.parent.parent # Root folder
    data_path       = base_path / 'Data' # Folder containing the Data
    ica_comp_path   = data_path / 'ICA_Components.csv'
    
    reg_pattern = re.compile(r'^sub-\d{2}$') # creat regular expression pattern to identify folders of processed subjects
    sub_folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)) and reg_pattern.match(d)] # find the folders starting with sub-XX where XX is the subject number with a leading 0
    
    # %% other parameters
    param_filt_lp   = 100   # filter parameters lowpass 
    
    rstate          = 97 # Seed for the ICA (see below)
    icf             = pd.read_csv(ica_comp_path, delimiter = ',') 
    
    # %% Loop over the processed subjects 
    for sub in sub_folders:
        folder_path = data_path / sub
        
        sub_nr = sub.split('-')[1]
        
        ses_folders = ['ses-1', 'ses-2'] # give options for two sessions
        
        for ses in ses_folders:
            ses_path = folder_path / ses / 'meg'
            
            if ses_path.exists():
                # find the number of the session to make sure the correct pattern is matched for the file names
                ses_nr = ses.split('-')[1]
                
                # make directory for all the downampled and filtered annotated files before the ICA
                downsampled_files   = []        
                downsampled_path    = ses_path / 'downsampled'
                data_pattern        = f"sub-{sub_nr}_ses-{ses_nr}_task-*_run-*_meg_tsss_ds-500Hz.fif" 
                data_files          = downsampled_path.glob(data_pattern)
                data_files_sorted   = sorted(data_files, key=extract_sort_key) # sort the files in the folder to have the runs in ascending order and load the SRT files first and only then the WL files        
                out_path            = ses_path / f'sub-{sub_nr}_ses-{ses_nr}_PostICA_Full.fif'
                
                if out_path.exists():
                    continue
                
                # select the correct row for this subject and session
                ica_row     = icf[(icf['subject'] == sub) & (icf['session'] == int(ses_nr))]
                if ica_row.empty:
                    print(f"No matching rows found for Subject: {sub}, Session: {ses}")
                    continue
                else:
                    # save the components as a list
                    ica_components_bad = literal_eval(ica_row['components'].iloc[0]) 
                    
                ica_dir     = ses_path / 'ica'
                ica_file    = ica_dir / f"ica_projsub-{sub_nr}_ses-{ses_nr}_rstate{rstate}.fif"
                
                # ICA component removal doesn't work for the string names. Converting to numbers instead where ICA001 == 1 etc.
                ica_ints = [int(comp[3:]) for comp in ica_components_bad]
                
                for data_file in data_files_sorted:
                     downsampled_files.append(data_file)
                
                data_list        = [mne.io.read_raw_fif(file, preload=True) for file in downsampled_files]
                data_combined    = mne.concatenate_raws(data_list, on_mismatch='warn')
                del data_list
                
                ica = mne.preprocessing.read_ica(ica_file, verbose=None)
                
                # apply ICA
                print(f"Removing bad ICA components for file: subject {sub} -- ses {ses}")
                ica.exclude = ica_ints
                ica.apply(data_combined)
                
                # low pass filter data post-ICA
                print(f"Applying {param_filt_lp} Hz low pass filter for subject {sub} -- ses {ses}")
                data_combined.filter(l_freq=None, h_freq=param_filt_lp, fir_design='firwin')
                
                data_combined.save(out_path, overwrite=True )
                