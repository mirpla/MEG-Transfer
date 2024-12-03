import os 
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def wl_performance(script_dir):
    #script_dir      = Path(__file__).resolve() # Location of current scripts
    data_path       = script_dir.parent.parent.parent / 'Data' # Root folder
    figsize = (12,6)
    
    #load wordist items
    wl = pd.read_csv(data_path / 'MEG_WL_ITEMS.csv').to_numpy()
    
    # load relevant information for behavioral analysis from central csv
    sub_info = pd.read_csv(data_path / 'Subject_Information.csv',encoding='ISO-8859-1')
    rel_info = pd.DataFrame({
        'sub': sub_info['sub'],
        'ID': sub_info['ID'],
        'Explicitness': sub_info['Explicitness'],
        'Order': sub_info['ID'] // 1000,  # Extract the order; 1 = Exp first, 2 = Cont first
        'SubID': sub_info['ID'] % 1000    # Extract the corresponding subject IDs
        })
    
    # make sure to analyse the correct session as the first session
    rel_info['Order'][0]
    
    sub_folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)) and d.startswith('sub-')]
    sub_names   = {} # initialize dict to keep track of all the respective sub names
    sub_conds   = {}
    sub_nums    = {}
    data        = [[None] for _ in range(len(sub_folders))] # initialise variable where data will go in
    
    for i, sub in enumerate(sub_folders):
        sub_number = int(sub.split('-')[1]) # Extract subject number
       
        idx = np.where(rel_info['sub'] == sub )[0][0] # find the condition
        ses_number = rel_info['Order'][idx] # find the right index to extract condition
       
        if sub_number < 10:# first 9 had a different process with 2 session etc 
            ses = f'ses-{ses_number}'
        else:
            ses = 'ses-1'
        
        beh_path = data_path / sub / ses / 'beh'
        
        if beh_path.exists(): 
            file_name = f'{ses_number}{sub_number:03d}.csv'
            file_path = beh_path / file_name
    
            # Check if the file exists
            if file_path.exists():
                # Load the content of the .csv file
                data[i] = pd.read_csv(file_path, usecols=range(1, 11),header=None).to_numpy()
                sub_names[i]    = file_name[:-4]
                sub_conds[i]    = int(sub_names[i][0])
                sub_nums[i]     = int(sub_names[i][-2:])
            else:
                print(f'No file for subject {sub_number} - Session {ses_number}.')
    
      #%%
    # Figures for First Session only
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    
    # Define colors
    colors = [(0.8, 0.8, 1), (1, 0.8, 0.8)]
    mean_colors = [(0, 0, 1), (1, 0, 0)]
    
    # Serial Recall Implicit
    axs[0].set_title('Serial Recall Implicit')
    axs[1].set_title('Serial Recall Explicit')
    
    sr_con          = []
    sr_incon        = []
    sr_con_exp      = []
    sr_incon_exp    = []
    sr = {}
    exp_trck = {}
    for i, sub in enumerate(data): # go through processed subjects
        try:
            exp_idx = rel_info.SubID == sub_nums[i]
            exp_trck[i] = rel_info['Explicitness'][exp_idx] 
            
            sr[i] = []    
            for t in range(10): # go through iterations           
                data[i][np.isnan(data[i][:, t]), t] = 0
                x = data[i][:, t]
                d = np.diff(x) == 1
                indices = np.where(np.diff(np.concatenate(([0], d, [0]))) != 0)[0]
                if len(indices) > 1:
                    segment_lengths = indices[1::2] - indices[::2]
                    sr[i].append(np.max(segment_lengths) + 1 if len(segment_lengths) > 0 else 0)
                else: 
                    sr[i].append(0)
            axs[int(exp_trck[i])].plot(sr[i], color=colors[sub_conds[i]-1], linewidth=2)
        
            if (sub_conds[i] == 1) & (int(exp_trck[i]) == 0):
                sr_con.append(sr[i])
            elif (sub_conds[i] == 2) & (int(exp_trck[i]) == 0):
                sr_incon.append(sr[i])
            elif (sub_conds[i] == 1) & (int(exp_trck[i]) == 1):
                sr_con_exp.append(sr[i])
            elif (sub_conds[i] == 2) & (int(exp_trck[i]) == 1):
                sr_incon_exp.append(sr[i])
        except: 
              continue  
    
    mean_sr_con    = np.nanmean(sr_con,    axis=0)  # Compute mean across subjects
    mean_sr_incon  = np.nanmean(sr_incon,  axis=0)
    
    mean_sr_con_exp    = np.nanmean(sr_con_exp,    axis=0)  # Compute mean across subjects
    mean_sr_incon_exp  = np.nanmean(sr_incon_exp,  axis=0)
    
    
    mean1, = axs[0].plot(mean_sr_con, color=mean_colors[0], linewidth=3, label = 'Congruent')
    mean2, = axs[0].plot(mean_sr_incon, color=mean_colors[1], linewidth=3, label = 'Incongruent')
    
    mean3, = axs[1].plot(mean_sr_con_exp, color=mean_colors[0], linewidth=3, label = 'Congruent')
    mean4, = axs[1].plot(mean_sr_incon_exp, color=mean_colors[1], linewidth=3, label = 'Incongruent')
    
    for f in [0,1]:
        axs[f].set_ylim([0, 12])
        axs[f].set_xlim([0, 9])
        current_ticks = axs[f].get_xticks() 
        axs[f].set_xticklabels([int(tick + 1) for tick in current_ticks])
        axs[f].set_xlabel('Trial')
        axs[f].set_ylabel('Serial Recall')
        axs[f].legend(handles = [mean1, mean2], loc = 'lower right')
        
    return sr, sub_nums