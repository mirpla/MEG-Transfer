import os 
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

script_dir = Path(__file__).resolve()
data_path = script_dir.parent.parent.parent / 'Data'
figsize = (12, 6)

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
    ses_folders = ['ses-1']#,'ses-2'] # only analyse 'ses-1' data   
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
plt.style.use('seaborn-v0_8-paper')  # Use a clean style for publication
fig, axs = plt.subplots(1, 1, figsize=figsize, dpi=300)  # Higher DPI for better SVG quality

# Define colors with transparency for shading
colors = ['#0000FF', '#FF0000']  # Blue and Red
alpha_fill = 0.2  # Transparency for the shading

# Serial Recall Implicit
axs.set_title('Serial Recall Learning Curves', pad=20, fontsize=12, fontweight='bold')

sr_con          = []
sr_incon        = []
sr_con_exp      = []
sr_incon_exp    = []
sr = {}
exp_trck = {}
for i, sub in enumerate(data): # go through processed subjects
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
    if (sub_conds[i] == 1) & (int(exp_trck[i]) == 0):
        sr_con.append(sr[i])
    elif (sub_conds[i] == 2) & (int(exp_trck[i]) == 0):
        sr_incon.append(sr[i])
    elif (sub_conds[i] == 1) & (int(exp_trck[i]) == 1):
        sr_con_exp.append(sr[i])
    elif (sub_conds[i] == 2) & (int(exp_trck[i]) == 1):
        sr_incon_exp.append(sr[i])

# Calculate means and standard errors
mean_sr_con = np.nanmean(sr_con, axis=0)
mean_sr_incon = np.nanmean(sr_incon, axis=0)
median_sr_con = np.nanmedian(sr_con, axis=0)
median_sr_incon = np.nanmedian(sr_incon, axis=0)


# Calculate standard errors
se_sr_con = np.nanstd(sr_con, axis=0) / np.sqrt(len(sr_con))
se_sr_incon = np.nanstd(sr_incon, axis=0) / np.sqrt(len(sr_incon))

# Create x-axis values
x = np.arange(10)

# Plot means with standard error bands
mean1, = axs.plot(x, mean_sr_con, color=colors[0], linewidth=2, label='Congruent')
mean2, = axs.plot(x, mean_sr_incon, color=colors[1], linewidth=2, label='Incongruent')

# Add standard error shading
axs.fill_between(x, 
                 mean_sr_con - se_sr_con, 
                 mean_sr_con + se_sr_con, 
                 color=colors[0], 
                 alpha=alpha_fill)
axs.fill_between(x, 
                 mean_sr_incon - se_sr_incon, 
                 mean_sr_incon + se_sr_incon, 
                 color=colors[1], 
                 alpha=alpha_fill)

# Customize the plot
axs.set_ylim([0, 12])
axs.set_xlim([-0.2, 9.2])  # Slightly extended for better visibility
axs.set_xticks(x)
axs.set_xticklabels([f'{int(i+1)}' for i in x])
axs.set_xlabel('Trial', fontsize=11, labelpad=10)
axs.set_ylabel('Serial Recall', fontsize=11, labelpad=10)

# Add grid for better readability
axs.grid(True, linestyle='--', alpha=0.3)

# Customize legend
axs.legend(handles=[mean1, mean2], 
          loc='lower right', 
          frameon=True, 
          framealpha=0.95,
          edgecolor='none')

# Adjust layout to prevent cutting off labels
plt.tight_layout()

# Save as SVG
# plt.savefig('learning_curves.svg', format='svg', bbox_inches='tight')
# plt.close()