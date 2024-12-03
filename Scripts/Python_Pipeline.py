from pathlib import Path
from Scripts.Import.Import_Data import Import_Data
from Scripts.Import.Import_Preproc import import_ER, coreg_subs
from Scripts.Preprocessing.Preprocess_Data import Preprocess_Data
from Scripts.Preprocessing.ICA_Manual import check_ICA_comp
from Scripts.Preprocessing.Post_ICA import apply_ICA
from Scripts.Preprocessing.RedoEvents import Events_Fix
from Scripts.Preprocessing.Preproc_Rest import  Crop_Rest_Events, Epoch_Rest
from Scripts.Preprocessing.Preproc_Functions import read_events

from Tools.Audio_Read import export_meg_audio

# %% Import the Raw MEG data and MRI's and perform Tsss and other first run preprocessing 
# Already done: ['2','4','5','6','7','8','9','10','11','13','14','15','16','18','19','20','21','22','23','24','25','29','30','31','32','33','35','36','37','39','42','43','45','47']
# MAKE SURE TO RUN AS ADMINISTRATOR! 
sub_codes = ['48','49']  # 17 24 still need 1 and 9, 9 has a naming problem
Import_Data(sub_codes)

#%%
subs = ['sub-47'] # 'sub-XX'
coreg_subs(subs)

# %% First run of preprocessing - HP filtering, downsampling and ICA
 
Preprocess_Data()

# %% Look at ICA components and write down which ones to exclude in a separate .csv (Data\ICA_Components.csv)

ses = 0 # select session for checking ICA
sub = Path('//analyse7/project0407/Data/sub-36') # select the subject folder to look at
check_ICA_comp(sub, ses)

# %% Reject the ICA components and apply a 100 Hz lp filter now that ICA components have been rejected

apply_ICA()

# %% Rest analyses:
Crop_Rest_Events(['sub-25','sub-29','sub-30','sub-31','sub-32','sub-33','sub-35','sub-36'])

# Make the Epochs
manual_rej = 1 # 0 = take previous artefacts; 1 = do artf rejection manually
epoch_dur = 4 # epoch window size in seconds
sessions = ['ses-1'] # give options for two sessions
Epoch_Rest(manual_rej,epoch_dur, sessions)

# %% Behavior
#WL_Analysis

# %% Other
import_ER() # import and process the empty room data to allow for NCM for source localisation, input is the data 'YYMMDD'
# Events_Fix Careful, running this will rerun all the event files and overwrite the existing ones

meg_file = '//raw/Project0407/ebs13/240919/MEG_1026_WL1.fif'
wav_file = 'C:/Users/mirceav/Desktop/audio.wav'
export_meg_audio(meg_file, wav_file)