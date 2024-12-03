# -*- coding: utf-8 -*-
"""
@author: gaby Cruz
"""

import mne
import os
#from mne.coreg import Coregistration
#from mne.io import read_info
from IPython import get_ipython
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'qt')

#%% Create folders for trans files
# literally just creates the empty trans folder within each session
# data_dir = r'Z:'
# mri_dir = 'Y:\MRI'
# out_dir = r'Y:\ANALYSIS\bem'
# plot_dir = r'Y:\PLOTS\bem'
#subfolders = [ f.path for f in os.scandir(maindir) if f.is_dir() ]
#subjects = [ f.name for f in os.scandir(maindir) if f.is_dir() ]

data_dir = r'Z:'
mri_dir = 'Y:\MRI'
out_dir = r'Y:\ANALYSIS'
plot_dir = r'Y:\PLOTS\bem'

# data_dir =  r'/raw/Project0349'
# mri_dir = r'/analyse/Project0349/MRI'
# out_dir = r'analyse/Project0349/ANALYSIS'
# plot_dir = r'analyse/Project0349/PLOTS/bem'




#subjects_list = ['ami28' ]#'ami28','jbe07','bsr27', 'dsa23', 'gto28',
subjects_list = [ f.name for f in os.scandir(data_dir) if f.is_dir() and f.name[0] != '_' and f.name[0].islower()]
#subjects_list = ['ade02', 'csi07', 'mwa29',  'rsg06', 'tdn02', 'uka11']
#subjects_list =  ['qqn19']
# Exclude participants that did not complete the experiment from subject list

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

#%% BEM stuff

conductivity = (0.3, 0.006, 0.3)  # for three layers
bem_problems = [] # note if there was a problem or not, 0 if okay, 1 if problem, can be used to index sub_list
for subject in subjects_list:
    print(subject)
    
    if subject not in folder_mapping:
        continue
    
    save_plot = os.path.join(plot_dir,subject)    
    if not os.path.exists(save_plot):
        os.makedirs(save_plot)
    # check if participant has mri already
    if not os.path.exists(os.path.join(mri_dir,subject)):
        print('Skipping ' + subject + ' because they have no MRI (yet).')
        continue
           
    # Create folder for BEM
    # this is done at the SUBJECT LEVEL as BEMs don't change session to session
    bem_folder = os.path.join(out_dir,folder_mapping[subject] + '_' + subject,'bem')  #sfpath + '\\bem\\'
    if not os.path.exists(bem_folder):
        os.makedirs(bem_folder)
    
    bemfile = os.path.join(bem_folder, subject + '_bem.h5')
    
    if os.path.isfile(bemfile):
         print('Skipping ' + subject + ' because they already have a BEM file.')
         continue
  
    # Set up source space
    src = mne.setup_source_space(
        subject=subject, subjects_dir=mri_dir,
        spacing='oct6', add_dist=False)
    
    # tried with 'with mne.viz.use_browser_backend('matplotlib'):', but couldn't save 
    # the figure below that way either: 

    # src_fig = mne.viz.plot_alignment(subject=subject, subjects_dir=mri_dir,
    #                             surfaces='white', coord_frame='mri',
    #                             src=src)
    # mne.viz.set_3d_view(src_fig, azimuth=173.78, elevation=101.75,
    #                 distance=0.35, focalpoint=(-0.03, -0.01, 0.03))
       
    # # Take a screenshot of the 3D scene
    # screenshot = src_fig.plotter.screenshot()
    
    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.imshow(screenshot, origin='upper')
    # ax.set_axis_off()  # Disable axis labels and ticks
    # fig.tight_layout()
    # fig_name = os.path.join(save_plot,subject + '_src.png')
    # fig.savefig(fig_name, dpi=150)
    # plt.close()
    # mne.viz.close_all_3d_figures()


     # Save source space
    src_name = os.path.join(bem_folder, subject + '-src.fif')
    src.save(src_name,overwrite=True)
    
    del src
    
    # Create and save BEM - IF there are no topo issues
    try:
        model = mne.make_bem_model(subject=subject, ico=4,
                                   conductivity=conductivity,
                                   subjects_dir=mri_dir)
        bem = mne.make_bem_solution(model)
        
        mne.write_bem_solution(bemfile, bem)
        bem_problems.append(0)
    except:
        bem_problems.append(1)
        print('#################### BEM ERROR FOR SUBJECT ' + subject + ' ####################')
  
    
    # Visualize surfaces - works EVEN IF previous BEM step failed
    plot_bem_kwargs = dict(
        subject=subject, subjects_dir=mri_dir,
        brain_surfaces='white', orientation='coronal',
        slices=[50,75, 100,125,130, 150,160,175])
    
    
    fig = mne.viz.plot_bem(**plot_bem_kwargs)
    figname = os.path.join(save_plot,subject + '_slices.png')
    fig.savefig(figname)
    plt.close('all')    


        