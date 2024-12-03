# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 12:58:20 2022

@author: gabriela
"""

import os.path as op
import os
import numpy as np
from mne.preprocessing import annotate_muscle_zscore
import mne
#import matplotlib.pyplot as plt

#%% set-up paths

# local path
# output_plot_folder = r'Y:\PLOTS\QUALITY_CHECK' 
# data_dir = r'Y:\ANALYSIS' # r'D:\PROJECTS\CAUSAL_NETWORKS\DATA'

# deepnet
output_plot_folder = r'/analyse/Project0349/PLOTS/QUALITY_CHECK' 
data_dir = r'/analyse/Project0349/ANALYSIS' # r'D:\PROJECTS\CAUSAL_NETWORKS\DATA'

input_folder = 'tsss'
output_folder ='annot'#r'D:\PROJECTS\CAUSAL_NETWORKS\ANALYSIS\tsss'

subjects_list = [ f.name for f in os.scandir(data_dir) if f.is_dir() and f.name[0] != '_' ]
subjects_list.sort()
print(subjects_list)


#%%
veog  = {
  "bsr27":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "dsa23":  {"session1": ['BIO001'], "session2": ['BIO002'], "session3": ['BIO002']},
  "mtr13":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "gto28":  {"session1": ['BIO002'], "session2": ['BIO001'], "session3": ['BIO002']},
  "ami28":  {"session1": ['EEG001'], "session2": ['EEG001'], "session3": ['EEG001']},
  "lka10":  {"session1": ['EEG001'], "session2": ['EEG001'], "session3": ['BIO002']},
  "qqn19":  {"session1": ['BIO002'], "session2": ['EEG001'], "session3": ['EEG001']},
  "mtr19":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "fha01":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "hwh21":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "rsh17":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "zwi25":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "tdn02":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "uka11":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "csi07":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "rsg06":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "mwa29":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "ade02":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "dtl05":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "rbe04":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "jmn22":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "dss19":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "fte25":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "hyr24":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "ank24":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "dja01":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "fmn28":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "omr03":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "amy20":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "ski23":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "jry29":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "jpa10":  {"session1": ['BIO001'], "session2": ['BIO002'], "session3": ['BIO002']},
  "epa14":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "crr22":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "jyg27":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "bbi29":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "ece24":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "hay06":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "awa19":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "mca10":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  "hky23":  {"session1": ['BIO002'], "session2": ['BIO002'], "session3": ['BIO002']},
  }  
     

#%%
bad_eeg_channels  = {
  "bsr27":  {"session1": {"block01": ['EEG002'], "block02": ['EEG002'], "block03": ['ok'],"block04": ['ok'], "block05": ['EEG011'], "block06": ['ok'],"block07": ['ok']},
             "session2": {"block09": ['lost'], "block10": ['lost'], "block11": ['lost'],"block12": ['lost'], "block13": ['lost'], "block14": ['ok'], "block15": ['ok'], "block16": ['ok'],"block17": ['ok'],"block18": ['ok'], "block19": ['EEG007'], "block20": ['ok']}, 
             "session3": {"block21": ['ok'], "block22": ['EEG007','EEG033'], "block23": ['EEG007','EEG033'],"block24": ['EEG023'], "block25": ['ok'], "block26": ['ok'],"block27": ['ok'],"block28": ['ok'], "block29": ['ok'], "block30": ['ok']}},
  "dsa23":  {"session1": {"block01": ['EEG008','EEG037','EEG041','EEG059'], "block02": ['EEG037','EEG059'], "block03": ['EEG037','EEG059'],"block04": ['EEG037','EEG059'], "block05": ['EEG037','EEG059'], "block06": ['EEG037','EEG041','EEG059'],"block07": ['EEG037','EEG059'],"block08": ['EEG037','EEG059'], "block09": ['EEG037','EEG041','EEG059'], "block10": ['EEG037','EEG059'],"block11": ['EEG037','EEG059','EEG041','EEG011','EEG021']},
             "session2": {"block12": ['EEG005'], "block13": ['ok'],"block14": ['EEG005','EEG037','EEG059'], "block15": ['EEG059','EEG037'], "block16": ['EEG037','EEG041','EEG054','EEG059','EEG061'],"block17": ['EEG041','EEG054','EEG037','EEG059']}, 
             "session3": {"block18": ['EEG061'], "block19": ['EEG059','EEG061'],"block20": ['EEG037','EEG059','EEG061'], "block21": ['EEG037','EEG059','EEG061'], "block22": ['EEG037','EEG059','EEG061'], "block23": ['EEG037','EEG059','EEG061'],"block24": ['EEG037','EEG059','EEG061'], "block25": ['EEG037','EEG059','EEG061'], "block26": ['EEG059','EEG061'],"block27": ['EEG037','EEG061'],"block28": ['EEG037','EEG059','EEG061']}},
  "mtr13":  {"session1": {"block01": ['EEG012','EEG033'], "block02": ['EEG012','EEG033'], "block03": ['EEG012','EEG033'],"block04": ['EEG012','EEG033'], "block05": ['EEG012','EEG033'], "block06": ['EEG012'],"block07": ['EEG012'],"block08": ['EEG012'], "block09": ['EEG012'], "block10": ['EEG012'],"block11": ['ok']},
             "session2": {"block12": ['EEG033'], "block13": ['EEG033'],"block14": ['EEG007','EEG033'], "block15": ['EEG008','EEG033'], "block16": ['ok'],"block17": ['EEG012','EEG033','EEG060'],"block18": ['EEG008','EEG033'], "block19": ['EEG008','EEG033'], "block20": ['EEG033'], "block21": ['EEG008','EEG033'],"block22": ['EEG008','EEG033']}, 
             "session3": { "block23": ['EEG007','EEG033','EEG058','EEG059'],"block24": ['EEG007','EEG033','EEG058','EEG059'], "block25": ['EEG007','EEG033','EEG058','EEG059'], "block26": ['EEG007','EEG033','EEG058','EEG059'],"block27": ['EEG007','EEG033','EEG058','EEG059'],"block28": ['EEG007','EEG033','EEG058','EEG059'], "block29": ['EEG007','EEG033','EEG058','EEG059'], "block30": ['EEG007','EEG033','EEG058','EEG059']}},
  "gto28":  {"session1": {"block01": ['ok'], "block02": ['ok'], "block03": ['ok'],"block04": ['ok'], "block05": ['ok'], "block06": ['ok'],"block07": ['ok'],"block08": ['ok'], "block09": ['ok'], "block10": ['EEG033','EEG037'],"block11": ['ok']},
             "session2": {"block12": ['ok'], "block13": ['ok'],"block14": ['ok'], "block15": ['ok'], "block16": ['ok'],"block17": ['ok'],"block18": ['ok'], "block19": ['ok'], "block20": ['ok'],"block21": ['ok'], "block22": ['ok']}, 
             "session3": {"block23": ['ok'],"block24": ['ok'], "block25": ['ok'], "block26": ['ok'],"block27": ['ok'],"block28": ['ok'], "block29": ['ok'], "block30": ['ok']}},
  "ami28":  {"session1": {"block01": ['EEG008','EEG012'], "block02": ['EEG012'], "block03": ['EEG012'],"block04": ['EEG012','EEG064'], "block05": ['EEG012','EEG064'], "block06": ['EEG012','EEG020','EEG064'],"block07": ['EEG012'],"block08": ['EEG012'], "block09": ['EEG012'], "block10": ['EEG012']},
             "session2": {"block11": ['EEG012'], "block12": ['EEG020','EEG033','EEG037','EEG038'], "block13": ['EEG012'],"block14": ['EEG012'], "block15": ['EEG012'], "block16": ['EEG012'],"block17": ['EEG012'],"block18": ['EEG012'], "block19": ['EEG012'], "block20": ['EEG012']}, 
             "session3": {"block21": ['EEG007','EEG012'], "block22": ['EEG007'], "block23": ['EEG007','EEG012'],"block24": ['EEG012'], "block25": ['EEG007','EEG012'], "block26": ['EEG012'],"block27": ['EEG012'],"block28": ['EEG012'], "block29": ['EEG012'], "block30": ['EEG012']}},
  "lka10":  {"session1": {"block01": ['EEG012','EEG058','EEG064'], "block02": ['EEG064'], "block03": ['EEG012','EEG024','EEG048','EEG058','EEG059','EEG064'],"block04": ['EEG012','EEG058','EEG059','EEG064'], "block05": ['EEG001','EEG012','EEG024','EEG058','EEG059'], "block06": ['EEG058','EEG063'],"block07": ['EEG012','EEG058','EEG059','EEG063','EEG064'],"block08": ['EEG012','EEG058','EEG059','EEG063','EEG064']},
             "session2": {"block09": ['EEG059'], "block10": ['EEG059'], "block11": ['EEG059'], "block12": ['EEG059'], "block13": ['EEG059'],"block14": ['EEG059','EEG064'], "block15": ['EEG059','EEG064'], "block16": ['EEG024','EEG059','EEG064']}, 
             "session3": {"block17": ['EEG024','EEG059','EEG064'],"block18": ['EEG002','EEG024','EEG048','EEG059','EEG064'], "block19": ['EEG002','EEG024','EEG048','EEG059','EEG064'], "block20": ['EEG002','EEG024','EEG063','EEG059','EEG064'],"block21": ['EEG002','EEG024','EEG048','EEG059','EEG064'], "block22": ['EEG002','EEG024','EEG048','EEG059','EEG064'], "block23": ['EEG002','EEG024','EEG048','EEG063','EEG064','EEG012','EEG059'],"block24": ['EEG002','EEG024','EEG048','EEG059','EEG064','EEG063'], "block25": ['EEG002','EEG024','EEG048','EEG059','EEG064','EEG063'], "block26": ['EEG002','EEG024','EEG048','EEG059','EEG064','EEG026']}},
  "qqn19":  {"session1": {"block01": ['EEG008','EEG013','EEG060','EEG061'], "block02": ['EEG060'], "block03": ['EEG020'],"block04": ['EEG020'], "block05": ['EEG020'], "block06": ['EEG020'],"block07": ['EEG020'],"block08": ['EEG020'], "block09": ['EEG020'], "block10": ['EEG020']},
             "session2": {"block11": ['EEG012'], "block12": ['ok'], "block13": ['EEG012'],"block14": ['EEG012'], "block15": ['EEG012'], "block16": ['EEG008','EEG012'],"block17": ['EEG012'],"block18": ['EEG008'], "block19": ['EEG008','EEG012'], "block20": ['EEG018','EEG032','EEG062']}, 
             "session3": {"block21": ['EEG008'], "block22": ['EEG008'], "block23": ['EEG008'],"block24": ['EEG008'], "block25": ['EEG008'], "block26": ['EEG008'],"block27": ['EEG008'],"block28": ['EEG008']}},
  "mtr19":  {"session1": {"block01": ['EEG008'], "block02": ['EEG008'], "block03": ['EEG001','EEG004','EEG008','EEG033'],"block04": ['EEG001','EEG004','EEG008','EEG033'], "block05": ['EEG001','EEG004','EEG008','EEG033'], "block06": ['EEG008'],"block07": ['EEG008'],"block08": ['EEG008']},
             "session2": {"block09": ['EEG008','EEG033','EEG064'], "block10": ['EEG008','EEG033','EEG064'],"block11": ['EEG008','EEG033','EEG064'], "block12": ['EEG008','EEG033','EEG064'], "block13": ['EEG008','EEG033','EEG064'],"block14": ['EEG008','EEG033','EEG064'], "block15": ['EEG008','EEG033','EEG064'], "block16": ['EEG008','EEG033','EEG026','EEG064'],"block17": ['EEG008','EEG033','EEG064','EEG026'],"block18": ['EEG008','EEG033','EEG064','EEG026'], "block19": ['EEG008','EEG033','EEG064','EEG026'], "block20": ['EEG008','EEG033','EEG026','EEG064']}, 
             "session3": {"block21": ['EEG008','EEG033','EEG064'], "block22": ['EEG008','EEG064','EEG033'], "block23": ['EEG008','EEG033','EEG064'],"block24": ['EEG008','EEG033','EEG064'], "block25": ['EEG008','EEG033','EEG064','EEG010'], "block26": ['EEG008','EEG033','EEG064','EEG010'],"block27": ['EEG008','EEG033','EEG010','EEG064','EEG018'],"block28": ['EEG008','EEG033','EEG064','EEG010'], "block29": ['EEG008','EEG033','EEG064','EEG010'], "block30": ['EEG008','EEG033','EEG064','EEG010']}},
  "fha01":  {"session1": {"block01": ['EEG064'], "block02": ['EEG064'], "block03": ['EEG064'],"block04": ['EEG064'], "block05": ['EEG064'], "block06": ['EEG064'],"block07": ['EEG064'],"block08": ['EEG064'], "block09": ['EEG064']},
             "session2": {"block10": ['EEG064'], "block11": ['EEG064'], "block12": ['EEG064'], "block13": ['EEG064'],"block14": ['EEG064'], "block15": ['EEG064'], "block16": ['EEG064'],"block17": ['EEG064'],"block18": ['EEG064']}, 
             "session3": {"block19": ['EEG064'], "block20": ['EEG064'], "block21": ['EEG064'], "block22": ['EEG064'], "block23": ['EEG064'],"block24": ['EEG064'], "block25": ['EEG064'], "block26": ['EEG064'],"block27": ['EEG064'],"block28": ['EEG064']}},
  "hwh21":  {"session1": {"block01": ['EEG008','EEG041','EEG044','EEG045','EEG058','EEG059','EEG021','EEG042'], "block02": ['EEG008','EEG019','EEG058','EEG059'], "block03": ['EEG008','EEG041','EEG058','EEG059','EEG064'],"block04": ['EEG008','EEG041','EEG058','EEG059','EEG064']},
             "session2": {"block05": ['EEG004','EEG038','EEG058','EEG059','EEG061','EEG064'], "block06": ['EEG004','EEG058','EEG059','EEG064','EEG042','EEG021','EEG038'],"block07": ['EEG004','EEG058','EEG061','EEG044'],"block08": ['EEG004','EEG061','EEG058','EEG059'], "block09": ['EEG061','EEG058','EEG004'], "block10": ['EEG007','EEG037','EEG061','EEG004'],"block11": ['EEG007','EEG037','EEG061','EEG004','EEG058','EEG059'], "block12": ['EEG004','EEG007','EEG058','EEG059','EEG021','EEG037','EEG061'], "block13": ['EEG007','EEG061','EEG004','EEG042','EEG038','EEG058','EEG059'],"block14": ['EEG004','EEG058','EEG059','EEG061','EEG021'], "block15": ['EEG004','EEG028','EEG058','EEG059','EEG061'], "block16": ['EEG038','EEG008','EEG042','EEG061','EEG004','EEG058','EEG059']}, 
             "session3": {"block17": ['EEG004','EEG012','EEG038','EEG058','EEG059','EEG061','EEG041','EEG042'],"block18": ['EEG004','EEG012','EEG028','EEG058','EEG059','EEG061'], "block19": ['EEG042','EEG021','EEG061','EEG004','EEG038','EEG058','EEG059'], "block20": ['EEG004','EEG058','EEG059','EEG061'],"block21": ['EEG004','EEG012','EEG058','EEG059','EEG021','EEG061'], "block22": ['EEG004','EEG012','EEG058','EEG059','EEG038'], "block23": ['EEG004','EEG058','EEG059','EEG061'],"block24": ['EEG004','EEG058','EEG059'], "block25": ['EEG004','EEG058','EEG059'], "block26": ['EEG004','EEG058','EEG059'],"block27": ['EEG004','EEG058','EEG059'],"block28": ['EEG004','EEG006','EEG058','EEG059']}},
  "rsh17":  {"session1": {"block01": ['EEG064'], "block02": ['EEG064'], "block03": ['EEG064'],"block04": ['EEG010','EEG064'], "block05": ['EEG064'], "block06": ['EEG022','EEG064'],"block07": ['EEG010','EEG064'],"block08": ['EEG010','EEG024','EEG008','EEG064']},
             "session2": {"block09": ['ok'], "block10": ['ok'],"block11": ['ok'], "block12": ['ok'], "block13": ['ok'],"block14": ['EEG010'], "block15": ['ok'], "block16": ['ok'],"block17": ['ok'],"block18": ['ok'], "block19": ['ok']}, 
             "session3": {"block20": ['EEG064'], "block21": ['EEG064'], "block22": ['EEG064'],"block23": ['EEG064'], "block24": ['EEG018','EEG064'], "block25": ['EEG064'],"block26": ['EEG064'],"block27": ['EEG064'], "block28": ['EEG064'], "block29": ['EEG064'], "block30": ['EEG064']}},
  "zwi25":  {"session1": {"block01": ['EEG001','EEG008','EEG012','EEG004'], "block02": ['EEG004','EEG008','EEG012','EEG042'], "block03": ['EEG001','EEG004','EEG008','EEG012','EEG033'],"block04": ['EEG001','EEG004','EEG008','EEG012','EEG033'], "block05": ['EEG001','EEG004','EEG008','EEG012','EEG033'], "block06": ['EEG001','EEG004','EEG008','EEG012','EEG033'],"block07": ['EEG001','EEG004','EEG008','EEG012','EEG033'],"block08": ['EEG001','EEG004','EEG008','EEG012','EEG033','EEG034'], "block09": ['EEG001','EEG004','EEG008','EEG012','EEG033','EEG020','EEG034','EEG039'], "block10": ['EEG001','EEG004','EEG008','EEG012','EEG033','EEG023','EEG002','EEG039','EEG034','EEG020']},
             "session2": {"block11": ['EEG004','EEG012','EEG034','EEG038'], "block12": ['EEG004','EEG059'], "block13": ['EEG004','EEG008','EEG034','EEG058'],"block14": ['EEG004','EEG058'], "block15": ['EEG004'], "block16": ['EEG004'],"block17": ['EEG004','EEG012','EEG034','EEG058'],"block18": ['EEG004','EEG034'], "block19": ['EEG004','EEG008','EEG012','EEG034'], "block20": ['EEG004','EEG008','EEG011','EEG035','EEG058'],"block21":['EEG004','EEG012','EEG035','EEG058']},
             "session3": {"block22": ['EEG001','EEG004','EEG038'], "block23": ['EEG004','EEG037','EEG038'], "block24": ['EEG001','EEG004','EEG038'],"block25": ['EEG004','EEG020','EEG034'], "block26": ['EEG001','EEG004','EEG037'], "block27": ['EEG001','EEG004','EEG034','EEG037'],"block28": ['EEG001','EEG004','EEG037'],"block29": ['EEG004'], "block30": ['EEG001','EEG004','EEG037','EEG020']}},
  "tdn02":  {"session1": {"block01": ['EEG004','EEG010','EEG020','EEG037'], "block02": ['EEG004','EEG010','EEG020','EEG037'], "block03": ['EEG004','EEG010','EEG020','EEG037'],"block04": ['EEG004','EEG010','EEG020','EEG037'], "block05": ['EEG004','EEG010','EEG020','EEG037'], "block06": ['EEG004','EEG010','EEG020','EEG037'],"block07": ['EEG004','EEG010','EEG020','EEG037','EEG058'],"block08": ['EEG004', 'EEG058','EEG010','EEG020','EEG037'], "block09": ['EEG002','EEG004', 'EEG058','EEG010','EEG020','EEG037','EEG033'], "block10": ['EEG004','EEG010','EEG020','EEG037']},
             "session2": {"block11": ['EEG004','EEG021','EEG024','EEG050','EEG026','EEG015'], "block12": ['EEG004','EEG021','EEG024','EEG050'], "block13": ['EEG004','EEG021','EEG024','EEG026','EEG052','EEG050','EEG049'],"block14": ['EEG004','EEG021','EEG024','EEG026','EEG050'], "block15": ['EEG002','EEG004','EEG046'], "block16": ['EEG002','EEG004','EEG021','EEG024','EEG026','EEG046'],"block17":['EEG002','EEG004','EEG020','EEG050','EEG053'],"block18": ['EEG002','EEG004','EEG021','EEG024','EEG026','EEG050','EEG053'], "block19":['EEG004','EEG027'], "block20": ['EEG010','EEG046','EEG049','EEG004','EEG021','EEG024']}, 
             "session3": {"block21":['EEG004','EEG050'], "block22":['EEG004','EEG050','EEG060'], "block23": ['EEG004','EEG050','EEG060'],"block24":['EEG004','EEG050','EEG060'], "block25": ['EEG004','EEG007','EEG050','EEG060'], "block26":['EEG004','EEG050','EEG060'],"block27": ['EEG004','EEG050','EEG010','EEG043','EEG026'],"block28": ['EEG004','EEG010','EEG050','EEG060'], "block29":['EEG004','EEG050','EEG060'], "block30": ['EEG004','EEG024','EEG050']}},
  "uka11": {"session1": {"block01": ['EEG004'], "block02": ['EEG004','EEG024'], "block03": ['EEG004','EEG024'],"block04": ['EEG004'], "block05": ['EEG004','EEG010','EEG056'], "block06": ['EEG004','EEG010'],"block07": ['EEG004','EEG010']},
            "session2": {"block08": ['EEG004'], "block09": ['EEG004'], "block10": ['EEG004'],"block11": ['EEG004'], "block12": ['EEG004'], "block13": ['EEG004'],"block14": ['EEG004'],"block15": ['EEG004'], "block16": ['EEG004'], "block17": ['EEG004']},
            "session3": {"block18": ['EEG004'], "block19": ['EEG004'], "block20": ['EEG004'],"block21": ['EEG004'], "block22": ['EEG004'], "block23": ['EEG004'],"block24": ['EEG004'],"block25": ['EEG004'],"block26": ['EEG004']}},
  "csi07":  {"session1": {"block01": ['ok'], "block02": ['ok'], "block03": ['ok'],"block04": ['ok'], "block05": ['ok'], "block06": ['ok'],"block07": ['ok'],"block08": ['ok']},
             "session2": {"block09": ['EEG012'], "block10": ['ok'], "block11": ['EEG012'],"block12": ['EEG048'], "block13": ['ok'], "block14": ['EEG059'],"block15": ['ok'],"block16": ['ok'], "block17": ['ok']},
             "session3": {"block18": ['ok'], "block19": ['EEG007'], "block20": ['ok'],"block21": ['ok'], "block22": ['ok'], "block23": ['ok'],"block24": ['ok'],"block25": ['EEG022']}},
  "rsg06":  {"session1": {"block01": ['EEG004','EEG008','EEG064'], "block02": ['EEG004','EEG008','EEG064'], "block03": ['EEG004','EEG008','EEG064'], "block05": ['EEG004','EEG008','EEG064'], "block06": ['EEG004','EEG008','EEG064'],"block07": ['EEG004','EEG008','EEG064'],"block08": ['EEG004','EEG008','EEG064'], "block09": ['EEG004','EEG008','EEG064'], "block10": ['EEG004','EEG008','EEG064']},
             "session2": {"block11": ['EEG004','EEG024','EEG064','EEG008'], "block12": ['EEG004','EEG012','EEG064'], "block13": ['EEG004','EEG064','EEG008'],"block14": ['EEG004','EEG008','EEG064'], "block15": ['EEG004','EEG008','EEG064'], "block16": ['EEG004','EEG008','EEG064'],"block17": ['EEG004','EEG008','EEG024','EEG064','EEG054'],"block18": ['EEG004','EEG008','EEG064'], "block19": ['EEG004','EEG064','EEG008','EEG009','EEG054']}, 
             "session3": {"block21": ['EEG004'], "block22": ['EEG004'], "block23": ['EEG004'],"block24": ['EEG004'], "block25": ['EEG004','EEG033'], "block26": ['EEG004'],"block27": ['EEG004'],"block28": ['EEG004','EEG011'], "block29": ['EEG004','EEG011'], "block30": ['EEG004','EEG011']}},
  "mwa29":  {"session1": {"block01": ['ok'], "block02": ['ok'], "block03": ['ok'],"block04": ['ok'], "block05": ['EEG059'], "block06": ['ok'],"block07": ['ok'],"block08": ['EEG059'], "block09": ['EEG047'], "block10": ['EEG047','EEG027']},
             "session2": {"block11": ['EEG046'], "block12": ['ok'], "block13": ['EEG048','EEG059'],"block14": ['ok'], "block15": ['EEG058'], "block16": ['ok'],"block17": ['ok'],"block18": ['ok'], "block19": ['EEG046']},
             "session3": {"block20": ['ok'],"block21": ['ok'], "block22": ['EEG058','EEG059'], "block23": ['EEG058','EEG059'],"block24": ['EEG058','EEG059'], "block25": ['EEG058','EEG059'], "block26": ['ok'],"block27": ['ok'],"block28": ['ok'], "block29": ['EEG059']}},
  "ade02":  {"session1": {"block01": ['EEG008'], "block02": ['ok'], "block03": ['ok'],"block04": ['ok'], "block05": ['ok'], "block06": ['ok'],"block07": ['EEG008'],"block08": ['ok'], "block09": ['ok'], "block10": ['ok'],"block11": ['ok']},
             "session2": {"block12": ['EEG007'], "block13": ['EEG058'], "block14": ['ok'],"block15": ['EEG024'], "block16": ['ok'], "block17": ['ok'],"block18": ['ok'],"block19": ['ok'], "block20": ['ok']},  
             "session3": {"block21": ['ok'], "block22": ['ok'], "block23": ['ok'],"block24": ['ok'], "block25": ['EEG024'], "block26": ['ok'],"block27": ['ok'],"block28": ['ok'], "block29": ['ok'], "block30": ['ok']}},
  "dtl05":  {"session1": {"block01": ['EEG008','EEG012','EEG001','EEG019'], "block02": ['EEG037','EEG019','EEG001','EEG018'], "block03": ['EEG036'],"block04": ['EEG019','EEG018'], "block05": ['EEG001','EEG020'], "block06": ['ok'],"block07": ['EEG008','EEG019','EEG018'],"block08": ['EEG019','EEG018'], "block09": ['EEG019','EEG018','EEG060'], "block10": ['EEG019','EEG018','EEG060','EEG001']},
             "session2": {"block11": ['EEG001','EEG018','EEG019','EEG064'] ,"block12": ['EEG018','EEG019','EEG064'], "block13": ['EEG018','EEG019','EEG064'], "block14": ['EEG012','EEG018','EEG019','EEG064'],"block15": ['EEG012','EEG018','EEG019','EEG064','EEG041'], "block16": ['EEG012','EEG018','EEG019','EEG064'], "block17": ['EEG012','EEG018','EEG019','EEG064'],"block18": ['EEG018','EEG019','EEG064'],"block19": ['EEG018','EEG019','EEG064'], "block20": ['EEG018','EEG019','EEG064','EEG008']},  
             "session3": {"block21": ['EEG020','EEG019'], "block22": ['EEG018','EEG019'], "block23": ['EEG018','EEG019'],"block24": ['EEG064'], "block25": ['EEG023','EEG018','EEG019','EEG064'], "block26": ['EEG018','EEG019','EEG064'],"block27": ['EEG018','EEG019'],"block28": ['EEG002','EEG020','EEG041','EEG018','EEG019','EEG064'], "block29": ['EEG001','EEG002','EEG020','EEG041','EEG018','EEG019','EEG064'], "block30": ['EEG018','EEG019','EEG064']}},
  "rbe04":  {"session1": {"block01": ['ok'], "block02": ['EEG012'], "block03": ['ok'],"block04": ['EEG012','EEG037','EEG059'], "block05": ['ok'], "block06": ['EEG012','EEG048','EEG059'],"block07": ['EEG058']},
             "session2": {"block08": ['EEG012'], "block09": ['EEG059'], "block10": ['EEG012','EEG037','EEG041'] ,"block11": ['EEG012'], "block12": ['EEG012','EEG024','EEG059'], "block13": ['EEG058'], "block14": ['EEG037'],"block15": ['EEG037','EEG059'], "block16": ['EEG037','EEG041','EEG059'], "block17": ['EEG012','EEG024','EEG037','EEG041','EEG059']},  
             "session3": {"block18": ['EEG012'],"block19": ['EEG024','EEG037'], "block20": ['EEG012'] , "block21": ['EEG037'], "block22": ['EEG037','EEG041'], "block23": ['EEG037'],"block24": ['EEG037','EEG041','EEG059'], "block25": ['EEG012','EEG037'], "block26": ['EEG037']}},
  "jmn22":  {"session1": {"block01": ['EEG020','EEG033','EEG037','EEG060'], "block02": ['EEG001','EEG006','EEG037','EEG061'], "block03": ['EEG006','EEG020','EEG033','EEG037','EEG060'],"block04": ['EEG020','EEG033','EEG037','EEG060'], "block05": ['EEG033'], "block06": ['EEG033','EEG037','EEG060'],"block07": ['EEG020','EEG033','EEG037','EEG041'],"block08": ['EEG020','EEG033'], "block09": ['EEG020','EEG033']},
             "session2": {"block10": ['EEG001','EEG006'], "block11": ['ok'],"block12": ['EEG007'], "block13": ['EEG058'], "block14": ['ok'],"block15": ['EEG024'], "block16": ['ok'], "block17": ['ok'],"block18": ['ok'],"block19": ['ok'], "block20": ['ok']},  
             "session3": {"block21": ['ok'], "block22": ['ok'], "block23": ['ok'],"block24": ['ok'], "block25": ['EEG024'], "block26": ['ok'],"block27": ['ok'],"block28": ['ok'], "block29": ['ok'], "block30": ['ok']}},
  "dss19":  {"session1": {"block01": ['ok'], "block02": ['ok'], "block03": ['ok'],"block04": ['ok'], "block05": ['ok'], "block06": ['ok'],"block07": ['ok'],"block08": ['ok'], "block09": ['ok'],"block10": ['ok']},
             "session2": {"block11": ['ok'], "block12": ['ok'], "block13": ['ok'], "block14": ['ok'],"block15": ['ok'], "block16": ['ok'], "block17": ['ok'],"block18": ['ok'],"block19": ['ok'], "block20": ['ok']},  
             "session3": {"block21": ['ok'], "block22": ['ok'], "block23": ['ok'],"block24": ['ok'], "block25": ['ok'], "block26": ['ok'],"block27": ['ok'],"block28": ['ok'], "block29": ['ok'], "block30": ['ok']}},
  "fte25":  {"session1": {"block01": ['lost'], "block02": ['lost'], "block03": ['lost'],"block04": ['lost'], "block05": ['lost'], "block06": ['lost'],"block07": ['lost'],"block08": ['lost'], "block09": ['lost']},
             "session2": {"block10": ['EEG007','EEG024','EEG033','EEG041','EEG058','EEG059','EEG005','EEG010','EEG012','EEG037'], "block11": ['EEG007','EEG024','EEG033','EEG041','EEG058','EEG059','EEG005','EEG010','EEG012','EEG037'], "block12": ['EEG007','EEG024','EEG033','EEG041','EEG058','EEG059','EEG005','EEG010','EEG012','EEG037'], "block13": ['EEG007','EEG024','EEG033','EEG041','EEG058','EEG059','EEG005','EEG010','EEG012','EEG020','EEG037'], "block14": ['EEG007','EEG024','EEG033','EEG041','EEG058','EEG059','EEG005','EEG010','EEG012','EEG020','EEG037'],"block15": ['EEG007','EEG024','EEG033','EEG041','EEG058','EEG059','EEG005','EEG010','EEG012','EEG037','EEG020','EEG055'], "block16": ['EEG007','EEG024','EEG033','EEG041','EEG058','EEG059','EEG005','EEG010','EEG012','EEG020','EEG037'], "block17": ['EEG007','EEG024','EEG033','EEG041','EEG058','EEG059','EEG005','EEG010','EEG012','EEG037','EEG020'],"block18": ['EEG007','EEG024','EEG033','EEG041','EEG058','EEG059','EEG005','EEG010','EEG012','EEG037','EEG020','EEG054'],"block19": ['EEG020','EEG007','EEG024','EEG033','EEG041','EEG058','EEG059','EEG005','EEG010','EEG012','EEG037']},  
             "session3": {"block20": ['EEG001','EEG012','EEG024','EEG033','EEG059'],"block21": ['EEG001','EEG012','EEG024','EEG033','EEG048','EEG059'], "block22": ['EEG001','EEG012','EEG024','EEG033','EEG059'], "block23": ['EEG001','EEG012','EEG024','EEG033','EEG059'],"block24": ['EEG001','EEG012','EEG021','EEG024','EEG033','EEG059'], "block25": ['EEG001','EEG012','EEG021','EEG024','EEG033','EEG059'], "block26": ['EEG001','EEG012','EEG024','EEG033','EEG059'],"block27": ['EEG001','EEG012','EEG033','EEG059'],"block28": ['EEG001','EEG012','EEG033','EEG059'], "block29": ['EEG012','EEG033','EEG059']}},
  "hyr24":  {"session1": {"block01": ['EEG003','EEG008','EEG009','EEG021'], "block02": ['EEG003','EEG008'], "block03": ['EEG003','EEG008'],"block04": ['EEG003','EEG008'], "block05": ['EEG003','EEG008'], "block06": ['EEG003','EEG008'],"block07": ['EEG003','EEG008'],"block08": ['EEG003','EEG008'], "block09": ['EEG003','EEG008'],"block10": ['EEG003','EEG008']},
             "session2": {"block11": ['EEG003','EEG008'], "block12": ['EEG003','EEG021','EEG033','EEG038','EEG042'], "block13": ['EEG008','EEG033','EEG038'], "block14": ['EEG008'],"block15": ['EEG008'], "block16": ['EEG008','EEG045'], "block17": ['EEG008'],"block18": ['EEG008','EEG045'],"block19": ['EEG038','EEG045'],"block20": ['EEG008']},  
             "session3": {"block21": ['EEG008'], "block22": ['EEG008','EEG024','EEG041','EEG044'], "block23": ['EEG008','EEG041','EEG044'],"block24": ['EEG008','EEG041','EEG045'], "block25": ['EEG008','EEG045'], "block26": ['EEG008','EEG041','EEG045'],"block27": ['EEG008'],"block28": ['EEG008','EEG020'], "block29": ['EEG008'],"block30": ['EEG008','EEG020','EEG045']}},
  "ank24":  {"session1": {"block01": ['EEG007','EEG059'], "block02": ['EEG059'], "block03": ['EEG001','EEG002','EEG020','EEG059'],"block04": ['EEG007','EEG059'], "block05": ['EEG007','EEG059'], "block06": ['EEG007','EEG059'],"block07": ['EEG007','EEG058','EEG059'],"block08": ['EEG007','EEG047','EEG059'], "block09": ['EEG007','EEG047','EEG059'],"block10": ['EEG007','EEG047','EEG059']},
             "session2": {"block11": ['EEG007','EEG059'], "block12": ['EEG007','EEG010','EEG059'], "block13": ['EEG007','EEG059'], "block14": ['EEG007','EEG010','EEG059'],"block15": ['EEG007','EEG059'], "block16": ['EEG007','EEG010','EEG059'], "block17": ['EEG007','EEG059'],"block18": ['EEG007','EEG010','EEG059'],"block19": ['EEG007','EEG010','EEG059'],"block20": ['EEG007','EEG010','EEG059']},  
             "session3": {"block21": ['EEG059'], "block22": ['EEG007','EEG012','EEG059'], "block23": ['EEG007','EEG012','EEG059'],"block24": ['EEG007','EEG059'], "block25": ['EEG007','EEG012','EEG059'], "block26": ['EEG007','EEG012','EEG059'],"block27": ['EEG007','EEG012','EEG059'],"block28": ['EEG007','EEG010','EEG059'], "block29": ['EEG007','EEG059'],"block30": ['EEG007','EEG012','EEG059']}},
  "dja01":  {"session1": {"block01": ['EEG008','EEG012','EEG021','EEG020','EEG033','EEG037','EEG058'], "block02": ['EEG008','EEG012','EEG021','EEG033','EEG037','EEG058'], "block03": ['EEG008','EEG012','EEG021','EEG033','EEG037','EEG058'],"block04": ['EEG008','EEG012','EEG033','EEG037','EEG045','EEG058'], "block05": ['EEG008','EEG012','EEG021','EEG033','EEG037','EEG058'], "block06": ['EEG008','EEG012','EEG021','EEG025','EEG033','EEG058'],"block07": ['EEG008','EEG012','EEG021','EEG025','EEG033','EEG037','EEG045','EEG058','EEG060'],"block08": ['EEG008','EEG012','EEG021','EEG025','EEG033','EEG037','EEG045','EEG058'], "block09": ['EEG008','EEG012','EEG033','EEG037','EEG058'],"block10": ['EEG008','EEG012','EEG033','EEG037']},
             "session2": {"block11": ['EEG001','EEG008','EEG021','EEG033','EEG037','EEG058'], "block12": ['EEG008','EEG012','EEG021','EEG033','EEG037','EEG038','EEG045','EEG058'], "block13": ['EEG008','EEG012','EEG021','EEG025','EEG033','EEG037','EEG038','EEG045','EEG058'], "block14": ['EEG001','EEG008','EEG021','EEG033','EEG037','EEG038','EEG045','EEG058'],"block15": ['EEG001','EEG008','EEG021','EEG033','EEG037','EEG038','EEG045','EEG058'], "block16": ['EEG001','EEG008','EEG021','EEG033','EEG037','EEG038','EEG045','EEG058'], "block17": ['EEG001','EEG008','EEG021','EEG033','EEG037','EEG038','EEG045','EEG058'],"block18": ['EEG001','EEG008','EEG021','EEG033','EEG037','EEG038','EEG045','EEG058'],"block19": ['EEG001','EEG008','EEG021','EEG033','EEG037','EEG038','EEG045','EEG058'],"block20": ['EEG001','EEG008','EEG021','EEG033','EEG037','EEG038','EEG045','EEG058']},  
             "session3": {"block21": ['EEG001','EEG008','EEG021','EEG033','EEG045','EEG058','EEG004','EEG042'], "block22": ['EEG008','EEG021','EEG024','EEG033','EEG045','EEG058','EEG004','EEG042'], "block23": ['EEG008','EEG021','EEG024','EEG033','EEG037','EEG045','EEG058','EEG004','EEG042'],"block24": ['EEG008','EEG021','EEG024','EEG033','EEG045','EEG058','EEG004','EEG042'], "block25": ['EEG008','EEG021','EEG024','EEG033','EEG037','EEG045','EEG058','EEG004','EEG042'], "block26": ['EEG008','EEG021','EEG024','EEG033','EEG045','EEG058','EEG004','EEG042'],"block27": ['EEG001','EEG008','EEG021','EEG024','EEG033','EEG045','EEG058','EEG004','EEG042'],"block28": ['EEG008','EEG021','EEG024','EEG033','EEG045','EEG058','EEG004','EEG042'], "block29": ['EEG008','EEG021','EEG024','EEG033','EEG045','EEG058'],"block30": ['EEG008','EEG021','EEG024','EEG033','EEG045','EEG058']}},
  "fmn28":  {"session1": {"block01": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG024', 'EEG058', 'EEG059', 'EEG061', 'EEG064'], "block02": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG059', 'EEG064'], "block03": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG059', 'EEG064'],"block04": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG059', 'EEG064'], "block05": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG024', 'EEG059', 'EEG064'], "block06": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG059', 'EEG064'],"block07": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG059', 'EEG064'],"block08": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG059', 'EEG064'], "block09": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG058', 'EEG059', 'EEG064'],"block10": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG048', 'EEG059', 'EEG064']},
             "session2": {"block11": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG024', 'EEG059', 'EEG064'], "block12": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG024', 'EEG059', 'EEG064'], "block13": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG024', 'EEG059', 'EEG064'], "block14": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG024', 'EEG059', 'EEG064'],"block15": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG024', 'EEG059', 'EEG064'], "block16": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG024', 'EEG059', 'EEG064'], "block17": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG024', 'EEG059', 'EEG064'],"block18": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG024', 'EEG059', 'EEG064'],"block19": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG024', 'EEG059', 'EEG064'], "block20": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG024', 'EEG059', 'EEG064']},  
             "session3": {"block21": ['EEG001', 'EEG002','EEG007', 'EEG008', 'EEG012', 'EEG058', 'EEG059', 'EEG064'], "block22": ['EEG001', 'EEG002','EEG007', 'EEG008', 'EEG012', 'EEG058', 'EEG059', 'EEG064']}},
  "omr03":  {"session1": {"block01": ['EEG005','EEG040','EEG052'], "block02": ['EEG005','EEG007', 'EEG033', 'EEG037', 'EEG041'], "block03": ['EEG033', 'EEG037', 'EEG041'],"block04": ['EEG012', 'EEG033', 'EEG037', 'EEG041', 'EEG058', 'EEG059', 'EEG060'], "block05": ['EEG007', 'EEG008', 'EEG024', 'EEG033', 'EEG037', 'EEG041', 'EEG058', 'EEG059'], "block06": ['EEG007', 'EEG021', 'EEG024', 'EEG033', 'EEG037', 'EEG038', 'EEG041', 'EEG048', 'EEG058', 'EEG059', 'EEG060'],"block07": ['ok'],"block08": ['EEG058', 'EEG059','EEG060']},
             "session2": {"block09": ['EEG007', 'EEG033', 'EEG037', 'EEG041'], "block10": ['EEG002', 'EEG007', 'EEG033', 'EEG037', 'EEG041', 'EEG058', 'EEG059'], "block11": ['EEG002'], "block12": ['EEG007', 'EEG058', 'EEG059'], "block13": ['EEG007', 'EEG021', 'EEG024', 'EEG033', 'EEG037', 'EEG041', 'EEG058', 'EEG059'], "block14": ['EEG007', 'EEG058', 'EEG059'],"block15": ['EEG007', 'EEG033', 'EEG037', 'EEG041', 'EEG042', 'EEG059'], "block16": ['EEG007', 'EEG033', 'EEG037', 'EEG041','EEG050'], "block17": ['EEG007', 'EEG033', 'EEG037', 'EEG041', 'EEG058', 'EEG059']},  
             "session3": {"block18": ['EEG007', 'EEG033', 'EEG037', 'EEG041'],"block19": ['EEG007', 'EEG033', 'EEG037', 'EEG041','EEG035'], "block20": ['EEG007', 'EEG033', 'EEG037', 'EEG041'], "block21": ['EEG007', 'EEG033', 'EEG037', 'EEG041'], "block22": ['EEG007', 'EEG033', 'EEG037', 'EEG041', 'EEG048', 'EEG058', 'EEG059'], "block23": ['EEG007', 'EEG008', 'EEG021', 'EEG024', 'EEG033', 'EEG037', 'EEG041', 'EEG058', 'EEG059', 'EEG060'],"block24": ['EEG007', 'EEG021', 'EEG033', 'EEG037', 'EEG041', 'EEG058', 'EEG059'], "block25": ['EEG007', 'EEG033', 'EEG037', 'EEG041', 'EEG058', 'EEG059', 'EEG060'], "block26": ['EEG007', 'EEG033', 'EEG037', 'EEG041', 'EEG059']}},
  "amy20":  {"session1": {"block01": ['EEG050'], "block02": ['ok'], "block03": ['ok'],"block04": ['EEG050'], "block05": ['ok'], "block06": ['ok'],"block07": ['EEG050'],"block08": ['ok'], "block09": ['ok']},
             "session2": {"block10": ['ok'], "block11": ['ok'], "block12": ['ok'], "block13": ['ok'], "block14": ['ok'],"block15": ['ok'], "block16": ['ok'], "block17": ['ok'],"block18": ['ok'],"block19": ['ok']},  
             "session3": {"block20": ['EEG003'], "block21": ['EEG003'], "block22": ['EEG003'], "block23": ['ok'],"block24": ['EEG003'], "block25": ['EEG003'], "block26": ['ok'],"block27": ['ok'],"block28": ['ok'], "block29": ['ok']}},
  "ski23":  {"session1": {"block01": ['EEG008','EEG021','EEG033','EEG037','EEG058'], "block02": ['EEG008','EEG024','EEG033','EEG037','EEG058'], "block03": ['EEG008','EEG021','EEG033','EEG037','EEG058'],"block04": ['EEG008','EEG021','EEG033','EEG037','EEG058'], "block05": ['EEG008','EEG021','EEG033','EEG037','EEG058'], "block06": ['EEG008','EEG021','EEG033','EEG037','EEG058'],"block07": ['EEG008','EEG024','EEG033','EEG037','EEG058'],"block08": ['EEG008','EEG021','EEG033','EEG037','EEG058'], "block09": ['EEG008','EEG021','EEG024','EEG033','EEG037','EEG058'],"block10": ['EEG008','EEG021','EEG024','EEG033','EEG037','EEG058']},
             "session2": {"block11": ['EEG010','EEG064'], "block12": ['EEG010','EEG064'], "block13": ['EEG010','EEG064'], "block14": ['EEG064'],"block15": ['EEG064'], "block16": ['EEG064'], "block17": ['EEG010','EEG064'],"block18": ['EEG064'],"block19": ['EEG064'],"block20": ['EEG064']},  
             "session3": {"block21": ['ok'], "block22": ['ok'], "block23": ['ok'],"block24": ['ok'], "block25": ['ok'], "block26": ['ok'],"block27": ['ok'],"block28": ['ok'], "block29": ['ok'],"block30": ['ok']}},
 "jry29":  {"session1": {"block01": ['ok'], "block02": ['ok'], "block03": ['EEG007','EEG024','EEG041','EEG059'],"block04": ['EEG007','EEG024','EEG041','EEG059'], "block05": ['EEG007','EEG024','EEG041','EEG059'], "block06": ['EEG007','EEG024','EEG041','EEG059'],"block07": ['EEG007','EEG024','EEG041','EEG059'],"block08": ['EEG007','EEG024','EEG041','EEG059'], "block09": ['EEG007','EEG024','EEG041','EEG059'],"block10": ['EEG007','EEG024','EEG041','EEG059']},
            "session2": {"block10": ['ok'], "block11": ['EEG007'], "block12": ['EEG007','EEG024','EEG059'], "block13": ['EEG007','EEG024','EEG059'], "block14": ['EEG007'],"block15": ['EEG007','EEG024','EEG059'], "block16": ['ok'], "block17": ['EEG024'],"block18": ['EEG007','EEG048','EEG059'],"block19": ['ok'],"block20": ['EEG007','EEG048','EEG059']},  
            "session3": {"block21": ['EEG007','EEG024','EEG059'], "block22": ['EEG007','EEG024','EEG059'], "block23": ['EEG007','EEG024'],"block24": ['EEG007','EEG024','EEG059'], "block25": ['EEG007','EEG024','EEG059'], "block26": ['EEG007','EEG024','EEG059'],"block27": ['EEG007','EEG024','EEG059'],"block28": ['EEG007','EEG024','EEG059'], "block29": ['EEG007','EEG024','EEG059'],"block30": ['EEG007','EEG024','EEG059'],}},
 "jpa10":  {"session1": {"block01": ['EEG001','EEG012','EEG033','EEG037','EEG041','EEG048','EEG060','EEG064'], "block02": ['EEG001','EEG007','EEG033','EEG037','EEG058','EEG060'], "block03": ['EEG001','EEG012','EEG033','EEG037','EEG058','EEG060'],"block04": ['EEG001','EEG007','EEG012','EEG033','EEG037','EEG045','EEG048','EEG058'], "block05": ['EEG001','EEG020','EEG033','EEG037','EEG058','EEG060'], "block06": ['EEG001','EEG007','EEG012','EEG033','EEG037','EEG058','EEG060'],"block07": ['EEG001','EEG007','EEG033','EEG037','EEG045','EEG058','EEG060'],"block08": ['EEG001','EEG007','EEG033','EEG037','EEG058','EEG060'], "block09": ['EEG001','EEG007','EEG012','EEG033','EEG037','EEG058','EEG060'],"block10": ['EEG001','EEG007','EEG033','EEG037','EEG058','EEG060']},
            "session2": {"block11": ['EEG001','EEG007','EEG021','EEG024','EEG033','EEG037','EEG058'], "block12": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058'], "block13": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058'], "block14": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058'],"block15": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058'], "block16": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058'], "block17": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058'],"block18": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058'],"block19": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058'],"block20": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058']},  
            "session3": {"block21": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058','EEG061','EEG064'], "block22": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058','EEG061','EEG064'], "block23": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058','EEG061','EEG064'],"block24": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058','EEG061','EEG064'], "block25": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058','EEG061','EEG064'], "block26": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058','EEG061','EEG064'],"block27": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058','EEG061','EEG064'],"block28": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058','EEG061','EEG064'], "block29": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058','EEG061','EEG064'],"block30": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058','EEG061','EEG064'],}},
 "epa14":  {"session1": {"block01": ['EEG012','EEG061'], "block02": ['EEG012','EEG061'], "block03": ['EEG012','EEG061'],"block04": ['EEG007','EEG012','EEG061'], "block05": ['EEG012','EEG061'], "block06": ['EEG012','EEG059','EEG061'],"block07": ['EEG012','EEG059'],"block08": ['EEG012','EEG059','EEG061'], "block09": ['EEG012','EEG059','EEG061'],"block10": ['EEG012','EEG059']},
            "session2": {"block11": ['EEG001','EEG033'], "block12": ['EEG001','EEG033'], "block13": ['EEG033'], "block14": ['EEG001','EEG033'],"block15": ['EEG001','EEG033'], "block16": ['EEG001','EEG033'], "block17": ['EEG001','EEG033'],"block18": ['EEG001','EEG033'],"block19": ['EEG001','EEG033'],"block20": ['EEG001','EEG033']},  
            "session3": {"block21": ['EEG012','EEG061'], "block22": ['EEG012','EEG061'], "block23": ['EEG012','EEG059','EEG061'],"block24": ['EEG001','EEG012','EEG061'], "block25": ['EEG001','EEG012','EEG061'], "block26": ['EEG012','EEG061'],"block27": ['EEG012','EEG061'],"block28": ['EEG001','EEG012','EEG059','EEG061'], "block29": ['EEG007','EEG012','EEG048','EEG059','EEG061'],"block30": ['EEG012','EEG061']}},
 "crr22":  {"session1": {"block01": ['EEG012','EEG033'], "block02": ['EEG012','EEG033'], "block03": ['EEG012','EEG033'],"block04": ['EEG012','EEG033'], "block05": ['EEG012','EEG033'], "block06": ['EEG012','EEG033','EEG037'],"block07": ['EEG012','EEG033','EEG037'],"block08": ['EEG012','EEG033','EEG037'], "block09": ['EEG012','EEG033','EEG037']},
            "session2": {"block10": ['EEG012','EEG059'],"block11": ['EEG003','EEG012','EEG033'], "block12": ['EEG012','EEG033','EEG038'], "block13": ['EEG012','EEG033','EEG059'], "block14": ['EEG012','EEG061'],"block15": ['EEG012','EEG059'], "block16": ['EEG012','EEG033','EEG061'], "block17": ['EEG012','EEG033'],"block18": ['EEG012','EEG033','EEG038'],"block19": ['EEG012','EEG033'],"block20": ['EEG012','EEG033','EEG061']},  
            "session3": {"block21": ['EEG001','EEG002','EEG012','EEG033','EEG037','EEG038','EEG058'], "block22": ['EEG003','EEG012','EEG033','EEG038','EEG050','EEG058'], "block23": ['EEG003','EEG012','EEG033'],"block24": ['EEG003','EEG012','EEG033','EEG050','EEG064'], "block25": ['EEG002','EEG012','EEG033'], "block26": ['EEG002','EEG012','EEG033','EEG045','EEG058'],"block27": ['EEG002','EEG012','EEG033','EEG045','EEG058'],"block28": ['EEG002','EEG012','EEG033','EEG045','EEG058'], "block29": ['EEG002','EEG012','EEG033','EEG050'],"block30": ['EEG002','EEG012','EEG033']}},
 "jyg27":  {"session1": {"block01": ['EEG012'], "block02": ['EEG007','EEG012'], "block03": ['EEG008','EEG012','EEG059'],"block04": ['EEG007','EEG012','EEG059'], "block05": ['EEG007','EEG012','EEG059'], "block06": ['EEG007','EEG012','EEG059'],"block07": ['EEG007','EEG012','EEG059']},
            "session2": {"block08": ['ok'], "block09": ['EEG061'],"block10": ['EEG041'], "block11": ['EEG041'], "block12": ['EEG012'], "block13": ['EEG012'], "block14": ['EEG012'],"block15": ['EEG012'], "block16": ['ok'], "block17": ['EEG007','EEG024']},  
            "session3": {"block18": ['EEG012','EEG061'],"block19": ['EEG012','EEG061'],"block20": ['EEG012','EEG061'],"block21": ['EEG012','EEG013','EEG061'], "block22": ['EEG012','EEG013','EEG061'], "block23": ['EEG012','EEG013','EEG061'],"block24": ['EEG012','EEG059','EEG061'], "block25": ['EEG012','EEG061'], "block26": ['EEG012','EEG061'],"block27": ['EEG012','EEG061']}},
 "bbi29":  {"session1": {"block01": ['EEG020','EEG033'], "block02": ['EEG020'], "block03": ['EEG020','EEG033','EEG037','EEG041'],"block04": ['EEG020','EEG033','EEG037','EEG041'], "block05": ['EEG020'], "block06": ['EEG020','EEG033','EEG037','EEG041'],"block07": ['EEG020','EEG033','EEG037','EEG041'],"block08": ['EEG020','EEG036'], "block09": ['EEG020'],"block10": ['EEG020','EEG037']},
            "session2": {"block11": ['EEG020','EEG033','EEG038'], "block12": ['EEG038','EEG033','EEG044'], "block13": ['EEG038','EEG033','EEG058'], "block14": ['EEG038','EEG033','EEG058'],"block15": ['EEG003','EEG033','EEG038'], "block16": ['EEG024','EEG033','EEG044'], "block17": ['EEG024','EEG044'],"block18": ['EEG020','EEG033'],"block19": ['EEG020','EEG033'],"block20": ['EEG020','EEG033','EEG037','EEG038','EEG041']},  
            "session3": {"block21": ['EEG020','EEG034','EEG038'], "block22": ['EEG020','EEG033','EEG038'], "block23": ['EEG033'],"block24": ['EEG020','EEG034','EEG038'], "block25": ['EEG020','EEG033','EEG038'], "block26": ['EEG020'],"block27": ['EEG020','EEG033','EEG038'],"block28": ['EEG020','EEG033','EEG034','EEG038'], "block29": ['EEG020','EEG033','EEG034','EEG038'],"block30": ['EEG020','EEG033','EEG034','EEG038']}},
 "ece24":  {"session1": {"block01": ['EEG008','EEG018','EEG033','EEG038','EEG060'], "block02": ['EEG019','EEG018','EEG032','EEG060'], "block03": ['EEG018','EEG019','EEG032','EEG033','EEG060','EEG064'],"block04": ['EEG018','EEG019','EEG032','EEG033','EEG060','EEG064'], "block05": ['EEG018','EEG019','EEG032','EEG033','EEG060','EEG064'], "block06": ['EEG018','EEG019','EEG032','EEG033','EEG060','EEG064'],"block07": ['EEG018','EEG019','EEG032','EEG033','EEG060','EEG064'],"block08": ['EEG018','EEG019','EEG032','EEG033','EEG060','EEG064'], "block09": ['EEG018','EEG019','EEG032','EEG033','EEG060','EEG064'],"block10": ['EEG018','EEG019','EEG032','EEG033','EEG060','EEG064']},
            "session2": {"block11": ['EEG018','EEG019','EEG032','EEG048','EEG060'], "block12": ['EEG018','EEG019','EEG032','EEG048','EEG060','EEG024','EEG059'], "block13": ['EEG018','EEG019','EEG032','EEG048','EEG060'], "block14": ['EEG018','EEG019','EEG032','EEG048','EEG060'],"block15": ['EEG018','EEG019','EEG032','EEG064','EEG060'], "block16": ['EEG018','EEG019','EEG032','EEG048','EEG060'], "block17": ['EEG018','EEG019','EEG032','EEG048','EEG060'],"block18": ['EEG018','EEG019','EEG032','EEG048','EEG060'],"block19": ['EEG018','EEG019','EEG024','EEG044','EEG060'],"block20": ['EEG018','EEG019','EEG032','EEG044','EEG060']},  
            "session3": {"block21": ['EEG018','EEG019','EEG032','EEG033','EEG064'], "block22": ['EEG018','EEG019','EEG032','EEG033','EEG064'], "block23": ['EEG011','EEG024','EEG032','EEG044','EEG060'],"block24": ['EEG018','EEG019','EEG033','EEG058','EEG060','EEG064'], "block25": ['EEG008','EEG018','EEG033','EEG048','EEG060'], "block26": ['EEG018','EEG019','EEG032','EEG060'],"block27": ['EEG008','EEG018','EEG024'],"block28": ['EEG018','EEG033'], "block29": ['EEG008','EEG018','EEG048','EEG060'],"block30": ['EEG008','EEG018','EEG048','EEG060']}},
 "awa19":  {"session1": {"block01": ['EEG012','EEG061'], "block02": ['EEG012','EEG061'], "block03": ['EEG012','EEG061'],"block04": ['EEG012','EEG061'], "block05": ['EEG012','EEG024','EEG048','EEG059','EEg061'], "block06": ['EEG012','EEG061'],"block07": ['EEG012','EEG061'],"block08": ['EEG012','EEG061'], "block09": ['EEG012','EEG061']},
            "session2": {"block10": ['EEG033','EEG037'],"block11": ['EEG012','EEG019'], "block12": ['EEG012','EEG019','EEG032'], "block13": ['EEG012','EEG019','EEG037','EEG041'], "block14": ['EEG012','EEG019','EEG037'],"block15": ['EEG012','EEG019','EEG037'], "block16": ['EEG012','EEG019'], "block17": ['EEG012','EEG019'],"block18": ['EEG012'],"block19": ['EEG012']},  
            "session3": {"block20": ['EEG002','EEG012','EEG037','EEG061'],"block21": ['EEG002','EEG012','EEG037'], "block22": ['EEG002','EEG012','EEG037','EEG041'], "block23": ['EEG002','EEG012','EEG037','EEG041'],"block24": ['EEG002','EEG012','EEG037','EEG041'], "block25": ['EEG012','EEG037'], "block26": ['EEG012','EEG037'],"block27": ['EEG012','EEG037'],"block28": ['EEG002','EEG012','EEG037','EEG048'], "block29": ['EEG002','EEG012','EEG037']}},
 "hay06":  {"session1": {"block01": ['EEG002','EEG020'], "block02": ['EEG002','EEG007','EEG020','EEG024','EEG041','EEG044'], "block03": ['EEG002','EEG007','EEG020','EEG041','EEG044'],"block04": ['EEG002','EEG008','EEG020','EEG048'], "block05": ['EEG002','EEG007','EEG020','EEG041'], "block06": ['EEG002','EEG007','EEG020'],"block07": ['EEG002','EEG007','EEG020'],"block08": ['EEG002','EEG007','EEG020'], "block09": ['EEG002','EEG007','EEG041'],"block10": ['EEG002','EEG007','EEG020','EEG048']},
            "session2": {"block11": ['EEG002','EEG008','EEG020'], "block12": ['EEG002','EEG020'], "block13": ['EEG002','EEG008','EEG020'], "block14": ['EEG002','EEG020'],"block15": ['EEG002','EEG008','EEG020'], "block16": ['EEG002','EEG008','EEG020'], "block17": [],"block18": ['EEG002'],"block19": [],"block20": []},  
            "session3": {"block21": ['EEG002','EEG020','EEG024'], "block22": [], "block23": ['EEG041'],"block24": [], "block25": ['EEG020'], "block26": ['EEG008','EEG012','EEG024'],"block27": ['EEG002','EEG007','EEG041'],"block28": ['EEG002','EEG012'], "block29": ['EEG002','EEG012'],"block30": ['EEG008']}},
 "mca10":  {"session1": {"block01": ['EEG003','EEG004','EEG008','EEG009','EEG021','EEG024','EEG025','EEG036','EEG038','EEG042','EEG045'], "block02": ['EEG003','EEG004','EEG008','EEG009','EEG021','EEG024','EEG025','EEG036','EEG038','EEG042','EEG045','EEG058'], "block03": ['EEG004','EEG021','EEG024','EEG036','EEG058'],"block04": ['EEG004','EEG021','EEG021','EEG024','EEG036','EEG058'], "block05": ['EEG004','EEG021','EEG021','EEG024','EEG036','EEG058'], "block06": ['EEG004','EEG021','EEG021','EEG024','EEG036','EEG058'],"block07": ['EEG004','EEG021','EEG021','EEG024','EEG036','EEG058'],"block08": ['EEG004','EEG021','EEG021','EEG024','EEG036','EEG058'], "block09": ['EEG004','EEG021','EEG021','EEG024','EEG036','EEG058'],"block10": ['EEG004','EEG021','EEG021','EEG024','EEG036','EEG058']},
            "session2": {"block11": ['EEG036'], "block12": ['EEG033','EEG036','EEG037','EEG038','EEG041'], "block13": ['EEG036','EEG059'], "block14": ['EEG036'],"block15": ['EEG033','EEG036','EEG037','EEG038'], "block16": ['EEG033','EEG036','EEG037','EEG038','EEG058'], "block17": ['EEG036'],"block18": ['EEG033','EEG036','EEG037','EEG038'],"block19": ['EEG036','EEG058','EEG059'],"block20": ['EEG036','EEG058','EEG059']}},
 "hky23":  {"session1": {"block01": ['EEG010','EEG033'], "block02": ['EEG010','EEG033'], "block03": ['EEG001','EEG002','EEG010','EEG033'],"block04": ['EEG001','EEG002','EEG010','EEG033'], "block05": ['EEG001','EEG002','EEG010','EEG033'], "block06": ['EEG010','EEG033'],"block07": ['EEG010','EEG033'],"block08": ['EEG010','EEG033'], "block09": ['EEG001','EEG002','EEG010','EEG033'],"block10": ['EEG001','EEG002','EEG010','EEG033']},
            "session2": {"block11": ['EEG010','EEG033'], "block12": ['EEG001','EEG010','EEG033'], "block13": ['EEG001','EEG010','EEG033'], "block14": ['EEG001','EEG010','EEG033'],"block15": ['EEG001','EEG010','EEG033'], "block16": ['EEG001','EEG010','EEG012','EEG033'], "block17": ['EEG001','EEG010','EEG021','EEG033'],"block18": ['EEG001','EEG010','EEG021','EEG033'],"block19": ['EEG001','EEG010','EEG011','EEG033'],"block20": ['EEG001','EEG010','EEG011','EEG021','EEG033']}},
 }





'''
NOTES:
fha01, first seconds in block22 need to be deleted
mtr13, first seconds in block29 need to be deleted

done in annotation and EEG interpolation code

hwh21, highfreq noise, may need to remove an ICA related to this!!!

some EEG blocks in jmn22 session 1 may be lost
'''
#%% Re-do
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


#%% load data
for subject_id in subjects_list:
    print(subject_id)
    #sub_idx = 8 #  
    #subject_id =subjects_list[sub_idx]
    print('Doing subject ' + subject_id + '\nSessions')
    subject_path = op.join(data_dir,subject_id,input_folder)
    session_folders = os.listdir(subject_path)
    sessions =  [x for x in session_folders if x.startswith('2')]
    print(sessions)
    sessions.sort()
    
    #%
    for idx,session in enumerate(sessions):
        if redo_list[subject_id[5:]]['session' + str(idx+1)] == False:
            continue
        # idx = 2
        # session = sessions[idx]
        print(session)
        save_dir = op.join(data_dir,subject_id,output_folder,session)  
        # create folder if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # session_idx = 0 # loop through sessions
        # session = sessions[session_idx]
        file_dir = op.join(subject_path,session)
        blocks =  os.listdir(file_dir)
        blocks = [x for x in blocks if x.startswith('B')]
        blocks.sort()
        print(blocks)
        # loop trhough the blocks annotate bad channels, compute PSD and check trial events
        # max filter
        # concatenate after maxfilter
        for block in blocks:
            print('\n\n\n======================================================\n' + block + '\n======================================================\n')

            save_annot_file = op.join(save_dir,block[:8] + '_saved-annotations.csv')
            # if os.path.isfile(save_annot_file):
            #     print('\n\nThis block has been done already!\nContinuing with next block\n\n')
            #     continue
            raw_file = op.join(file_dir,block)
            raw = mne.io.read_raw_fif(raw_file,preload = True)            
            # if len(raw.info['bads']) == 64:
            #     continue
            # interpolate bad EEG channels
            channels_to_interpolate = bad_eeg_channels[subject_id[5:]]['session'+str(idx+1)]['block'+block[6:8]]
            raw.info['bads'] = []
            raw.info['bads'] = channels_to_interpolate# bad_eeg_channels[subject_id[5:]]['session'+str(idx+1)]['block'+block[6:8]]
            
            if channels_to_interpolate == ['ok']: # if the components are empty, skip this iteration of the for loop
                print('All EEG channels looking good/n there is no need for interpolation')
                raw.info['bads'] = []
            elif channels_to_interpolate == ['lost']:
                print('no EEG data for this block')
                raw.info['bads'] =  ['EEG001','EEG002','EEG003','EEG004','EEG005','EEG006','EEG007','EEG008','EEG009','EEG010','EEG011','EEG012','EEG013',
                 'EEG014','EEG015','EEG016','EEG017','EEG018','EEG019','EEG020','EEG021','EEG022','EEG023','EEG024','EEG025','EEG026','EEG027','EEG028',
                 'EEG029','EEG030','EEG031','EEG032','EEG033','EEG034','EEG035','EEG036','EEG037','EEG038','EEG039','EEG040','EEG041','EEG042','EEG043',
                 'EEG044','EEG045','EEG046','EEG047','EEG048','EEG049','EEG050','EEG051','EEG052','EEG053','EEG054','EEG055','EEG056','EEG057','EEG058',
                 'EEG059','EEG060','EEG061','EEG062','EEG063','EEG064']
            else:        
                print('\n======================================================\n Interpolating bad EEG channels')
                raw.interpolate_bads(reset_bads=True)
            
            # cut bad segment for subject
            if subject_id == 'S003_mtr13' and block == 'Block_29_tsss_mc.fif':
                print('cutting initial bad segment')
                raw.crop(tmin=23) #<- remove 23 first seconds  # example 10 seconds form start and end raw.crop(tmin=10, tmax=raw_filt.times[-1] - 10) 
                #raw.plot()
            
            if subject_id == 'S009_fha01' and block == 'Block_22_tsss_mc.fif':
                print('cutting initial bad segment')
                raw.crop(tmin=25) #<- remove 25 first seconds  # example 10 seconds form start and end raw.crop(tmin=10, tmax=raw_filt.times[-1] - 10) 
                #raw.plot()
            
            # crop end of file fpr bsr27
            if subject_id == 'dss19' and block == 'Block_11_tsss.fif':
                print('cutting final bad segment')
                raw.crop(tmax=723)    
            
            print('\n======================================================\n Annotating artefacts')
            # find blinks
            if subject_id == 'S001_bsr27' and block == 'Block_01_tsss.fif':
                print('true')
                eog_events = mne.preprocessing.find_eog_events(raw, ch_name= 'EEG001')  #'BIO002' 'EEG001 for ami28 session 1, no eog electrodes 
            else:
                eog_events = mne.preprocessing.find_eog_events(raw, ch_name= veog[subject_id[5:]]['session'+str(idx+1)])
                        
            n_blinks = len(eog_events)
            onset = eog_events[:, 0] / raw.info['sfreq'] - 0.25  
            onset -= raw._first_time  
            duration = np.repeat(0.5, n_blinks)  
            description = ['blink'] * n_blinks
            annotations_blink = mne.Annotations(onset,duration,description)#orig_time=raw.info['meas_date']
            raw.set_annotations(annotations_blink)
            #find musccle
            threshold_muscle = 10 
            annotations_muscle, scores_muscle = annotate_muscle_zscore(raw, ch_type="mag", threshold=threshold_muscle, min_length_good=0.2,filter_freq=[110, 140])
            
            # add to add the blik annotations to raw, then extract them and add the extracted annotation together with muscle
            # otherwise, I had an error with orig_time
            annotations_blink = raw.annotations
            raw.set_annotations(annotations_blink + annotations_muscle)
            
            #raw.set_annotations(annotations_blink+annotations_muscle);
            # fig, ax = plt.subplots()
            # ax.plot(raw.times, scores_muscle)
            # ax.axhline(y=threshold_muscle, color='r')
            # ax.set(xlabel='time, (s)', ylabel='zscore', title='Muscle activity')
            
            # add annotation
            #raw.set_annotations(annotations_blink+annotations_muscle);
            
            # check 
            #raw.plot(start=50)
            #plt.close()
            
            # save annotations
            raw.annotations.save(save_annot_file, overwrite=True)
            
            events_from_annot, event_dict = mne.events_from_annotations(raw)
            #annot_from_file = mne.read_annotations('saved-annotations.csv')
            #print(annot_from_file)
            raw.save(raw_file,overwrite=True) 
        
    





#%%
