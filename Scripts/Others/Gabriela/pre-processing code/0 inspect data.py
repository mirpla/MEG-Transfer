

#%%
# import sys
import os.path as op
import numpy as np
import mne
import os
import re
#import matplotlib.pyplot as plt

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt') # so MNE plots are in a pop-up window, not inline

# %matplotlib qt
'''
# method to see all of the fields available in info
raw.info.keys()

# Raw is a Python class, and any instance of the Raw class (such as our raw data here) is a Python object. 
# Any Python object has an atrribute __dict__, and this attribute contains a dictionary of all of the object’s attributes, 
# with keys being the attribute names, and values being the information stored for that attribute.
raw.__dict__

# check data and size
print(type(raw._data))
print(raw._data.shape)

# scan duration

scan_durn = raw._data.shape[1] / raw.info['sfreq']
print('Duration of EEG recording = ', scan_durn, 's, or', scan_durn / 60, 'min.')
'''

#%%

 
def check_block_name(dir_list):
    blocks_list = []
    for block in dir_list:
        string = re.findall("[a-zA-Z]+", block)
        pattern = "^Block_[0-9]{2,2}.fif$"
        state = bool(re.match(pattern, block))
        
        if state and block not in blocks_list: 
                blocks_list.append(block)
        else:
            num = re.findall(r'\d+', block) # extract number from string
            newstring = string[0].capitalize()
            if num and newstring.startswith('B'):
                num = "%02d" % (int(num[0]),) # makes sure it is a two digit number
                if string[1] == 'fif':
                    new_name = 'Block_' + num + '.' + string[1]
                    if new_name not in blocks_list: 
                        blocks_list.append(new_name)
            else:
                print('File name removed from block list. Not a task file')
    return blocks_list


#%%
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

#%% Channel list
default_channel_list = ['BIO001', 'BIO002', 'BIO003' ,'EEG001', 'EEG002', 'EEG003', 'EEG004', 'EEG005', 'EEG006', 'EEG007', 'EEG008', 'EEG009', 'EEG010', 'EEG011', 'EEG012', 'EEG013', 'EEG014', 'EEG015', 'EEG016', 'EEG017', 'EEG018', 'EEG019', 'EEG020', 'EEG021', 'EEG022', 'EEG023', 'EEG024', 'EEG025', 'EEG026', 'EEG027', 'EEG028', 'EEG029', 'EEG030', 'EEG031', 'EEG032', 'EEG033', 'EEG034', 'EEG035', 'EEG036', 'EEG037', 'EEG038', 'EEG039', 'EEG040', 'EEG041', 'EEG042', 'EEG043', 'EEG044', 'EEG045', 'EEG046', 'EEG047', 'EEG048', 'EEG049', 'EEG050', 'EEG051', 'EEG052', 'EEG053', 'EEG054', 'EEG055', 'EEG056', 'EEG057', 'EEG058', 'EEG059', 'EEG060', 'EEG061', 'EEG062', 'EEG063', 'EEG064', 'MEG0111', 'MEG0112', 'MEG0113', 'MEG0121', 'MEG0122', 'MEG0123', 'MEG0131', 'MEG0132', 'MEG0133', 'MEG0141', 'MEG0142', 'MEG0143', 'MEG0211', 'MEG0212', 'MEG0213', 'MEG0221', 'MEG0222', 'MEG0223', 'MEG0231', 'MEG0232', 'MEG0233', 'MEG0241', 'MEG0242', 'MEG0243', 'MEG0311', 'MEG0312', 'MEG0313', 'MEG0321', 'MEG0322', 'MEG0323', 'MEG0331', 'MEG0332', 'MEG0333', 'MEG0341', 'MEG0342', 'MEG0343', 'MEG0411', 'MEG0412', 'MEG0413', 'MEG0421', 'MEG0422', 'MEG0423', 'MEG0431', 'MEG0432', 'MEG0433', 'MEG0441', 'MEG0442', 'MEG0443', 'MEG0511', 'MEG0512', 'MEG0513', 'MEG0521', 'MEG0522', 'MEG0523', 'MEG0531', 'MEG0532', 'MEG0533', 'MEG0541', 'MEG0542', 'MEG0543', 'MEG0611', 'MEG0612', 'MEG0613', 'MEG0621', 'MEG0622', 'MEG0623', 'MEG0631', 'MEG0632', 'MEG0633', 'MEG0641', 'MEG0642', 'MEG0643', 'MEG0711', 'MEG0712', 'MEG0713', 'MEG0721', 'MEG0722', 'MEG0723', 'MEG0731', 'MEG0732', 'MEG0733', 'MEG0741', 'MEG0742', 'MEG0743', 'MEG0811', 'MEG0812', 'MEG0813', 'MEG0821', 'MEG0822', 'MEG0823', 'MEG0911', 'MEG0912', 'MEG0913', 'MEG0921', 'MEG0922', 'MEG0923', 'MEG0931', 'MEG0932', 'MEG0933', 'MEG0941', 'MEG0942', 'MEG0943', 'MEG1011', 'MEG1012', 'MEG1013', 'MEG1021', 'MEG1022', 'MEG1023', 'MEG1031', 'MEG1032', 'MEG1033', 'MEG1041', 'MEG1042', 'MEG1043', 'MEG1111', 'MEG1112', 'MEG1113', 'MEG1121', 'MEG1122', 'MEG1123', 'MEG1131', 'MEG1132', 'MEG1133', 'MEG1141', 'MEG1142', 'MEG1143', 'MEG1211', 'MEG1212', 'MEG1213', 'MEG1221', 'MEG1222', 'MEG1223', 'MEG1231', 'MEG1232', 'MEG1233', 'MEG1241', 'MEG1242', 'MEG1243', 'MEG1311', 'MEG1312', 'MEG1313', 'MEG1321', 'MEG1322', 'MEG1323', 'MEG1331', 'MEG1332', 'MEG1333', 'MEG1341', 'MEG1342', 'MEG1343', 'MEG1411', 'MEG1412', 'MEG1413', 'MEG1421', 'MEG1422', 'MEG1423', 'MEG1431', 'MEG1432', 'MEG1433', 'MEG1441', 'MEG1442', 'MEG1443', 'MEG1511', 'MEG1512', 'MEG1513', 'MEG1521', 'MEG1522', 'MEG1523', 'MEG1531', 'MEG1532', 'MEG1533', 'MEG1541', 'MEG1542', 'MEG1543', 'MEG1611', 'MEG1612', 'MEG1613', 'MEG1621', 'MEG1622', 'MEG1623', 'MEG1631', 'MEG1632', 'MEG1633', 'MEG1641', 'MEG1642', 'MEG1643', 'MEG1711', 'MEG1712', 'MEG1713', 'MEG1721', 'MEG1722', 'MEG1723', 'MEG1731', 'MEG1732', 'MEG1733', 'MEG1741', 'MEG1742', 'MEG1743', 'MEG1811', 'MEG1812', 'MEG1813', 'MEG1821', 'MEG1822', 'MEG1823', 'MEG1831', 'MEG1832', 'MEG1833', 'MEG1841', 'MEG1842', 'MEG1843', 'MEG1911', 'MEG1912', 'MEG1913', 'MEG1921', 'MEG1922', 'MEG1923', 'MEG1931', 'MEG1932', 'MEG1933', 'MEG1941', 'MEG1942', 'MEG1943', 'MEG2011', 'MEG2012', 'MEG2013', 'MEG2021', 'MEG2022', 'MEG2023', 'MEG2031', 'MEG2032', 'MEG2033', 'MEG2041', 'MEG2042', 'MEG2043', 'MEG2111', 'MEG2112', 'MEG2113', 'MEG2121', 'MEG2122', 'MEG2123', 'MEG2131', 'MEG2132', 'MEG2133', 'MEG2141', 'MEG2142', 'MEG2143', 'MEG2211', 'MEG2212', 'MEG2213', 'MEG2221', 'MEG2222', 'MEG2223', 'MEG2231', 'MEG2232', 'MEG2233', 'MEG2241', 'MEG2242', 'MEG2243', 'MEG2311', 'MEG2312', 'MEG2313', 'MEG2321', 'MEG2322', 'MEG2323', 'MEG2331', 'MEG2332', 'MEG2333', 'MEG2341', 'MEG2342', 'MEG2343', 'MEG2411', 'MEG2412', 'MEG2413', 'MEG2421', 'MEG2422', 'MEG2423', 'MEG2431', 'MEG2432', 'MEG2433', 'MEG2441', 'MEG2442', 'MEG2443', 'MEG2511', 'MEG2512', 'MEG2513', 'MEG2521', 'MEG2522', 'MEG2523', 'MEG2531', 'MEG2532', 'MEG2533', 'MEG2541', 'MEG2542', 'MEG2543', 'MEG2611', 'MEG2612', 'MEG2613', 'MEG2621', 'MEG2622', 'MEG2623', 'MEG2631', 'MEG2632', 'MEG2633', 'MEG2641', 'MEG2642', 'MEG2643', 'MISC001', 'MISC002', 'STI001', 'STI002', 'STI003', 'STI004', 'STI005', 'STI006', 'STI007', 'STI008', 'STI009', 'STI010', 'STI011', 'STI012', 'STI013', 'STI014', 'STI015', 'STI016', 'STI101', 'STI102', 'SYS201']

#%% bad channels notes
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
  "ank24":  {"session1": ['MEG0133','MEG0922', 'MEG1243', 'MEG1042',  'MEG1813', 'MEG1432','MEG1633','MEG2332','MEG2433'], "session2":['MEG0133', 'MEG0922', 'MEG1243', 'MEG1042',  'MEG1813', 'MEG1432','MEG1633','MEG2332','MEG2433'], "session3": ['MEG1243', 'MEG1042',   'MEG2332']},
  "dja01":  {"session1": ['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433'], "session2":['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433'], "session3": ['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433']},
  "fmn28":  {"session1": ['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433'], "session2":['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433'], "session3": ['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433']},
  "omr03":  {"session1": ['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433'], "session2":['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433'], "session3": ['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433']},
  "amy20":  {"session1": ['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433'], "session2":['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433'], "session3": ['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433']},
  "ski23":  {"session1": ['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433'], "session2":['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433'], "session3": ['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433']},
  "jry29":  {"session1": ['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433'], "session2":['MEG0723','MEG1042', 'MEG1243', 'MEG2332', 'MEG2433'], "session3": ['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433']},
  "jpa10":  {"session1": ['MEG1042', 'MEG1243', 'MEG2332', 'MEG2433'], "session2":['MEG1042', 'MEG1243', 'MEG2332'], "session3": ['MEG1042', 'MEG1243', 'MEG2332']},
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



#%%
bad_eeg_channels  = {
  "bsr27":  {"session1": {"block01": ['EEG002'], "block02": ['EEG002'], "block03": ['ok'],"block04": ['ok'], "block05": ['EEG011'], "block06": ['ok'],"block07": ['ok']},
             "session2": {"block09": ['lost'], "block10": ['lost'], "block11": ['lost'],"block12": ['lost'], "block13": ['lost'], "block14": ['ok'], "block15": ['ok'], "block16": ['ok'],"block17": ['ok'],"block18": ['ok'], "block19": ['EEG007'], "block20": ['ok']}, 
             "session3": {"block21": ['ok'], "block22": ['EEG007','EEG033'], "block23": ['EEG007','EEG033'],"block24": ['EEG023'], "block25": ['ok'], "block26": ['ok'],"block27": ['ok'],"block28": ['ok'], "block29": ['ok'], "block30": ['ok']}},
  "dsa23":  {"session1": {"block01": ['EEG008','EEG037','EEG041','EEG059'], "block02": ['EEG037','EEG059'], "block03": ['EEG037','EEG059'],"block04": ['EEG037','EEG059'], "block05": ['EEG037','EEG059'], "block06": ['EEG037','EEG041','EEG059'],"block07": ['EEG037','EEG059'],"block08": ['EEG037','EEG059'], "block09": ['EEG037','EEG041','EEG059'], "block10": ['EEG037','EEG059'],"block11": ['EEG037','EEG059']},
             "session2": {"block12": ['EEG005'], "block13": ['ok'],"block14": ['EEG005','EEG037','EEG059'], "block15": ['EEG059','EEG037'], "block16": ['EEG037','EEG059','EEG061','EEG054','EEG041'],"block17": ['EEG037','EEG059','EEG054','EEG041']}, 
             "session3": {"block18": ['EEG061'], "block19": ['EEG059','EEG061'],"block20": ['EEG037','EEG059','EEG061'], "block21": ['EEG037','EEG059','EEG061'], "block22": ['EEG037','EEG059','EEG061'], "block23": ['EEG037','EEG059','EEG061'],"block24": ['EEG037','EEG059','EEG061'], "block25": ['lost'], "block26": ['EEG037','EEG059','EEG061'],"block27": ['EEG037','EEG061'],"block28": ['EEG037','EEG059','EEG061']}},
  "mtr13":  {"session1": {"block01": ['EEG012','EEG033'], "block02": ['EEG012','EEG033'], "block03": ['EEG012','EEG033'],"block04": ['EEG012','EEG033'], "block05": ['EEG012','EEG033'], "block06": ['EEG012'],"block07": ['EEG012'],"block08": ['EEG012'], "block09": ['EEG012'], "block10": ['EEG012'],"block11": ['ok']},
             "session2": {"block12": ['EEG033'], "block13": ['EEG033'],"block14": ['EEG007','EEG033'], "block15": ['EEG008','EEG033'], "block16": ['ok'],"block17": ['EEG012','EEG033','EEG060'],"block18": ['EEG008','EEG033'], "block19": ['EEG008','EEG033'], "block20": ['EEG033'], "block21": ['EEG008','EEG033'],"block22": ['EEG008','EEG033']}, 
             "session3": { "block23": ['EEG007','EEG033','EEG058','EEG059'],"block24": ['EEG007','EEG033','EEG058','EEG059'], "block25": ['EEG007','EEG033','EEG058','EEG059'], "block26": ['EEG007','EEG033','EEG058','EEG059'],"block27": ['EEG007','EEG033','EEG058','EEG059'],"block28": ['EEG007','EEG033','EEG058','EEG059'], "block29": ['EEG007','EEG008','EEG033','EEG058','EEG059'], "block30": ['EEG007','EEG033','EEG058','EEG059']}},
  "gto28":  {"session1": {"block01": ['ok'], "block02": ['ok'], "block03": ['ok'],"block04": ['ok'], "block05": ['ok'], "block06": ['ok'],"block07": ['ok'],"block08": ['ok'], "block09": ['ok'], "block10": ['EEG033','EEG037'],"block11": ['ok']},
             "session2": {"block12": ['ok'], "block13": ['ok'],"block14": ['ok'], "block15": ['ok'], "block16": ['ok'],"block17": ['ok'],"block18": ['ok'], "block19": ['ok'], "block20": ['ok'],"block21": ['ok'], "block22": ['ok']}, 
             "session3": {"block23": ['ok'],"block24": ['ok'], "block25": ['ok'], "block26": ['ok'],"block27": ['ok'],"block28": ['ok'], "block29": ['ok'], "block30": ['ok']}},
  "ami28":  {"session1": {"block01": ['EEG008','EEG012'], "block02": ['EEG012'], "block03": ['EEG012'],"block04": ['EEG012','EEG064'], "block05": ['EEG012','EEG064'], "block06": ['EEG012','EEG020','EEG064'],"block07": ['EEG012'],"block08": ['EEG012'], "block09": ['EEG012'], "block10": ['EEG012']},
             "session2": {"block11": ['EEG012'], "block12": ['EEG020','EEG033','EEG037','EEG038'], "block13": ['EEG012'],"block14": ['EEG012'], "block15": ['EEG012'], "block16": ['EEG012'],"block17": ['EEG012'],"block18": ['EEG012'], "block19": ['EEG012'], "block20": ['EEG012']}, 
             "session3": {"block21": ['EEG007','EEG012'], "block22": ['EEG007'], "block23": ['EEG007','EEG012'],"block24": ['EEG012'], "block25": ['EEG007','EEG012'], "block26": ['EEG012'],"block27": ['EEG012'],"block28": ['EEG012'], "block29": ['EEG012'], "block30": ['EEG012']}},
  "lka10":  {"session1": {"block01": ['EEG012','EEG058','EEG064'], "block02": ['EEG064'], "block03": ['EEG012','EEG024','EEG048','EEG058','EEG059','EEG064'],"block04": ['EEG012','EEG058','EEG059','EEG064'], "block05": ['EEG001','EEG012','EEG024','EEG058','EEG059'], "block06": ['EEG058','EEG063'],"block07": ['EEG012','EEG058','EEG059','EEG063','EEG064'],"block08": ['EEG012','EEG058','EEG059','EEG063','EEG064']},
             "session2": {"block09": ['EEG059'], "block10": ['EEG059'], "block11": ['EEG059'], "block12": ['EEG059'], "block13": ['EEG059'],"block14": ['EEG059','EEG064'], "block15": ['EEG059','EEG064'], "block16": ['EEG024','EEG059','EEG064']}, 
             "session3": {"block17": ['EEG024','EEG059','EEG064'],"block18": ['EEG002','EEG024','EEG048','EEG059','EEG064'], "block19": ['EEG002','EEG024','EEG048','EEG059','EEG064'], "block20": ['EEG002','EEG024','EEG063','EEG059','EEG064'],"block21": ['EEG002','EEG024','EEG048','EEG059','EEG064'], "block22": ['EEG002','EEG024','EEG048','EEG059','EEG064'], "block23": ['EEG002','EEG024','EEG048','EEG063','EEG064''EEG012','EEG059'],"block24": ['EEG002','EEG024','EEG048','EEG059','EEG064','EEG063'], "block25": ['EEG002','EEG024','EEG048','EEG059','EEG064','EEG063'], "block26": ['EEG002','EEG024','EEG048','EEG059','EEG064','EEG026']}},
  "qqn19":  {"session1": {"block01": ['EEG060','EEG061'], "block02": ['EEG060'], "block03": ['EEG020'],"block04": ['EEG020'], "block05": ['EEG020'], "block06": ['EEG020'],"block07": ['EEG020'],"block08": ['EEG020'], "block09": ['EEG020'], "block10": ['EEG020']},
             "session2": {"block11": ['EEG012'], "block12": ['ok'], "block13": ['EEG012'],"block14": ['EEG012'], "block15": ['EEG012'], "block16": ['EEG008','EEG012'],"block17": ['EEG012'],"block18": ['EEG008'], "block19": ['EEG008','EEG012'], "block20": ['EEG018','EEG032','EEG062']}, 
             "session3": {"block21": ['EEG008'], "block22": ['EEG008'], "block23": ['EEG008'],"block24": ['EEG008'], "block25": ['EEG008'], "block26": ['EEG008'],"block27": ['EEG008'],"block28": ['EEG008']}},
  "mtr19":  {"session1": {"block01": ['EEG008'], "block02": ['EEG008'], "block03": ['EEG001','EEG004','EEG008','EEG033'],"block04": ['EEG001','EEG004','EEG008','EEG033'], "block05": ['EEG001','EEG004','EEG008','EEG033'], "block06": ['EEG008'],"block07": ['EEG008'],"block08": ['EEG008']},
             "session2": {"block09": ['EEG064'], "block10": ['EEG064'],"block11": ['EEG064'], "block12": ['EEG064'], "block13": ['EEG064'],"block14": ['EEG064'], "block15": ['EEG064'], "block16": ['EEG064'],"block17": ['EEG064','EEG026'],"block18": ['EEG064','EEG026'], "block19": ['EEG064','EEG026'], "block20": ['EEG064']}, 
             "session3": {"block21": ['EEG064'], "block22": ['EEG064','EEG033'], "block23": ['EEG064'],"block24": ['EEG064'], "block25": ['EEG064','EEG010'], "block26": ['EEG064','EEG010'],"block27": ['EEG064','EEG018'],"block28": ['EEG064','EEG010'], "block29": ['EEG064','EEG010'], "block30": ['EEG064','EEG010']}},
  "fha01":  {"session1": {"block01": ['EEG064'], "block02": ['EEG064'], "block03": ['EEG064'],"block04": ['EEG064'], "block05": ['EEG064'], "block06": ['EEG064'],"block07": ['EEG064'],"block08": ['EEG064'], "block09": ['EEG064']},
             "session2": {"block10": ['EEG064'], "block11": ['EEG064'], "block12": ['EEG064'], "block13": ['EEG064'],"block14": ['EEG064'], "block15": ['EEG064'], "block16": ['EEG064'],"block17": ['EEG064'],"block18": ['EEG064']}, 
             "session3": {"block19": ['EEG064'], "block20": ['EEG064'], "block21": ['EEG064'], "block22": ['EEG064'], "block23": ['EEG064'],"block24": ['EEG064'], "block25": ['EEG064'], "block26": ['EEG064'],"block27": ['EEG064'],"block28": ['EEG064']}},
  "hwh21":  {"session1": {"block01": ['EEG058','EEG059'], "block02": ['EEG058','EEG059'], "block03": ['EEG059'],"block04": ['ok']},
             "session2": {"block05": ['EEG004','EEG058','EEG059'], "block06": ['EEG004','EEG058','EEG059'],"block07": ['EEG004','EEG058'],"block08": ['EEG004'], "block09": ['EEG004'], "block10": ['EEG004'],"block11": ['EEG004','EEG058','EEG059'], "block12": ['EEG004','EEG007','EEG058','EEG059'], "block13": ['EEG004','EEG038','EEG058','EEG059'],"block14": ['EEG004','EEG058','EEG059'], "block15": ['EEG004','EEG028','EEG058','EEG059'], "block16": ['EEG004','EEG058','EEG059']}, 
             "session3": {"block17": ['EEG004','EEG012','EEG038','EEG058','EEG059'],"block18": ['EEG004','EEG012','EEG028','EEG058','EEG059'], "block19": ['EEG004','EEG038','EEG058','EEG059'], "block20": ['EEG004','EEG058','EEG059'],"block21": ['EEG004','EEG012','EEG058','EEG059'], "block22": ['EEG004','EEG012','EEG058','EEG059'], "block23": ['EEG004','EEG058','EEG059'],"block24": ['EEG004','EEG058','EEG059'], "block25": ['EEG004','EEG058','EEG059'], "block26": ['EEG004','EEG058','EEG059'],"block27": ['EEG004','EEG058','EEG059'],"block28": ['EEG004','EEG058','EEG059']}},
  "rsh17":  {"session1": {"block01": ['ok'], "block02": ['ok'], "block03": ['ok'],"block04": ['EEG010'], "block05": ['ok'], "block06": ['ok'],"block07": ['EEG010'],"block08": ['EEG010']},
             "session2": {"block09": ['ok'], "block10": ['ok'],"block11": ['ok'], "block12": ['ok'], "block13": ['ok'],"block14": ['EEG010'], "block15": ['ok'], "block16": ['ok'],"block17": ['ok'],"block18": ['ok'], "block19": ['ok']}, 
             "session3": {"block20": ['ok'], "block21": ['ok'], "block22": ['ok'],"block23": ['ok'], "block24": ['ok'], "block25": ['ok'],"block26": ['ok'],"block27": ['ok'], "block28": ['ok'], "block29": ['ok'], "block30": ['ok']}},
  "zwi25":  {"session1": {"block01": ['EEG004'], "block02": ['EEG004'], "block03": ['EEG004','EEG012'],"block04": ['EEG004','EEG012'], "block05": ['EEG004'], "block06": ['EEG004'],"block07": ['EEG004'],"block08": ['EEG004','EEG034'], "block09": ['EEG004'], "block10": ['EEG004']},
             "session2": {"block11": ['EEG004'], "block12": ['EEG004','EEG059'], "block13": ['EEG004','EEG058'],"block14": ['EEG004'], "block15": ['EEG004'], "block16": ['EEG004'],"block17": ['EEG004'],"block18": ['EEG004'], "block19": ['EEG004'], "block20": ['EEG004','EEG058'],"block21":['EEG004','EEG035']},
             "session3": {"block22": ['EEG004'], "block23": ['EEG004'], "block24": ['EEG004'],"block25": ['EEG004'], "block26": ['EEG004'], "block27": ['EEG004'],"block28": ['EEG004'],"block29": ['EEG004'], "block30": ['EEG004']}},
  "tdn02":  {"session1": {"block01": ['EEG004'], "block02": ['EEG004'], "block03": ['EEG004'],"block04": ['EEG004'], "block05": ['EEG004'], "block06": ['EEG004'],"block07": ['EEG004'],"block08": ['EEG004', 'EEG058'], "block09": ['EEG004', 'EEG058'], "block10": ['EEG004']},
             "session2": {"block11": ['EEG004','EEG021','EEG024'], "block12": ['EEG004','EEG021','EEG024'], "block13": ['EEG004','EEG021','EEG024'],"block14": ['EEG004','EEG021','EEG024'], "block15": ['EEG004'], "block16": ['EEG004','EEG021','EEG024'],"block17":['EEG004'],"block18": ['EEG004','EEG021','EEG024'], "block19":['EEG004'], "block20": ['EEG004','EEG021','EEG024']}, 
             "session3": {"block21":['EEG004','EEG050'], "block22":['EEG004','EEG050'], "block23": ['EEG004','EEG050'],"block24":['EEG004','EEG050'], "block25": ['EEG004','EEG050'], "block26":['EEG004','EEG050'],"block27": ['EEG004','EEG050'],"block28": ['EEG004','EEG050'], "block29":['EEG004','EEG050'], "block30": ['EEG004','EEG050']}},
  "uka11": {"session1": {"block01": ['EEG004'], "block02": ['EEG004','EEG024'], "block03": ['EEG004','EEG024'],"block04": ['EEG004'], "block05": ['EEG004'], "block06": ['EEG004','EEG010'],"block07": ['EEG004','EEG010']},
            "session2": {"block08": ['EEG004'], "block09": ['EEG004'], "block10": ['EEG004'],"block11": ['EEG004'], "block12": ['EEG004'], "block13": ['EEG004'],"block14": ['EEG004'],"block15": ['EEG004'], "block16": ['EEG004'], "block17": ['EEG004']},
            "session3": {"block18": ['EEG004'], "block19": ['EEG004'], "block20": ['EEG004'],"block21": ['EEG004'], "block22": ['EEG004'], "block23": ['EEG004'],"block24": ['EEG004'],"block25": ['EEG004'],"block26": ['EEG004']}},
  "csi07":  {"session1": {"block01": ['ok'], "block02": ['ok'], "block03": ['ok'],"block04": ['ok'], "block05": ['ok'], "block06": ['ok'],"block07": ['ok'],"block08": ['ok']},
             "session2": {"block09": ['EEG012'], "block10": ['ok'], "block11": ['EEG012'],"block12": ['EEG048'], "block13": ['ok'], "block14": ['EEG059'],"block15": ['ok'],"block16": ['ok'], "block17": ['ok']},
             "session3": {"block18": ['ok'], "block19": ['EEG007'], "block20": ['ok'],"block21": ['ok'], "block22": ['ok'], "block23": ['ok'],"block24": ['ok'],"block25": ['ok']}},
  "rsg06":  {"session1": {"block01": ['EEG004','EEG008'], "block02": ['EEG004','EEG008'], "block03": ['EEG004','EEG008'], "block05": ['EEG004','EEG008'], "block06": ['EEG004','EEG008'],"block07": ['EEG004','EEG008'],"block08": ['EEG004','EEG008'], "block09": ['EEG004','EEG008'], "block10": ['EEG004','EEG008']},
             "session2": {"block11": ['EEG004','EEG024'], "block12": ['EEG004','EEG012'], "block13": ['EEG004'],"block14": ['EEG004','EEG008'], "block15": ['EEG004','EEG008'], "block16": ['EEG004','EEG008'],"block17": ['EEG004','EEG008','EEG024'],"block18": ['EEG004','EEG008'], "block19": ['EEG004']}, 
             "session3": {"block21": ['EEG004'], "block22": ['EEG004'], "block23": ['EEG004'],"block24": ['EEG004'], "block25": ['EEG004','EEG033'], "block26": ['EEG004'],"block27": ['EEG004'],"block28": ['EEG004','EEG011'], "block29": ['EEG004','EEG011'], "block30": ['EEG004','EEG011']}},
  "mwa29":  {"session1": {"block01": ['ok'], "block02": ['ok'], "block03": ['ok'],"block04": ['ok'], "block05": ['EEG059'], "block06": ['ok'],"block07": ['ok'],"block08": ['EEG059'], "block09": ['EEG047'], "block10": ['EEG047']},
             "session2": {"block11": ['EEG046'], "block12": ['ok'], "block13": ['EEG048','EEG059'],"block14": ['ok'], "block15": ['EEG058'], "block16": ['ok'],"block17": ['ok'],"block18": ['ok'], "block19": ['ok']},
             "session3": {"block20": ['ok'],"block21": ['ok'], "block22": ['EEG058','EEG059'], "block23": ['EEG058','EEG059'],"block24": ['EEG058','EEG059'], "block25": ['EEG058','EEG059'], "block26": ['ok'],"block27": ['ok'],"block28": ['ok'], "block29": ['EEG059']}},
  "ade02":  {"session1": {"block01": ['EEG008'], "block02": ['ok'], "block03": ['ok'],"block04": ['ok'], "block05": ['ok'], "block06": ['ok'],"block07": ['EEG008'],"block08": ['ok'], "block09": ['ok'], "block10": ['ok'],"block11": ['ok']},
             "session2": {"block12": ['EEG007'], "block13": ['EEG058'], "block14": ['ok'],"block15": ['EEG024'], "block16": ['ok'], "block17": ['ok'],"block18": ['ok'],"block19": ['ok'], "block20": ['ok']},  
             "session3": {"block21": ['ok'], "block22": ['ok'], "block23": ['ok'],"block24": ['ok'], "block25": ['EEG024'], "block26": ['ok'],"block27": ['ok'],"block28": ['ok'], "block29": ['ok'], "block30": ['ok']}},
  "dtl05":  {"session1": {"block01": ['EEG008','EEG012'], "block02": ['EEG037','EEG019'], "block03": ['ok'],"block04": ['ok'], "block05": ['ok'], "block06": ['ok'],"block07": ['EEG008'],"block08": ['ok'], "block09": ['ok'], "block10": ['ok']},
             "session2": {"block11": ['ok'] ,"block12": ['ok'], "block13": ['ok'], "block14": ['EEG012'],"block15": ['EEG012'], "block16": ['EEG012','EEG018'], "block17": ['EEG012'],"block18": ['ok'],"block19": ['ok'], "block20": ['EEG019']},  
             "session3": {"block21": ['EEG020'], "block22": ['ok'], "block23": ['ok'],"block24": ['ok'], "block25": ['EEG023'], "block26": ['ok'],"block27": ['ok'],"block28": ['EEG002','EEG020','EEG041'], "block29": ['EEG001','EEG002','EEG020','EEG041'], "block30": ['EEG019']}},
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
  "hyr24":  {"session1": {"block01": ['EEG003','EEG008'], "block02": ['EEG003','EEG008'], "block03": ['EEG003','EEG008'],"block04": ['EEG003','EEG008'], "block05": ['EEG003','EEG008'], "block06": ['EEG003','EEG008'],"block07": ['EEG003','EEG008'],"block08": ['EEG003','EEG008'], "block09": ['EEG003','EEG008'],"block10": ['EEG003','EEG008']},
             "session2": {"block11": ['EEG003','EEG008'], "block12": ['EEG003','EEG021','EEG033','EEG038','EEG042'], "block13": ['EEG008','EEG033','EEG038'], "block14": ['EEG008'],"block15": ['EEG008'], "block16": ['EEG008','EEG045'], "block17": ['EEG008'],"block18": ['EEG008','EEG045'],"block19": ['EEG038','EEG045'],"block20": ['EEG008']},  
             "session3": {"block21": ['EEG008'], "block22": ['EEG008','EEG024','EEG041','EEG044'], "block23": ['EEG008','EEG041','EEG044'],"block24": ['EEG008','EEG041','EEG045'], "block25": ['EEG008','EEG045'], "block26": ['EEG008','EEG041','EEG045'],"block27": ['EEG008'],"block28": ['EEG008','EEG020'], "block29": ['EEG008'],"block30": ['EEG008','EEG020','EEG045']}},
  "ank24":  {"session1": {"block01": ['EEG007','EEG059'], "block02": ['EEG059'], "block03": ['EEG059'],"block04": ['EEG007','EEG059'], "block05": ['EEG007','EEG059'], "block06": ['EEG007','EEG059'],"block07": ['EEG007','EEG058','EEG059'],"block08": ['EEG007','EEG047','EEG059'], "block09": ['EEG007','EEG047','EEG059'],"block10": ['EEG007','EEG047','EEG059']},
             "session2": {"block11": ['EEG007','EEG059'], "block12": ['EEG007','EEG010','EEG059'], "block13": ['EEG007','EEG059'], "block14": ['EEG007','EEG010','EEG059'],"block15": ['EEG007','EEG059'], "block16": ['EEG007','EEG010','EEG059'], "block17": ['EEG007','EEG059'],"block18": ['EEG007','EEG010','EEG059'],"block19": ['EEG007','EEG010','EEG059'],"block20": ['EEG007','EEG010','EEG059']},  
             "session3": {"block21": ['EEG059'], "block22": ['EEG007','EEG012','EEG059'], "block23": ['EEG007','EEG012','EEG059'],"block24": ['EEG007','EEG059'], "block25": ['EEG007','EEG012','EEG059'], "block26": ['EEG007','EEG012','EEG059'],"block27": ['EEG007','EEG012','EEG059'],"block28": ['EEG007','EEG010','EEG059'], "block29": ['EEG007','EEG059'],"block30": ['EEG007','EEG012','EEG059']}},
  "dja01":  {"session1": {"block01": ['EEG008','EEG012','EEG021','EEG020','EEG033','EEG037','EEG058'], "block02": ['EEG008','EEG012','EEG021','EEG033','EEG037','EEG058'], "block03": ['EEG008','EEG012','EEG021','EEG033','EEG037','EEG058'],"block04": ['EEG008','EEG012','EEG033','EEG037','EEG045','EEG058'], "block05": ['EEG008','EEG012','EEG021','EEG033','EEG037','EEG058'], "block06": ['EEG008','EEG012','EEG021','EEG025','EEG033','EEG058'],"block07": ['EEG008','EEG012','EEG021','EEG025','EEG033','EEG037','EEG045','EEG058','EEG060'],"block08": ['EEG008','EEG012','EEG021','EEG025','EEG033','EEG037','EEG045','EEG058'], "block09": ['EEG008','EEG012','EEG033','EEG037','EEG058'],"block10": ['EEG008','EEG012','EEG033','EEG037']},
             "session2": {"block11": ['EEG001','EEG008','EEG021','EEG033','EEG037','EEG058'], "block12": ['EEG008','EEG012','EEG021','EEG033','EEG037','EEG038','EEG045','EEG058'], "block13": ['EEG008','EEG012','EEG021','EEG025','EEG033','EEG037','EEG038','EEG045','EEG058'], "block14": ['EEG001','EEG008','EEG021','EEG033','EEG037','EEG038','EEG045','EEG058'],"block15": ['EEG001','EEG008','EEG021','EEG033','EEG037','EEG038','EEG045','EEG058'], "block16": ['EEG001','EEG008','EEG021','EEG033','EEG037','EEG038','EEG045','EEG058'], "block17": ['EEG001','EEG008','EEG021','EEG033','EEG037','EEG038','EEG045','EEG058'],"block18": ['EEG001','EEG008','EEG021','EEG033','EEG037','EEG038','EEG045','EEG058'],"block19": ['EEG001','EEG008','EEG021','EEG033','EEG037','EEG038','EEG045','EEG058'],"block20": ['EEG001','EEG008','EEG021','EEG033','EEG037','EEG038','EEG045','EEG058']},  
             "session3": {"block21": ['EEG001','EEG008','EEG021','EEG033','EEG045','EEG058'], "block22": ['EEG008','EEG021','EEG024','EEG033','EEG045','EEG058'], "block23": ['EEG008','EEG021','EEG024','EEG033','EEG0037','EEG045','EEG058'],"block24": ['EEG008','EEG021','EEG024','EEG033','EEG045','EEG058'], "block25": ['EEG008','EEG021','EEG024','EEG033','EEG037','EEG045','EEG058'], "block26": ['EEG008','EEG021','EEG024','EEG033','EEG045','EEG058'],"block27": ['EEG001','EEG008','EEG021','EEG024','EEG033','EEG045','EEG058'],"block28": ['EEG008','EEG021','EEG024','EEG033','EEG045','EEG058'], "block29": ['EEG008','EEG021','EEG024','EEG033','EEG045','EEG058'],"block30": ['EEG008','EEG021','EEG024','EEG033','EEG045','EEG058']}},
  "fmn28":  {"session1": {"block01": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG024', 'EEG058', 'EEG059', 'EEG061', 'EEG064'], "block02": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG059', 'EEG064'], "block03": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG059', 'EEG064'],"block04": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG059', 'EEG064'], "block05": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG024', 'EEG059', 'EEG064'], "block06": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG059', 'EEG064'],"block07": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG059', 'EEG064'],"block08": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG059', 'EEG064'], "block09": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG058', 'EEG059', 'EEG064'],"block10": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG048', 'EEG059', 'EEG064']},
             "session2": {"block11": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG024', 'EEG059', 'EEG064'], "block12": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG024', 'EEG059', 'EEG064'], "block13": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG024', 'EEG059', 'EEG064'], "block14": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG024', 'EEG059', 'EEG064'],"block15": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG024', 'EEG059', 'EEG064'], "block16": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG024', 'EEG059', 'EEG064'], "block17": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG024', 'EEG059', 'EEG064'],"block18": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG024', 'EEG059', 'EEG064'],"block19": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG024', 'EEG059', 'EEG064'], "block20": ['EEG001', 'EEG002','EEG007', 'EEG012', 'EEG024', 'EEG059', 'EEG064']},  
             "session3": {"block21": ['EEG001', 'EEG002','EEG007', 'EEG008', 'EEG012', 'EEG058', 'EEG059', 'EEG064'], "block22": ['EEG001', 'EEG002','EEG007', 'EEG008', 'EEG012', 'EEG058', 'EEG059', 'EEG064']}},
  "omr03":  {"session1": {"block01": ['EEG005'], "block02": ['EEG007', 'EEG033', 'EEG037', 'EEG041'], "block03": ['EEG033', 'EEG037', 'EEG041'],"block04": ['EEG012', 'EEG033', 'EEG037', 'EEG041', 'EEG058', 'EEG059', 'EEG060'], "block05": ['EEG007', 'EEG008', 'EEG024', 'EEG033', 'EEG037', 'EEG041', 'EEG058', 'EEG059'], "block06": ['EEG007', 'EEG021', 'EEG024', 'EEG033', 'EEG037', 'EEG038', 'EEG041', 'EEG048', 'EEG058', 'EEG059', 'EEG060'],"block07": ['ok'],"block08": ['EEG058', 'EEG059','EEG060']},
             "session2": {"block09": ['EEG007', 'EEG033', 'EEG037', 'EEG041'], "block10": ['EEG002', 'EEG007', 'EEG033', 'EEG037', 'EEG041', 'EEG058', 'EEG059'], "block11": ['EEG002'], "block12": ['EEG007', 'EEG058', 'EEG059'], "block13": ['EEG007', 'EEG021', 'EEG024', 'EEG033', 'EEG037', 'EEG041', 'EEG058', 'EEG059'], "block14": ['EEG007', 'EEG058', 'EEG059'],"block15": ['EEG007', 'EEG033', 'EEG037', 'EEG041', 'EEG042', 'EEG059'], "block16": ['EEG007', 'EEG033', 'EEG037', 'EEG041'], "block17": ['EEG007', 'EEG033', 'EEG037', 'EEG041', 'EEG058', 'EEG059']},  
             "session3": {"block18": ['EEG007', 'EEG033', 'EEG037', 'EEG041'],"block19": ['EEG007', 'EEG033', 'EEG037', 'EEG041'], "block20": ['EEG007', 'EEG033', 'EEG037', 'EEG041'], "block21": ['EEG007', 'EEG033', 'EEG037', 'EEG041'], "block22": ['EEG007', 'EEG033', 'EEG037', 'EEG041', 'EEG048', 'EEG058', 'EEG059'], "block23": ['EEG007', 'EEG008', 'EEG021', 'EEG024', 'EEG033', 'EEG037', 'EEG041', 'EEG058', 'EEG059', 'EEG060'],"block24": ['EEG007', 'EEG021', 'EEG033', 'EEG037', 'EEG041', 'EEG058', 'EEG059'], "block25": ['EEG007', 'EEG033', 'EEG037', 'EEG041', 'EEG058', 'EEG059', 'EEG060'], "block26": ['EEG007', 'EEG033', 'EEG037', 'EEG041', 'EEG059']}},
  "amy20":  {"session1": {"block01": ['ok'], "block02": ['ok'], "block03": ['ok'],"block04": ['ok'], "block05": ['ok'], "block06": ['ok'],"block07": ['ok'],"block08": ['ok'], "block09": ['ok']},
             "session2": {"block10": ['ok'], "block11": ['ok'], "block12": ['ok'], "block13": ['ok'], "block14": ['ok'],"block15": ['ok'], "block16": ['ok'], "block17": ['ok'],"block18": ['ok'],"block19": ['ok']},  
             "session3": {"block20": ['EEG003'], "block21": ['EEG003'], "block22": ['EEG003'], "block23": ['ok'],"block24": ['EEG003'], "block25": ['EEG003'], "block26": ['ok'],"block27": ['ok'],"block28": ['ok'], "block29": ['ok']}},
  "ski23":  {"session1": {"block01": ['EEG008','EEG021','EEG033','EEG037','EEG058'], "block02": ['EEG008','EEG024','EEG033','EEG037','EEG058'], "block03": ['EEG008','EEG021','EEG033','EEG037','EEG058'],"block04": ['EEG008','EEG021','EEG033','EEG037','EEG058'], "block05": ['EEG008','EEG021','EEG033','EEG037','EEG058'], "block06": ['EEG008','EEG021','EEG033','EEG037','EEG058'],"block07": ['EEG008','EEG024','EEG033','EEG037','EEG058'],"block08": ['EEG008','EEG021','EEG033','EEG037','EEG058'], "block09": ['EEG008','EEG021','EEG024','EEG033','EEG037','EEG058'],"block10": ['EEG008','EEG021','EEG024','EEG033','EEG037','EEG058']},
             "session2": {"block10": ['ok'], "block11": ['ok'], "block12": ['ok'], "block13": ['ok'], "block14": ['ok'],"block15": ['ok'], "block16": ['ok'], "block17": ['ok'],"block18": ['ok'],"block19": ['ok']},  
             "session3": {"block20": ['ok'], "block21": ['ok'], "block22": ['ok'], "block23": ['ok'],"block24": ['ok'], "block25": ['ok'], "block26": ['ok'],"block27": ['ok'],"block28": ['ok'], "block29": ['ok']}},
 "jry29":  {"session1": {"block01": ['ok'], "block02": ['ok'], "block03": ['EEG007','EEG024','EEG041','EEG059'],"block04": ['EEG007','EEG024','EEG041','EEG059'], "block05": ['EEG007','EEG024','EEG041','EEG059'], "block06": ['EEG007','EEG024','EEG041','EEG059'],"block07": ['EEG007','EEG024','EEG041','EEG059'],"block08": ['EEG007','EEG024','EEG041','EEG059'], "block09": ['EEG007','EEG024','EEG041','EEG059'],"block10": ['EEG007','EEG024','EEG041','EEG059']},
            "session2": {"block10": ['ok'], "block11": ['EEG007'], "block12": ['EEG007','EEG024','EEG059'], "block13": ['EEG007','EEG024','EEG059'], "block14": ['EEG007'],"block15": ['EEG007','EEG024','EEG059'], "block16": ['ok'], "block17": ['EEG024'],"block18": ['EEG007','EEG048','EEG059'],"block19": ['ok'],"block20": ['EEG007','EEG048','EEG059']},  
            "session3": {"block21": ['EEG007','EEG024','EEG059'], "block22": ['EEG007','EEG024','EEG059'], "block23": ['EEG007','EEG024'],"block24": ['EEG007','EEG024','EEG059'], "block25": ['EEG007','EEG024','EEG059'], "block26": ['EEG007','EEG024','EEG059'],"block27": ['EEG007','EEG024','EEG059'],"block28": ['EEG007','EEG024','EEG059'], "block29": ['EEG007','EEG024','EEG059'],"block30": ['EEG007','EEG024','EEG059'],}},
 "jpa10":  {"session1": {"block01": ['EEG001','EEG012','EEG033','EEG037','EEG041','EEG048'], "block02": ['EEG001','EEG007','EEG033','EEG037','EEG058','EEG060'], "block03": ['EEG001','EEG012','EEG033','EEG037','EEG058','EEG060'],"block04": ['EEG001','EEG007','EEG012','EEG033','EEG037','EEG045','EEG048','EEG058'], "block05": ['EEG001','EEG020','EEG033','EEG037','EEG058','EEG060'], "block06": ['EEG001','EEG007','EEG012','EEG033','EEG037','EEG058','EEG060'],"block07": ['EEG001','EEG007','EEG033','EEG037','EEG045','EEG058','EEG060'],"block08": ['EEG001','EEG007','EEG033','EEG037','EEG058','EEG060'], "block09": ['EEG001','EEG007','EEG012','EEG033','EEG037','EEG058','EEG060'],"block10": ['EEG001','EEG007','EEG033','EEG037','EEG058','EEG060']},
            "session2": {"block11": ['EEG001','EEG007','EEG021','EEG024','EEG033','EEG037','EEG058'], "block12": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058'], "block13": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058'], "block14": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058'],"block15": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058'], "block16": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058'], "block17": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058'],"block18": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058'],"block19": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058'],"block20": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058']},  
            "session3": {"block21": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058','EEG061'], "block22": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058','EEG061'], "block23": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058','EEG061'],"block24": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058','EEG061'], "block25": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058','EEG061'], "block26": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058','EEG061'],"block27": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058','EEG061'],"block28": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058','EEG061'], "block29": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058','EEG061'],"block30": ['EEG001','EEG007','EEG020','EEG021','EEG024','EEG033','EEG037','EEG058','EEG061'],}},
 "epa14":  {"session1": {"block01": ['EEG012','EEG061'], "block02": ['EEG012','EEG061'], "block03": ['EEG012','EEG061'],"block04": ['EEG007','EEG012','EEG061'], "block05": ['EEG012','EEG061'], "block06": ['EEG012','EEG059','EEG061'],"block07": ['EEG012','EEG059'],"block08": ['EEG012','EEG059','EEG061'], "block09": ['EEG012','EEG059','EEG061'],"block10": ['EEG012','EEG059']},
            "session2": {"block11": ['EEG001','EEG033'], "block12": ['EEG001','EEG033'], "block13": ['EEG033'], "block14": ['EEG001','EEG033'],"block15": ['EEG001','EEG033'], "block16": ['EEG001','EEG033'], "block17": ['EEG001','EEG033'],"block18": ['EEG001','EEG033'],"block19": ['EEG001','EEG033'],"block20": ['EEG001','EEG033']},  
            "session3": {"block21": ['EEG012','EEG061'], "block22": ['EEG012','EEG061'], "block23": ['EEG012','EEG059','EEG061'],"block24": ['EEG001','EEG012','EEG061'], "block25": ['EEG001','EEG012','EEG061'], "block26": ['EEG012','EEG061'],"block27": ['EEG012','EEG061'],"block28": ['EEG001','EEG012','EEG059','EEG061'], "block29": ['EEG007','EEG012','EEG048','EEG059','EEG061'],"block30": ['EEG012','EEG061']}},
 "crr22":  {"session1": {"block01": ['EEG012','EEG033'], "block02": ['EEG012','EEG033'], "block03": ['EEG012','EEG033'],"block04": ['EEG012','EEG033'], "block05": ['EEG012','EEG033'], "block06": ['EEG012','EEG033','EEG037'],"block07": ['EEG012','EEG033','EEG037'],"block08": ['EEG012','EEG033','EEG037'], "block09": ['EEG012','EEG033','EEG037']},
            "session2": {"block10": ['EEG012','EEG059'],"block11": ['EEG003','EEG012','EEG033'], "block12": ['EEG012','EEG033','EEG038'], "block13": ['EEG012','EEG033','EEG059'], "block14": ['EEG012','EEG061'],"block15": ['EEG012','EEG059'], "block16": ['EEG012','EEG033','EEG061'], "block17": ['EEG012','EEG033'],"block18": ['EEG012','EEG033','EEG038'],"block19": ['EEG012','EEG033'],"block20": ['EEG012','EEG033','EEG061']},  
            "session3": {"block21": ['EEG001','EEG002','EEG012','EEG033','EEG037','EEG038','EEG058'], "block22": ['EEG003','EEG012','EEG033','EEG038','EEG050','EEG058'], "block23": ['EEG003','EEG012','EEG033'],"block24": ['EEG003','EEG012','EEG033','EEG050','EEG064'], "block25": ['EEG002','EEG012','EEG033'], "block26": ['EEG002','EEG012','EEG033','EEG045','EEG058'],"block27": ['EEG002','EEG012','EEG033','EEG045','EEG058'],"block28": ['EEG002','EEG012','EEG033','EEG045','EEG058'], "block29": ['EEG002','EEG012','EEG033','EEG050'],"block30": ['EEG002','EEG012','EEG033']}},
 "jyg27":  {"session1": {"block01": ['EEG012'], "block02": ['EEG007','EEG012'], "block03": ['EEG008','EEG012','EEG059'],"block04": ['EEG007','EEG012','EEG059'], "block05": ['EEG007','EEG012','EEG059'], "block06": ['EEG007','EEG012','EEG059'],"block07": ['EEG007','EEG012','EEG059']},
            "session2": {"block08": [], "block09": ['EEG061'],"block10": ['EEG041'], "block11": ['EEG041'], "block12": ['EEG012'], "block13": ['EEG012'], "block14": ['EEG012'],"block15": ['EEG012'], "block16": [], "block17": ['EEG007','EEG024']},  
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




#  "ank24":  {"session1": {"block01": [], "block02": [], "block03": [],"block04": [], "block05": [], "block06": [],"block07": [],"block08": [], "block09": [],"block10": []},
#              "session2": {"block11": [], "block12": [], "block13": [], "block14": [],"block15": [], "block16": [], "block17": [],"block18": [],"block19": [],"block20": []},  
#              "session3": {"block21": [], "block22": [], "block23": [],"block24": [], "block25": [], "block26": [],"block27": [],"block28": [], "block29": [],"block30": []}},
# }

'''
NOTES:
fha01, first seconds in block22 need to be deleted
mtr13, first seconds in block29 need to be deleted

done in annotation and EEG interpolation code

hwh21, highfreq noise, may need to remove an ICA related to this!!!

jmn22 check picks in channel MEG0243
'''
#%%

data_dir = r'Z:' # r'D:\PROJECTS\CAUSAL_NETWORKS\DATA'
#output_folder = r'Y:\ANALYSIS\QUALITY_CHECK'#r'D:\PROJECTS\CAUSAL_NETWORKS\ANALYSIS\QUALITY_CHECK'

subjects_list = [ f.name for f in os.scandir(data_dir) if f.is_dir() and f.name[0] != '_' and f.name[0].islower()]
subjects_list.sort()
print(subjects_list) 

#%%
# loop through participants

sub_idx =20 #-13 # 5  
subject_nb = folder_mapping[subjects_list[sub_idx]]
subject_id =subjects_list[sub_idx]
print('Doing subject ' + subject_id + '\nSessions')
subject_path = op.join(data_dir,subjects_list[sub_idx])
session_folders = os.listdir(subject_path)
sessions =  [x for x in session_folders if x.startswith('2')]
print(sessions)

meg = False # to inspect MEG data, if set to False will inspect eeg data

#%% Loop through sessions check 

#for idx,session in enumerate(sessions):
idx = 1
session = sessions[idx]
print(session)
#save_dir = op.join(output_folder,subject_id,session) # set path for output files
# create folder if it doesn't exist
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# session_idx = 0 # loop through sessions
# session = sessions[session_idx]
file_dir = op.join(subject_path,session)
blocks =  os.listdir(file_dir)
#blocks = [x for x in blocks if x.startswith('B')]
blocks = check_block_name(blocks)
blocks.sort() 
print(blocks)
# loop trhough the blocks annotate bad channels, compute PSD and check trial events
# max filter
# concatenate after maxfilter
if meg:
    blocks = blocks[0::2]
    
for block in blocks:
    print(block)
    # block=blocks[0]
    #-----------------------------------------------------------------------------
    # Load file
    #-----------------------------------------------------------------------------
    #file_name = blocks[0] # loop through blocks
    #empty_room = op.join(file_dir,'empty_room.fif')
    #raw = mne.io.read_raw_fif(empty_room,preload = True)
    #raw.filter(l_freq=0,h_freq=120,h_trans_bandwidth=2) 
    raw_file = op.join(file_dir,block)
    raw = mne.io.read_raw_fif(raw_file,preload = True)
    
    # raw.info['dig']
    #% Check data info
    print(raw.info)
    
    #check channels
    ch_names = raw.info['ch_names']
    #print(ch_names)
    if ch_names != default_channel_list:
        print('UNEXPECTED NUMBER OF CHANNELS!!!\n')
        if len(ch_names) > len(default_channel_list):
            extra_chan = list(set(ch_names) - set(default_channel_list))
            print('Deleting extra channels!....Double Check')
            raw.drop_channels(extra_chan) # save raw, overwrite
        else:
            extra_chan = list(set(default_channel_list)-set(ch_names))
            if extra_chan == ['BIO003']:
                print('No heart data for this participant')
            else:           
                print('Missing channels! probably wrong settings during aquisition, check what is missing')
    raw.info['bads'] = []
    raw.notch_filter(np.arange(50, 251, 50))
    raw.filter(l_freq=1,h_freq=40,h_trans_bandwidth=2) 
    
    if meg:
            
        raw.info['bads'] = bad_meg_channels[subject_id]['session'+str(idx+1)]
        
        #print(raw.info['bads'] )
        #raw.plot_psd(tmin=10) # before filtering
        #raw.plot_psd(tmin=10, fmin=0, fmax=50)
        # use 148 for MEG, 50 for EEG
        
        raw.pick('meg')
        raw.plot()
        # If you want to check eeg only coment out the line above and run this instead
        # eeg = raw.copy().pick_types(meg=False, eeg=True)
        # eeg.plot(scalings = 40e-6)
        # alternatively do: 
        #raw.pick('eeg').plot(scalings = 40e-6) # use this to only plot a especific type of data, scale if neccesary 
        #raw.plot()
        # To plot only MEG run this. For some reason marking baad channels do not work, so I prefer to plot raw
        # meg = raw.copy().pick_types(meg=True, eeg=False)
        # meg.plot()
    else:
        #raw.info['bads'] = bad_meg_channels[subject_id]['session'+str(idx+1)]
        #raw.info['bads'] = bad_eeg_channels[subject_id]['session'+str(idx+1)]['b' + block.replace("_", "")[1:-4]]
        
        #print(raw.info['bads'] )
        #raw.plot_psd(tmin=10) # before filtering
        #raw.plot_psd(tmin=10, fmin=0, fmax=50)
        raw.notch_filter(np.arange(50, 251, 50))
        raw.filter(l_freq=1,h_freq=40,h_trans_bandwidth=2) # use 148 for MEG, 50 for EEG
        
        #raw.pick('meg')
        #raw.plot()
        # If you want to check eeg only coment out the line above and run this instead
        # eeg = raw.copy().pick_types(meg=False, eeg=True)
        # eeg.plot(scalings = 40e-6)
        # alternatively do: 
        raw.pick('eeg').plot(scalings = 40e-6) # use this to only plot a especific type of data, scale if neccesary 
        #raw.plot()
        # To plot only MEG run this. For some reason marking baad channels do not work, so I prefer to plot raw
        # meg = raw.copy().pick_types(meg=True, eeg=False)
        # meg.plot()    
    
    print('Subject ' + subject_id + '\nSession ' + str(idx+1) + '\nBlock ' + block)
    
#%% cHPI off
'''
ami28, session1, all blacks no cHPI



'''

#%% comments
'''
bsr27
Block07_08, block_06, block_09 very bag magnometers
block_12, block_13, block_14, block_17, block_19, block_24, block_27 some very bad segments in MAG
block09, all EEG channels bad

dsa23
some bad segments in MAG: block_04, block_08, block_18, block_22, block_25
several bad channel sin block_16, block_17, check after maxfilter
all EEG channels bad block_25

gto28
EEG, bad channels at the end of Block-01, block_05,
check block04 after maxfilter
 block_10 (very bad MAG), block_18, block_19, block_26 very bad MAG, block 28 (very bad), block_29

 {"session1": ['MEG0922', 'MEG1243',  'MEG1042', 'MEG2332','MEG0243','MEG1823'], "session2":['MEG0922', 'MEG1243',  'MEG1042', 'MEG2332','MEG0243','MEG1823'], "session3": ['MEG0922', 'MEG1243',  'MEG1042', 'MEG2332','MEG0243','MEG1823']}
 
'''

