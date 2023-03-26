import glob
import logging
import pandas as pd
import numpy as np
import scipy.io
from pathlib import Path
import itertools
import matplotlib.pyplot as plt

def detrend(data):
    """
    Detrends the data
    :param data: data
    :return: detrended data
    """
    return scipy.signal.detrend(data, axis=0, type='linear')

def get_functional_connectivity(data):
    """
    Get functional connectivity between regions
    :return: correlation matrix, z-scores
    """
    corr = np.corrcoef(data)  # correlation matrix
    zscores = np.arctanh(corr) #convert to tanh space
    #set diagonal elements to NaN
    for i in range(corr.shape[0]):
        corr[i, i] = np.NaN 
    return corr, zscores

def load_nirs(filepath):
    """
    Load nirs data from filepath
    :return: data, stims
    """
    nirs = scipy.io.loadmat(filepath) 
    stims = np.array(nirs['s'], dtype=np.int64) # stimulus data
    data = np.array(nirs['procResult']['dc'][0][0], dtype=np.float64) # HbO, HbR, HbT values
    logging.info("Successfully loaded data from {}".format(filepath))
    logging.info("Data shape: {}, Stimulus data shape: {}".format(data.shape, stims.shape))
    return data, stims



input_dir = './stem_data/'
output_dir = './output/'
# output_dir = './output_{}'.format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
# create output directory
Path(output_dir).mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=output_dir+'/logs.log',
                    filemode='w', 
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    level=logging.INFO, 
                    encoding='utf-8')

# get all the files in the input directory with format *_C.nirs
files = glob.glob(input_dir + '*_C.nirs')
print(files)

data, stims = load_nirs(files[0])
print(data.shape)
data = data[:,0,:] # 0 for HbO, 1 for HbR, 2 for HbT
print(data.shape)
print(stims.shape)

# get the stim locations for stim 0
loc = np.where(stims[:, 0] == 1)[0]
print(loc)
# get the data for stim 0
data_x = data[loc[0]:loc[-1], :]
print(data_x.shape)

# get the stim locations for stim 1
loc = np.where(stims[:, 1] == 1)[0]
#  check if number of stim locations is even
assert len(loc) % 2 == 0, "Number of stim locations is not even"
print(loc)
# get the data for stim 1
data_y = data[loc[0]:loc[-1], :]
print(data_y.shape)

# detrend the data
data_x_detrend = detrend(data_x)

# # plot the non-detrended and detrended data
# plt.plot(data_x, 'b')
# plt.plot(data_x_detrend, 'r')
# plt.show()

# get the functional connectivity
corr, zscores = get_functional_connectivity(data_x_detrend.T)
print(corr.shape)




# loop through all the files and conditions to calculate functional connectivity
stim_0_fc = []
stim_1_fc = []  
for file in files:
    data, stims = load_nirs(file)
    data = data[:,0,:] # 0 for HbO, 1 for HbR, 2 for HbT
    ID = file.split('/')[-1].split('_')[0]
    print('processing ID: {}'.format(ID))
    for stim in [0,1]:
        try:
            # get the stim locations
            loc = np.where(stims[:, stim] == 1)[0]
            #  check if number of stim locations is even
            assert len(loc) % 2 == 0, "Number of stim locations is not even"
            # get the data for stim
            data_stim = data[loc[0]:loc[-1], :]
            # detrend the data
            data_stim_detrend = detrend(data_stim)
            # print(data_stim_detrend.shape)
            # get the functional connectivity
            corr, zscores = get_functional_connectivity(data_stim_detrend.T)
            # store the functional connectivity
            if stim == 0:
                stim_0_fc.append(corr)
            elif stim == 1:
                stim_1_fc.append(corr)
        except Exception as e:
            print("Error in file: {}, error: {}".format(file, e))
        
# save the functional connectivity
stim_0_fc = np.array(stim_0_fc)
stim_1_fc = np.array(stim_1_fc)
print(stim_0_fc.shape)

# mean across subjects
stim_0_fc_mean = np.nanmean(stim_0_fc, axis=0)
stim_1_fc_mean = np.nanmean(stim_1_fc, axis=0)
print(stim_0_fc_mean.shape)

# save the mean functional connectivity as .csv file
np.savetxt(output_dir+'/stim_0_fc_mean.csv', stim_0_fc_mean, delimiter=',')
np.savetxt(output_dir+'/stim_1_fc_mean.csv', stim_1_fc_mean, delimiter=',')


