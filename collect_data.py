# -*- coding: utf-8 -*-
import numpy as np
import time
from datetime import timedelta
import datetime
import numpy.random as nr
import pandas as pd
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import itertools

def elapsed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        f = func(*args, **kwargs)  # Call the method
        end = time.time()
        elapsed = end-start
        print('Elapsed: {}'.format(str(timedelta(seconds=elapsed))))
        return f  # Return whatever the method returns
    return wrapper

# Collect all files from the dataset folder
    
# Make all permutations of choose N DOF per action
# e.g. choose 2: alas_cubic_HEAD_PAN & _HEAD_TILT 
#                alas_cubic_HEAD_PAN & L_ELBOW ... etc...
    
path_path = '/home/mathias/Projects/spyder_projects/motion_generator_v2/dataset/'
onlyfiles = [f for f in listdir(path_path) if isfile(join(path_path, f))]

# Print the filenames found
#for file in onlyfiles:
#    print(file)
    
filefiles = [f for f in onlyfiles if '.csv' in f]
#for ff in filefiles:
#    print(ff)
    
jimmy_dof = ['HEAD_PAN', 'HEAD_TILT', 'L_ELBOW', 'L_SHO_PITCH', 'L_SHO_ROLL',
             'R_ELBOW', 'R_SHO_PITCH', 'R_SHO_ROLL']
mocap_dof = ['Hips_x', 'Hips_y', 'Hips_z', 'LeftArm_x', 'LeftArm_y', 'LeftArm_z',
             'LeftForeArm_x', 'LeftForeArm_y', 'LeftForeArm_z', 'RightArm_x',
             'RightArm_y', 'RightArm_z', 'RightForeArm_x', 'RightForeArm_y',
             'RightForeArm_z', 'Spine1_x', 'Spine1_y', 'Spine1_z']

jimmy_clips = []
mocap_clips = []

#=================
# Separate data in dataset into two bins
#=================
@elapsed
def split_data(filenames):
    j = []
    m = []
    for ff in filenames:
        for d in jimmy_dof:
            if d in ff:
                j.append(ff)
                break
        else:
            m.append(ff)
    return j, m

print("Splitting dataset ...")
jimmy_clips, mocap_clips = split_data(filefiles)
print("done!")
#print(len(jimmy_clips))
#print(len(mocap_clips))

#=========================================
# Collect clips into dict for permutations
#=========================================
@elapsed
def collect_motion_set(clips, dofset):
    """
        Collect clips per DOF into dictionaries, keyed by the action name
    """
    clipset = {}
    for c in clips:
        for dof in dofset:
            dof = '_'+dof
            name = c
            if dof in c:
                name = name.replace(dof, '')  
                break
        if name in clipset:
            clipset[name].append(c)
        else:
            clipset[name] = [c]
    return clipset

print("Collecting dataset ...")
jimhjim = collect_motion_set(jimmy_clips, jimmy_dof)
mocamoc = collect_motion_set(mocap_clips, mocap_dof)
print("done!")

#====================
# Create permutations
#====================
@elapsed
def collect_motionset_combinations(collection, choose=2, klaas=1.0):
    collected = {}
    for c in collection:
        
        for f in itertools.combinations(collection[c], choose):
            data = []
            columns = []
            fname = c.replace('.csv','') + '_'
            name = '_'.join(f).replace(fname,'').replace('.csv','')
            name = fname + '_' + name
            
            for joint_data in f:
                
                joint_name = joint_data.replace(fname,'').replace('.csv','')
                
                columns.append(joint_name)
                readdata = pd.read_csv(path_path + joint_data).values
                if len(data) == 0:
                    data = readdata
                else:
                    data = np.hstack((data, readdata))
            
            if len(data) > 0:
                klaas_col = np.full((len(data),1), klaas)
                data = np.hstack((data, klaas_col))
                columns += ['class']
                collected[name] = pd.DataFrame(data, columns=columns)                
    return collected

print("Collecting dataset combinations...")
collected_data_jimhjim = collect_motionset_combinations(jimhjim, klaas=0.0)
collected_data_mocamoc = collect_motionset_combinations(mocamoc, klaas=1.0)
print("Collected {} jimmy dataset. and {} mocap datasets.".format(
        len(collected_data_jimhjim), len(collected_data_mocamoc)))

#=======================
# Collect slices of data
#=======================
#@elapsed
def get_slices(data, slice_size=30, step=10):
    """
        Get subset of data for training/test/validation
    """
    yumyum = []
    
    for i in np.arange(0, len(data)-step+1, step):
        yumyum.append(data[i:i+slice_size])
    return yumyum

@elapsed
def get_collected_slices(data, slice_size=30, step=10):
    """
        Collect all slices and organize by name
    """
    collected_slices = {}
    
    for key in data:
               
        slices = get_slices(data[key], slice_size, step)
#        print("key: {} ({})".format(key, len(slices)))
        for j in range(len(slices)):
            index = "{}_{}".format(key, j)
            if len(slices[j]) < slice_size:
                continue
            collected_slices[index] = slices[j]
            
    return collected_slices
            
slice_size = 30
step = 10
#jimhjim_slices = {key: get_slices(collected_data_jimhjim[key]) for key in collected_data_jimhjim}
#mocamoc_slices = {key: get_slices(collected_data_mocamoc[key]) for key in collected_data_mocamoc}
print("Collecting slices of size: {} and step size: {} ...".format(slice_size, step))    
jimhjim_slices = get_collected_slices(collected_data_jimhjim, slice_size=slice_size, step=step)
mocamoc_slices = get_collected_slices(collected_data_mocamoc, slice_size=slice_size, step=step)
print("Dataset size:")
print("jimmy data w/ slices: {}".format(len(jimhjim_slices)))
print("mocap_data w/ slices: {}".format(len(mocamoc_slices)))

#=========================================
# Determine training/test/validation sizes
#=========================================
print("Collecting training/test/validation datasets ...")
base = min(len(jimhjim_slices), len(mocamoc_slices))

train, validation, test = 0.8, 0.1, 0.1
train_size = int(train * base)
validation_size = int(validation * base)
test_size = int(test * base)

print("train_size: {}".format(train_size))
print("test_size: {}".format(test_size))
print("validation_size: {}".format(validation_size))

print("\nCollecting jimmy datasets ...")
jimmy_dataset = [data for k, data in jimhjim_slices.items()]
np.random.shuffle(jimmy_dataset)
jimmy_train = jimmy_dataset[:train_size]
jimmy_test = jimmy_dataset[train_size:train_size+test_size]
jimmy_validation = jimmy_dataset[train_size+test_size:train_size+test_size+validation_size]

print("\nCollecting mocap datasets ...")
mocap_dataset = [data for k, data in mocamoc_slices.items()]
np.random.shuffle(mocap_dataset)
mocap_train = mocap_dataset[:train_size]
mocap_test = mocap_dataset[train_size:train_size+test_size]
mocap_validation = mocap_dataset[train_size+test_size:train_size+test_size+validation_size]

print("Done!")

