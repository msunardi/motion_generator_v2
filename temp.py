# -*- coding: utf-8 -*-
# https://gist.github.com/msunardi/f00745948aaeb456f6a34ec48159bc33
from os import listdir
from os.path import isfile, join

import time
from datetime import timedelta

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

def elapsed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        f = func(*args, **kwargs)  # Call the method
        end = time.time()
        elapsed = end-start
        print('Elapsed: {}'.format(str(timedelta(seconds=elapsed))))
        return f  # Return whatever the method returns
    return wrapper

#path_path = '/home/mathias/Projects/motion_data'
#path_path = '/home/mathias/Projects/Blender/combined_dataset'
path_path = '/home/mathias/Projects/Blender/dataset'
onlyfiles = [f for f in listdir(ros_path) if isfile(join(ros_path, f))]

# Print the filenames found
for file in onlyfiles:
    print(file)
    
filefiles = [f for f in onlyfiles if '.csv' in f]
for ff in filefiles:
    print(ff)

# =============================================================================
# with open(path_path + '/train_derivative.csv') as f:
#     fileread = f.read()
#     print(len(fileread))
# =============================================================================
    
@elapsed
def normalize(file):
    rad = 6.2831853
    out = []
    with open(path_path + '/male2_a3_swingarms.csv') as f:
        fileread = f.read()
        i = 0
        for ea in fileread.split('\n'):
            if len(ea) > 1:            
                print(ea.split(','))
                ea = [float(x) for x in ea.split(',')]
                mean = np.mean(np.array(ea).astype(np.float16))
                tmp = map(lambda x: (x-mean)/mean, ea)
                out.append(tmp)
                i += 1
    return out

ros_path = '/home/mathias/catkin_ws/src/motion_generator/src/csv'
#normd = normalize(path_path)
filename = '/male2_a3_swingarms.csv'
#filename = 'male2_e8_uppercutright.csv'
filepath = '{}/{}'.format(path_path, filename)
normd = pd.read_csv(filepath)
#normd['RightForeArm_x'].plot()
filename = 'alas2_linear.csv'
filename = 'alas2_cubic.csv'
filename = 'brah_cubic_0.csv'
filename = 'brah_linear_0.csv'
filename = 'clap_cubic_0.csv'
filename = 'clap_linear_0.csv'
filepath = '{}/{}'.format(ros_path, filename)
normd = pd.read_csv(filepath)
normd['R_ELBOW'].plot()
normd.plot(figsize=(15,10))

np.mean(normd['RightForeArm_x'])
mean = np.mean(normd['LeftForeArm_x'])
print(type(normd['LeftForeArm_x']))
mm = map(lambda x: x - mean, normd['LeftForeArm_x'])
normd['LeftForeArm_x'] = [z for z in mm]
print(type(normd['LeftForeArm_x']))
plt.plot(zoop)

mavg = normd['LeftForeArm_x'].rolling(30).mean()
mavg.plot(figsize=(15,10))
normd['LeftForeArm_x'].plonlyfiles = [f for f in listdir(ros_path) if isfile(join(ros_path, f))]

# Print the filenames found
for file in onlyfiles:
    print(file)
    
filefiles = [f for f in onlyfiles if '.csv' in f]
for ff in filefiles:
    print(ff)t(figsize=(15,10), kind='kde')
dir(normd)
normd.info

for zx in normd.columns:
    print(zx)


@elapsed
def collect_data(sourcepath, target_dir = './dataset'):
    onlyfiles = [f for f in listdir(sourcepath) if isfile(join(sourcepath, f))]        
    filefiles = [f for f in onlyfiles if '.csv' in f]
    
    num = 0
    for filename in filefiles:
        filepath = '{}/{}'.format(sourcepath, filename)
        fubar = pd.read_csv(filepath)
        
        for col in fubar.columns:
            mean = np.mean(fubar[col])
            new_data = [f for f in map(lambda x: x - mean, fubar[col])]
            if 'Blender' in sourcepath:
                new_data.append(0.0)
            elif 'catkin_ws' in sourcepath:
                new_data.append(1.0)
            new_data = pd.DataFrame(new_data)
            new_filename = filename.replace('+','_')
            for zz in ['.csv','(',')',' ']:
                new_filename = new_filename.replace(zz, '')
            new_filepath = '{}/{}_{}.csv'.format(target_dir, new_filename, col)
            print('Writing: {} ...'.format(new_filepath))
            new_data.to_csv(new_filepath, index=False)
            num += 1
    
    print('Written {} files.'.format(num))
            
# =============================================================================
# Keras
# =============================================================================
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, GRU, Dropout, SimpleRNN
import numpy as np
import time
from datetime import timedelta
import numpy.random as nr
import pandas as pd
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

# =============================================================================
# Helper methods
# =============================================================================
@elapsed
def get_data(path='./dataset'):
    # list all the files in path
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]   
    # only collect .csv files     
    filefiles = [f for f in onlyfiles if '.csv' in f]
    
    print("Found {} files.".format(len(filefiles)))
    
    fubar = []
    
    for filename in filefiles:
        filepath = '{}/{}'.format(path, filename)
        adata = pd.read_csv(filepath)
        fubar.append(adata)
        
    return pd.DataFrame(fubar)
    
@elapsed
def get_data_plot(path='./dataset', limit=None, drange=None, min_length=100, offset=15):
    # list all the files in path
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]   
    # only collect .csv files     
    filefiles = [f for f in onlyfiles if '.csv' in f]
    #print(filefiles)
    #return

    print("Found {} files.".format(len(filefiles)))

    fubar = []
    fu_cols = []
    
    if limit:
        ff = filefiles[:limit]
    else:
        ff = filefiles
        
    if bool(drange) and type(drange) == tuple:
        ff = filefiles[drange[0]:drange[1]]


    old_label = None
    for filename in ff:
        filepath = '{}/{}'.format(path, filename)
        adata = pd.read_csv(filepath)

        fname = filename.replace('.csv','')
        if len(adata) < min_length:
            print("Skipping: {} len: {} < {}".format(fname, len(adata), min_length))
            continue
        walks = walk(adata.values, offset=offset)
        
        for i in range(len(walks)):
            a_walk = walks[i]               
            label = "{}_{}".format(fname, i)

            
            # Check for array with static values
            if list(a_walk).count(a_walk[0]) == len(list(a_walk)):
                print("Skipping: {} - identical values".format(label))
                continue       
            
            fubar.append(a_walk)  
#            zup = pd.DataFrame(data=a_walk, columns=[label]) 
            fu_cols.append(label)
            if old_label != label:
                old_label = label
            else:
                raise ValueError("old label == new label!")
            # zup.plot()          
        
    #return fubar, fu_cols
    
    # Convert data into rows (sequence) x columns (samples)
    fubark = np.array([fu.T[0] for fu in fubar])
    assert fubark.T.shape[1] == len(fu_cols)
    return pd.DataFrame(data=fubark.T, columns=fu_cols)

def walk(data, steps=50, offset=5):
    rr = []
    for i in range(len(data)):
        tmp = data[i*offset:i*offset+steps]
        if len(tmp) < steps:
            break
        rr.append(tmp)
    return rr

def get_target_index(target):
    return 0 if target[0] == 1.0 else 1

def calculate_error(actual, prediction):
    return 1.0 - prediction[int(actual)]

def prediction_confidence(actual, prediction):
    return prediction[int(actual)]

def get_label(key):
    if any([k in key for k in ['HEAD_PAN', 'HEAD_TILT', 'L_ELBOW',
                               'L_SHO_PITCH', 'L_SHO_ROLL', 'R_ELBOW',
                               'R_SHO_PITCH', 'R_SHO_ROLL']]):
        return 1.0
    return 0.0

# =============================================================================
# Parameters
# =============================================================================
timesteps = 50
num_classes = 2
batch_size = 50
hidden_size = 512
data_dim = 1
epochs = 30

# =============================================================================
# Model
# =============================================================================

model = Sequential()
model.add(GRU(hidden_size, return_sequences=True, stateful=True,
               batch_input_shape=(batch_size, timesteps, data_dim)))
model.add(GRU(hidden_size, return_sequences=True, stateful=True))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# =============================================================================
# Collect some data
# =============================================================================
all_data = get_data()
all_data = get_data_plot(offset=20)

train_data = []
validation_data = []
test_data = []

# Set ratio of train, validation, and test data
r_train, r_validation, r_test = 0.8, 0.1, 0.1

tot_size = len(all_data.columns)

n_train = int(r_train * tot_size)
n_validation = int(r_validation * tot_size)
n_test = int(r_test * tot_size)

randomized_index = np.random.choice(all_data.columns, size=len(all_data.columns), replace=False)
#randomized_index = np.random.choice(range(len(all_data)), size=5, replace=False)
randomed_data = np.array([all_data[col].values for col in randomized_index]).T
randomized_data = pd.DataFrame(data=randomed_data, columns=randomized_index)

train_cols = randomized_index[:n_train]
train_data = pd.DataFrame(data=np.array([randomized_data[col].values for col in train_cols]).T, columns=train_cols)
val_cols = randomized_index[n_train:n_train+n_validation]
validation_data =pd.DataFrame(data=np.array([randomized_data[col].values for col in val_cols]).T, columns=val_cols)
test_cols = randomized_index[n_train+n_validation:]
test_data = pd.DataFrame(data=np.array([randomized_data[col].values for col in test_cols]).T, columns=test_cols)

@elapsed
def get_labels2(some_data):
    class0 = 0
    class1 = 0
    datata0 = []
    datata1 = []
    datata= []
    labels = []
    le_bin = None

    for col in some_data.columns:
        thedata = some_data[col].values
        l = len(thedata)
        lab = get_label(col) # Get label based on column/filename

        if lab == 0.0:
            le_bin = datata0
            class0 += 1
        else:
            le_bin = datata1
            class1 += 1

        # label is two-vector (softmax)
        lablab = [lab, 1 - lab]
        tmp_lab = np.array([lablab for i in range(l)])

        le_bin.append({'data': thedata, 
                 'label': tmp_lab})

    dat_dat = min(class0, class1)
#    print("class0x1: {} x {}".format(class0, class1))
#    print("class0: {}".format(datata0))
#    print("class1: {}".format(datata1))
    for fdata in [datata0, datata1]:
        pick = np.random.choice(fdata, size=dat_dat, replace=False)
        datata.extend([k['data'] for k in fdata])
        labels.extend([k['label'] for k in fdata])

    print('Collected {} data and {} labels.'.format(len(datata), len(labels)))
    return datata, labels

# Collect data vs. labels *OBSOLETE*
@elapsed
def get_labels(some_data):
    datata = []
    
    # Bins to collect data of the two classes; used to balance the dataset
    datata0 = []
    datata1 = []    
    le_bin = None
    class0 = 0
    class1 = 0
    labels = []
    skipped = []
    for d in some_data:
        l = len(d)
        lab = d['0'].values[-1]
        if lab not in [0.0, 1.0] or l < timesteps:
            skipped.append(d['0'])
            continue
        if lab == 0.0:
            le_bin = datata0
            class0 += 1
        else:
            le_bin = datata1
            class1 += 1
            
        val = d['0'].values
        # last working version
#        lablab = []
#        start = i * timesteps
#        end = start + timesteps
#        for x in range(timesteps):
#            lablab.append([lab, 1 - lab])
##        print(lablab)
#        lablab = np.array(lablab, dtype=np.float32)
#        datata.append(np.array(val[:timesteps], dtype=np.float32).reshape(timesteps,1))
#        labels.append(np.array(lablab, dtype=np.float32).reshape(timesteps,2))
        
        # See how many chunks of size <timesteps> can be extracted from each data (working)
#        for i in range(len(val)//timesteps):
#            lablab = []
#            start = i * timesteps
#            end = start + timesteps
#            for x in range(timesteps):
#                lablab.append([lab, 1 - lab])
#                
#            lablab = np.array(lablab, dtype=np.float32)
#            datata.append(np.array(val[start:end], dtype=np.float32).reshape(timesteps,1))
#            labels.append(np.array(lablab, dtype=np.float32).reshape(timesteps,2))
        for i in range(len(val)//timesteps):
            lablab = []
            start = i * timesteps
            end = start + timesteps
            for x in range(timesteps):
                lablab.append([lab, 1 - lab])
                
            lablab = np.array(lablab, dtype=np.float32)
            data_to_store = np.array(val[start:end], dtype=np.float32).reshape(timesteps,1)
            label_to_store = np.array(lablab, dtype=np.float32).reshape(timesteps,2)
            le_bin.append({'data': data_to_store, 
                 'label': label_to_store})
    
    # Now pick the data so the number of data points for each class is balanced
    dat_dat = min(class0, class1)
    for fdata in [datata0, datata1]:
        pick = np.random.choice(range(dat_dat), size=dat_dat, replace=False)
        datata.extend([k['data'] for k in fdata])
        labels.extend([k['label'] for k in fdata])
    # Shuffle the data
    total_data = 0
    if len(datata) == len(labels):
        total_data = len(datata)
        shuffle = np.random.choice(range(total_data), size=total_data, replace=False)
        datata = [datata[i] for i in shuffle]
        labels = [labels[i] for i in shuffle]
    else:
        raise ValueError('datata =/= labels')
    
    print('Collected {} data and {} labels. Skipped {}.'.format(len(datata), len(labels), len(skipped)))
    return datata, labels, skipped

#train_x, train_y, train_skipped = get_labels(train_data)
#validate_x, validate_y, validate_skipped = get_labels(validation_data)
#test_x, test_y, test_skipped = get_labels(test_data)

train_x, train_y = get_labels2(train_data)
validate_x, validate_y = get_labels2(validation_data)
test_x, test_y = get_labels2(test_data)

# Just get batch-divisable data, and convert to numpy arrays
t_size = len(train_x)//batch_size * batch_size
train_x = np.array([t.reshape(t.shape[0],1) for t in train_x[:t_size]], dtype=np.float32)
train_y = np.array([t.reshape(t.shape[0],2) for t in train_y[:t_size]], dtype=np.float32)
v_size = len(validate_x)//batch_size * batch_size
validate_x = np.array([v.reshape(v.shape[0],1) for v in validate_x[:v_size]], dtype=np.float32)
validate_y = np.array([v.reshape(v.shape[0],2) for v in validate_y[:v_size]], dtype=np.float32)
ts_size = len(test_x)//batch_size * batch_size
test_x = np.array([s.reshape(s.shape[0],1) for s in test_x[:ts_size]], dtype=np.float32)
test_y = np.array([s.reshape(s.shape[0],2) for s in test_y[:ts_size]], dtype=np.float32)
# =============================================================================
# Train!
# =============================================================================
model.fit(train_x, train_y,
          batch_size=batch_size, epochs=epochs, shuffle=True,
          validation_data=(validate_x, validate_y))

# =============================================================================
# Test!
# =============================================================================
predictions = model.predict_on_batch(test_x[:100])

gt = get_target_index
print("sample\tactual\t\tpredicted\t\t\tprediction error (over correct class)")
true_positive = 0
false_positive = 0
true_negative= 0
false_negative = 0

for k in range(len(predictions)):
    p = predictions[k]
    actual = gt(test_y[k][-1])
    
    error = calculate_error(actual, p[-1])
    if error > 0.4:
        error = '**' + str(error) + '**'
        if actual == 0:
            false_positive += 1
        else:
            false_negative += 1
    else:
        if actual == 0:
            true_negative += 1
        else:
            true_positive += 1
    print("%s\t%s\t%s  \t%s" % (k, actual, p[-1], error))
print("\n================\n")
print("Confusion matrix\n")
print("================\n")
print("\tP\tN")
print("P\t{}\t{}".format(true_positive, false_negative))
print("N\t{}\t{}".format(false_positive, true_negative))