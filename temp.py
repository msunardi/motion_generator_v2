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
from keras.callbacks import TensorBoard
import numpy as np
import time
from datetime import timedelta
import datetime
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
    for i in range(len(data)-1):
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

@elapsed
def get_labels2(some_data):
    class0 = 0
    class1 = 0
    datata0 = []
    datata1 = []
    datata= []
    labels = []
    samples = []
    le_bin = None

    for col in some_data.columns:
        thedata = some_data[col].values
        l = len(thedata)
        lab = get_label(col) # Get label based on column/filename
        
        print("{} labeled: {}".format(col, lab))
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
                 'label': tmp_lab,
                 'sample': col})

    dat_dat = min(class0, class1)
#    print("class0x1: {} x {}".format(class0, class1))
#    print("class0: {}".format(datata0))
#    print("class1: {}".format(datata1))
    for fdata in [datata0, datata1]:
        if len(fdata) > 0:
            pick = np.random.choice(fdata, size=dat_dat, replace=False)
            datata.extend([k['data'] for k in pick])
            labels.extend([k['label'] for k in pick])
            samples.extend([k['sample'] for k in pick])

    shuffled_index = np.random.choice(range(len(datata)), size=len(datata), replace=False)
    datatax = [datata[i] for i in shuffled_index]
    labelsx = [labels[i] for i in shuffled_index]
    samplesx = [samples[i] for i in shuffled_index]

    print('Collected {} data and {} labels.'.format(len(datatax), len(labelsx)))
    return datatax, labelsx, samplesx

@elapsed
def shuffle(data, labels, samples=None, size=None):
    # Check data and labels are the same size
    assert len(data) == len(labels)
    if not size:
        size = len(data)
    idx = np.random.choice(range(len(data)), size=size, replace=False)
    print("Shuffle: {}".format(idx))
    datax = np.array([data[i].reshape(timesteps,1) for i in idx], dtype=np.float32)
    labelsx = np.array([labels[i] for i in idx], dtype=np.float32)
    print("Shuffled: {} data and {} labels".format(len(datax), len(labelsx)))
    if samples:
        samplesx = [samples[i] for i in idx]
        return datax, labelsx, samplesx
    return datax, labelsx


# Plot by first converting to pd.DataFrames
# Example: pdplot_data(test_x_shuffled, [(42,1), (0,1), (2,0), (3,1), (4,0)])
# (42,1) = test_x_shuffled[42], class=1
def pdplot_data(data, labels=None, samples=[], sample_names=[], figsize=()):
    # samples should be list of tuples: (index, class)
    if len(labels) > 0:
        np_toplot = np.array([data[i].T[0] for i in samples])
        pd_labels = [get_target_index(labels[i][0]) for i in samples]
        pd_sample_label = [z for z in zip(samples, pd_labels)]
        pd_toplot = pd.DataFrame(data=np_toplot.T, columns=["{} ({})".format(s[0],s[1]) for s in pd_sample_label])
    elif all([type(s)==tuple for s in samples]):
        np_toplot = np.array([data[i[0]].T[0] for i in samples])
        pd_toplot = pd.DataFrame(data=np_toplot.T, columns=["{} ({})".format(s[0],s[1]) for s in samples])
    else:
        np_toplot = np.array([data[i].T[0] for i in samples])
        pd_toplot = pd.DataFrame(data=np_toplot.T, columns=samples)
    if bool(figsize):
        pd_toplot.plot(figsize=figsize)
    else:
        pd_toplot.plot()
    if len(sample_names) > 0:
        for i in samples:
            print('{} - {}'.format(i, sample_names[i]))

# Stupid boring plot from numpy array (data), and index (sample)
def plot_from_test(data, sample):
    plt.plot(data[sample].T[0])  

def verify(labels, samples, n=20):
    for i in range(n):
        print('{}: {}'.format(samples[i], labels[i][0]))    

# =============================================================================
# Parameters
# =============================================================================
timesteps = 50
num_classes = 2
batch_size = 50
hidden_size8 = 8
hidden_size64 = 64
hidden_size128 = 128
hidden_size256 = 256
hidden_size512 = 512

hidden_size = hidden_size128
data_dim = 1
epochs = 20

# =============================================================================
# Model (GRU)
# =============================================================================
model = None
model = Sequential()
model.add(GRU(hidden_size64, return_sequences=True, stateful=True,
               batch_input_shape=(batch_size, timesteps, data_dim)))
model.add(GRU(hidden_size128, return_sequences=True, stateful=True))
model.add(GRU(hidden_size256, return_sequences=True, stateful=True))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# =============================================================================
# Model (LSTM)
# =============================================================================
model = None
model = Sequential()
model.add(LSTM(hidden_size64, return_sequences=True, stateful=True,
               batch_input_shape=(batch_size, timesteps, data_dim)))
model.add(LSTM(hidden_size128, return_sequences=True, stateful=True))
model.add(LSTM(hidden_size256, return_sequences=True, stateful=True))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# =============================================================================
# Collect some data
# =============================================================================
#all_data = get_data()
all_data = get_data_plot(offset=10)

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


# =============================================================================
# Split to train, validation, and test data
# =============================================================================
#train_x, train_y, train_skipped = get_labels(train_data)
#validate_x, validate_y, validate_skipped = get_labels(validation_data)
#test_x, test_y, test_skipped = get_labels(test_data)

train_x, train_y, train_samples = get_labels2(train_data)
validate_x, validate_y, validate_samples = get_labels2(validation_data)
test_x, test_y, test_samples = get_labels2(test_data)

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

#train_x, train_y = shuffle(train_x, train_y, size=t_size)
#validate_x, validate_y = shuffle(validate_x, validate_y, size=v_size)
train_x, train_y, train_samplesx = shuffle(train_x, train_y, samples=train_samples, size=t_size)
validate_x, validate_y, v_samples = shuffle(validate_x, validate_y, samples=validate_samples, size=v_size)

# =============================================================================
# Tensorboard - Ref: http://fizzylogic.nl/2017/05/08/monitor-progress-of-your-keras-based-neural-network-using-tensorboard/
# =============================================================================
tensorboard = TensorBoard(log_dir="logs/{}".format(datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")))
# =============================================================================
# Train!
# =============================================================================
start = time.time()

print(model.summary())
model.fit(train_x, train_y,
          batch_size=batch_size, epochs=epochs, shuffle=True,
          validation_data=(validate_x, validate_y), callbacks=[tensorboard])

end = time.time()
elapsed = end-start
print('Training Elapsed: {}'.format(str(timedelta(seconds=elapsed))))
# =============================================================================
# Test!
# =============================================================================
#predictions = model.predict_on_batch(test_x[:batch_size]) # old
# Shuffle test data
test_shuffle = np.random.choice(range(len(test_x)), size=batch_size, replace=False)
test_x_shuffled = np.array([test_x[i].reshape(50,1) for i in test_shuffle], dtype=np.float32)
test_y_shuffled = np.array([test_y[i] for i in test_shuffle], dtype=np.float32)
test_samples_shuffled = [test_samples[i] for i in test_shuffle]
test_x_shuffled, test_y_shuffled, test_samples_shuffled = shuffle(test_x, test_y, test_samples, size=batch_size)

#test_x_shuffled, test_y_shuffled = shuffle(test_x, test_y, size=batch_size)

predictions = model.predict_on_batch(test_x_shuffled)


gt = get_target_index
print("sample\t\t\t\t\tactual\t\tpredicted\t\tprediction error (over correct class)")
true_positive = 0
false_positive = 0
true_negative= 0
false_negative = 0

for k in range(len(predictions)):
    prediction = predictions[k][-1] # Only choose the last prediction
    actual = gt(test_y_shuffled[k][-1])
    
    error = calculate_error(actual, prediction)
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
    sample_name = test_samples_shuffled[k]
    print("%s\t%s\t%s  \t%s" % (sample_name[:15] + '...' + sample_name[-15:], actual, prediction, error))
print("\n================\n")
print("Confusion matrix\n")
print("================\n")
print("\tP\tN")
print("P\t{}\t{}".format(true_positive, false_negative))
print("N\t{}\t{}".format(false_positive, true_negative))

precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)

print("Precision: {}".format(precision))
print("Recall: {}".format(recall))

# =============================================================================
# Model saving and loading - Ref; https://keras.io/models/about-keras-models/
# =============================================================================
from keras.models import model_from_json, model_from_yaml
json_string = model.to_json() # only save architecture
model.save_weights('models/gru_64_128_256_v4_weights.h5') # saves weights
model.save('models/gru_64_128_256_v4.h5')

