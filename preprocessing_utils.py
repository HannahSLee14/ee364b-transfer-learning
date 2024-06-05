import mat73
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from pyriemann.utils.covariance import normalize

# copied from original paper code (Link: https://github.com/Kyungho-Won/EEG-dataset-for-RSVP-P300-speller/blob/main/Python/Load_Won2021dataset.ipynb)
def butter_lowpass_filter(data, lowcut, fs, order):
  nyq = fs/2
  low = lowcut/nyq
  b, a = butter(order, low, btype='low')
  y = filtfilt(b, a, data) # zero-phase filter # data: [ch x time]
  return y

# copied from original paper code (Link: https://github.com/Kyungho-Won/EEG-dataset-for-RSVP-P300-speller/blob/main/Python/Load_Won2021dataset.ipynb)
def butter_highpass_filter(data, highcut, fs, order):
  nyq = fs/2
  high = highcut/nyq
  b, a = butter(order, high, btype='high')
  y = filtfilt(b, a, data) # zero-phase filter
  return y

# copied from original paper code (Link: https://github.com/Kyungho-Won/EEG-dataset-for-RSVP-P300-speller/blob/main/Python/Load_Won2021dataset.ipynb)
def butter_bandpass_filter(data, lowcut, highcut, fs, order):
  nyq = fs/2
  low = lowcut/nyq
  high = highcut/nyq
  b, a = butter(order, [low, high], btype='band')
  # demean before filtering
  meandat = np.mean(data, axis=1)
  data = data - meandat[:, np.newaxis]
  y = filtfilt(b, a, data)
  return y

# copied from original paper code (Link: https://github.com/Kyungho-Won/EEG-dataset-for-RSVP-P300-speller/blob/main/Python/Load_Won2021dataset.ipynb)
def extractEpoch3D(data, event, srate, baseline, frame, opt_keep_baseline):
  # extract epoch from 2D data into 3D [ch x time x trial]
  # input: event, baseline, frame
  # extract epoch = baseline[0] to frame[2]

  # for memory pre-allocation
  if opt_keep_baseline == True:
    begin_tmp = int(np.floor(baseline[0]/1000*srate))
    end_tmp = int(begin_tmp+np.floor(frame[1]-baseline[0])/1000*srate)
  else:
    begin_tmp = int(np.floor(frame[0]/1000*srate))
    end_tmp = int(begin_tmp+np.floor(frame[1]-frame[0])/1000*srate)
  
  epoch3D = np.zeros((data.shape[0], end_tmp-begin_tmp, len(event)))
  nth_event = 0

  for i in event:
    if opt_keep_baseline == True:
      begin_id = int(i + np.floor(baseline[0]/1000 * srate))
      end_id = int(begin_id + np.floor((frame[1]-baseline[0])/1000*srate))
    else:
      begin_id = int(i + np.floor(frame[0]/1000 * srate))
      end_id = int(begin_id + np.floor((frame[1]-frame[0])/1000*srate))
    
    tmp_data = data[:, begin_id:end_id]

    begin_base = int(np.floor(baseline[0]/1000 * srate))
    end_base = int(begin_base + np.floor(np.diff(baseline)/1000 * srate)-1)
    base = np.mean(tmp_data[:, begin_base:end_base], axis=1)

    rmbase_data = tmp_data - base[:, np.newaxis]
    epoch3D[:, :, nth_event] = rmbase_data
    nth_event = nth_event + 1

  return epoch3D

# modified from original paper code (Link: https://github.com/Kyungho-Won/EEG-dataset-for-RSVP-P300-speller/blob/main/Python/Load_Won2021dataset.ipynb)
def extractSession(eeg, type, sess_num, baseline, frame, elec_to_keep=[], opt_keep_baseline=False):
    # extract session of data organized into epochs [ch x time x trial]
    # input: eeg = eeg data
    #       type = 'train' or 'test'
    #       sess_num = 0-1 for 'train', 0-3 for 'test'
    #       baseline = baseline window in ms
    #       frame = frame window in ms
    # output: target EEG data, nontarget EEG data

    # get the data
    data = np.asarray(eeg[type][sess_num]['data'])

    # only keep relevant electrodes - similar to used in https://ieeexplore.ieee.org/document/7318721
    # elec_to_keep = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T7', 'Cz', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2']
    # based on visual inspection:
    if len(elec_to_keep) == 0:
        elec_to_keep = ['FP1', 'FP2', 'F3', 'FZ', 'F4', 'T7', 'Cz', 'T8', 'P7', 'P8', 'O1', 'Oz', 'O2']
    idx_to_keep = []
    elec = eeg[type][sess_num]['chanlocs']
    for i in range(len(elec)):
        if elec[i]['labels'] in elec_to_keep:
            idx_to_keep.append(i)
    data = data[idx_to_keep]
    
    # BP filter data from 1-20 Hz using Butterworth filter 
    srate = eeg[type][sess_num]['srate']
    data = butter_bandpass_filter(data, 1, 20, srate, 5) # matching filter from https://arxiv.org/abs/1409.0107
    markers = eeg[type][sess_num]['markers_target']

    targetID = np.where(markers==1)[0]
    nontargetID = np.where(markers==2)[0]

    targetEEG = extractEpoch3D(data, targetID, srate, baseline, frame, opt_keep_baseline)
    nontargetEEG = extractEpoch3D(data, nontargetID, srate, baseline, frame, opt_keep_baseline)
    
    return targetEEG, nontargetEEG

# copied from original paper code (Link: https://github.com/Kyungho-Won/EEG-dataset-for-RSVP-P300-speller/blob/main/Python/Load_Won2021dataset.ipynb)
def extractAllSessions(eeg, type, baseline, frame, elec_to_keep=[], opt_keep_baseline=False):
    # extract all data session trials and concatanates them [ch x time x trial]
    # input: eeg = eeg data
    #       type = 'train' or 'test'
    #       sess_num = 0-1 for 'train', 0-3 for 'test'
    #       baseline = baseline window in ms
    #       frame = frame window in ms
    # output: all target EEG data, all nontarget EEG data
    for n_sess in range(len(eeg[type])):
        tmp_targetEEG, tmp_nontargetEEG = extractSession(eeg, type, n_sess, baseline, frame, elec_to_keep, opt_keep_baseline)

    if n_sess == 0:
        all_targetEEG = tmp_targetEEG
        all_nontargetEEG = tmp_nontargetEEG
    else:
        all_targetEEG = np.dstack((all_targetEEG, tmp_targetEEG))
        all_nontargetEEG = np.dstack((all_nontargetEEG, tmp_nontargetEEG))

    return all_targetEEG, all_nontargetEEG

def normalize_data(data):
    n_elec, n_samp, n_trial = data.shape
    max_over_trial_per_elec = np.max(np.abs(data), axis=1)
    max_over_trial_per_elec = np.tile(max_over_trial_per_elec, (n_samp, 1, 1,))
    max_over_trial_per_elec = np.moveaxis(max_over_trial_per_elec, 0, 1)

    return data / max_over_trial_per_elec

# adapted from this paper: https://ieeexplore.ieee.org/document/8013808, https://arxiv.org/abs/1409.0107 
def prepare_covmat(sess_data_target, sess_data_nontarget):
    # prepares the covariance matrices that are used as features for classification
    # input: sess_data_target, sess_data_nontarget (output of extractSession function)
    # output: all target covariance matrices, all nontarget covariance matrices
    
    n_elec, n_samp, n_trial_target = sess_data_target.shape
    n_elec, n_samp, n_trial_nontarget = sess_data_nontarget.shape
    n_trial = n_trial_target + n_trial_nontarget

    # normalize the data
    target_data = sess_data_target.copy()
    target_data = normalize_data(target_data)
    nontarget_data = sess_data_nontarget.copy()
    nontarget_data = normalize_data(nontarget_data)

    # prepare augmented matrices
    E_mat = np.mean(target_data, axis=2)
    E_mat_target = np.tile(E_mat, (n_trial_target, 1, 1))
    E_mat_target = np.moveaxis(E_mat_target, 0, -1)
    aug_data_target = np.vstack([E_mat_target, target_data])

    E_mat_nontarget = np.tile(E_mat, (n_trial_nontarget, 1, 1))
    E_mat_nontarget = np.moveaxis(E_mat_nontarget, 0, -1)
    aug_data_nontarget = np.vstack([E_mat_nontarget, nontarget_data])

    # create augmented covariance matrices
    target_cov = np.asarray([((1 / (n_samp - 1)) * aug_data_target[:,:,i] @ aug_data_target[:,:,i].T) for i in range(n_trial_target)])
    nontarget_cov = np.asarray([((1 / (n_samp - 1)) * aug_data_nontarget[:,:,i] @ aug_data_nontarget[:,:,i].T) for i in range(n_trial_nontarget)])

    return target_cov, nontarget_cov

def processSessionsIndiv(eeg, baseline, frame, elec_to_keep=[], opt_keep_baseline=False):
  train_target_cov = []
  train_nontarget_cov = []
  test_target_cov = []
  test_nontarget_cov = []

  for n_train in range(len(eeg['train'])):
      target_data, nontarget_data = extractSession(eeg, 'train', n_train, baseline, frame, elec_to_keep, opt_keep_baseline)
      target_cov, nontarget_cov = prepare_covmat(target_data, nontarget_data)
      train_target_cov.append(target_cov)
      train_nontarget_cov.append(nontarget_cov)

  train_target_cov = np.array(train_target_cov)
  train_nontarget_cov = np.array(train_nontarget_cov)

  for n_test in range(len(eeg['test'])):
      target_data, nontarget_data = extractSession(eeg, 'test', n_test, baseline, frame, elec_to_keep, opt_keep_baseline)
      target_cov, nontarget_cov = prepare_covmat(target_data, nontarget_data)
      test_target_cov.append(target_cov)
      test_nontarget_cov.append(nontarget_cov)

  test_target_cov = np.array(test_target_cov)
  test_nontarget_cov = np.array(test_nontarget_cov)

  # normalize the data
  train_target_cov = normalize(train_target_cov, "corr")
  train_nontarget_cov = normalize(train_nontarget_cov, "corr")
  test_target_cov = normalize(test_target_cov, "corr")
  test_nontarget_cov = normalize(test_nontarget_cov, "corr")

  return train_target_cov, train_nontarget_cov, test_target_cov, test_nontarget_cov

# modified from: https://github.com/Kyungho-Won/EEG-dataset-for-RSVP-P300-speller/blob/main/Python/RSVP_visualization_ERP.py
def processRSVP(eeg, baseline, frame, elec_to_keep=[], opt_keep_baseline=False):
  cur_EEG = eeg['RSVP']
  data = np.asarray(cur_EEG['data'])
  srate = cur_EEG['srate']
  data = butter_bandpass_filter(data, 1, 20, srate,5)
  markers = cur_EEG['markers_target']

  if len(elec_to_keep) == 0:
      elec_to_keep = ['FP1', 'FP2', 'F3', 'FZ', 'F4', 'T7', 'Cz', 'T8', 'P7', 'P8', 'O1', 'Oz', 'O2']
  idx_to_keep = []
  elec = cur_EEG['chanlocs']
  for i in range(len(elec)):
      if elec[i]['labels'] in elec_to_keep:
          idx_to_keep.append(i)
  data = data[idx_to_keep]

  targetID = np.where(markers==1)[0]
  nontargetID = np.where(markers==2)[0]

  targetEEG = extractEpoch3D(data, targetID, srate, baseline, frame, opt_keep_baseline)
  nontargetEEG = extractEpoch3D(data, nontargetID, srate, baseline, frame, opt_keep_baseline)

  target_cov, nontarget_cov = prepare_covmat(targetEEG, nontargetEEG)

  # normalize the data
  target_cov = normalize(target_cov, "corr")
  nontarget_cov = normalize(nontarget_cov, "corr")

  return target_cov, nontarget_cov
