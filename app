#This is a condensed verison of the code in the colab notebook used to load the streamlit app - does not include addiitonal ui elements



import streamlit as st
import numpy as np
import scipy.io as sio
import mne
import pandas as pd
import os

# Global Constants
CH_NAMES = ['FC3', 'C3', 'CP3', 'FC4', 'C4', 'CP4']
SFREQ = 250

def load_mat_data(file_path, file_type='mat'):
    mat_data = sio.loadmat(file_path)
    raw_data = mat_data['RawEEGData']
    labels = mat_data['Labels']
    if raw_data.ndim != 3:
        raw_data = np.transpose(raw_data, (2, 0, 1))
    n_trials, n_channels_in_file, n_samples = raw_data.shape
    if n_channels_in_file > len(CH_NAMES):
        raw_data = raw_data[:, :len(CH_NAMES), :]
    return raw_data, labels

def load_csv_to_mne(csv_file, sfreq=250):
    df = pd.read_csv(csv_file)
    data = df[CH_NAMES].values.T
    data = data * 1e-6
    info = mne.create_info(ch_names=CH_NAMES, sfreq=SFREQ, ch_types='eeg')
    raw_data = mne.io.RawArray(data, info)
    return raw_data

@st.cache_data
def load_and_clean_data(file_path, file_type='mat'):
    if file_type == 'mat':
        raw_data, labels = load_mat_data(file_path)
        info = mne.create_info(ch_names=CH_NAMES, sfreq=SFREQ, ch_types='eeg')
        events_matrix = np.zeros((len(labels), 3), dtype=int)
        events_matrix[:, 0] = np.arange(len(labels)) * 1000
        events_matrix[:, 2] = labels.flatten()
        epochs = mne.EpochsArray(data=raw_data*1e-6, info=info, events=events_matrix, tmin=-3.0)
    else:
        raw_mne = load_csv_to_mne(file_path)
        events = mne.make_fixed_length_events(raw_mne, duration=8.0)
        labels = np.ones(len(events))
        epochs = mne.Epochs(raw_mne, events, tmin=-3.0, tmax=4.5, baseline=(-3.0, -1.5), preload=True)
    return epochs, labels

def calculate_psd(epochs_segmented):
    contra_motorstrip = epochs_segmented.copy().pick(['FC4', 'C4', 'CP4'])
    ipsi_motorstrip = epochs_segmented.copy().pick(['FC3', 'C3', 'CP3'])
    c_base = contra_motorstrip.compute_psd(tmin=-3, tmax=-1.5, fmin=8, fmax=30).get_data().mean(axis=(1, 2))
    c_task = contra_motorstrip.compute_psd(tmin=0.5, tmax=4.5, fmin=8, fmax=30).get_data().mean(axis=(1, 2))
    i_base = ipsi_motorstrip.compute_psd(tmin=-3, tmax=-1.5, fmin=8, fmax=30).get_data().mean(axis=(1, 2))
    i_task = ipsi_motorstrip.compute_psd(tmin=0.5, tmax=4.5, fmin=8, fmax=30).get_data().mean(axis=(1, 2))
    erddrop_contra = ((c_task - c_base) / c_base) * 100
    erddrop_ipsi = ((i_task - i_base) / i_base) * 100
    return erddrop_contra, erddrop_ipsi

def success_rate(erddrop_ipsi, erddrop_contra, labels, ideal_li_slider):
    li_value = (erddrop_ipsi - erddrop_contra) / (np.abs(erddrop_ipsi) + np.abs(erddrop_contra))
    predictions = []

    # these are the default values set for intent and confidence that are changed via for loop conditions being met
    intent = False
    confidence = 0.0

    for contra, ipsi, li in zip(erddrop_contra, erddrop_ipsi, li_value):
        weighted_score = 0.7*contra + 0.3*ipsi
        if (weighted_score) <= -20 or (li < -0.2 and weighted_score < -5):
            predictions.append(2)
            intent = True
        else:
            predictions.append(1)
            intent = False

        # Condition 1 Confidence
        passing_score = -20
        ideal_score = -60
        score_dist_ratio = abs((weighted_score - passing_score)/(ideal_score - passing_score))
        score_dist_ratio = min(max(score_dist_ratio, 0), 1)
        confidence_scores = 50 + (score_dist_ratio)*50

        # Condition 2 Confidence
        passing_li = -0.2
        ideal_li = ideal_li_slider
        lidist_ratio = abs(li - passing_li) / abs(ideal_li - passing_li)
        lidist_ratio = min(max(lidist_ratio, 0), 1)  #for this min/max - it takes max bc if ratio is less than 0, will assume 0, then takes min of whole statement so that you get exact value(assuming it's less than 1)

        passing_weight = -5
        ideal_weight = -20
        weight_ratio = abs((weighted_score - passing_weight)/(ideal_weight - passing_weight))
        weight_ratio = min(max(weight_ratio, 0), 1)

        combined_ratio = min(lidist_ratio, weight_ratio)
        confidence_liandweight = 50 + (combined_ratio)*50 #this is *50 bc we want confidence percentage from 50 to 100

        confidence = (confidence_scores + confidence_liandweight)/2

    matches = np.sum(np.array(predictions) == labels.flatten()) #condensed version of before - number of times the prediction label([1]/[2]) = actual label
    accuracy_percentage = (matches / len(labels)) * 100
    return accuracy_percentage, intent, confidence

# --- MASTER FUNCTION --- basically returns the intent(l/r), confidence level, and accuracy - but why would we need it?
@st.cache_data
def produce_response(file_path, ideal_li_slider):
    epochs, labels = load_and_clean_data(file_path)
    drop_c, drop_i = calculate_psd(epochs)
    acc, intent, conf = success_rate(drop_i, drop_c, labels, ideal_li_slider)
    return acc, intent, conf
