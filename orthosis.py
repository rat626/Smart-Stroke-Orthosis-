#dependencies:
#mne
#scipy
#numpy
#pandas
#matplotlib

import mne
import scipy.io as sio
from scipy import stats
import numpy as np
import pandas as pd
import os 

SAMP_FREQ = 500


#MAP OF ALL CHANNELS TO INDICES FROM DATASET 
EEG_MAP = eeg_names = [
    'Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FCz', 'FC3', 'FC4',
    'FT7', 'FT8', 'Cz', 'C3', 'C4', 'T3', 'T4', 'CPz', 'CP3', 'CP4',
    'TP7', 'TP8', 'Pz', 'P3', 'P4', 'T5', 'T6', 'Oz', 'O1', 'O2'
]

ALL_CHANNELS = EEG_MAP + ['HEOG', 'VEOG', 'STIM']

instruction_tmin = 0.0
instruction_tmax = 2.0
imagery_tmin = 2.0
imagery_tmax = 6.0
break_tmin = 6.0
break_tmax = 8.0

bands = {
    'mu': (8, 13),
    'beta': (13, 30),
    'both': (8, 30)
}

stdev_threshold_gate1 = 1.5
stdev_threshold_gate2 = 1.5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_CANDIDATE = os.path.join(BASE_DIR, 'data')
DATA_DIR = _DATA_CANDIDATE if os.path.isdir(_DATA_CANDIDATE) else BASE_DIR
PARTICIPANTS_TSV = os.path.join(BASE_DIR, 'participants.tsv')

GATE1_FAIL_REASON = (
    "Gate 1 failed: imagery power on paretic/contralateral motor channels was not "
    "significantly lower than break-period power — imagery did not elicit a "
    "significant contralateral drop."
)
GATE2_FAIL_REASON = (
    "Gate 2 failed: laterality index was not above 0 (ipsilateral power must exceed "
    "contralateral), or was below the healthy-hand imagery floor (mean − k·SD)."
)

#open metadata of all participants
metadata = pd.read_csv(PARTICIPANTS_TSV, sep='\t')


def _print(verbose, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)


def gate1_passes(curr_power, mean_paretic_break, std_paretic_break, k):
    return float(curr_power) <= float(mean_paretic_break) - float(k) * float(std_paretic_break)


def gate2_passes(curr_li, healthy_li_mean, healthy_li_std, k):
    li = float(curr_li)
    return li > 0.0 and li >= float(healthy_li_mean) - float(k) * float(healthy_li_std)

#this function applies notch and bandpass filter to raw object, then segments into 40 epochs(trials)
def filter_and_segment_data(raw):
    all_channel_names = eeg_names + ['HEOG', 'VEOG', 'STIM']
    all_channel_types = (['eeg'] * 30) + (['eog'] * 2) + ['stim']

        # 2. Extract labels and the full 3D data block (40, 33, 4000)
    eeg_struct = raw['eeg'][0, 0]
    labels = eeg_struct['label'].flatten().astype(int)
    raw_data_matrix = eeg_struct['rawdata'] / 1e6

    #since there isnt an mne notch filter for epochsarray - we do it to raw matrix
    raw_data_matrix[:, :32, :] = mne.filter.notch_filter(
        raw_data_matrix[:, :32, :],
        Fs=SAMP_FREQ,
        freqs=60.0,
        method='fir',
        phase='zero',
        verbose=False
    )

    # 3. Load directly into an MNE EpochsArray (No continuous stitching - this accepts 3d!!
    info = mne.create_info(ch_names=all_channel_names, 
                            sfreq=SAMP_FREQ, 
                            ch_types=all_channel_types)

    n_trials = len(labels)
    events = np.zeros((n_trials, 3), dtype=int)
    for i in range(n_trials):
        events[i, 0] = i * 4000
        events[i, 2] = int(labels[i])

    epochs = mne.EpochsArray(
        raw_data_matrix,
        info=info,
        events=events,
        event_id={'Left_Hand': 1, 'Right_Hand': 2},
        tmin=0.0,
        baseline=(0.0, 2.0)  # <-- Forces MNE to use the first 2 seconds as the baseline window
    )

    # Band-pass filter Brain Waves (8-30 Hz)
    epochs.filter(l_freq=8.0, h_freq=30.0, picks='eeg', method='fir', phase='zero', verbose=False)

    # Band-pass filter Eye Tracks (1-40 Hz)
    epochs.filter(l_freq=1.0, h_freq=40.0, picks='eog', method='fir', phase='zero', verbose=False)

    return epochs


#this function makes a distribution plot for all eeg and eog channels across all trials for the participant, and classifies outliers based on 1.5IQR
def remove_bad_trials(epochs, verbose=True):
    eeg_ch_names = ALL_CHANNELS[:30]
    eog_ch_names = ['HEOG', 'VEOG']

    # 2. Extract the filtered data arrays (Trials x Channels x Samples)
    eeg_data = epochs.copy().pick(eeg_ch_names).get_data()
    eog_data = epochs.copy().pick(eog_ch_names).get_data()

    # 3. Calculate Peak-to-Peak (Max - Min) per trial per channel in microvolts (uV)
    # We flatten these matrices so we look at every single channel-trial combination
    eeg_p2p_all = ((np.max(eeg_data, axis=2) - np.min(eeg_data, axis=2)) * 1e6).flatten()
    eog_p2p_all = ((np.max(eog_data, axis=2) - np.min(eog_data, axis=2)) * 1e6).flatten()

    # 4. Compute standard Box-and-Whisker metrics and dynamically build reject_criteria
    reject_criteria = {}

    for key, label, data in [("eeg", "EEG (Brain)", eeg_p2p_all), ("eog", "EOG (Eyes)", eog_p2p_all)]:
        q1 = np.percentile(data, 25) # Bottom of the box
        q3 = np.percentile(data, 75) # Top of the box
        iqr = q3 - q1                # Height of the box

        # Statistical definition of a mild outlier (Q3 + 1.5 * IQR) in uV
        upper_whisker_uv = q3 + (1.5 * iqr)
        extreme_cutoff_uv = q3 + (3.0 * iqr)

        # Save to dictionary in Volts (required by MNE)
        reject_criteria[key] = upper_whisker_uv * 1e-6

        _print(verbose, f"\n=== {label} BOXPLOT STATISTICS ===")
        _print(verbose, f"  Box Bottom (Q1)     : {q1:.1f} uV")
        _print(verbose, f"  Box Top (Q3)        : {q3:.1f} uV")
        _print(verbose, f"  Box Height (IQR)    : {iqr:.1f} uV")
        _print(verbose, f"  Mild Outlier Cutoff : {upper_whisker_uv:.1f} uV")
        _print(verbose, f"  Extreme Cutoff      : {extreme_cutoff_uv:.1f} uV")

    # =====================================================================
    # APPLY DYNAMIC OUTLIERS TO REJECT BAD EPOCHS
    # =====================================================================

    # Screen the array using dynamically computed thresholds
    n_orig_trials = len(epochs)
    epochs.drop_bad(reject=reject_criteria, verbose=verbose)

    # Pull surviving clean trial numbers and labels
    clean_labels = epochs.events[:, 2]
    n_clean_trials = len(epochs)
    kept_trials = [i + 1 for i, log in enumerate(epochs.drop_log) if not log]
    rejected_trials = [i + 1 for i, log in enumerate(epochs.drop_log) if log]

    _print(verbose, "\n=== TRIAL REJECTION SUMMARY ===")
    _print(verbose, f"  Kept     ({len(kept_trials)}/{n_orig_trials}): {kept_trials}")
    _print(verbose, f"  Rejected ({len(rejected_trials)}/{n_orig_trials}): {rejected_trials}")

    return clean_labels, n_clean_trials, epochs

#this calculates the laterality index using absolute power for imagery and break, and r^2 of the imagery period - used to choose frequency band 
def calculate_laterality_and_r2(epochs, fmin, fmax, imagery_tmin, imagery_tmax, break_tmin, break_tmax):
    # 1. Isolate left and right channels
    left_epochs = epochs.copy().pick(['FC3', 'CP3', 'C3'])
    right_epochs = epochs.copy().pick(['FC4', 'CP4', 'C4'])

    # 2. Calculate PSD for Imagery & Break
    psd_img_left = left_epochs.compute_psd(tmin=imagery_tmin, tmax=imagery_tmax, fmin=fmin, fmax=fmax, picks='eeg', verbose=False)
    psd_img_right = right_epochs.compute_psd(tmin=imagery_tmin, tmax=imagery_tmax, fmin=fmin, fmax=fmax, picks='eeg', verbose=False)
    psd_brk_left = left_epochs.compute_psd(tmin=break_tmin, tmax=break_tmax, fmin=fmin, fmax=fmax, picks='eeg', verbose=False)
    psd_brk_right = right_epochs.compute_psd(tmin=break_tmin, tmax=break_tmax, fmin=fmin, fmax=fmax, picks='eeg', verbose=False)

    # 3. Calculate absolute power (mean across frequencies and channels)
    pwr_img_left = psd_img_left.get_data().mean(axis=(1, 2))
    pwr_img_right = psd_img_right.get_data().mean(axis=(1, 2))
    
    pwr_brk_left = psd_brk_left.get_data().mean(axis=(1, 2))
    pwr_brk_right = psd_brk_right.get_data().mean(axis=(1, 2))

    # 4. Standard Laterality Index formula using absolute power: (R - L) / (R + L)
    laterality_imagery = (pwr_img_right - pwr_img_left) / (pwr_img_right + pwr_img_left)
    laterality_break = (pwr_brk_right - pwr_brk_left) / (pwr_brk_right + pwr_brk_left)

    # 5. Calculate R^2 between the two conditions (Imagery vs. Break)
    labels = np.concatenate([np.ones(len(laterality_imagery)), np.zeros(len(laterality_break))])
    features = np.concatenate([laterality_imagery, laterality_break])

    r_value, _ = stats.pointbiserialr(labels, features)
    r_squared = r_value ** 2

    return r_squared

def _participant_row(metadata_df, participant_id):
    """Return the participants.tsv row whose Participant_ID matches this subject."""
    pid = str(participant_id).strip()
    ids = metadata_df['Participant_ID'].astype(str).str.strip()
    rows = metadata_df.loc[ids == pid]
    if rows.empty:
        return None
    return rows.iloc[0]


def get_paralysis_side(metadata_df, participant_id):
    row = _participant_row(metadata_df, participant_id)
    if row is None:
        raise ValueError(f"No participants.tsv row for {participant_id}")
    side = str(row['ParalysisSide']).strip().lower()
    if side not in ('left', 'right'):
        raise ValueError(f"Invalid paralysis side for {participant_id}: {side}")
    return side


def get_handedness(metadata_df, participant_id):
    row = _participant_row(metadata_df, participant_id)
    if row is None:
        return None
    return str(row['Handedness']).strip().lower()


def get_stroke_location(metadata_df, participant_id):
    row = _participant_row(metadata_df, participant_id)
    if row is None:
        return None
    value = row['StrokeLocation']
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().strip('"')
    return text or None


def choose_frequency_band(epochs, imagery_tmin, imagery_tmax, break_tmin, break_tmax, participant_paralysis_side, verbose=True):
    if participant_paralysis_side == 'right':
        healthy_epochs = epochs['Left_Hand']
    else:
        healthy_epochs = epochs['Right_Hand']

    r2_results = {}
    for band_name, (fmin, fmax) in bands.items():
        r2_results[band_name] = calculate_laterality_and_r2(
        healthy_epochs, fmin, fmax, 
        imagery_tmin, imagery_tmax, break_tmin, break_tmax
    )

    # 7. Find the condition with the maximum R^2
    best_condition = max(r2_results, key=r2_results.get)
    downstream_fmin, downstream_fmax = bands[best_condition]

    _print(verbose, "\n=== FREQUENCY BAND SELECTION ===")
    for band_name, r2 in r2_results.items():
        bmin, bmax = bands[band_name]
        marker = "  <-- CHOSEN" if band_name == best_condition else ""
        _print(verbose, f"  {band_name:5s} ({bmin}-{bmax} Hz): R^2 = {r2:.4f}{marker}")
    _print(verbose, f"  Chosen frequency band: {best_condition} ({downstream_fmin}-{downstream_fmax} Hz)")

    return downstream_fmin, downstream_fmax, best_condition, r2_results

#NOTE - BOTH GATES ARE EVALUATED ON EVERY TRIAL; BOTH MUST PASS FOR THE ORTHOSIS TO OPEN 


#this is the first gate - makes distribution of power in break periods for paretic channels - checks if current trial is below mean - gate1 stdevs
def first_gate_evaluation(current_trial_paretic_channels, mean_paretic_break, std_paretic_break, downstream_fmin, downstream_fmax):
  # current_trial_paretic_channels is already picked for paretic channels for a single trial
  curr_power = current_trial_paretic_channels.compute_psd(
      tmin=imagery_tmin, tmax=imagery_tmax, fmin=downstream_fmin, fmax=downstream_fmax, picks='eeg', verbose=False
  ).get_data().mean(axis=(1, 2)) # This will be a single value for one trial, averaged across channels
  curr_power_val = float(np.asarray(curr_power).ravel()[0])
  if curr_power <= mean_paretic_break - stdev_threshold_gate1 * std_paretic_break:
    return True, curr_power_val
  else:
    return False, curr_power_val


#this is the second gate - makes distribution of power in imagery periods for healthy imagery - sees if current li is within this range - using absolute power
def second_gate_evaluation(healthy_side_for_li_arg, current_trial_all_channels, healthy_li_mean, healthy_li_std, downstream_fmin, downstream_fmax):
  # 1. Compute PSD for the current single trial (with all channels initially)
  psd_current_single_trial = current_trial_all_channels.compute_psd(
      tmin=imagery_tmin, tmax=imagery_tmax, fmin=downstream_fmin, fmax=downstream_fmax, picks='eeg', verbose=False
  )

  # 2. Pick channels and extract power based on healthy_side_for_li_arg for LI calculation
  if healthy_side_for_li_arg == 'right': # Right motor cortex ipsilateral to healthy movement
    ipsi_power_curr = psd_current_single_trial.copy().pick(['FC4', 'CP4', 'C4']).get_data().mean(axis=(1, 2))
    contra_power_curr = psd_current_single_trial.copy().pick(['FC3', 'CP3', 'C3']).get_data().mean(axis=(1, 2))
  else: # Left motor cortex ipsilateral to healthy movement
    ipsi_power_curr = psd_current_single_trial.copy().pick(['FC3', 'CP3', 'C3']).get_data().mean(axis=(1, 2))
    contra_power_curr = psd_current_single_trial.copy().pick(['FC4', 'CP4', 'C4']).get_data().mean(axis=(1, 2))

  # 3. Calculate LI (will be a single value for one trial)
  curr_li = (ipsi_power_curr - contra_power_curr) / (ipsi_power_curr + contra_power_curr)

  # 4. Evaluate Gate: LI must be above 0 and at/above the healthy-hand floor
  curr_li_val = float(np.asarray(curr_li).ravel()[0])
  passed = gate2_passes(curr_li_val, healthy_li_mean, healthy_li_std, stdev_threshold_gate2)
  return passed, curr_li_val


def list_participant_ids(base_dir=None):
    base_dir = base_dir or DATA_DIR
    ids = []
    for folder in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path) or not folder.startswith('sub-'):
            continue
        if any(name.endswith('.mat') for name in os.listdir(folder_path)):
            ids.append(folder)
    return ids


def _result_shell(participant_id, **kwargs):
    result = {
        'participant_id': participant_id,
        'file': None,
        'skipped': False,
        'skip_reason': None,
        'paralysis_side': None,
        'handedness': None,
        'stroke_location': None,
        'n_orig_trials': None,
        'n_kept': None,
        'n_rejected': None,
        'kept_trials': [],
        'rejected_trials': [],
        'chosen_band': None,
        'chosen_band_hz': None,
        'chosen_band_r2': None,
        'cal_trial_nums': [],
        'test_trial_nums': [],
        'test_paretic_trial_nums': [],
        'n_gate1_cal': None,
        'n_gate2_cal': None,
        'n_test_paretic': None,
        'paretic_channels': [],
        'healthy_side_for_li': None,
        'gate1_break_powers': [],
        'gate1_mean': None,
        'gate1_std': None,
        'gate2_healthy_lis': [],
        'gate2_mean': None,
        'gate2_std': None,
        'trials': [],
        'gate1_fail_reason': GATE1_FAIL_REASON,
        'gate2_fail_reason': GATE2_FAIL_REASON,
    }
    result.update(kwargs)
    return result


def rescore_result(result, k1, k2):
    """Re-apply gate thresholds using the same inequalities as the pipeline."""
    scored = dict(result)
    if result.get('skipped') or not result.get('trials'):
        scored['stdev_threshold_gate1'] = float(k1)
        scored['stdev_threshold_gate2'] = float(k2)
        return scored
    trials = []
    for trial in result['trials']:
        g1 = gate1_passes(trial['gate1_value'], result['gate1_mean'], result['gate1_std'], k1)
        g2 = gate2_passes(trial['gate2_value'], result['gate2_mean'], result['gate2_std'], k2)
        updated = dict(trial)
        updated['gate1_passed'] = g1
        updated['gate2_passed'] = g2
        updated['opened'] = bool(g1 and g2)
        trials.append(updated)
    scored['trials'] = trials
    scored['stdev_threshold_gate1'] = float(k1)
    scored['stdev_threshold_gate2'] = float(k2)
    return scored


def run_subject(participant_id, metadata_df=None, base_dir=None, verbose=True):
    base_dir = base_dir or DATA_DIR
    if metadata_df is None:
        metadata_df = pd.read_csv(PARTICIPANTS_TSV, sep='\t')

    folder_path = os.path.join(base_dir, participant_id)
    mat_files = sorted(name for name in os.listdir(folder_path) if name.endswith('.mat')) if os.path.isdir(folder_path) else []
    if not mat_files:
        reason = f"WARNING: {participant_id} failed (FileNotFoundError: no .mat file). Skipping subject."
        _print(verbose, "\n" + "=" * 70)
        _print(verbose, f"SUBJECT: {participant_id}  |  FILE: none")
        _print(verbose, "=" * 70)
        _print(verbose, reason)
        return _result_shell(
            participant_id,
            skipped=True,
            skip_reason=reason,
            handedness=get_handedness(metadata_df, participant_id),
            stroke_location=get_stroke_location(metadata_df, participant_id),
        )

    file = mat_files[0]
    handedness = get_handedness(metadata_df, participant_id)
    stroke_location = get_stroke_location(metadata_df, participant_id)

    _print(verbose, "\n" + "=" * 70)
    _print(verbose, f"SUBJECT: {participant_id}  |  FILE: {file}")
    _print(verbose, "=" * 70)

    try:
        raw = sio.loadmat(os.path.join(folder_path, file))
        epochs = filter_and_segment_data(raw)
        clean_labels, n_clean_trials, cleaned_epochs = remove_bad_trials(epochs, verbose=verbose)
        n_orig_trials = len(cleaned_epochs.drop_log)
        kept_trials = [i + 1 for i, log in enumerate(cleaned_epochs.drop_log) if not log]
        rejected_trials = [i + 1 for i, log in enumerate(cleaned_epochs.drop_log) if log]
        common = dict(
            file=file,
            handedness=handedness,
            stroke_location=stroke_location,
            n_orig_trials=int(n_orig_trials),
            n_kept=int(n_clean_trials),
            n_rejected=int(len(rejected_trials)),
            kept_trials=kept_trials,
            rejected_trials=rejected_trials,
        )

        if n_clean_trials == 0:
            reason = "WARNING: all trials rejected. Skipping subject."
            _print(verbose, reason)
            return _result_shell(participant_id, skipped=True, skip_reason=reason, **common)

        participant_paralysis_side = get_paralysis_side(metadata_df, participant_id)
        _print(verbose, f"  Paralysis side (participants.tsv): {participant_paralysis_side}")
        common['paralysis_side'] = participant_paralysis_side

        downstream_fmin, downstream_fmax, best_condition, r2_results = choose_frequency_band(
            cleaned_epochs, imagery_tmin, imagery_tmax, break_tmin, break_tmax,
            participant_paralysis_side, verbose=verbose
        )
        common['chosen_band'] = best_condition
        common['chosen_band_hz'] = (int(downstream_fmin), int(downstream_fmax))
        common['chosen_band_r2'] = float(r2_results[best_condition])

        if participant_paralysis_side == 'right':
            paretic_hand_label = 2
            healthy_hand_label = 1
            paretic_channels_for_gate1 = ['FC3', 'CP3', 'C3'] #bc right hand -- left brain lesion
            healthy_side_for_li = 'right'

        elif participant_paralysis_side == 'left': # Left hand impaired -> Right brain lesion
            paretic_hand_label = 1
            healthy_hand_label = 2
            paretic_channels_for_gate1 = ['FC4', 'CP4', 'C4'] # Right motor strip (ipsilesional/paretic for Left Hand MI)
            healthy_side_for_li = 'left'

        else:
            raise ValueError(f"Invalid paralysis side: {participant_paralysis_side}")

        common['paretic_channels'] = list(paretic_channels_for_gate1)
        common['healthy_side_for_li'] = healthy_side_for_li

        # Chronological 50/50 split of cleaned trials (no leakage into gate stats)
        n_clean = len(cleaned_epochs)
        n_cal = n_clean // 2
        calibration_epochs = cleaned_epochs[:n_cal]
        test_epochs = cleaned_epochs[n_cal:]
        cal_trial_nums = (calibration_epochs.events[:, 0] // 4000 + 1).tolist() if n_cal else []
        test_trial_nums = (test_epochs.events[:, 0] // 4000 + 1).tolist() if len(test_epochs) else []
        common['cal_trial_nums'] = [int(x) for x in cal_trial_nums]
        common['test_trial_nums'] = [int(x) for x in test_trial_nums]

        _print(verbose, "\n=== CHRONOLOGICAL 50/50 SPLIT ===")
        _print(verbose, f"  Cleaned trials            : {n_clean}")
        _print(verbose, f"  Calibration (first half)  : {cal_trial_nums}")
        _print(verbose, f"  Test (second half)        : {test_trial_nums}")

        if n_cal == 0 or len(test_epochs) == 0:
            reason = "WARNING: not enough cleaned trials to split 50/50. Skipping subject."
            _print(verbose, reason)
            return _result_shell(participant_id, skipped=True, skip_reason=reason, **common)

        #first - make the distribution for gate 1 using calibration paretic trials only
        paretic_epochs_for_gate1_dist = calibration_epochs[calibration_epochs.events[:, 2] == paretic_hand_label]
        healthy_epochs_for_gate2_dist = calibration_epochs[calibration_epochs.events[:, 2] == healthy_hand_label]
        paretic_testing_epochs = test_epochs[test_epochs.events[:, 2] == paretic_hand_label]
        test_paretic_trial_nums = (
            (paretic_testing_epochs.events[:, 0] // 4000 + 1).tolist()
            if len(paretic_testing_epochs) else []
        )
        common['test_paretic_trial_nums'] = [int(x) for x in test_paretic_trial_nums]

        common['n_gate1_cal'] = int(len(paretic_epochs_for_gate1_dist))
        common['n_gate2_cal'] = int(len(healthy_epochs_for_gate2_dist))
        common['n_test_paretic'] = int(len(paretic_testing_epochs))

        _print(verbose, f"  Gate 1 cal (paretic)      : {len(paretic_epochs_for_gate1_dist)} trials")
        _print(verbose, f"  Gate 2 cal (healthy hand) : {len(healthy_epochs_for_gate2_dist)} trials")
        _print(verbose, f"  Test (paretic)            : {len(paretic_testing_epochs)} trials")

        if len(paretic_epochs_for_gate1_dist) == 0:
            reason = "WARNING: no paretic trials in calibration half. Skipping subject."
            _print(verbose, reason)
            return _result_shell(participant_id, skipped=True, skip_reason=reason, **common)
        if len(healthy_epochs_for_gate2_dist) == 0:
            reason = "WARNING: no healthy-hand trials in calibration half. Skipping subject."
            _print(verbose, reason)
            return _result_shell(participant_id, skipped=True, skip_reason=reason, **common)
        if len(paretic_testing_epochs) == 0:
            reason = "WARNING: no paretic trials in test half. Skipping subject."
            _print(verbose, reason)
            return _result_shell(participant_id, skipped=True, skip_reason=reason, **common)

        paretic_epochs_for_gate1_dist = paretic_epochs_for_gate1_dist.copy().pick(paretic_channels_for_gate1)

        # Compute PSD in the break region for these paretic trials (this forms the baseline distribution for Gate 1)
        psd_paretic_break = paretic_epochs_for_gate1_dist.compute_psd(tmin=break_tmin,
                                                                    tmax=break_tmax,
                                                                    fmin=downstream_fmin,
                                                                    fmax=downstream_fmax,
                                                                    picks='eeg',
                                                                    verbose=False)
        trial_powers_break = psd_paretic_break.get_data().mean(axis=(1, 2))

        # Single scalar floats: mean and std of break power for Gate 1 threshold
        global_mean_break = trial_powers_break.mean()
        global_std_break = trial_powers_break.std()

        #NOW - WE INITIALIZE THE HEALTHY DISTRIBUTION FOR GATE 2 USING CALIBRATION "HEALTHY HAND TRIALS"
        psd_healthy_imagery_for_gate2_dist = healthy_epochs_for_gate2_dist.compute_psd(tmin = imagery_tmin,
                                                                                        tmax = imagery_tmax,
                                                                                        fmin = downstream_fmin,
                                                                                        fmax = downstream_fmax,
                                                                                        picks = 'eeg',
                                                                                        verbose = False)
        if healthy_side_for_li == 'right': # Right motor cortex ipsilateral to healthy movement (e.g., Left Hand MI)
            psd_healthy_ipsi = psd_healthy_imagery_for_gate2_dist.copy().pick(['FC4', 'CP4', 'C4'])
            psd_healthy_contra = psd_healthy_imagery_for_gate2_dist.copy().pick(['FC3', 'CP3', 'C3'])

        else: # Left motor cortex ipsilateral to healthy movement (e.g., Right Hand MI)
            psd_healthy_ipsi = psd_healthy_imagery_for_gate2_dist.copy().pick(['FC3', 'CP3', 'C3'])
            psd_healthy_contra = psd_healthy_imagery_for_gate2_dist.copy().pick(['FC4', 'CP4', 'C4'])

        ipsi_power = psd_healthy_ipsi.get_data().mean(axis=(1, 2))
        contra_power = psd_healthy_contra.get_data().mean(axis=(1, 2))

        healthy_li_array = (ipsi_power - contra_power) / (ipsi_power + contra_power)
        healthy_li_mean = healthy_li_array.mean()
        healthy_li_std = healthy_li_array.std()

        common['gate1_break_powers'] = np.asarray(trial_powers_break, dtype=float).ravel().tolist()
        common['gate1_mean'] = float(global_mean_break)
        common['gate1_std'] = float(global_std_break)
        common['gate2_healthy_lis'] = np.asarray(healthy_li_array, dtype=float).ravel().tolist()
        common['gate2_mean'] = float(healthy_li_mean)
        common['gate2_std'] = float(healthy_li_std)

                        # =====================================================================
        # TESTING PHASE: Simulating the Real-Time BCI Pipeline
        # =====================================================================

        total_trials = len(paretic_testing_epochs)
        successful_opens = 0
        trial_records = []

        _print(verbose, f"\n--- Starting Real-Time Simulation for {total_trials} Test Paretic Trials ---")
        _print(verbose, f"    Frequency band in use: {downstream_fmin}-{downstream_fmax} Hz")
        _print(verbose, "    Gate rule: both gates are checked on every trial; BOTH must pass to open")
        _print(verbose, "    Open rate denominator: all paretic trials in the test half")

        gate1_fail_reason = GATE1_FAIL_REASON
        gate2_fail_reason = GATE2_FAIL_REASON

        for i in range(total_trials):
            # 1. Extract a single trial (mimicking a real-time 4-second incoming data buffer)
            current_trial = paretic_testing_epochs[i]
            orig_trial_num = int(current_trial.events[0, 0] // 4000) + 1

            # 2. Extract just the paretic channels for Gate 1
            current_trial_paretic_channels = current_trial.copy().pick(paretic_channels_for_gate1)

            # 3. Evaluate both gates independently (no short-circuit)
            gate1_passed, curr_power = first_gate_evaluation(current_trial_paretic_channels, global_mean_break, global_std_break, downstream_fmin, downstream_fmax)
            gate2_passed, curr_li = second_gate_evaluation(healthy_side_for_li, current_trial, healthy_li_mean, healthy_li_std, downstream_fmin, downstream_fmax)

            trial_records.append({
                'trial_num': int(orig_trial_num),
                'gate1_value': float(curr_power),
                'gate2_value': float(curr_li),
                'gate1_passed': bool(gate1_passed),
                'gate2_passed': bool(gate2_passed),
                'opened': bool(gate1_passed and gate2_passed),
            })

            if gate1_passed and gate2_passed:
                _print(verbose, f"Trial {orig_trial_num}: [SUCCESS] ORTHOSIS OPENED (Gate 1 & 2 Passed)")
                successful_opens += 1
            else:
                _print(verbose, f"Trial {orig_trial_num}: [LOCKED] Gate 1={'PASS' if gate1_passed else 'FAIL'}, Gate 2={'PASS' if gate2_passed else 'FAIL'}")
                if not gate1_passed:
                    _print(verbose, f"    Reason: {gate1_fail_reason}")
                if not gate2_passed:
                    _print(verbose, f"    Reason: {gate2_fail_reason}")

        # Calculate clinical performance on test-half paretic trials only
        open_rate = (successful_opens / total_trials) * 100 if total_trials else 0.0
        _print(verbose, f"\nFinal Orthosis Open Rate: {open_rate:.1f}% ({successful_opens}/{total_trials} test paretic trials)")
        _print(verbose, f"SUBJECT {participant_id} COMPLETE")

        return _result_shell(
            participant_id,
            skipped=False,
            skip_reason=None,
            trials=trial_records,
            **common
        )
    except Exception as e:
        reason = f"WARNING: {participant_id} failed ({type(e).__name__}: {e}). Skipping subject."
        _print(verbose, reason)
        return _result_shell(
            participant_id,
            file=file,
            handedness=handedness,
            stroke_location=stroke_location,
            skipped=True,
            skip_reason=reason,
        )


def main():
    #open metadata tsv - applies to all participants
    metadata_df = pd.read_csv(PARTICIPANTS_TSV, sep='\t')

    #ask agent to make main so that it opens every folder and loops thru the mat files
    for folder in list_participant_ids(DATA_DIR):
        run_subject(folder, metadata_df=metadata_df, base_dir=DATA_DIR, verbose=True)


if __name__ == "__main__":
    main()
 