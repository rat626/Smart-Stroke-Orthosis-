# BrainTrain - an EEG - Based Orthosis to Train Contralateral Motor Pathways in Hemiparetic Stroke Patients
In this project, I created a pipeline to decode intended hand movement and control a motorized orthosis for hemiparetic stroke patients. The inspiration for this idea came from my experience shadowing in a physical therapy clinic, where I saw multiple patients with left/right brain stroke who were impaired in various degrees, from an inability to open the left fist from a clenched position, needing to use the right hand to help pry open their thumb, to subdural hemmhorage left untreated, which induced stroke that impaired the regulation of tone in the left leg, immobility in the left hand, and in some cases, speech difficulty caused by residual damage to the left brain. For context, stroke commonly involves a loss of bloodflow to a particular region of the brain, commonly in motor - relevant regions, and requires an immediate removal of blood clots to prevent near - instant death, with later impacts involving a paralysis/lack of regulation of movement in one side of the body. 

While several methods, such as functional electrical stimulation, robotic arms, and EEG - based ipsilateral systems used to bypass injured pathways already exist to help such patients, I was inspired by the use of mirror therapy to mimic the proper functionality of the impaired hand, as well as PT exercises that stroke patients would traditionally struggle with - some examples include running on a treadmill with weight attached to the leg opposite to the side of brain damage(used to force patients to apply increased knee drive to the stroke - affected leg in natural gait), and moving the impaired leg over obstacles to promote a habit of increasing stride length in the impaired side of the body. 

Therefore, when thinking of an idea for my system, I wanted to develop a system that didn't bypass injured brain pathways, but rather, reconstruct them over time through continual usage of the contralateral side of the brain to move the impaired hand. Several current systems utilize motor - relevant signals from the same side of the brain to control the same side arm/leg, but this is unnatural - and involves building alternative neural connections, rather than strengthening impaired pathways. 

I intend to attach a motor to a natural orthosis/glove, which patients use to keep their hands in the right positions to easen movement. A motor would decode intent from this heuristic pipeline, which identifies the hand a patient intends to move by measuring the event - related desynchronization(ERD) observed in the alpha - beta(8-30 hz) frequency range via an EEG headset. Brain rhythms in the alpha - beta frequency, specifically mu rhythms in the alpha range, are commonly linked to motor actions, and drops in the "power" of these frequencies(lowered PSD), or contribution of these frequencies to an overall brain signal, are linked to the intent, planning, and execution of motor action - this is referred to as ERD. Mu/beta ERD drops in the right hemisphere correspond to left hand motor intent/action, while mu/beta drops in the left hemisphere correspond to right hand motor intent/action. However, for patients with severe brain damage, it is hard to produce a high enough contralateral drop, and therefore, ipsilateral/same - side signals must be harnessed to aid in movement. While some methods purely utilize ipsilateral system, my pipeline rewards patients with movement of the orthosis when the percentage of ERD drop from the opposite - side/contralateral hemisphere is greater than the left, which stimulates a positive feedback loop over time. 


Here is the detailed breakdown of how I made the pipeline - with attached diagrams for explanation:

1. Loaded the training files using scipy to turn the .mat file into a dataframe, and later convert it into a numpy array(for csv file - this would involve pandas - pd.read, .to_numpy().T

2. Label the array using mne - involves creating an info object with the channel names(unique numbers), channel types(eeg), and sampling frequency(512 hz - from dataset):
   sampling_freq = 512
   channel_names = ['F3', 'FC3', 'C3', 'CP3', 'P3', 'FCz', 'CPz', 'F4', 'FC4', 'C4', 'CP4', 'P4']
   channel_types = ['eeg'] * 12

3. Filtered the data - used a bandpass filter to only include frequencies between 8 and 30 hz(alpha(mu rhythm included) and beta frequencies) and a notch filter at 60 hz(generally good practice for cleaning signal)

4. Made a segmented EpochsArray object with labeled left and right hand motor trials - the "constructor" is below:
   epochs_segmented = mne.EpochsArray(
    data = rawdata * 1e-6,      # Your 3D cube (scaled to Volts)
    info = info,                # Your metadata
    events = events_matrix,     # The 3-column array
    event_id = event_dict,      # The dictionary
    tmin = -3.0                 # Matching the timing diagram start
)

- The data(multiplied by 1e-6 to put in units of microvolts) and info components are the same - events_matrix is created by 1) creating a 3 - dimensional array of 0's, 2) populating the 1st dimension with all samples linked to the trial they correspond to(for context, I used the 1st dimension of the provided rawdata structure from the CBCI dataset, of which the 1st dimension is # of trials, second is # of channels, and third is # of samples), and 3) flattening the 3rd dimension and populating the remaining second dimension with the labels array - also provided in the dataset, which contains [1] and [2] - with [1] corresponding to right hand movement, and [2] corresponding to left

6. Create a "contralateral motorstrip" and "ipsilateral motorstrip" - for our purposes, I considered the left hand to be the affected hand - meaning that the right hemisphere is contralateal and left is ipsilateral
   - Made a copy of segmented/filtered epochs array and used .pick() to isolate FC4, C4, and CP4 for the contralateral motor - relevant electrodes, which capture neuron group involved in motor planning and the reciept of feedback from motor execution - did same for the ipsilateral side, using FC3, C3, and CP3 instead
  
7. Calculate the power spectral density in these "motorstrips" - with separate baseline/evaluation PSD variables created based on the time segment they refer to. As the diagram shows, I calculated the PSD measured from -3.5s to -1.5s in a trial as the "baseline", given that there was not yet direction to move a particular hand, making it the best possible time segment within the trial for measuring what the EEG data would be in the absence of motor intent. As indicated by the text box, the sweet spot for identifying motor intent was between 0.5 and 4.5s, which I used as the "execution" part of the trial from which I calculated PSD

<img width="698" height="266" alt="Screenshot 2026-01-03 at 12 45 08 AM" src="https://github.com/user-attachments/assets/965332a8-24ab-438a-92c0-2547ec9a59fa" />


8. Calculate the % ERD drop in the contralateral and ipsilateral side using % change formula(( (PSD_execution - PSD_baseline)/PSD_baseline) * 100 ) - instead of just plugging in the values from the .compute_psd() function, I created a separate data spectrum and collected the mean across each of the 3 electrodes in the motorstrip from which I obtained the baseline/execution values for the contralateral and ipsilateral sides, using this function: baseline_(contra/ipsi)= (contra/ipsi)_psd_baseline.get_data().mean(axis = (1, 2))


9. Create a gated logic/threshold to ensure that movement of the orthosis is only triggered when sufficient contralateral motor intent is detected:
  - Step 1: Calculate lateralization index(LI) between contralateral and ipsilateral side ERD drop(LI = (ERDdrop_ipsilateral - ERDdrop_contralateral) / (ERDdrop_ipsilateral) +   (ERDdrop_contralateral))

  - Step 2: Here is the gated logic: for contra, li in zip(erddrop_contra, li_value):
     for contra, ipsi, li in zip(erddrop_contra, erddrop_ipsi, li_value):
        weighted_score = 0.7*contra + 0.3*ipsi
        if (weighted_score) <= -20 or (li < -0.2 and weighted_score < -5):
            predictions.append(2)
            intent = True
        else:
            predictions.append(1)
            intent = False

    Pulls from every baolue in the contralateral and ipsilateral ERDdrop arrays and LI value array(because these values have been calculated for each of the 80 trials - see CBCI dataset structure for info)

    If the weighted score(0.7*contra + 0.3*ipsi) is less than -20 - this means that the ERD drop produced was <= 20%, and 70% of it came from the contralateral side - we want to promote contralateral side motor intent - therefore, if the weighted score is 70% contralateral, the patient is rewarded with movement of the orthosis

    To account for the fact that not all patients may be able to produce a 20% ERD drop or more(bc. the occurence of subdural hemmhorage or tbi can result in complete dysfunction of contralateral - side areas, meaning that there has to be reliance on ipsilateral motor intent, at least in initial stages of rehabilitation), the second part of the logic - li < -0.2 and weighted_score < -5, signifies that if the LI is less than -0.2, meaning that the contralateral side must have a greater ERD drop han the ipsilateral side, and the weighted score must be less than -5, to ensure that the ERD drop is nonzero to a significant extent

10. Generate an averafe confidence of measurement based on the amount of deviation of the LI and weighted score from "passing range":
    - Code for weighted_score confidence ratio:

      passing_score = -20
        ideal_score = -60
        score_dist_ratio = abs((weighted_score - passing_score)/(ideal_score - passing_score))
        score_dist_ratio = min(max(score_dist_ratio, 0), 1)
        confidence_scores = 50 + (score_dist_ratio)*50

      For the first condition, the weighted score must be equal to -20(or less), meaning than an ideal score would be around -60(can be adjusted - but idea is to be far enough away from -20 in the right direction) - a ratio of the distance between the actual measured weighted score and passing score, versus the ideal score and passing score, is calculated, which is then processed to ensure it is greater than 0 and less than 1(set to 0 automatically via the max(score_dist_ratio,0) line if less than 0) and 1 via the min(_______, 1) if the inside functon yield a score less than 1. The final score is then turned into a percentage value between 50 and 100


       passing_li = -0.2
        ideal_li = ideal_li_slider
        lidist_ratio = abs(li - passing_li) / abs(ideal_li - passing_li)
        lidist_ratio = min(max(lidist_ratio, 0), 1) 

        passing_weight = -5
        ideal_weight = -20
        weight_ratio = abs((weighted_score - passing_weight)/(ideal_weight - passing_weight))
        weight_ratio = min(max(weight_ratio, 0), 1)

For the second condition - involves 2 parts(li < -0.2 AND weighted_score < -5) - with the li, I incorporated a slider within the streamlit app that can be used to adjust the ideal li score used for calculating the ratio. The second part, involving passing_weight, uses the same logic as condition 1 - but now, only involves a passing score of -5, accounting for patients that are unable to produce an ERD drop(with 70% contralateral effort) that is 20% or more - which in this case, would be the ideal score)

11. Wrap all code into helper functions(included in attached file), and a streamlit - ready "master function". The master function is attached below:

    @st.cache_data
def produce_response(file_path, ideal_li_slider):
    epochs, labels = load_and_clean_data(file_path)
    drop_c, drop_i = calculate_psd(epochs)
    acc, intent, conf = success_rate(drop_i, drop_c, labels, ideal_li_slider)
    return acc, intent, conf

    @st.cache_data is used to cache stored data in variables to prevent loss of data when re-running the function several times - the data is first loaded, then cleaned/processed, after which the ERDdrop values for the contralateral(c) and ipsilateral(i) sides are calculated, after which the success_rate function calculates accuracy(# of matches between the prediction made by the gate logic and the actual value found in the labels part of the dataset), intent(left or right hand movement), and confidence score



The data used to train and evaluate the pipeline's accuracy comes from the WCCI 2020 Glasgow Clinical Brain Computer Interfaces Challenge(CBCI 2020): https://github.com/SulemanRasheed/CBCI-Competition-2020/blob/master/Data/README.md - from which I utilized the dataset structure, and participant training/testing data

These are some additional research publications I used to develop my veto and dual gate logic for triggering movement: 

Cantillo-Negrete, Jessica, et al. “The ReHand-BCI Trial: A Randomized Controlled Trial of a Brain-Computer Interface for Upper Extremity Stroke Neurorehabilitation.” Frontiers in Neuroscience, vol. 19, Frontiers Media SA, June 2025, https://doi.org/10.3389/fnins.2025.1579988.

Dodd, Keith C., et al. “Role of the Contralesional vs. Ipsilesional Hemisphere in Stroke Recovery.” Frontiers in Human Neuroscience, vol. 11, Sept. 2017, https://doi.org/10.3389/fnhum.2017.00469.

Zhang, Y.; Gao, Y.; Zhou, J.; Zhang, Z.; Feng, M.; Liu, Y. Advances in Brain-Computer Interface Controlled Functional Electrical Stimulation for Upper Limb Recovery after Stroke. Brain Research Bulletin 2025, 111354. https://doi.org/10.1016/j.brainresbull.2025.111354.
‌

‌

