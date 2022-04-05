import io
import os
import re
import gzip
import numpy as np
import pandas as pd
from scipy import spatial
from ast import literal_eval

DATA_VOXCELEB_META = "data/vox1_meta.csv"
DATA_VOXCELEB_UTTERANCES = "data/utt2num_frames"
DATA_TRIALS_O = "data/voxceleb1_o_cleaned.trials"
DATA_TRIALS_EASY = "data/voxceleb1_e_cleaned.trials"
DATA_TRIALS_HARD = "data/voxceleb1_h_cleaned.trials"
DATA_XVECTOR_RAW = "data/xvector.txt"
DATA_XVECTOR_COMPRESSED = "data/xvector.h5"
DATA_XVECTOR_SAMPLE_RAW = "data/xvector-sample.txt"
DATA_XVECTOR_SAMPLE_COMPRESSED = "data/xvector-sample.h5"

def load_x_vectors(in_file:str):
    out_file = DATA_XVECTOR_RAW.replace('.txt', '.h5')

    # Open cleaned data if it already exists
    if(os.path.isfile(out_file)):
        print('> Load X Vectors from compressed file ({})...'.format(out_file))

        # Load gzip compressed dictionary
        with gzip.open(out_file, 'rb') as f:
            return np.load(f, allow_pickle=True).item()
    else:
        print('> Load X Vectors from raw file ({})...'.format(in_file))

        xvecs = {}

        # Read raw data, line by line
        with io.open(in_file, 'r', encoding='utf-8') as fr:
            for line in fr.readlines():
                # Remove trailing newline characters and extract id + values
                id, values = line.strip().split('  ')

                # Interpret values as numpy array
                values = re.sub(r'(?u)(?<=[\d])\ (?=[\d\-]+)', ',', values)
                values = np.array(literal_eval(values))

                # Insert values into the dictionary
                xvecs[id] = values

        # Save gzip compressed dictionary
        print('> Save X Vectors to compressed file ({})...'.format(out_file))
        with gzip.open(out_file, 'wb') as fw:
            np.save(fw, xvecs)

    return xvecs

def load_metadata_clusters(in_file:str):
    print('> Load VoxCeleb metadata from raw file ({})...'.format(in_file))

    df = pd.read_csv(in_file, sep='\t', encoding='utf-8')
    df.rename(columns={
        'VoxCeleb1 ID' : 'speaker_id', 
        'VGGFace1 ID' : 'speaker_name', 
        'Gender' : 'gender', 
        'Nationality' : 'nationality', 
        'Set' : 'set', 
    }, inplace=True)

    # Replace gender values with full word
    df.gender.replace({'m':'Male', 'f':'Female'}, inplace=True)

    return df

def load_utterance_clusters(in_file:str):
    print('> Load VoxCeleb utterance metadata from raw file ({})...'.format(in_file))

    df_uterrances = pd.read_csv(in_file, delim_whitespace=True, encoding='utf-8', names=['utterance_id', 'length'])

    df_utterance_clusters = \
        df_uterrances.groupby(
            pd.qcut(df_uterrances.length, 4)
        ).size().reset_index(name='count').length.unique().astype(str)

    utterance_thresholds = [
        df_uterrances.length.quantile(0.25), 
        df_uterrances.length.quantile(0.50), 
        df_uterrances.length.quantile(0.75), 
        df_uterrances.length.max()
    ]

    return (df_uterrances, df_utterance_clusters, utterance_thresholds)

def load_trials(in_file:str):
    # Read raw data, line by line
    with io.open(in_file, 'r', encoding='utf-8') as fr:
        # Split on spaces to extract data
        return [line.strip().split(' ') for line in fr.readlines()]

def get_speaker(line:str):
    speaker_id, _, gender, nationality, _ = line.strip().split('\t')
    return (speaker_id, ('Male' if gender == 'm' else 'Female', nationality))

def load_speakers(in_file:str):
    # Read raw data, line by line
    with io.open(in_file, 'r', encoding='utf-8') as fr:
        # Split on spaces to extract data
        return dict(get_speaker(line) for line in fr.readlines())

def load_utterances(in_file:str):
    # Read raw data, line by line
    with io.open(in_file, 'r', encoding='utf-8') as fr:
        # Split on spaces to extract data
        return dict(line.strip().split(' ') for line in fr.readlines())

def get_utterance_cluster(length:int, utterance_clusters):
    for cluster, threshold in utterance_clusters:
        if (length <= threshold):
            return cluster
    
    raise Exception('ERROR')

def is_target_speaker(x_vectors:np.array, enrollment_id, test_id):
    return 1 - spatial.distance.cosine(x_vectors[test_id], x_vectors[enrollment_id])

def load_processed_trials(trials_file:str, utterance_clusters, x_vecs:dict):
    out_file = trials_file.replace('.trials', '.h5')

    # Open cleaned data if it already exists
    if(os.path.isfile(out_file)):
        print('> Load trials from compressed file ({})...'.format(out_file))

        df = pd.read_csv(out_file, sep=',', encoding='utf-8')
    else:
        print('> Load trials from raw file ({})...'.format(trials_file))

        trials_clean = {
            'speaker_id' : [],
            'enrollment' : [],
            'test' : [],
            'label' : [],
            'score' : [],
        }
        for enrollment, test, label in load_trials(trials_file):
            speaker_id = enrollment[:7]
            trials_clean['speaker_id'].append(speaker_id)
            trials_clean['enrollment'].append(enrollment)
            trials_clean['test'].append(test)
            trials_clean['label'].append(1 if label == 'target' else 0)
            trials_clean['score'].append(is_target_speaker(x_vecs, enrollment, test))

        df = pd.DataFrame.from_dict(trials_clean)
        
        # Load VoxCeleb metadata for speakers
        print('> Load VoxCeleb metadata from raw file ({})...'.format(DATA_VOXCELEB_META))
        speakers = load_speakers(DATA_VOXCELEB_META)
        df['gender'] = df.apply(lambda row: speakers[row['speaker_id']][0], axis=1)
        df['nationality'] = df.apply(lambda row: speakers[row['speaker_id']][1], axis=1)

        # Load utterance metadata
        print('> Load VoxCeleb utterance metadata from raw file ({})...'.format(DATA_VOXCELEB_UTTERANCES))
        utterances = load_utterances(DATA_VOXCELEB_UTTERANCES)
        df['enrollment_length'] = df.apply(
            lambda row: get_utterance_cluster(
                int(utterances[row['enrollment']]),
                utterance_clusters,
            ), 
            axis=1
        )
        df['test_length'] = df.apply(
            lambda row: get_utterance_cluster(
                int(utterances[row['test']]), 
                utterance_clusters,
            ), 
            axis=1
        )

        print('> Save cleaned trials to compressed file ({})...'.format(out_file))
        df.to_csv(out_file, sep=',', index=False, encoding='utf-8')

    return df

def save_trials_per_cluster(df:pd.DataFrame):
    for gender in df.gender.sort_values().unique():
        df.loc[df.gender.isin([gender])].to_csv(
            'data/voxceleb1_gender_{}.trials'.format(gender), 
            sep=',', index=False, encoding='utf-8',
        )
    for country in df.nationality.sort_values().unique():
        df.loc[df.nationality.isin([country])].to_csv(
            'data/voxceleb1_nationality_{}.trials'.format(country), 
            sep=',', index=False, encoding='utf-8',
        )