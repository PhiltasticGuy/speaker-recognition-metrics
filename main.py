import io
import os
import re
import gzip
from matplotlib import pyplot as plt
import pandas as pd
from scipy import spatial
from ast import literal_eval
import numpy as np
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import DetCurveDisplay

DATA_VOXCELEB_META = "data/vox1_meta.csv"
DATA_TRIALS_O = "data/voxceleb1_o_cleaned.trials"
DATA_TRIALS_EASY = "data/voxceleb1_e_cleaned.trials"
DATA_TRIALS_HARD = "data/voxceleb1_h_cleaned.trials"
DATA_XVECTOR_RAW = "data/xvector.txt"
DATA_XVECTOR_COMPRESSED = "data/xvector.h5"
DATA_XVECTOR_SAMPLE_RAW = "data/xvector-sample.txt"
DATA_XVECTOR_SAMPLE_COMPRESSED = "data/xvector-sample.h5"

CLUSTER_ALL_SPEAKERS = 'All Speakers'

def load_x_vectors(in_file:str, out_file:str):
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
                id, values = line.replace('\n', '').split('  ')

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

def load_voxceleb_metadata(in_file:str):
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

def load_trials(in_file:str):
    # Read raw data, line by line
    with io.open(in_file, 'r', encoding='utf-8') as fr:
        # Split on spaces to extract data
        return [line.replace('\n', '').split(' ') for line in fr.readlines()]

def load_processed_trials(trials_file:str, metadata:pd.DataFrame, x_vecs:dict):
    out_file = trials_file.replace('.trials', '.h5')
    trials = load_trials(trials_file)

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
            'gender' : [],
            'nationality' : [],
            'score' : [],
        }
        for enrollment, test, label in trials:
            speaker_id = enrollment[:7]
            trials_clean['speaker_id'].append(speaker_id)
            trials_clean['enrollment'].append(enrollment)
            trials_clean['test'].append(test)
            trials_clean['label'].append(1 if label == 'target' else 0)
            trials_clean['gender'].append(None)
            trials_clean['nationality'].append(None)
            trials_clean['score'].append(is_target_speaker(x_vecs, enrollment, test))

        df = pd.DataFrame.from_dict(trials_clean)
        
        for index, data in metadata.iterrows():
            # Remove trailing newline characters and extract id + values
            speaker_id, _, gender, nationality, _ = data

            # Insert values into the dictionary
            df.loc[df['speaker_id'].isin([speaker_id]), 'gender'] = gender
            df.loc[df['speaker_id'].isin([speaker_id]), 'nationality'] = nationality

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

def print_clusters(df:pd.DataFrame):
    print(df.groupby('gender').size().reset_index(name='count'))
    print()
    print(df.groupby('nationality').size().reset_index(name='count'))
    print()
    print(df.groupby('nationality')['enrollment'].nunique())
    print()
    print(df.groupby('nationality')['test'].nunique())

def is_target_speaker(x_vectors:np.array, enrollment_id, test_id):
    return 1 - spatial.distance.cosine(x_vectors[test_id], x_vectors[enrollment_id])

def get_eer(labels, scores, method:str='roc', display:bool=False):
    if method == 'roc':
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
        fnr = 1 - tpr
    elif method == 'det':
        fpr, fnr, thresholds = metrics.det_curve(labels, scores)

    idx_eer = np.nanargmin(np.abs(fnr - fpr))
    eer = max(fpr[idx_eer], fnr[idx_eer]) * 100

    if display:
        print_roc_det_curves(labels, scores, method)

    return eer, fpr, fnr, thresholds

def print_roc_det_curves(labels, scores, method:str='roc'):
    if method == 'roc':
        d:RocCurveDisplay = RocCurveDisplay.from_predictions(labels, scores)
        d.ax_.set_title("Receiver Operating Characteristic (ROC) curves")
        x = [
            np.min([d.ax_.get_xlim(), d.ax_.get_ylim()]),  # min of both axes
            np.max([d.ax_.get_xlim(), d.ax_.get_ylim()]),  # max of both axes
        ]
        y = x.copy()
        y.reverse()
    elif method == 'det':
        d:DetCurveDisplay = DetCurveDisplay.from_predictions(labels, scores)
        d.ax_.set_title("Detection Error Tradeoff (DET) curves")
        x = y = [
            np.min([d.ax_.get_xlim(), d.ax_.get_ylim()]),  # min of both axes
            np.max([d.ax_.get_xlim(), d.ax_.get_ylim()]),  # max of both axes
        ]

    d.ax_.set_aspect('equal')
    d.ax_.plot(x, y, 'k-', alpha=0.75, zorder=0)
    d.ax_.grid(linestyle="--")
    plt.savefig('plots/{}_curve.png'.format(method), bbox_inches='tight')
    plt.close()

# SOURCE: https://github.com/clovaai/voxceleb_trainer
# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
def get_min_dcf(fnrs, fprs, thresholds, p_target=0.05, c_miss=1, c_fa=1):    
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]

    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    
    return min_dcf, min_c_det_threshold

def calculate_metrics(labels, scores):
    val_eer_roc, fpr, fnr, thresholds = get_eer(labels, scores, method='roc', display=trace_eer_plots)
    val_eer_det, fpr, fnr, thresholds = get_eer(labels, scores, method='det', display=trace_eer_plots)
    val_min_dcf, thresholds = get_min_dcf(fpr, fnr, thresholds)
    return val_eer_roc, val_eer_det, val_min_dcf

def update_metrics(df:pd.DataFrame, cluster_id:str, dataset_prefix:str, eer_roc, eer_det, min_dcf):
    df.loc[df.clusters.isin([cluster_id]), f'{dataset_prefix}_eer_roc'] = eer_roc
    df.loc[df.clusters.isin([cluster_id]), f'{dataset_prefix}_eer_det'] = eer_det
    df.loc[df.clusters.isin([cluster_id]), f'{dataset_prefix}_min_dcf'] = min_dcf
    return

##############################################################################
# [ENTRYPOINTS]
##############################################################################
if __name__ == "__main__":
    verbose = False
    trace_eer_plots = False

    x_vecs = load_x_vectors(DATA_XVECTOR_RAW, DATA_XVECTOR_RAW.replace('.txt', '.h5'))
    metadata = load_voxceleb_metadata(DATA_VOXCELEB_META)

    if (verbose):
        print_clusters(metadata)

    # Create DataFrame for the calculated metrics
    clusters = [
        CLUSTER_ALL_SPEAKERS,
        *metadata.gender.sort_values().unique(),
        *metadata.nationality.sort_values().unique(),
    ]
    datasets = [
        # ('test', DATA_TRIALS_O,
        ('easy', DATA_TRIALS_EASY),
        ('hard', DATA_TRIALS_HARD),
    ]
    cols = [
        'clusters',
        # 'test_eer_roc',
        # 'test_eer_det',
        # 'test_min_dcf',
        'easy_eer_roc',
        'easy_eer_det',
        'easy_min_dcf',
        'hard_eer_roc',
        'hard_eer_det',
        'hard_min_dcf',
    ]
    data = [[c, *[np.nan]*(len(cols)-1)] for c in clusters]
    df_metrics = pd.DataFrame(data=data, columns=cols)

    for prefix, trials_file in datasets:
        print()
        print('> Processing \'{}\' dataset...'.format(trials_file))

        # Load data and metadata from files
        df = load_processed_trials(trials_file=trials_file, metadata=metadata, x_vecs=x_vecs)
        
        if (verbose):
            print()
            print_clusters(df)
            print()

        # ALL
        print('> Calculate metrics for \'{}\' cluster...'.format(CLUSTER_ALL_SPEAKERS))
        labels = df['label'].to_list()
        scores = df['score'].to_list()
        update_metrics(
            df_metrics, CLUSTER_ALL_SPEAKERS, prefix,
            *calculate_metrics(labels, scores)
        )
        
        # Gender
        for gender in df.gender.sort_values().unique():
            print('> Calculate metrics for \'{}\' cluster...'.format(gender))
            labels = df[df.gender.isin([gender])]['label'].to_list()
            scores = df[df.gender.isin([gender])]['score'].to_list()
            update_metrics(
                df_metrics, gender, prefix,
                *calculate_metrics(labels, scores)
            )

        # Nationality
        for country in df.nationality.sort_values().unique():
            print('> Calculate metrics for \'{}\' cluster...'.format(country))
            labels = df[df.nationality.isin([country])]['label'].to_list()
            scores = df[df.nationality.isin([country])]['score'].to_list()
            update_metrics(
                df_metrics, country, prefix,
                *calculate_metrics(labels, scores)
            )

    if (verbose):
        print()
        print(df_metrics.info())
        print()
        print(df_metrics.head(25))

    # Prepare DataFrame for CSV file output
    df_metrics.rename(columns={
        'clusters':'Clusters',
        # 'test_eer_roc':'EER (ROC)',
        # 'test_eer_det':'EER (DET)',
        # 'test_min_dcf':'minDCF',
        'easy_eer_roc':'EER (ROC)',
        'easy_eer_det':'EER (DET)',
        'easy_min_dcf':'minDCF',
        'hard_eer_roc':'EER (ROC)',
        'hard_eer_det':'EER (DET)',
        'hard_min_dcf':'minDCF',
    }, inplace=True)

    # Save the DataFrame to CSV file
    print()
    print('> Save DataFrame to CSV file...')
    df_metrics.to_csv('data/metrics_per_cluster.csv', sep=',', encoding='utf-8', index=False)

    # Save the DataFrame to latex file
    # More styles available at:
    # https://pandas.pydata.org/docs/reference/api/pandas.io.formats.style.Styler.to_latex.html
    # df_metrics.style.to_latex('data/metrics_per_cluster.latex', index=False, bold_rows=True)
    print('> Save DataFrame to latex file...')
    # df_metrics.style.format(decimal=',', thousands='.', precision=3)
    df_metrics.to_latex(
        'data/metrics_per_cluster.latex', 
        index=False,
        label='tab:CompareMetrics',
        na_rep='-',
        caption='Comparison of speaker recognition metrics per dataset and cluster.',
        bold_rows=True,
        float_format="%.5f"
    )

    # # Calculate metrics using SpeechBrain librairies
    # import torch, torchaudio
    # torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
    # torchaudio.set_audio_backend("soundfile")  # The new interface
    # from speechbrain.utils.metric_stats import EER, minDCF
    # positives = [score for score, label in pairs if label == 1]
    # negatives = [score for score, label in pairs if label == 0]
    # val_eer, threshold = EER(torch.tensor(positives), torch.tensor(negatives))
    # val_minDCF, threshold = minDCF(torch.tensor(positives), torch.tensor(negatives))
    # print()
    # print('> SpeechBrain...')
    # print('   EER: {}'.format(val_eer * 100))
    # print('minDCF: {}'.format(val_minDCF * 100))