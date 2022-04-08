
import numpy as np
import pandas as pd

import datasets as ds
import metrics

CLUSTER_ALL_SPEAKERS = 'All Speakers'
NA_LATEX_OUTPUT = '\\textemdash'

def print_clusters(df:pd.DataFrame):
    print(df.groupby('gender').size().reset_index(name='count'))
    print()
    print(df.groupby('nationality').size().reset_index(name='count'))
    print()
    print(df.groupby('nationality')['enrollment'].nunique())
    print()
    print(df.groupby('nationality')['test'].nunique())

##############################################################################
# [ENTRYPOINTS]
##############################################################################
if __name__ == "__main__":
    verbose = False
    trace_eer_plots = False
    include_target_ratio = False

    # Load datasets
    x_vecs = ds.load_x_vectors(ds.DATA_XVECTOR_RAW)
    metadata = ds.load_metadata_clusters(ds.DATA_VOXCELEB_META)
    utterances, utterance_clusters, utterance_thresholds = \
        ds.load_utterance_clusters(ds.DATA_VOXCELEB_UTTERANCES)

    if (verbose):
        print_clusters(metadata)
        print()
        print(utterances.info())
        print()
        print(utterances.describe().transpose())
        print()
        print(utterances.groupby(pd.qcut(utterances.length, 4)).size().reset_index(name='count'))

    # Create DataFrame for the calculated metrics
    clusters = [
        CLUSTER_ALL_SPEAKERS,
        *metadata.gender.sort_values().unique(),
        *metadata.nationality.sort_values().unique(),
        *utterance_clusters,
    ]
    datasets = [
        # ('test', DATA_TRIALS_O,
        ('easy', ds.DATA_TRIALS_EASY),
        ('hard', ds.DATA_TRIALS_HARD),
    ]
    cols = [
        'clusters',
        # 'test_eer',
        # 'test_min_dcf',
        'easy_count',
        'easy_eer',
        'easy_min_dcf',
        'easy_cllr',
        'easy_min_cllr',
        'hard_count',
        'hard_eer',
        'hard_min_dcf',
        'hard_cllr',
        'hard_min_cllr',
    ]
    data = [[c, *[np.nan]*(len(cols)-1)] for c in clusters]
    df_metrics = pd.DataFrame(data=data, columns=cols)

    for prefix, trials_file in datasets:
        print()
        print('> Processing \'{}\' dataset...'.format(trials_file))

        # Load data and metadata from files
        df = ds.load_processed_trials(
            trials_file=trials_file, 
            utterance_clusters=list(zip(utterance_clusters, utterance_thresholds)), 
            x_vecs=x_vecs
        )
        
        if (verbose):
            print()
            print_clusters(df)
            print()

        # ALL
        print('> Calculate metrics for \'{}\' cluster...'.format(CLUSTER_ALL_SPEAKERS))
        labels = df['label'].to_list()
        scores = df['score'].to_list()

        
        if (include_target_ratio):
            targets = df.loc[(df['label'] == 1), 'speaker_id'].count()
            df_metrics.loc[df_metrics.clusters.isin([CLUSTER_ALL_SPEAKERS]), f'{prefix}_count'] = df_metrics.apply(
                lambda row: '{:.0f} \hfill(\SI{{{}}}{{\percent}})'.format(len(scores), targets/len(scores)*100), 
                axis=1
            )
        else:
            df_metrics.loc[df_metrics.clusters.isin([CLUSTER_ALL_SPEAKERS]), f'{prefix}_count'] = len(scores)

        metrics.update_metrics(
            df_metrics, CLUSTER_ALL_SPEAKERS, prefix,
            *metrics.calculate_metrics(labels, scores)
        )
        
        # Calculate metrics per clusters
        metrics.process_cluster_subsets(
            df_data = df, 
            df_metrics = df_metrics, 
            dataset_prefix = prefix, 
            cluster_subsets = df.gender.sort_values().unique(), 
            groupby_column = 'gender', 
            trace_eer_plots= trace_eer_plots,
        )
        metrics.process_cluster_subsets(
            df_data = df, 
            df_metrics = df_metrics, 
            dataset_prefix = prefix, 
            cluster_subsets = df.nationality.sort_values().unique(), 
            groupby_column = 'nationality', 
            trace_eer_plots= trace_eer_plots,
        )
        metrics.process_cluster_subsets(
            df_data = df, 
            df_metrics = df_metrics, 
            dataset_prefix = prefix, 
            cluster_subsets = utterance_clusters, 
            groupby_column = 'test_length', 
            trace_eer_plots= trace_eer_plots,
        )

    if (verbose):
        print()
        print(df_metrics.info())
        print()
        print(df_metrics.head(25))

    # Prepare DataFrame for CSV file output
    df_metrics['easy_count'] = df_metrics['easy_count'].map(
        lambda x: x if not pd.isna(x) else NA_LATEX_OUTPUT
    ) 
    df_metrics['hard_count'] = df_metrics['hard_count'].map(
        lambda x: x if not pd.isna(x) else NA_LATEX_OUTPUT
    ) 
    df_metrics['easy_count'] = df_metrics['easy_count'].astype(str)
    df_metrics['hard_count'] = df_metrics['hard_count'].astype(str)
    df_metrics['easy_eer'] = df_metrics['easy_eer'].map(
        lambda x: '{:.2f}'.format(x) if not pd.isna(x) else NA_LATEX_OUTPUT
    ) 
    df_metrics['hard_eer'] = df_metrics['hard_eer'].map(
        lambda x: '{:.2f}'.format(x) if not pd.isna(x) else NA_LATEX_OUTPUT
    ) 
    df_metrics['easy_min_dcf'] = df_metrics['easy_min_dcf'].map(
        lambda x: '{:.3f}'.format(x) if not pd.isna(x) else NA_LATEX_OUTPUT
    ) 
    df_metrics['hard_min_dcf'] = df_metrics['hard_min_dcf'].map(
        lambda x: '{:.3f}'.format(x) if not pd.isna(x) else NA_LATEX_OUTPUT
    ) 
    df_metrics['easy_cllr'] = df_metrics['easy_cllr'].map(
        lambda x: '{:.3f}'.format(x) if not pd.isna(x) else NA_LATEX_OUTPUT
    ) 
    df_metrics['hard_cllr'] = df_metrics['hard_cllr'].map(
        lambda x: '{:.3f}'.format(x) if not pd.isna(x) else NA_LATEX_OUTPUT
    ) 
    df_metrics['easy_min_cllr'] = df_metrics['easy_min_cllr'].map(
        lambda x: '{:.3f}'.format(x) if not pd.isna(x) else NA_LATEX_OUTPUT
    ) 
    df_metrics['hard_min_cllr'] = df_metrics['hard_min_cllr'].map(
        lambda x: '{:.3f}'.format(x) if not pd.isna(x) else NA_LATEX_OUTPUT
    ) 
    df_metrics.rename(columns={
        'clusters':'Clusters',
        # 'test_eer_roc':'EER',
        # 'test_min_dcf':'minDCF',
        'easy_count':'Trials',
        'easy_eer':'EER',
        'easy_min_dcf':'minDCF',
        'easy_cllr':'$C_{llr}$',
        'easy_min_cllr':'$minC_{llr}$',
        'hard_count':'Trials',
        'hard_eer':'EER',
        'hard_min_dcf':'minDCF',
        'hard_cllr':'$C_{llr}$',
        'hard_min_cllr':'$minC_{llr}$',
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
        escape=False,
        label='tab:CompareMetrics',
        na_rep=NA_LATEX_OUTPUT,
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