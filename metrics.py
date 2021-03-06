import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import DetCurveDisplay
from pyllr.quick_eval import scoreslabels_2_eer_cllr_mincllr

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

# SOURCE: https://github.com/clovaai/voxceleb_trainer/blob/master/tuneThreshold.py
#    @inproceedings{chung2020in,
#        title={In defence of metric learning for speaker recognition},
#        author={Chung, Joon Son and Huh, Jaesung and Mun, Seongkyu and Lee, Minjae and Heo, Hee Soo and Choe, Soyeon and Ham, Chiheon and Jung, Sunghwan and Lee, Bong-Jin and Han, Icksang},
#        booktitle={Interspeech},
#        year={2020}
#    }
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

# SOURCE: https://gitlab.eurecom.fr/nautsch/cllr
#    Niko Br??mmer, Luciana Ferrer and Albert Swart, 
#    "Out of a hundred trials, how many errors does your speaker verifier make?", 
#    2011, 
#    https://arxiv.org/abs/2104.00732
# SOURCE: https://github.com/bsxfan/PYLLR/blob/main/pyllr/quick_eval.py
def get_cllr(scores, labels):    
    eer, Cllr, minCllr = scoreslabels_2_eer_cllr_mincllr(np.asarray(scores), np.asarray(labels))
    return Cllr, minCllr

def process_cluster_subsets(
    df_data:pd.DataFrame, 
    df_metrics:pd.DataFrame, 
    dataset_prefix:str, 
    cluster_subsets, 
    groupby_column:str, 
    trace_eer_plots=False,
    include_target_ratio=False,
):
    for cluster in cluster_subsets:
        print('> Calculate metrics for \'{}\' cluster...'.format(cluster))
        labels = df_data[df_data[groupby_column].isin([cluster])]['label'].to_list()
        scores = df_data[df_data[groupby_column].isin([cluster])]['score'].to_list()

        if (include_target_ratio):
            targets = df_data.loc[df_data[groupby_column].isin([cluster]) & (df_data['label'] == 1), 'speaker_id'].count()
            df_metrics.loc[df_metrics.clusters.isin([cluster]), f'{dataset_prefix}_count'] = df_metrics.apply(
                lambda row: '{:.0f} \hfill(\SI{{{}}}{{\percent}})'.format(len(scores), targets/len(scores)*100), 
                axis=1
            )
        else:
            df_metrics.loc[df_metrics.clusters.isin([cluster]), f'{dataset_prefix}_count'] = len(scores)

        update_metrics(
            df_metrics, cluster, dataset_prefix,
            *calculate_metrics(labels, scores, trace_eer_plots=trace_eer_plots)
        )

def calculate_metrics(labels, scores, eer_method:str='roc', trace_eer_plots=False):
    # Calculate Equal Error Rate (EER)
    if (eer_method == 'roc'):
        val_eer, fpr, fnr, thresholds = get_eer(labels, scores, method='roc', display=trace_eer_plots)
    else:
        val_eer, fpr, fnr, thresholds = get_eer(labels, scores, method='det', display=trace_eer_plots)

    # Calculate minimum Detection Cost Function (minDCF)
    val_min_dcf, thresholds = get_min_dcf(fpr, fnr, thresholds)

    # Calculate Calibrated Log Likelyhood Ratio (Cllr)
    val_cllr, val_min_cllr = get_cllr(scores, labels)

    return val_eer, val_min_dcf, val_cllr, val_min_cllr

def update_metrics(df:pd.DataFrame, cluster_id:str, dataset_prefix:str, eer, min_dcf, cllr, min_cllr):
    df.loc[df.clusters.isin([cluster_id]), f'{dataset_prefix}_eer'] = eer
    df.loc[df.clusters.isin([cluster_id]), f'{dataset_prefix}_min_dcf'] = min_dcf
    df.loc[df.clusters.isin([cluster_id]), f'{dataset_prefix}_cllr'] = cllr
    df.loc[df.clusters.isin([cluster_id]), f'{dataset_prefix}_min_cllr'] = min_cllr
    return