import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import DetCurveDisplay

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

def calculate_metrics(labels, scores, eer_method:str='roc', trace_eer_plots=False):
    # Calculate Equal Error Rate (EER)
    if (eer_method == 'roc'):
        val_eer, fpr, fnr, thresholds = get_eer(labels, scores, method='roc', display=trace_eer_plots)
    else:
        val_eer, fpr, fnr, thresholds = get_eer(labels, scores, method='det', display=trace_eer_plots)

    # Calculate minimum Detection Cost Function (minDCF)
    val_min_dcf, thresholds = get_min_dcf(fpr, fnr, thresholds)
    
    return val_eer, val_min_dcf

def update_metrics(df:pd.DataFrame, cluster_id:str, dataset_prefix:str, eer, min_dcf):
    df.loc[df.clusters.isin([cluster_id]), f'{dataset_prefix}_eer'] = eer
    df.loc[df.clusters.isin([cluster_id]), f'{dataset_prefix}_min_dcf'] = min_dcf
    return