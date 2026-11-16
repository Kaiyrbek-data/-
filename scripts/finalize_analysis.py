"""Readable publication figures and a reproducible validation threshold audit."""
from pathlib import Path
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,precision_recall_curve,roc_curve
from sklearn.calibration import calibration_curve

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.config import *
from src.features import INPUT_COLS
from src.modeling import write_input_schema,choose_threshold
from src.eda import save_figure

def finalize_analysis():
    selection=json.loads((ARTIFACTS/'selection.json').read_text(encoding='utf-8'))
    winner=selection['winner']
    cohort=pd.read_parquet(DATA/'cohort.parquet')
    manifest=pd.read_parquet(DATA/'split_manifest.parquet')
    frame=cohort.merge(manifest,on=['encounter_id','patient_nbr'],validate='one_to_one')
    fit=frame.query("split=='fit'"); validation=frame.query("split=='validation'")
    write_input_schema(fit)
    bundle=joblib.load(ARTIFACTS/'best_model.joblib')
    p_val=bundle['model'].predict_proba(validation[INPUT_COLS])[:,1]
    threshold=choose_threshold(validation[TARGET],p_val)
    if not np.isclose(threshold,bundle['threshold']):
        raise AssertionError('Persisted threshold does not reproduce on validation')
    precision,recall,thresholds=precision_recall_curve(validation[TARGET],p_val)
    scores=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-15)
    pd.DataFrame({'threshold':thresholds,'precision':precision[:-1],'recall':recall[:-1],'f1':scores}).to_csv(TABLES/'validation_threshold_curve.csv',index=False)
    plt.figure(figsize=(8,4.5))
    for values,label in [(precision[:-1],'Precision'),(recall[:-1],'Recall'),(scores,'F1')]:
        plt.plot(thresholds,values,label=label)
    plt.axvline(threshold,color='black',linestyle='--',label=f'Chosen threshold {threshold:.3f}')
    plt.xlim(0,min(.7,float(np.quantile(p_val,.995))+.1));plt.ylim(0,1)
    plt.xlabel('Probability threshold');plt.ylabel('Metric');plt.title('Threshold chosen on validation only');plt.legend()
    save_figure('validation_threshold_curve')
    pred=pd.read_parquet(DATA/'test_predictions.parquet').query('model==@winner')
    y=pred.y_true.to_numpy(); p=pred.probability.to_numpy()
    matrix=confusion_matrix(y,p>=threshold,labels=[0,1])
    plt.figure(figsize=(7,4.3))
    sns.heatmap(matrix,annot=True,fmt='d',cmap='Blues',cbar=False,annot_kws={'size':18},
                xticklabels=['Predicted 0','Predicted 1'],yticklabels=['Actual 0','Actual 1'])
    plt.title(f'{winner}: sealed test confusion matrix')
    save_figure('selected_confusion')
    fpr,tpr,_=roc_curve(y,p)
    plt.figure(figsize=(7,4.3));plt.plot(fpr,tpr,color='#147D92',lw=2);plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False positive rate');plt.ylabel('True positive rate');plt.title(f'{winner}: test ROC curve');save_figure('selected_roc')
    pr,re,_=precision_recall_curve(y,p)
    plt.figure(figsize=(7,4.3));plt.plot(re,pr,color='#147D92',lw=2);plt.axhline(y.mean(),linestyle='--',color='gray')
    plt.xlabel('Recall');plt.ylabel('Precision');plt.title(f'{winner}: test precision-recall curve');save_figure('selected_pr')
    observed,mean=calibration_curve(y,p,n_bins=8,strategy='quantile')
    plt.figure(figsize=(7,4.3));plt.plot(mean,observed,'o-',color='#147D92');plt.plot([0,.6],[0,.6],'k--')
    plt.xlabel('Predicted probability');plt.ylabel('Observed frequency');plt.title(f'{winner}: test calibration');save_figure('selected_calibration')
    errors=pd.read_parquet(DATA/'error_analysis.parquet')
    errors.groupby('error_type').agg(encounters=(TARGET,'size'),patients=('patient_nbr','nunique'),
        mean_prior_inpatient=('number_inpatient','mean'),median_prior_inpatient=('number_inpatient','median'),
        mean_stay=('time_in_hospital','mean'),mean_probability=('probability','mean')).to_csv(TABLES/'error_profiles.csv')
    print('Threshold reproduced; publication figures and input schema are ready.',flush=True)

if __name__=='__main__':finalize_analysis()
