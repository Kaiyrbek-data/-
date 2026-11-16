"""Grouped CV, disjoint calibration, validation-only selection, sealed test evaluation."""
import gc
import importlib.metadata
import json
import os
import platform
import threading
import time
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import psutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score, brier_score_loss, log_loss, precision_recall_curve,
    roc_curve, confusion_matrix)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from .config import *
from .features import INPUT_COLS, build_pipeline
from .database import save_experiments

class MemoryMonitor:
    """Sample resident memory of the current process; estimators use threads."""
    def __enter__(self):
        self.peak=0; self.stop=threading.Event()
        def sample():
            process=psutil.Process()
            while not self.stop.is_set():
                self.peak=max(self.peak,process.memory_info().rss)
                self.stop.wait(.05)
        self.thread=threading.Thread(target=sample,daemon=True); self.thread.start()
        return self
    def __exit__(self,*args):
        self.stop.set(); self.thread.join()

def model_candidates(quick=False):
    jobs=min(4,os.cpu_count() or 1)
    return {
      'Dummy':(DummyClassifier(strategy='prior'),None,0),
      'LogisticRegression':(LogisticRegression(max_iter=1000,solver='liblinear',random_state=SEED),
        {'model__C':[.01,.1,1.,10.], 'model__class_weight':[None,'balanced']},2 if quick else 4),
      'RandomForest':(RandomForestClassifier(n_estimators=100 if quick else 180,n_jobs=jobs,random_state=SEED),
        {'model__max_depth':[10,16,None], 'model__min_samples_leaf':[10,30],
         'model__max_features':['sqrt',.5], 'model__class_weight':[None,'balanced_subsample']},2 if quick else 4),
      'LightGBM':(LGBMClassifier(n_jobs=jobs,random_state=SEED,verbosity=-1,deterministic=True,
                     force_col_wise=True,subsample_freq=1),
        {'model__n_estimators':[180,350,550], 'model__learning_rate':[.03,.06],
         'model__num_leaves':[15,31], 'model__min_child_samples':[40,100],
         'model__reg_lambda':[1.,5.], 'model__colsample_bytree':[.8,1.],
         'model__subsample':[.8,1.], 'model__scale_pos_weight':[1.,4.]},2 if quick else 6),
    }

def choose_threshold(y, probabilities):
    precision,recall,thresholds=precision_recall_curve(y,probabilities)
    f1=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-15)
    best=np.flatnonzero(np.isclose(f1,f1.max()))[-1]
    return float(thresholds[best])

def expected_calibration_error(y,p,bins=10):
    groups=np.minimum((np.asarray(p)*bins).astype(int),bins-1)
    return float(sum((groups==b).mean()*abs(np.asarray(y)[groups==b].mean()-np.asarray(p)[groups==b].mean())
                     for b in range(bins) if (groups==b).any()))

def evaluate(y,p,threshold):
    prediction=np.asarray(p)>=threshold
    return dict(roc_auc=float(roc_auc_score(y,p)),pr_auc_ap=float(average_precision_score(y,p)),
      f1=float(f1_score(y,prediction,zero_division=0)),precision=float(precision_score(y,prediction,zero_division=0)),
      recall=float(recall_score(y,prediction,zero_division=0)),accuracy=float(accuracy_score(y,prediction)),
      brier=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,p,labels=[0,1])),
      ece_10bins=expected_calibration_error(y,p),threshold=float(threshold))

def group_bootstrap(y,p,groups,threshold,n_bootstrap=300):
    y=np.asarray(y); p=np.asarray(p); codes,unique=pd.factorize(groups)
    rng=np.random.default_rng(SEED); rows=[]
    for _ in range(n_bootstrap):
        weights=np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))[codes]
        if len(np.unique(y[weights>0]))<2: continue
        rows.append([roc_auc_score(y,p,sample_weight=weights),
                     average_precision_score(y,p,sample_weight=weights),
                     f1_score(y,p>=threshold,sample_weight=weights,zero_division=0)])
    values=np.asarray(rows)
    return {f'{metric}_{bound}':float(np.quantile(values[:,i],q))
            for i,metric in enumerate(['roc_auc','pr_auc_ap','f1'])
            for bound,q in [('ci_low',.025),('ci_high',.975)]}

def write_input_schema(fit):
    defaults={}; schema={}
    for column in INPUT_COLS:
        counts=fit[column].value_counts(dropna=False)
        default=counts.index[0] if len(counts) else None
        default=None if pd.isna(default) else default
        if column in NUM_COLS:
            default=int(round(fit[column].median()))
            schema[column]={'type':'integer','min':1 if column in ['time_in_hospital','num_lab_procedures','num_medications','number_diagnoses'] else 0,
                            'max':14 if column=='time_in_hospital' else 1000}
        else:
            schema[column]={'type':'category','values':sorted(str(x) for x in fit[column].dropna().unique())}
        defaults[column]=default.item() if isinstance(default,np.generic) else default
    (ARTIFACTS/'input_schema.json').write_text(json.dumps({'defaults':defaults,'fields':schema},ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'examples'/'patient.json').write_text(json.dumps(defaults,ensure_ascii=False,indent=2),encoding='utf-8')

def train_models(parts,quick=False):
    ensure_directories()
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    fit,cal,val,test=[parts[k] for k in ['fit','calibration','validation','test']]
    X,y=fit[INPUT_COLS],fit[TARGET]
    cv=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SEED)
    folds=list(cv.split(X,y,groups=fit.patient_nbr))
    fold_manifest=fit[['encounter_id','patient_nbr']].copy(); fold_manifest['cv_fold']=-1
    fold_summary=[]
    for fold,(a,b) in enumerate(folds):
        assert set(fit.iloc[a].patient_nbr).isdisjoint(fit.iloc[b].patient_nbr)
        fold_manifest.iloc[b,fold_manifest.columns.get_loc('cv_fold')]=fold
        fold_summary.append(dict(fold=fold,train_rows=len(a),validation_rows=len(b),
            train_positive_rate=float(y.iloc[a].mean()),validation_positive_rate=float(y.iloc[b].mean()),patient_overlap=0))
    fold_manifest.to_parquet(DATA/'cv_manifest.parquet',index=False)
    pd.DataFrame(fold_summary).to_csv(TABLES/'cv_split_audit.csv',index=False)
    validation_rows=[]; models={}; details={}; probabilities={}; search_frames=[]
    for name,(estimator,params,n_iter) in model_candidates(quick).items():
        print(f'[TRAIN] {name}: '+(f'{n_iter} candidates x 3 grouped folds' if params else 'fixed baseline, no hyperparameter search'),flush=True)
        pipeline=build_pipeline(estimator)
        started=time.perf_counter()
        with MemoryMonitor() as memory, threadpool_limits(limits=1,user_api='blas'):
            if params:
                search=RandomizedSearchCV(pipeline,params,n_iter=n_iter,cv=folds,
                    scoring={'ap':'average_precision','roc':'roc_auc'},refit='ap',
                    n_jobs=1,random_state=SEED,error_score='raise',return_train_score=False)
                search.fit(X,y)
                fitted=search.best_estimator_; cv_ap=float(search.best_score_)
                fit_seconds=search.refit_time_; best_params=search.best_params_
                cv_frame=pd.DataFrame(search.cv_results_).assign(model=name)
                parquet_frame=cv_frame.copy()
                for column in parquet_frame.columns:
                    if column.startswith('param'):
                        parquet_frame[column]=parquet_frame[column].map(lambda value: json.dumps(value,default=str))
                parquet_frame.to_parquet(ARTIFACTS/f'cv_{name}.parquet',index=False)
                cv_frame[['params','mean_test_ap','std_test_ap','mean_test_roc','mean_fit_time','rank_test_ap']].to_csv(TABLES/f'cv_{name}.csv',index=False)
                search_frames.append(cv_frame)
            else:
                before=time.perf_counter(); fitted=pipeline.fit(X,y)
                fit_seconds=time.perf_counter()-before; cv_ap=None; best_params={}
        search_seconds=time.perf_counter()-started
        cal_started=time.perf_counter()
        if name=='Dummy':
            calibrated=fitted
        else:
            calibrated=CalibratedClassifierCV(FrozenEstimator(fitted),method='sigmoid')
            calibrated.fit(cal[INPUT_COLS],cal[TARGET])
        calibration_seconds=time.perf_counter()-cal_started
        p_raw=fitted.predict_proba(val[INPUT_COLS])[:,1]
        p_val=calibrated.predict_proba(val[INPUT_COLS])[:,1]
        threshold=choose_threshold(val[TARGET],p_val)
        row=dict(model=name,**evaluate(val[TARGET],p_val,threshold),cv_ap=cv_ap,
                 fit_time_s=fit_seconds,search_time_s=search_seconds,
                 calibration_time_s=calibration_seconds,peak_process_rss_MiB=memory.peak/1024**2,
                 raw_brier=float(brier_score_loss(val[TARGET],p_raw)),
                 raw_log_loss=float(log_loss(val[TARGET],p_raw)))
        validation_rows.append(row)
        details[name]={'best_params':best_params,'threshold':threshold,'cv_ap':cv_ap}
        bundle={'model':calibrated,'threshold':threshold,'name':name,'run_id':run_id,
                'prediction_time':'discharge','excluded_discharge_codes':DEATH_CODES+HOSPICE_CODES,
                'selection':'validation average precision; threshold maximizes validation F1'}
        path=ARTIFACTS/f'{name}.joblib'; joblib.dump(bundle,path,compress=3)
        row['model_size_MiB']=path.stat().st_size/1024**2
        models[name]=calibrated
        print(f'[VALIDATION] {name} AP={row["pr_auc_ap"]:.4f}, ROC={row["roc_auc"]:.4f}, F1={row["f1"]:.4f}',flush=True)
        if params: del search
        gc.collect()
    validation=pd.DataFrame(validation_rows)
    # Selection is frozen BEFORE any test predictions are evaluated.
    winner=validation.query("model != 'Dummy'").sort_values('pr_auc_ap',ascending=False).iloc[0]['model']
    selection={'winner':winner,'criterion':'validation average precision','run_id':run_id,
               'details':details,'quick_mode':quick,'test_not_used_for_selection':True}
    (ARTIFACTS/'selection.json').write_text(json.dumps(selection,indent=2,ensure_ascii=False),encoding='utf-8')
    selected=joblib.load(ARTIFACTS/f'{winner}.joblib')
    joblib.dump(selected,ARTIFACTS/'best_model.joblib',compress=3)
    validation.to_csv(TABLES/'validation_metrics.csv',index=False)
    write_input_schema(fit)
    test_rows=[]; pred_frames=[]
    for name,model in models.items():
        started=time.perf_counter(); p=model.predict_proba(test[INPUT_COLS])[:,1]
        elapsed=time.perf_counter()-started; probabilities[name]=p
        threshold=details[name]['threshold']
        row=dict(model=name,**evaluate(test[TARGET],p,threshold))
        row.update(group_bootstrap(test[TARGET],p,test.patient_nbr,threshold,100 if quick else 300))
        row.update({c:validation.set_index('model').loc[name,c] for c in [
            'fit_time_s','search_time_s','calibration_time_s','peak_process_rss_MiB','model_size_MiB']})
        row['predict_ms_per_1000']=elapsed/len(test)*1e6
        row['selected_on_validation']=int(name==winner)
        test_rows.append(row)
        pred_frames.append(pd.DataFrame({'encounter_id':test.encounter_id.to_numpy(),
             'patient_nbr':test.patient_nbr.to_numpy(),'model':name,'y_true':test[TARGET].to_numpy(),
             'probability':p,'prediction':(p>=threshold).astype(int),'threshold':threshold}))
    metrics=pd.DataFrame(test_rows); predictions=pd.concat(pred_frames,ignore_index=True)
    metrics.to_csv(TABLES/'test_metrics.csv',index=False)
    metrics.to_parquet(ARTIFACTS/'experiment_results.parquet',index=False)
    predictions.to_parquet(DATA/'test_predictions.parquet',index=False)
    save_experiments(run_id,metrics,predictions)
    version_names=['pandas','numpy','scikit-learn','lightgbm','scipy','joblib','pyarrow','streamlit']
    versions={name:importlib.metadata.version(name) for name in version_names}
    versions.update(python=platform.python_version(),platform=platform.platform(),cpu=platform.processor(),threads=min(4,os.cpu_count() or 1))
    (ARTIFACTS/'environment.json').write_text(json.dumps(versions,indent=2),encoding='utf-8')
    plot_evaluation(test,probabilities,details,metrics,winner)
    interpret_and_analyze(models[winner],winner,details[winner]['threshold'],val,test,probabilities[winner],quick)
    print('[SELECTED]',winner,flush=True)
    print(metrics[['model','roc_auc','pr_auc_ap','f1','precision','recall']].to_string(index=False),flush=True)
    return metrics

def plot_evaluation(test,probabilities,details,metrics,winner):
    from .eda import save_figure
    fig,axes=plt.subplots(1,3,figsize=(16,4.8))
    for name,p in probabilities.items():
        fpr,tpr,_=roc_curve(test[TARGET],p); axes[0].plot(fpr,tpr,label=name)
        precision,recall,_=precision_recall_curve(test[TARGET],p); axes[1].plot(recall,precision,label=name)
        obs,pred=calibration_curve(test[TARGET],p,n_bins=8,strategy='quantile')
        axes[2].plot(pred,obs,marker='o',label=name)
    axes[0].plot([0,1],[0,1],'k--'); axes[1].axhline(test[TARGET].mean(),linestyle='--',color='gray')
    axes[2].plot([0,1],[0,1],'k--')
    for ax,title,x,y in zip(axes,['ROC curves','Precision-recall curves','Calibration'],
          ['False positive rate','Recall','Mean predicted probability'],['True positive rate','Precision','Observed frequency']):
        ax.set(title=title,xlabel=x,ylabel=y); ax.legend(fontsize=8)
    save_figure('model_curves')
    threshold=details[winner]['threshold']; p=probabilities[winner]
    matrix=confusion_matrix(test[TARGET],p>=threshold,labels=[0,1])
    pd.DataFrame(matrix,index=['Actual 0','Actual 1'],columns=['Predicted 0','Predicted 1']).to_csv(TABLES/'confusion_matrix.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    sns.heatmap(matrix,annot=True,fmt='d',cmap='Blues',ax=axes[0],cbar=False)
    axes[0].set(title=f'{winner}: test confusion matrix',xlabel='Predicted class',ylabel='Actual class')
    view=metrics.set_index('model')[['roc_auc','pr_auc_ap','f1']]
    view.plot.bar(ax=axes[1]); axes[1].set(title='Sealed test comparison',ylim=(0,1)); axes[1].tick_params(axis='x',rotation=25)
    save_figure('model_comparison')
    val=pd.read_csv(TABLES/'validation_metrics.csv')
    val.set_index('model')[['raw_brier','brier']].plot.bar(figsize=(9,4),color=['#8A9AA9','#147D92'])
    plt.title('Validation Brier score: before / after sigmoid calibration'); plt.xticks(rotation=20)
    save_figure('calibration_comparison')

def interpret_and_analyze(model,winner,threshold,val,test,p,quick):
    from .eda import save_figure
    sample=val.sample(n=min(1500 if quick else 3000,len(val)),random_state=SEED)
    importance=permutation_importance(model,sample[INPUT_COLS],sample[TARGET],
        scoring='average_precision',n_repeats=2 if quick else 4,n_jobs=1,random_state=SEED)
    table=pd.DataFrame({'feature':INPUT_COLS,'importance_mean':importance.importances_mean,
                        'importance_std':importance.importances_std}).sort_values('importance_mean',ascending=False)
    table.to_csv(TABLES/'permutation_importance.csv',index=False)
    top=table.head(15).sort_values('importance_mean')
    plt.figure(figsize=(10,6)); plt.barh(top.feature,top.importance_mean,xerr=top.importance_std,color='#147D92')
    plt.xlabel('Decrease in validation average precision'); plt.title(f'{winner}: permutation importance')
    save_figure('permutation_importance')
    errors=test[['encounter_id','patient_nbr','age','gender','race','number_inpatient','time_in_hospital',TARGET]].copy()
    errors['probability']=p; errors['prediction']=(p>=threshold).astype(int)
    errors['error_type']=np.select([(errors[TARGET]==1)&(errors.prediction==0),
        (errors[TARGET]==0)&(errors.prediction==1),(errors[TARGET]==1)&(errors.prediction==1)],['FN','FP','TP'],default='TN')
    errors.to_parquet(DATA/'error_analysis.parquet',index=False)
    errors.groupby('error_type').agg(
        encounters=(TARGET,'size'),patients=('patient_nbr','nunique'),
        mean_prior_inpatient=('number_inpatient','mean'),
        median_prior_inpatient=('number_inpatient','median'),
        mean_stay=('time_in_hospital','mean'),mean_probability=('probability','mean')
    ).to_csv(TABLES/'error_profiles.csv')
    errors.query("error_type=='FN'").sort_values('probability').head(20).to_csv(TABLES/'false_negative_examples.csv',index=False)
    rows=[]
    for column in ['age','gender','race']:
        for value,group in errors.groupby(column,dropna=False):
            result={'feature':column,'group':str(value),'n':len(group),'patients':group.patient_nbr.nunique(),
                    'positives':int(group[TARGET].sum()),'observed_rate':float(group[TARGET].mean()),
                    'recall':float(recall_score(group[TARGET],group.prediction,zero_division=0)),
                    'precision':float(precision_score(group[TARGET],group.prediction,zero_division=0)),
                    'f1':float(f1_score(group[TARGET],group.prediction,zero_division=0)),
                    'brier':float(brier_score_loss(group[TARGET],group.probability))}
            result['roc_auc']=float(roc_auc_score(group[TARGET],group.probability)) if group[TARGET].nunique()==2 else np.nan
            result['interpretation']='small group: descriptive only' if len(group)<200 or result['positives']<20 else 'descriptive; no causal interpretation'
            rows.append(result)
    pd.DataFrame(rows).to_csv(TABLES/'subgroup_metrics.csv',index=False)
