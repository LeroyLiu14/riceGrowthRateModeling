#!/usr/bin/env python3
import argparse, gc, json, math, platform, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, balanced_accuracy_score, matthews_corrcoef, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

p=argparse.ArgumentParser()
p.add_argument('--input-zip',type=Path,required=True); p.add_argument('--work-dir',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True)
p.add_argument('--datasets',nargs='+',default=['B0','D0','D1','D2']); p.add_argument('--tasks',nargs='+',default=['classification','regression'])
p.add_argument('--cv-folds',type=int,default=5); p.add_argument('--n-estimators',type=int,default=1); p.add_argument('--device',default='cpu'); p.add_argument('--skip-chronological',action='store_true')
a=p.parse_args(); a.work_dir.mkdir(parents=True,exist_ok=True); a.output_dir.mkdir(parents=True,exist_ok=True)
if not (a.work_dir/'manifest.json').exists():
    with zipfile.ZipFile(a.input_zip) as z: z.extractall(a.work_dir)
m=json.load(open(a.work_dir/'manifest.json',encoding='utf-8')); z=np.load(a.work_dir/'data.npz',allow_pickle=False)
ni={c:i for i,c in enumerate(m['full_numeric'])}; ci={c:i for i,c in enumerate(m['full_categorical'])}

def load(ds):
    d=m['datasets'][ds]; data={}
    for c in d['features']:
        if c in d['categorical']:
            vals=m['category_values'][c]; data[c]=pd.Series([vals[i] for i in z['Xcat'][:,ci[c]].astype(int)],dtype='string')
        else: data[c]=z['Xnum'][:,ni[c]].astype(np.float32)
    X=pd.DataFrame(data); yc=pd.Series(z['y_cls'].astype(int)); yr=z['y_reg'].astype(float)*m['y_reg_std']+m['y_reg_mean']
    return X,yc,pd.Series(yr),pd.Series(z['birth_group'].astype(str)),pd.Series(z['pig_id'].astype(str))

def cls_metric(y,s,pred):
    k=int(np.sum(y==1)); idx=np.argsort(-s)[:k]
    return dict(auc=roc_auc_score(y,s),ap=average_precision_score(y,s),topk_precision=float(np.mean(y[idx]==1)),balanced_accuracy=balanced_accuracy_score(y,pred),mcc=matthews_corrcoef(y,pred))
def reg_metric(y,p):
    rmse=math.sqrt(mean_squared_error(y,p)); rho=spearmanr(y,p,nan_policy='omit').statistic
    return dict(r2=r2_score(y,p),rmse=rmse,mae=mean_absolute_error(y,p),rrmse=rmse/np.mean(np.abs(y)),spearman=rho)
def clf():
    from tabicl import TabICLClassifier
    return TabICLClassifier(n_estimators=a.n_estimators,batch_size=1,n_jobs=2,random_state=42,device=a.device,use_amp=False,use_fa3=False,checkpoint_version='tabicl-classifier-v2-20260212.ckpt',allow_auto_download=True,verbose=True)
def reg():
    from tabicl import TabICLRegressor
    return TabICLRegressor(n_estimators=a.n_estimators,batch_size=1,n_jobs=2,random_state=42,device=a.device,use_amp=False,use_fa3=False,checkpoint_version='tabicl-regressor-v2-20260212.ckpt',allow_auto_download=True,verbose=True)
def score1(model,X):
    pr=np.asarray(model.predict_proba(X)); classes=np.asarray(model.classes_); return pr[:,int(np.flatnonzero(classes==1)[0])]
def chrono(groups):
    u=sorted(groups.unique()); cut=max(1,min(len(u)-1,int(len(u)*.7))); trset=set(u[:cut]); return np.flatnonzero(groups.isin(trset)),np.flatnonzero(~groups.isin(trset))
def summ(df):
    if df.empty:return df
    nums=df.select_dtypes(include=[np.number]).columns; g=df.groupby(['dataset','model'])[list(nums)]
    return g.mean().add_suffix('_mean').join(g.std(ddof=1).add_suffix('_sd')).reset_index()

cr=[]; rr=[]; cc=[]; rc=[]; oof_all=None
for ds in a.datasets:
    print('DATASET',ds,flush=True); X,yc,yr,g,ids=load(ds); splits=list(StratifiedGroupKFold(n_splits=a.cv_folds,shuffle=True,random_state=42).split(X,yc,g))
    o=pd.DataFrame({'pig_id':ids,'group':g,'y_cls':yc,'y_reg':yr}); ccol='CLS_'+ds; rcol='REG_'+ds; o[ccol]=np.nan; o[rcol]=np.nan
    for i,(tr,te) in enumerate(splits,1):
        print(ds,'fold',i,'train',len(tr),'test',len(te),flush=True)
        if 'classification' in a.tasks:
            t=time.time(); q=clf(); q.fit(X.iloc[tr],yc.iloc[tr]); s=score1(q,X.iloc[te]); pred=np.asarray(q.predict(X.iloc[te])).astype(int)
            cr.append(dict(dataset=ds,model='TabICLv2',fold=i,n_train=len(tr),n_test=len(te),positives_test=int(yc.iloc[te].sum()),**cls_metric(yc.iloc[te].to_numpy(),s,pred),seconds=time.time()-t)); o.loc[te,ccol]=s; del q; gc.collect()
        if 'regression' in a.tasks:
            t=time.time(); q=reg(); q.fit(X.iloc[tr],yr.iloc[tr]); pred=np.asarray(q.predict(X.iloc[te]),dtype=float).reshape(-1)
            rr.append(dict(dataset=ds,model='TabICLv2',fold=i,n_train=len(tr),n_test=len(te),**reg_metric(yr.iloc[te].to_numpy(),pred),seconds=time.time()-t)); o.loc[te,rcol]=pred; del q; gc.collect()
    if not a.skip_chronological:
        tr,te=chrono(g)
        if 'classification' in a.tasks:
            t=time.time(); q=clf(); q.fit(X.iloc[tr],yc.iloc[tr]); s=score1(q,X.iloc[te]); pred=np.asarray(q.predict(X.iloc[te])).astype(int); cc.append(dict(dataset=ds,model='TabICLv2',n_train=len(tr),n_test=len(te),positives_test=int(yc.iloc[te].sum()),**cls_metric(yc.iloc[te].to_numpy(),s,pred),seconds=time.time()-t)); del q; gc.collect()
        if 'regression' in a.tasks:
            t=time.time(); q=reg(); q.fit(X.iloc[tr],yr.iloc[tr]); pred=np.asarray(q.predict(X.iloc[te]),dtype=float).reshape(-1); rc.append(dict(dataset=ds,model='TabICLv2',n_train=len(tr),n_test=len(te),**reg_metric(yr.iloc[te].to_numpy(),pred),seconds=time.time()-t)); del q; gc.collect()
    oof_all=o if oof_all is None else oof_all.merge(o[['pig_id',ccol,rcol]],on='pig_id',how='outer')

cdf=pd.DataFrame(cr); rdf=pd.DataFrame(rr); cdf.to_csv(a.output_dir/'classification_folds.csv',index=False); rdf.to_csv(a.output_dir/'regression_folds.csv',index=False); summ(cdf).to_csv(a.output_dir/'classification_summary.csv',index=False); summ(rdf).to_csv(a.output_dir/'regression_summary.csv',index=False); pd.DataFrame(cc).to_csv(a.output_dir/'classification_chrono.csv',index=False); pd.DataFrame(rc).to_csv(a.output_dir/'regression_chrono.csv',index=False); oof_all.to_csv(a.output_dir/'oof_predictions.csv',index=False)
import tabicl
json.dump(dict(tabicl=getattr(tabicl,'__version__','unknown'),torch=torch.__version__,device=a.device,n_estimators=a.n_estimators,cv_folds=a.cv_folds,n=m['n_common'],datasets=a.datasets),open(a.output_dir/'metadata.json','w'),indent=2)
print('COMPLETED',a.output_dir.resolve(),flush=True)
