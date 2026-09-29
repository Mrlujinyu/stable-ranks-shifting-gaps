#!/usr/bin/env python3
"""Verify manuscript evidence from CSV data. Use --resample and --gee for full checks."""
import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
import argparse,json,sys,itertools
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
R=Path(__file__).resolve().parents[1];D=R/'data';A=R/'analysis_outputs/final_checks'
# Exact round-trip parsing preserves probability masses in discrete null tails.
_read_csv = pd.read_csv
def read_csv(*args, **kwargs):
    kwargs.setdefault('float_precision', 'round_trip')
    return _read_csv(*args, **kwargs)
checks=[]
def check(name,value,expected,atol=1e-8):
    a=np.asarray(value);b=np.asarray(expected)
    ok=bool(np.allclose(a,b,atol=atol,rtol=1e-8,equal_nan=True))
    checks.append({'check':name,'passed':ok,'max_absolute_error':float(np.nanmax(abs(a-b))) if a.size else 0.0})
    if not ok:print('FAILED',name,value,expected,flush=True)
def load_main():
    d=read_csv(D/'main_panel.csv');cfg=read_csv(D/'table2_configurations.csv');cfg=cfg[cfg.primary].sort_values('submission')
    return d[cfg.figure_id].to_numpy(float),d,list(cfg.submission)
def load_pro():
    d=read_csv(D/'pro_labels_analysis.csv');meta=read_csv(D/'pro_submission_metadata.csv');names=sorted(meta.loc[meta.included,'submission'])
    o=read_csv(D/'pro_observed_matrix.csv');m=o.pivot(index='task_id',columns='submission',values='resolved').reindex(d.task_id)[names]
    assert m.notna().all().all()
    return m.to_numpy(float),d,names

def audit_panel(Y,d,names,bad,label):
    main=label=='main';pairs=read_csv(D/('all91_pairs.csv' if main else 'pro_all36_pairs.csv'));scores=read_csv(D/('all14_scores.csv' if main else 'pro_score_distortion.csv'))
    ij=np.triu_indices(len(names),1);raw=Y.mean(0)*100;f=Y[~bad].mean(0)*100
    idx={a:i for i,a in enumerate(names)}
    for row in scores.itertuples():
        j=idx[row.agent];check(label+':score:'+row.agent,[raw[j],f[j]],[row.raw_score_pp,row.clean_score_pp])
    tr=pairs if main else read_csv(D/'pro_statistical_transitions.csv')
    disc=Y[:,ij[0]]-Y[:,ij[1]];gr=raw[ij[0]]-raw[ij[1]];gf=f[ij[0]]-f[ij[1]]
    for s,x,g in [('raw',disc,gr),('clean',disc[~bad],gf)]:
        p=np.array([stats.binomtest(int((x[:,j]>0).sum()),int((x[:,j]!=0).sum()),.5).pvalue if np.any(x[:,j]) else 1.0 for j in range(x.shape[1])]);q=multipletests(p,method='fdr_bh')[1]
        lookup={(r.agent_a,r.agent_b):r for r in tr.itertuples()}
        for k,(i,j) in enumerate(zip(*ij)):
            row=lookup[names[i],names[j]];check(label+':'+s+':p:'+str(k),p[k],getattr(row,s+'_p'));check(label+':'+s+':q:'+str(k),q[k],getattr(row,s+'_q'))
    lookup={(r.agent_a,r.agent_b):r for r in pairs.itertuples()}
    for k,(i,j) in enumerate(zip(*ij)):
        row=lookup[names[i],names[j]];check(label+':gap:'+str(k),[gr[k],gf[k],gf[k]-gr[k]],[row.raw_gap_pp,row.clean_gap_pp,row.delta_gap_pp])
    obs=json.loads((D/('core_summary.json' if main else 'pro_summary.json')).read_text());obs=obs[0] if main else obs
    check(label+':N',len(Y),obs['tasks']);check(label+':removed',bad.sum(),obs['removed'])
    check(label+':median_gap',np.median(abs(gf-gr)),obs['median_abs_gap_pp']);check(label+':tau',stats.kendalltau(raw,f).statistic,obs['kendall_tau'])
    null=read_csv(D/('random_removal.csv' if main else 'pro_random_removal.csv'))
    for k in ['mean_abs_gap_pp','median_abs_gap_pp','pairs_ge_2pp_rate','practical_1pp_rate','practical_2pp_rate','reversals','statistical_change_rate','kendall_tau']:
        ext=(null[k]<=obs[k]) if k=='kendall_tau' else (null[k]>=obs[k]);v=(1+ext.sum())/(1+len(null));check(label+':archived_null_p:'+k,v,obs[k+'_null_p'])
    return obs

def audit_controls(Y,d,names,bad):
    ij=np.triu_indices(len(names),1);raw=np.zeros(len(names));f=raw.copy();support=0;ws=[]
    for _,ix in d.groupby(['repo','human_difficulty']).indices.items():
        good=ix[~bad[ix]]
        if len(good)==0 or bad[ix].sum()==0:continue
        raw+=Y[ix].sum(0);f+=Y[good].mean(0)*len(ix);support+=len(ix);ws.extend([len(ix)/len(good)]*len(good))
    raw=raw/support*100;f=f/support*100;dg=(f-raw)[ij[0]]-(f-raw)[ij[1]];obs=json.loads((D/'reweighted_summary.json').read_text())
    check('weighted:support',support,obs['support_tasks']);check('weighted:ESS',sum(ws)**2/np.sum(np.array(ws)**2),obs['clean_ess']);check('weighted:median_gap',np.median(abs(dg)),obs['median_abs_gap_pp'])
    original=Y.mean(0)*100;rowidx=dict(zip(d.task_id,range(len(d))))
    saved=json.loads((D/'matched_full_reference_summary.json').read_text())
    for n in ['exact_repo_difficulty','exact_repo_difficulty_nearest_metadata']:
        mm=read_csv(D/(n+'_matches.csv'));a=np.array([rowidx[t] for t in mm.defect_task]);b=np.array([rowidx[t] for t in mm.clean_task]);assert len(set(a))==len(a) and len(set(b))==len(b)
        assert np.all(d.repo.iloc[a].to_numpy()==d.repo.iloc[b].to_numpy());assert np.all(d.human_difficulty.iloc[a].to_numpy()==d.human_difficulty.iloc[b].to_numpy())
        sq=np.delete(Y,a,axis=0).mean(0)*100-original;sc=np.delete(Y,b,axis=0).mean(0)*100-original;gq=sq[ij[0]]-sq[ij[1]];gc=sc[ij[0]]-sc[ij[1]]
        rec=next(x for x in saved if x['method']==n)
        check(n+':full_reference',[np.mean(abs(gq)),np.mean(abs(gc)),np.mean(abs(gq-gc))],[rec['mean_abs_quality_delta_gap_pp'],rec['mean_abs_clean_removal_delta_gap_pp'],rec['mean_abs_contrast_pp']])
    disc=Y[:,ij[0]]-Y[:,ij[1]];inf=abs(disc-disc.mean(0)).mean(1)*100/(len(Y)-1);savedinf=read_csv(D/'task_influence.csv').set_index('task_id').reindex(d.task_id)
    check('influence:all_tasks',inf,savedinf.influence_pp.to_numpy())

def audit_gee(Y,d,names):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import patsy
    n,m=Y.shape
    lo=pd.DataFrame({'success':Y.ravel(),'agent':np.tile([f'A{i:02}' for i in range(m)],n),'task':np.repeat(np.arange(n),m),'repo':np.repeat(d.repo,m),'difficulty':np.repeat(d.human_difficulty,m),'defect':np.repeat(d.severity.ge(2).astype(int),m)})
    fit=smf.gee('success ~ C(agent)*(defect+C(difficulty)) + C(repo)',groups='task',data=lo,family=sm.families.Binomial(),cov_struct=sm.cov_struct.Independence()).fit(maxiter=150)
    ix=[i for i,k in enumerate(fit.params.index) if k.endswith(':defect')];w=fit.wald_test(np.eye(len(fit.params))[ix],scalar=True)
    ref=next(x for x in json.loads((D/'interaction_summary.json').read_text()) if x['model']=='primary')
    check('GEE:Wald',float(w.statistic),ref['interaction_wald'],atol=1e-5);check('GEE:p',float(w.pvalue),ref['p'],atol=1e-7)
    profiles=read_csv(D/'primary_profiles.csv').set_index('agent')
    for i,agent in enumerate(names):
        z=lo.iloc[::m].copy();z['agent']=f'A{i:02}';a=z.copy();b=z.copy();a['defect']=1;b['defect']=0
        xa=np.asarray(patsy.build_design_matrices([fit.model.data.design_info],a)[0]);xb=np.asarray(patsy.build_design_matrices([fit.model.data.design_info],b)[0])
        pa=1/(1+np.exp(-xa@fit.params));pb=1/(1+np.exp(-xb@fit.params));effect=(pa-pb).mean()*100
        grad=((pa*(1-pa))[:,None]*xa-(pb*(1-pb))[:,None]*xb).mean(0)*100;se=np.sqrt(max(0,grad@fit.cov_params().to_numpy()@grad))
        r=profiles.loc[agent];check('GEE:profile:'+agent,[effect,effect-1.96*se,effect+1.96*se],[r.adjusted_probability_delta_pp,r.ci_low,r.ci_high],atol=1e-5)
    return {'converged':bool(fit.converged),'finite_covariance':bool(np.isfinite(fit.cov_params()).all().all()),'wald':float(w.statistic),'p':float(w.pvalue)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--resample',action='store_true');ap.add_argument('--gee',action='store_true');args=ap.parse_args();A.mkdir(parents=True,exist_ok=True)
    y,d,n=load_main();bad=d.severity.ge(2).to_numpy();obs=audit_panel(y,d,n,bad,'main');audit_controls(y,d,n,bad)
    py,pd_,pn=load_pro();pb=pd_.corrected.to_numpy(bool);pobs=audit_panel(py,pd_,pn,pb,'pro');extra={}
    if args.resample:
        from statistics_core import run
        for label,Y,L,N,B,S in [('main',y,d,n,bad,obs),('pro',py,pd_,pn,pb,pobs)]:
            result=run(Y,L,N,'severity_ge2' if label=='main' else 'known_corrected',B,dataset='A' if label=='main' else 'Pro')
            for k in S:
                if isinstance(S[k],(float,int)) and k in result[-1]:check(label+':resampled:'+k,result[-1][k],S[k],atol=1e-7)
            extra[label+'_resampling']=result[-1]
    if args.gee:extra['primary_gee']=audit_gee(y,d,n)
    report={'checks':len(checks),'passed':sum(c['passed'] for c in checks),'failed':[c for c in checks if not c['passed']],'resampling_reexecuted':args.resample,'primary_GEE_refitted':args.gee,'agent_runs_reexecuted':False,'checks_detail':checks,'numerical_results':extra}
    (A/'verification_results.json').write_text(json.dumps(report,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
    print('CHECKS',report['checks'],'PASSED',report['passed'],'FAILED',len(report['failed']))
    return 0 if not report['failed'] else 1
if __name__=='__main__':sys.exit(main())
