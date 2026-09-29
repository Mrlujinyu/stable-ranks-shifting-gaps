import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']: os.environ[key]='2'
from pathlib import Path
import json,itertools
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from statsmodels.stats.multitest import multipletests
R=Path(__file__).resolve().parents[1]; O=R/'results'; B=10000; SEED=20260919
def js(name,obj): (O/name).write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')
def ci(x): return np.quantile(x,[.025,.975],axis=0)
def gaps(s,ij): return s[...,ij[0]]-s[...,ij[1]]
def bh(p):
    p=np.asarray(p); order=np.argsort(p,axis=-1); sortedp=np.take_along_axis(p,order,axis=-1); k=p.shape[-1]
    q=np.minimum.accumulate((sortedp*k/np.arange(1,k+1))[...,::-1],axis=-1)[...,::-1].clip(0,1)
    return np.take_along_axis(q,np.argsort(order,axis=-1),axis=-1)
def mcn_p(diff):
    n=np.sum(diff!=0,axis=0); z=np.abs(np.sum(diff,axis=0)); return np.minimum(1,2*stats.binom.cdf((n-z)/2,n,.5))
def states(p,g): return np.where(p<.05,np.sign(g),0).astype(int)
def metrics(raw,clean,ij):
    gr=gaps(raw,ij); gc=gaps(clean,ij); dg=gc-gr; rev=(gr*gc < -1e-10)
    out=dict(mean_abs_gap_pp=np.mean(abs(dg),axis=-1),median_abs_gap_pp=np.median(abs(dg),axis=-1),pairs_ge_1pp_rate=np.mean(abs(dg)>=1,axis=-1),pairs_ge_2pp_rate=np.mean(abs(dg)>=2,axis=-1),pairs_ge_5pp_rate=np.mean(abs(dg)>=5,axis=-1),reversals=np.sum(rev,axis=-1),practical_1pp_rate=np.mean(rev & (np.maximum(abs(gr),abs(gc))>=1),axis=-1),practical_2pp_rate=np.mean(rev & (np.maximum(abs(gr),abs(gc))>=2),axis=-1))
    # tau-b from pairwise signs, with actual score ties respected.
    a=np.sign(np.round(gr,10)); b=np.sign(np.round(gc,10)); denom=np.sqrt(np.sum(a*a,axis=-1)*np.sum(b*b,axis=-1)); out['kendall_tau']=np.sum(a*b,axis=-1)/denom
    r1=stats.rankdata(-raw,axis=-1); r2=stats.rankdata(-clean,axis=-1); r1=r1-r1.mean(axis=-1,keepdims=True); r2=r2-r2.mean(axis=-1,keepdims=True)
    out['spearman_rho']=np.sum(r1*r2,axis=-1)/np.sqrt(np.sum(r1*r1,axis=-1)*np.sum(r2*r2,axis=-1))
    for k in [3,5,10]:
        cut1=np.sort(raw,axis=-1)[...,-min(k,raw.shape[-1])]; cut2=np.sort(clean,axis=-1)[...,-min(k,clean.shape[-1])]
        t1=raw>=np.expand_dims(cut1,-1); t2=clean>=np.expand_dims(cut2,-1); out[f'top{k}_symmetric_difference']=np.sum(t1!=t2,axis=-1)
    return out
def bootstrap_scores(Y,bad,rng):
    n=len(Y); raw=[]; clean=[]
    for start in range(0,B,250):
        w=rng.multinomial(n,np.full(n,1/n),size=min(250,B-start)); raw.append(w@Y/n*100); wc=w*(~bad); clean.append(wc@Y/wc.sum(axis=1)[:,None]*100)
    return np.concatenate(raw),np.concatenate(clean)
def run(Y,d,names,maskname,bad,dataset='A'):
    rng=np.random.default_rng(SEED); n,m=Y.shape; ij=np.triu_indices(m,1); D=Y[:,ij[0]]-Y[:,ij[1]]; keep=~bad; nk=keep.sum()
    raw=Y.mean(0)*100; clean=Y[keep].mean(0)*100; gr=gaps(raw,ij); gc=gaps(clean,ij); dg=gc-gr
    br,bc=bootstrap_scores(Y,bad,rng); bgraw=gaps(br,ij); bgclean=gaps(bc,ij); bgdelta=bgclean-bgraw
    sr=[]; pr=[]; tr=[]; rr=[]
    lo,hi=ci(bc-br); rlo,rhi=ci(br); clo,chi=ci(bc); rel=(clean-raw)/raw*100; rell,relh=ci((bc-br)/np.where(br!=0,br,np.nan)*100)
    for j,a in enumerate(names): sr.append(dict(agent=a,raw_score_pp=raw[j],clean_score_pp=clean[j],delta_pp=clean[j]-raw[j],delta_ci_low=lo[j],delta_ci_high=hi[j],relative_change_percent=rel[j],relative_ci_low=rell[j],relative_ci_high=relh[j],raw_ci_low=rlo[j],raw_ci_high=rhi[j],clean_ci_low=clo[j],clean_ci_high=chi[j]))
    rp=mcn_p(D); cp=mcn_p(D[keep]); rq=bh(rp); cq=bh(cp); pooled=bh(np.r_[rp,cp]); rs=states(rq,gr); cs=states(cq,gc)
    gl,gh=ci(bgdelta); rgl,rgh=ci(bgraw); cgl,cgh=ci(bgclean)
    for k,(a,b) in enumerate(zip(*ij)):
        pr.append(dict(agent_a=names[a],agent_b=names[b],raw_gap_pp=gr[k],clean_gap_pp=gc[k],delta_gap_pp=dg[k],abs_delta_gap_pp=abs(dg[k]),delta_ci_low=gl[k],delta_ci_high=gh[k],reversal=gr[k]*gc[k]<-1e-10,practical_1pp=(gr[k]*gc[k]<-1e-10 and max(abs(gr[k]),abs(gc[k]))>=1),practical_2pp=(gr[k]*gc[k]<-1e-10 and max(abs(gr[k]),abs(gc[k]))>=2)))
        tr.append(dict(agent_a=names[a],agent_b=names[b],raw_p=rp[k],clean_p=cp[k],raw_q=rq[k],clean_q=cq[k],raw_state=int(rs[k]),clean_state=int(cs[k]),changed=rs[k]!=cs[k],raw_uncorrected_state=int(states(rp[k],gr[k])),clean_uncorrected_state=int(states(cp[k],gc[k])),raw_ci_low=rgl[k],raw_ci_high=rgh[k],clean_ci_low=cgl[k],clean_ci_high=cgh[k],raw_combined_state=int(rs[k]) if rgl[k]*rgh[k]>0 else 0,clean_combined_state=int(cs[k]) if cgl[k]*cgh[k]>0 else 0,raw_pooled_bh_state=int(states(pooled[k],gr[k])),clean_pooled_bh_state=int(states(pooled[k+len(rp)],gc[k]))))
    obs=metrics(raw,clean,ij); obs['statistical_change_rate']=np.mean(rs!=cs)
    obs['uncorrected_change_rate']=np.mean(states(rp,gr)!=states(cp,gc)); obs['pairs']=len(gr)
    mm=np.mean(abs(bgdelta),axis=1); med=np.median(abs(bgdelta),axis=1)
    obs.update(mean_abs_gap_ci_low=ci(mm)[0],mean_abs_gap_ci_high=ci(mm)[1],median_abs_gap_ci_low=ci(med)[0],median_abs_gap_ci_high=ci(med)[1])
    rank1=stats.rankdata(-raw); rank2=stats.rankdata(-clean)
    for j,a in enumerate(names): rr.append(dict(agent=a,raw_rank=rank1[j],clean_rank=rank2[j],rank_shift=rank2[j]-rank1[j],**obs))
    nullscores=[]; nullchanges=[]
    for start in range(0,B,200):
        z=min(200,B-start); w=np.zeros((z,n))
        for row in w: row[rng.choice(n,nk,replace=False)]=1
        sc=w@Y/nk*100; nullscores.append(sc)
        disc=w@(D!=0); sm=w@D; p=np.minimum(1,2*stats.binom.cdf((disc-abs(sm))/2,disc,.5)); st=states(bh(p),sm)
        nullchanges.append(np.mean(st!=rs,axis=1))
    ns=np.concatenate(nullscores); null=metrics(raw,ns,ij); null['statistical_change_rate']=np.concatenate(nullchanges)
    nd=pd.DataFrame(null); nd['iteration']=np.arange(B)
    for key in ['mean_abs_gap_pp','median_abs_gap_pp','pairs_ge_2pp_rate','practical_1pp_rate','practical_2pp_rate','reversals','statistical_change_rate','kendall_tau']:
        low=key=='kendall_tau'; obs[key+'_null_p']=(1+np.sum(nd[key]<=obs[key] if low else nd[key]>=obs[key]))/(B+1); obs[key+'_null95']=np.quantile(nd[key],.05 if low else .95)
    # Store per-agent random score changes alongside ranking/comparison nulls.
    for j,a in enumerate(names): nd['delta_pp__'+a]=ns[:,j]-raw[j]
    for k,r in enumerate(sr): r['random_two_sided_p']=(1+np.sum(abs(ns[:,k]-raw[k])>=abs(clean[k]-raw[k])))/(B+1)
    for frame in [sr,pr,tr,rr]:
        for row in frame: row.update(dataset=dataset,definition=maskname,tasks=n,removed=int(bad.sum()),retained=int(nk))
    nd['dataset']=dataset; nd['definition']=maskname
    obs.update(dataset=dataset,definition=maskname,tasks=n,agents=m,removed=int(bad.sum()))
    print(json.dumps(obs,default=float),flush=True)
    return sr,pr,tr,rr,nd,obs
def masks(d): return {'severity_ge2':d.severity.ge(2).to_numpy(),'severity_ge1':d.severity.ge(1).to_numpy(),'severity_eq3':d.severity.eq(3).to_numpy(),'all_defects':d.any_problem.to_numpy(),'problem_statement':d.problem_statement_problem.to_numpy(),'test_validity':d.test_validity_problem.to_numpy(),'other_problem':d.other_problem.to_numpy()}
def load():
    d=pd.read_parquet(O/'task_quality_labels.parquet'); a=pd.read_parquet(O/'agent_matrix.parquet'); am=pd.read_parquet(O/'submission_metadata.parquet'); names=sorted(am.loc[am.primary,'submission'])
    mat=a[a.dataset.eq('A') & a.submission.isin(names)].pivot(index='task_id',columns='submission',values='resolved').reindex(d.task_id)[names].astype(float)
    good=mat.notna().all(1).to_numpy(); return mat.to_numpy()[good],d.loc[good].reset_index(drop=True),names,mat,d
def pilot_available(mat,d,names):
    use=d.pilot_eligible.to_numpy(); z=mat.to_numpy()[use]; q=d.loc[use].severity.ge(2).to_numpy(); rows=[]
    for i,j in itertools.combinations(range(len(names)),2):
        ok=np.isfinite(z[:,i]) & np.isfinite(z[:,j]); raw=np.mean(z[ok,i]-z[ok,j])*100; clean=np.mean((z[:,i]-z[:,j])[ok & ~q])*100
        rows.append(dict(agent_a=names[i],agent_b=names[j],tasks=ok.sum(),raw_gap_pp=raw,clean_gap_pp=clean,delta_gap_pp=clean-raw))
    pd.DataFrame(rows).to_parquet(O/'pilot_all1693_pairwise.parquet',index=False)
def main():
    Y,d,names,mat,all_d=load(); pilot_available(mat,all_d,names)
    out=[[],[],[],[],[],[]]
    for name,bad in masks(d).items():
        v=run(Y,d,names,name,bad)
        for i in range(4): out[i].extend(v[i])
        out[4].append(v[4]); out[5].append(v[5])
        for filename,rows in zip(['score_distortion','pairwise_distortion','statistical_transitions','ranking_stability'],out[:4]): pd.DataFrame(rows).to_parquet(O/(filename+'.parquet'),index=False)
        pd.concat(out[4],ignore_index=True).to_parquet(O/'random_removal.parquet',index=False); js('core_summary.json',out[5])
        if name=='severity_ge2': js('fast_empirical_gate_core.json',v[5])
if __name__=='__main__': main()
