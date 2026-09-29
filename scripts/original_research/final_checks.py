from analyze import *
from statsmodels.stats.contingency_tables import mcnemar
import hashlib
def main():
    Y,d,names,mat,all_d=load(); bad=d.severity.ge(2).to_numpy(); ij=np.triu_indices(len(names),1); rng=np.random.default_rng(SEED)
    out=[]; summaries=[]
    for method in ['exact_repo_difficulty','exact_repo_difficulty_nearest_metadata']:
        pairs=pd.read_parquet(O/(method+'_matches.parquet')); idx=pd.Series(np.arange(len(d)),index=d.task_id); a=idx[pairs.defect_task].to_numpy(); b=idx[pairs.clean_task].to_numpy(); k=len(a); n=len(d)
        raw=Y.mean(0)*100; q=(Y.sum(0)-Y[a].sum(0))/(n-k)*100; c=(Y.sum(0)-Y[b].sum(0))/(n-k)*100
        quality=gaps(q-raw,ij); clean=gaps(c-raw,ij); contrast=quality-clean; bs=[]; null=[]
        for start in range(0,B,250):
            z=min(250,B-start); w=rng.multinomial(k,np.full(k,1/k),size=z); delta=Y[b]-Y[a]; bs.append(gaps(w@delta/(n-k)*100,ij)); sign=rng.choice([-1,1],size=(z,k)); null.extend(np.mean(abs(gaps(sign@delta/(n-k)*100,ij)),axis=1))
        bs=np.concatenate(bs); lo,hi=ci(bs); meanboot=np.mean(abs(bs),axis=1); observed=np.mean(abs(contrast))
        for t,(i,j) in enumerate(zip(*ij)): out.append(dict(method=method,agent_a=names[i],agent_b=names[j],removed_each=k,retained=n-k,quality_gap_change_pp=quality[t],matched_clean_gap_change_pp=clean[t],contrast_pp=contrast[t],contrast_ci_low=lo[t],contrast_ci_high=hi[t]))
        summaries.append(dict(method=method,removed_each=k,retained=n-k,mean_abs_quality_delta_gap_pp=np.mean(abs(quality)),mean_abs_clean_removal_delta_gap_pp=np.mean(abs(clean)),mean_abs_contrast_pp=observed,contrast_ci_low=ci(meanboot)[0],contrast_ci_high=ci(meanboot)[1],permutation_p=(1+np.sum(np.array(null)>=observed))/(B+1),contrast_direction_agreement=np.mean(np.sign(contrast)==np.sign(gaps(Y[~bad].mean(0)-Y.mean(0),ij)))))
    pd.DataFrame(out).to_parquet(O/'matched_full_reference_analysis.parquet',index=False); js('matched_full_reference_summary.json',summaries)
    # Independent finite-sample checks of stored primary metrics.
    p=pd.read_parquet(O/'pairwise_distortion.parquet'); p=p[p.definition.eq('severity_ge2')]; t=pd.read_parquet(O/'statistical_transitions.parquet'); t=t[t.definition.eq('severity_ge2')]; score=pd.read_parquet(O/'score_distortion.parquet'); score=score[score.definition.eq('severity_ge2')]
    checks={}
    np.testing.assert_allclose(p.delta_gap_pp,gaps(Y[~bad].mean(0)-Y.mean(0),ij)*100,atol=1e-12); checks['pairwise_equations']=True
    for j,(a,b) in enumerate(zip(*ij)):
        table=pd.crosstab(pd.Series(Y[:,a]),pd.Series(Y[:,b])).reindex(index=[0.,1.],columns=[0.,1.],fill_value=0).to_numpy(); actual=mcnemar(table,exact=True).pvalue
        np.testing.assert_allclose(t.raw_p.iloc[j],actual,atol=1e-12)
    checks['mcnemar_all91_independent_library']=True
    influence=pd.read_parquet(O/'task_influence.parquet').influence_pp.to_numpy()
    for k in [0,100,700,len(Y)-1]:
        direct=gaps(np.delete(Y,k,axis=0).mean(0),ij)-gaps(Y.mean(0),ij); np.testing.assert_allclose(np.mean(abs(direct))*100,influence[k],atol=1e-12)
    checks['influence_direct_deletion']=True
    summary=json.loads((O/'core_summary.json').read_text())[0]; assert abs(stats.kendalltau(Y.mean(0),Y[~bad].mean(0)).statistic-summary['kendall_tau'])<1e-10; checks['tau_scipy']=True
    old=R.parent/'benchmark-audit-triage'; hashes=json.loads((O/'input_checksums.json').read_text()); checks['old_inputs_unchanged']=all(hashlib.sha256((old/f).read_bytes()).hexdigest()==h for f,h in hashes.items()); assert checks['old_inputs_unchanged']
    # Same harness/model exploratory contrasts and newest-eight sensitivity, no mechanistic inference.
    ids=sorted(range(len(names)),key=lambda j:names[j],reverse=True)[:8]; yy=Y[:,ids]; checks['newest_eight_summary']=metrics(yy.mean(0)*100,yy[~bad].mean(0)*100,np.triu_indices(8,1))
    prof=pd.read_parquet(O/'agent_sensitivity_profiles.parquet'); am=pd.read_parquet(O/'submission_metadata.parquet'); prof=prof.merge(am[['submission','model','scaffold']],left_on='agent',right_on='submission',suffixes=('_estimator','_underlying')); prof.to_parquet(O/'family_exploratory.parquet',index=False)
    checks['disk_bytes']=sum(p.stat().st_size for p in R.rglob('*') if p.is_file()); checks['resource_limits']=checks['disk_bytes']<5_000_000_000
    js('verification.json',checks); print(json.dumps(checks,default=lambda x:x.item()),flush=True)
if __name__=='__main__': main()
