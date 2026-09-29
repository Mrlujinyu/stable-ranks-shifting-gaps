from analyze import *
import re,hashlib
def norm(x): return re.sub(r'-v(?:\d+|[0-9a-f]{6,}|nan)$','',re.sub('^instance_','',x),flags=re.I)
def main():
    # Main method and result hashes are frozen BEFORE this script examines Pro outcomes.
    frozen={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'reports/protocol.md',O/'core_summary.json',O/'matched_summary.json',O/'reweighted_summary.json',O/'interaction_summary.json',R/'scripts/analyze.py']}
    js('main_freeze_before_pro.json',frozen)
    x=pd.read_csv(R/'sources/pro_original_rows.csv'); x['task_id']=x['metadata.instance_id'].map(norm); x['submission']=x['metadata.model_name']; x['resolved']=x['metadata.resolved'].astype(int)
    assert not x.duplicated(['task_id','submission']).any()
    lab=pd.read_parquet(R.parent/'benchmark-audit-triage/results/dataset_b_labels.parquet').rename(columns={'task_id':'verified_task_id','canonical_id':'task_id'})
    lab['status']=np.where(lab.corrected,'known-corrected','no-known-correction')
    lab.to_parquet(O/'pro_task_quality_labels.parquet',index=False)
    # Outcome-blind coverage criterion and ambiguous same-model run exclusions.
    manifest=x.groupby('submission').agg(observed_tasks=('task_id','nunique'),source_timestamp=('created_at','min')).reset_index()
    manifest['exclusion']=''; manifest.loc[manifest.submission.eq('Claude 4 Sonnet - 10132025'),'exclusion']='Same-model configuration provenance unclear; retain paper entry with higher coverage'
    manifest.loc[manifest.submission.eq('Gemini 2.5 Pro Preview -- debug-oct22'),'exclusion']='Same-preview model/config identity unclear; retain paper entry'
    manifest['included']=manifest.exclusion.eq('') & manifest.observed_tasks.ge(700)
    manifest['harness']='Not recorded in source CSV'; manifest['model']=manifest.submission
    names=sorted(manifest.loc[manifest.included,'submission']); mat=x[x.submission.isin(names)].pivot(index='task_id',columns='submission',values='resolved').reindex(lab.task_id)[names]
    valid=mat.notna().all(1).to_numpy(); Y=mat.to_numpy()[valid]; d=lab.loc[valid].reset_index(drop=True); bad=d.corrected.to_numpy(); d['status']=np.where(bad,'known-corrected','no-known-correction')
    manifest.to_parquet(O/'pro_submission_metadata.parquet',index=False); x[['task_id','submission','resolved','created_at']].to_parquet(O/'pro_observed_matrix.parquet',index=False); d.to_parquet(O/'pro_labels_analysis.parquet',index=False)
    old=pd.read_parquet(O/'agent_matrix.parquet'); old=old[old.dataset.eq('B')]
    merged=old.merge(x[['task_id','submission','resolved']],on=['task_id','submission'],how='left',suffixes=('_old','_observed'))
    audit=dict(archived_matrix_cells=len(old),observed_source_rows=len(x),zero_filled_missing_cells=int(merged.resolved_observed.isna().sum()),observed_disagreements=int((merged.resolved_old.ne(merged.resolved_observed)&merged.resolved_observed.notna()).sum()),tasks=len(Y),agents=len(names),known_corrected=int(bad.sum()),repositories=int(d.repo.nunique()),numeric_gate=len(Y)>=500 and len(names)>=8,provenance='partial: model labels and source rows audited; harness/settings not independently documented',independent_configurations='9 distinct named model/reasoning configurations; ambiguous same-model runs excluded',not_a_corrected_rerun=True)
    js('pro_data_gate.json',audit)
    sr,pr,tr,rr,nd,summary=run(Y,d,names,'known_corrected',bad,dataset='Pro')
    for filename,rows in [('score_distortion',sr),('pairwise_distortion',pr),('statistical_transitions',tr),('ranking_stability',rr)]: pd.DataFrame(rows).to_parquet(O/('pro_'+filename+'.parquet'),index=False)
    nd.to_parquet(O/'pro_random_removal.parquet',index=False); js('pro_summary.json',summary)
    ij=np.triu_indices(len(names),1); DD=Y[:,ij[0]]-Y[:,ij[1]]; v=abs(DD-DD.mean(0)).mean(1)*100/(len(Y)-1)
    d['influence_pp']=v; d.to_parquet(O/'pro_task_influence.parquet',index=False); a=v[bad]; b=v[~bad]; u=stats.mannwhitneyu(a,b); rng=np.random.default_rng(SEED); boot=[]
    for _ in range(B): boot.append(rng.choice(a,len(a)).mean()-rng.choice(b,len(b)).mean())
    js('pro_influence_summary.json',dict(known_corrected_median=np.median(a),known_corrected_iqr=np.quantile(a,[.25,.75]).tolist(),no_known_correction_median=np.median(b),no_known_correction_iqr=np.quantile(b,[.25,.75]).tolist(),cliffs_delta=2*u.statistic/(len(a)*len(b))-1,mw_p=u.pvalue,mean_difference=a.mean()-b.mean(),difference_ci_low=ci(boot)[0],difference_ci_high=ci(boot)[1]))
if __name__=='__main__': main()
