from analyze import *
def influence(Y,d,names):
    ij=np.triu_indices(Y.shape[1],1); D=Y[:,ij[0]]-Y[:,ij[1]]; g=D.mean(0)
    v=abs(D-g).mean(1)*100/(len(Y)-1); x=d[['task_id','repo','human_difficulty','severity','problem_statement_problem','test_validity_problem','other_problem','any_problem']].copy(); x['influence_pp']=v
    x['maximum_gap_change_pp']=abs(D-g).max(1)*100/(len(Y)-1); x.to_parquet(O/'task_influence.parquet',index=False)
    rng=np.random.default_rng(SEED); rows=[]
    for name,bad in masks(d).items():
        a=v[bad]; b=v[~bad]; u=stats.mannwhitneyu(a,b,alternative='two-sided'); delta=2*u.statistic/(len(a)*len(b))-1
        boot=[]
        for j in range(B):
            ab=rng.choice(a,len(a)); bb=rng.choice(b,len(b)); boot.append((ab.mean()-bb.mean(),2*stats.mannwhitneyu(ab,bb).statistic/(len(ab)*len(bb))-1))
        boot=np.asarray(boot); lo,hi=ci(boot)
        rows.append(dict(definition=name,defect_median=np.median(a),defect_q25=np.quantile(a,.25),defect_q75=np.quantile(a,.75),clean_median=np.median(b),clean_q25=np.quantile(b,.25),clean_q75=np.quantile(b,.75),cliffs_delta=delta,cliffs_ci_low=lo[1],cliffs_ci_high=hi[1],mean_difference=a.mean()-b.mean(),mean_ci_low=lo[0],mean_ci_high=hi[0],mw_p=u.pvalue))
    out=pd.DataFrame(rows); out['mw_q']=bh(out.mw_p); out.to_parquet(O/'influence_summary.parquet',index=False); return v
def matched_indices(d,bad,rng,nearest=False):
    left=[]; right=[]; strata=[]
    feats=np.log1p(d[['problem_length','patch_lines','f2p_count']].to_numpy(float)); feats=(feats-np.nanmean(feats,0))/np.nanstd(feats,0); feats=np.nan_to_num(feats)
    for key,idx in d.groupby(['repo','human_difficulty']).indices.items():
        a=idx[bad[idx]]; b=idx[~bad[idx]]; k=min(len(a),len(b))
        if not k: continue
        if nearest:
            dist=((feats[a,None,:]-feats[None,b,:])**2).sum(2); ia,ib=linear_sum_assignment(dist); a=a[ia]; b=b[ib]
        else: a=rng.choice(a,k,False); b=rng.choice(b,k,False)
        left.extend(a); right.extend(b); strata.extend([str(key)]*k)
    return np.array(left),np.array(right),strata
def adjusted(Y,d,bad):
    raw=np.zeros(Y.shape[1]); clean=raw.copy(); ids=[]; w=[]; total=0
    for _,idx in d.groupby(['repo','human_difficulty']).indices.items():
        c=idx[~bad[idx]]
        if len(c)==0 or bad[idx].sum()==0: continue
        raw+=Y[idx].sum(0); clean+=Y[c].mean(0)*len(idx); total+=len(idx); ids.extend(c); w.extend([len(idx)/len(c)]*len(c))
    return raw/total*100,clean/total*100,total,np.array(ids),np.array(w)
def main():
    Y,d,names,mat,ad=load(); rng=np.random.default_rng(SEED); ij=np.triu_indices(len(names),1); bad=d.severity.ge(2).to_numpy(); rows=[]; summaries=[]
    infl=abs((Y[:,ij[0]]-Y[:,ij[1]])-gaps(Y.mean(0),ij)).mean(1)*100/(len(Y)-1)
    for nearest in [False,True]:
        a,b,strata=matched_indices(d,bad,rng,nearest); k=len(a); qa=Y[a]; qb=Y[b]; raw=(qa.mean(0)+qb.mean(0))*50; quality=qb.mean(0)*100; cleanremove=qa.mean(0)*100
        dgq=gaps(quality-raw,ij); dgc=gaps(cleanremove-raw,ij); contrast=gaps(quality-cleanremove,ij)
        boots=[]; perm=[]; iboot=[]; iperm=[]
        # Pair bootstrap and within-pair swaps preserve exact repo/difficulty.
        for start in range(0,B,250):
            z=min(250,B-start); w=rng.multinomial(k,np.full(k,1/k),size=z); diff=(qb-qa)
            bs=w@diff/k*100; boots.append(gaps(bs,ij)); sign=rng.choice([-1,1],size=(z,k)); pm=sign@diff/k*100; perm.append(np.mean(abs(gaps(pm,ij)),axis=1))
            iv=infl[a]-infl[b]; iboot.extend(w@iv/k); iperm.extend(sign@iv/k)
        boots=np.concatenate(boots); perm=np.concatenate(perm); lo,hi=ci(boots); agg=np.mean(abs(contrast)); mag=np.mean(abs(boots),axis=1)
        method='exact_repo_difficulty_nearest_metadata' if nearest else 'exact_repo_difficulty'
        for t,(i,j) in enumerate(zip(*ij)): rows.append(dict(method=method,agent_a=names[i],agent_b=names[j],matched_pairs=k,removed_each=k,quality_gap_change_pp=dgq[t],clean_removal_gap_change_pp=dgc[t],differential_contrast_pp=contrast[t],contrast_ci_low=lo[t],contrast_ci_high=hi[t]))
        u=stats.mannwhitneyu(infl[a],infl[b]); cliff=2*u.statistic/k**2-1
        summary=dict(method=method,matched_pairs=k,defect_coverage=k/bad.sum(),clean_coverage=k/(~bad).sum(),mean_abs_contrast_pp=agg,mean_abs_contrast_ci_low=ci(mag)[0],mean_abs_contrast_ci_high=ci(mag)[1],mean_abs_quality_gap_shift_pp=np.mean(abs(dgq)),pairs_quality_shift_ge2_rate=np.mean(abs(dgq)>=2),permutation_p=(1+np.sum(perm>=agg))/(B+1),matched_influence_difference=np.mean(iv),influence_ci_low=ci(iboot)[0],influence_ci_high=ci(iboot)[1],influence_permutation_p=(1+np.sum(np.asarray(iperm)>=np.mean(iv)))/(B+1),influence_cliffs_delta=cliff)
        summaries.append(summary); print(json.dumps(summary,default=float),flush=True)
        pd.DataFrame({'defect_task':d.task_id.iloc[a].to_numpy(),'clean_task':d.task_id.iloc[b].to_numpy(),'stratum':strata}).to_parquet(O/(method+'_matches.parquet'),index=False)
    pd.DataFrame(rows).to_parquet(O/'matched_analysis.parquet',index=False); js('matched_summary.json',summaries)
    raw,clean,total,ids,w=adjusted(Y,d,bad); delta=clean-raw
    br=[]; bc=[]
    for _ in range(B):
        # Resample each repo/difficulty and defect-status cell, preserving weights and support.
        ar=np.zeros(len(names)); ac=ar.copy(); n=0
        for _,ix in d.groupby(['repo','human_difficulty']).indices.items():
            g=ix[~bad[ix]]; f=ix[bad[ix]]
            if not len(g) or not len(f): continue
            cg=Y[rng.choice(g,len(g))].mean(0); fg=Y[rng.choice(f,len(f))].mean(0)
            ar+=cg*len(g)+fg*len(f); ac+=cg*len(ix); n+=len(ix)
        br.append(ar/n*100); bc.append(ac/n*100)
    br=np.array(br); bc=np.array(bc); gl,gh=ci(gaps(bc-br,ij)); out=[]
    for t,(i,j) in enumerate(zip(*ij)): out.append(dict(agent_a=names[i],agent_b=names[j],raw_gap_pp=raw[i]-raw[j],weighted_clean_gap_pp=clean[i]-clean[j],adjusted_delta_gap_pp=delta[i]-delta[j],ci_low=gl[t],ci_high=gh[t],support_tasks=total,support_mass=total/len(d),clean_effective_sample_size=w.sum()**2/np.sum(w*w)))
    pd.DataFrame(out).to_parquet(O/'reweighted_analysis.parquet',index=False)
    summary=dict(support_tasks=total,support_mass=total/len(d),clean_ess=w.sum()**2/np.sum(w*w),**metrics(raw,clean,ij)); js('reweighted_summary.json',summary); print(json.dumps(summary,default=float),flush=True)
    # Repository and agent exclusions, without changing labels or adjustment method.
    robust=[]
    for kind,values in [('repository',sorted(d.repo.unique())),('agent',names)]:
        for value in values:
            iy=d.repo.ne(value).to_numpy() if kind=='repository' else np.ones(len(d),bool); cols=[i for i,a in enumerate(names) if kind!='agent' or a!=value]
            yy=Y[iy][:,cols]; dd=d[iy].reset_index(drop=True); bb=dd.severity.ge(2).to_numpy(); ra,cl,nt,_,_=adjusted(yy,dd,bb); jj=np.triu_indices(len(cols),1)
            robust.append(dict(exclusion_type=kind,excluded=value,support_tasks=nt,**metrics(ra,cl,jj)))
    pd.DataFrame(robust).to_parquet(O/'leave_one_out.parquet',index=False)
    # Missingness/expanded configurations and the original 1693-task pilot.
    sens=[]; allm=pd.read_parquet(O/'agent_matrix.parquet'); am=pd.read_parquet(O/'submission_metadata.parquet')
    for label,subs,fill in [('primary_all1693_missing_failure',names,True),('expanded16_complete',sorted(am.loc[am.included,'submission']),False)]:
        mm=allm[allm.dataset.eq('A') & allm.submission.isin(subs)].pivot(index='task_id',columns='submission',values='resolved').reindex(ad.task_id)[subs].astype(float)
        valid=ad.pilot_eligible.to_numpy() if fill else mm.notna().all(axis=1).to_numpy(); yy=mm.to_numpy()[valid]; yy=np.nan_to_num(yy) if fill else yy; bb=ad.loc[valid].severity.ge(2).to_numpy()
        sens.append(dict(analysis=label,tasks=len(yy),agents=len(subs),**metrics(yy.mean(0)*100,yy[~bb].mean(0)*100,np.triu_indices(len(subs),1))))
    js('sensitivity_summary.json',sens)
    influence(Y,d,names)
if __name__=='__main__': main()
