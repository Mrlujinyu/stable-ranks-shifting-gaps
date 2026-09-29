from analyze import *
import statsmodels.api as sm
import statsmodels.formula.api as smf
def main():
    _,_,names,mat,d=load(); use=d.pilot_eligible.to_numpy(); Y=mat.to_numpy()[use]; d=d.loc[use].reset_index(drop=True); n,m=Y.shape; bad=d.severity.ge(2).to_numpy(); ij=np.triu_indices(m,1)
    ok=np.isfinite(Y[:,ij[0]])&np.isfinite(Y[:,ij[1]]); D=np.nan_to_num(Y[:,ij[0]]-Y[:,ij[1]])*ok; keep=~bad
    gr=D.sum(0)/ok.sum(0)*100; gc=D[keep].sum(0)/ok[keep].sum(0)*100; dg=gc-gr; rng=np.random.default_rng(SEED)
    null=[]; boot=[]; bootraw=[]; bootclean=[]
    for start in range(0,B,200):
        z=min(200,B-start); w=np.zeros((z,n))
        for row in w: row[rng.choice(n,keep.sum(),False)]=1
        gn=w@D/(w@ok)*100; dd=gn-gr; rv=gn*gr < -1e-10
        null.append(pd.DataFrame({'median_abs_gap_pp':np.median(abs(dd),1),'mean_abs_gap_pp':np.mean(abs(dd),1),'pairs_ge2_rate':np.mean(abs(dd)>=2,1),'practical1_rate':np.mean(rv & (np.maximum(abs(gn),abs(gr))>=1),1)}))
        w=rng.multinomial(n,np.full(n,1/n),size=z); wc=w*keep
        rb=w@D/(w@ok)*100; cb=wc@D/(wc@ok)*100; bootraw.append(rb); bootclean.append(cb); boot.append(cb-rb)
    null=pd.concat(null,ignore_index=True); boot=np.concatenate(boot); lo,hi=ci(boot); br=np.concatenate(bootraw); bc=np.concatenate(bootclean)
    summary=dict(tasks=n,agents=m,repositories=int(d.repo.nunique()),pair_task_min=int(ok.sum(0).min()),pair_task_max=int(ok.sum(0).max()),median_abs_gap_pp=np.median(abs(dg)),median_ci_low=ci(np.median(abs(boot),1))[0],median_ci_high=ci(np.median(abs(boot),1))[1],mean_abs_gap_pp=np.mean(abs(dg)),pairs_ge2_rate=np.mean(abs(dg)>=2),practical1_rate=np.mean((gr*gc < -1e-10)&(np.maximum(abs(gr),abs(gc))>=1)))
    for key in ['median_abs_gap_pp','pairs_ge2_rate','practical1_rate']: summary[key+'_null_p']=(1+np.sum(null[key]>=summary[key]))/(B+1)
    rows=[]
    for k,(i,j) in enumerate(zip(*ij)): rows.append(dict(agent_a=names[i],agent_b=names[j],tasks=int(ok[:,k].sum()),raw_gap_pp=gr[k],clean_gap_pp=gc[k],delta_gap_pp=dg[k],ci_low=lo[k],ci_high=hi[k]))
    pd.DataFrame(rows).to_parquet(O/'pilot_all1693_pairwise.parquet',index=False); null.to_parquet(O/'pilot_all1693_random_removal.parquet',index=False)
    long=pd.DataFrame({'success':Y.ravel(),'agent':np.tile([f'A{i:02}' for i in range(m)],n),'task':np.repeat(np.arange(n),m),'repo':np.repeat(d.repo,m),'difficulty':np.repeat(d.human_difficulty,m),'defect':np.repeat(bad.astype(int),m),'statement':np.repeat(d.problem_statement_problem.astype(int),m),'test':np.repeat(d.test_validity_problem.astype(int),m),'other':np.repeat(d.other_problem.astype(int),m)}).dropna(subset=['success'])
    tests=[]
    for label,terms in [('primary',['defect']),('types',['statement','test','other'])]:
        fit=smf.gee('success ~ C(agent)*('+'+'.join(terms)+'+C(difficulty)) + C(repo)',groups='task',data=long,family=sm.families.Binomial()).fit(maxiter=100)
        for term in terms:
            ix=[i for i,k in enumerate(fit.params.index) if k.endswith(':'+term)]; C=np.eye(len(fit.params))[ix]; t=fit.wald_test(C,scalar=True); tests.append(dict(defect=term,wald=float(t.statistic),df=len(ix),p=float(t.pvalue),converged=bool(fit.converged)))
    ts=pd.DataFrame(tests); ts['q']=bh(ts.p); ts.to_parquet(O/'pilot_all1693_interactions.parquet',index=False); summary['interactions']=ts.to_dict('records'); js('pilot_all1693_summary.json',summary); print(json.dumps(summary,default=lambda x:x.item()),flush=True)
if __name__=='__main__': main()
