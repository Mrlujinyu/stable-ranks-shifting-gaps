from analyze import *
import statsmodels.api as sm
import statsmodels.formula.api as smf
def main():
    Y,d,names,_,_=load(); n,m=Y.shape
    long=pd.DataFrame({'success':Y.ravel(),'agent':np.tile([f'A{i:02}' for i in range(m)],n),'task':np.repeat(np.arange(n),m),'repo':np.repeat(d.repo,m),'difficulty':np.repeat(d.human_difficulty,m),'defect':np.repeat(d.severity.ge(2).astype(int),m)})
    rows=[]
    scenarios=[('none','none')]+[('repo',x) for x in sorted(d.repo.unique())]+[('agent',f'A{i:02}') for i in range(m)]
    for kind,value in scenarios:
        z=long if kind=='none' else long[long[kind].ne(value)]
        # Stable parsimonious nuisance specification avoids agent-by-small-repository separation.
        model=smf.gee('success ~ C(agent)*(defect+C(difficulty)) + C(repo)',groups='task',data=z,family=sm.families.Binomial()).fit(maxiter=100)
        idx=[i for i,k in enumerate(model.params.index) if k.endswith(':defect')]; C=np.eye(len(model.params))[idx]; test=model.wald_test(C,scalar=True)
        rows.append(dict(exclusion_type=kind,excluded=value,wald=float(test.statistic),df=len(idx),p=float(test.pvalue),converged=model.converged,finite_covariance=bool(np.isfinite(model.cov_params()).all().all()),max_abs_coefficient=float(abs(model.params).max())))
        pd.DataFrame(rows).to_parquet(O/'interaction_robustness.parquet',index=False); print(json.dumps(rows[-1],default=lambda x:x.item()),flush=True)
    js('interaction_robustness.json',rows)
if __name__=='__main__': main()
