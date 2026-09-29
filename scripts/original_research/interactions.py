from analyze import *
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
import patsy
def main():
    Y,d,names,_,_=load(); n,m=Y.shape
    long=pd.DataFrame({'success':Y.ravel(),'agent':np.tile([f'A{i:02}' for i in range(m)],n),'task':np.repeat(np.arange(n),m),'repo':np.repeat(d.repo,m),'difficulty':np.repeat(d.human_difficulty,m),'statement':np.repeat(d.problem_statement_problem.astype(int),m),'test':np.repeat(d.test_validity_problem.astype(int),m),'other':np.repeat(d.other_problem.astype(int),m),'high':np.repeat(d.severity.eq(3).astype(int),m),'defect':np.repeat(d.severity.ge(2).astype(int),m)})
    rows=[]; summaries=[]; profiles=[]
    for version in ['types','primary','high']:
        terms=['statement','test','other'] if version=='types' else (['defect'] if version=='primary' else ['high'])
        formula='success ~ C(agent)*('+'+'.join(terms)+') + C(difficulty)'
        mod=BinomialBayesMixedGLM.from_formula(formula,{'repository':'0+C(repo)'},long,vcp_p=.5,fe_p=2)
        fit=mod.fit_vb(minim_opts={'maxiter':500},scale_fe=False)
        converged=bool(fit.optim_retvals['success']); print(version,'mixed convergence',converged,flush=True)
        for name,mean,sd in zip(mod.exog_names,fit.fe_mean,fit.fe_sd): rows.append(dict(model='mixed_logistic_'+version,term=name,estimate=mean,se=sd,interval_low=mean-1.96*sd,interval_high=mean+1.96*sd,interval_type='VB_approximate_credible',converged=converged,p=np.nan,q=np.nan))
        rows.append(dict(model='mixed_logistic_'+version,term='repository_log_sd',estimate=fit.vcp_mean[0],se=fit.vcp_sd[0],interval_low=fit.vcp_mean[0]-1.96*fit.vcp_sd[0],interval_high=fit.vcp_mean[0]+1.96*fit.vcp_sd[0],interval_type='VB_approximate_credible',converged=converged,p=np.nan,q=np.nan))
        # Agent-specific difficulty and repository main effects; the earlier saturated
        # agent-by-repository diagnostic had separated nuisance cells and is not used.
        gf='success ~ C(agent)*('+'+'.join(terms)+'+C(difficulty)) + C(repo)'
        gee=smf.gee(gf,groups='task',data=long,family=sm.families.Binomial(),cov_struct=sm.cov_struct.Independence()).fit(maxiter=150)
        ci95=gee.conf_int(); interaction=[k for k in gee.params.index if ':' in k and any(k.endswith(':'+t) for t in terms)]; q=bh(gee.pvalues[interaction].to_numpy())
        for name,mean in gee.params.items(): rows.append(dict(model='GEE_adjusted_'+version,term=name,estimate=mean,se=gee.bse[name],interval_low=ci95.loc[name,0],interval_high=ci95.loc[name,1],interval_type='task_cluster_robust_95CI',converged=gee.converged,p=gee.pvalues[name],q=q[interaction.index(name)] if name in interaction else np.nan))
        for t in terms:
            idx=[i for i,k in enumerate(gee.params.index) if ':' in k and k.endswith(':'+t)]; C=np.eye(len(gee.params))[idx]; test=gee.wald_test(C,scalar=True)
            summaries.append(dict(model=version,defect=t,interaction_wald=float(test.statistic),df=len(idx),p=float(test.pvalue),converged=gee.converged))
            for ai,agent in enumerate(names):
                # Common task covariate distribution; toggle defect while retaining other labels.
                z=long.iloc[::m].copy(); z['agent']=f'A{ai:02}'; a=z.copy(); b=z.copy(); a[t]=1; b[t]=0
                xa=np.asarray(patsy.build_design_matrices([gee.model.data.design_info],a)[0]); xb=np.asarray(patsy.build_design_matrices([gee.model.data.design_info],b)[0])
                pa=1/(1+np.exp(-xa@gee.params)); pb=1/(1+np.exp(-xb@gee.params)); effect=(pa-pb).mean()*100
                gradient=((pa*(1-pa))[:,None]*xa-(pb*(1-pb))[:,None]*xb).mean(0)*100; se=np.sqrt(max(0,gradient@gee.cov_params().to_numpy()@gradient))
                term=t if ai==0 else f'C(agent)[T.A{ai:02}]:{t}'; logor=gee.params[t]+(gee.params.get(term,0) if ai else 0)
                profiles.append(dict(agent=agent,defect=t,adjusted_probability_delta_pp=effect,ci_low=effect-1.96*se,ci_high=effect+1.96*se,conditional_odds_ratio=np.exp(logor),model='GEE_adjusted_'+version))
        pd.DataFrame(rows).to_parquet(O/'interaction_models.parquet',index=False); pd.DataFrame(profiles).to_parquet(O/'agent_sensitivity_profiles.parquet',index=False)
        print(json.dumps(summaries,default=float),flush=True)
    sd=pd.DataFrame(summaries); sd['q']=bh(sd.p); sd.to_parquet(O/'interaction_summary.parquet',index=False)
    js('interaction_summary.json',sd.to_dict('records'))
if __name__=='__main__': main()
