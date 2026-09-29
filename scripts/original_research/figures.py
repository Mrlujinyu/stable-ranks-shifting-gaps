from analyze import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
F=R/'figures'; F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
BLUE='#245B85'; ORANGE='#B96527'; GREY='#6C7278'
def save(fig,name):
    for ext in ['pdf','svg','png']: fig.savefig(F/(name+'.'+ext),bbox_inches='tight')
    plt.close(fig)
def main():
    s=pd.read_parquet(O/'score_distortion.parquet'); s=s[s.definition.eq('severity_ge2')].sort_values('raw_score_pp').reset_index(drop=True)
    names=s.agent.tolist(); codes={a:f'A{i+1:02}' for i,a in enumerate(names)}; pd.DataFrame({'code':list(codes.values()),'submission':names}).to_csv(F/'agent_key.csv',index=False)
    ys=np.arange(len(s)); fig,axs=plt.subplots(1,2,figsize=(10,5.5),gridspec_kw={'width_ratios':[1.4,1]},layout='constrained')
    ax=axs[0]
    for i,row in s.iterrows(): ax.plot([row.raw_score_pp,row.clean_score_pp],[i,i],color='#BBC2C9',lw=1.7)
    ax.scatter(s.raw_score_pp,ys,color=BLUE,marker='o',label='Raw'); ax.scatter(s.clean_score_pp,ys,color=ORANGE,marker='s',label='Filtered')
    ax.set(yticks=ys,yticklabels=[codes[a] for a in names],xlabel='Resolution rate (%)',xlim=(0,90),title='(a) Raw and filtered scores'); ax.legend(frameon=False); ax.grid(axis='x',alpha=.2)
    ax=axs[1]; ax.errorbar(s.delta_pp,ys,xerr=[s.delta_pp-s.delta_ci_low,s.delta_ci_high-s.delta_pp],fmt='o',color=BLUE,capsize=2); ax.set(yticks=ys,yticklabels=[],xlabel='Filtered − raw (percentage points)',title='(b) Score shifts with 95% bootstrap CI'); ax.axvline(0,color=GREY,lw=1); ax.grid(axis='x',alpha=.2)
    save(fig,'figure1_scores')
    p=pd.read_parquet(O/'pairwise_distortion.parquet'); p=p[p.definition.eq('severity_ge2')]; z=np.zeros((len(names),len(names))); lookup={a:i for i,a in enumerate(names)}
    for row in p.itertuples(): z[lookup[row.agent_a],lookup[row.agent_b]]=row.delta_gap_pp; z[lookup[row.agent_b],lookup[row.agent_a]]=-row.delta_gap_pp
    fig,ax=plt.subplots(figsize=(7.6,6.5),layout='constrained'); limit=np.max(abs(z)); im=ax.imshow(z,cmap='PuOr_r',norm=TwoSlopeNorm(0,-limit,limit)); ax.set(xticks=ys,yticks=ys,xticklabels=list(codes.values()),yticklabels=list(codes.values()),xlabel='Agent B',ylabel='Agent A',title='Pairwise gap distortion: Δ(A − B), percentage points'); plt.setp(ax.get_xticklabels(),rotation=45,ha='right'); fig.colorbar(im,ax=ax,shrink=.75,label='Filtered gap − raw gap (pp)'); save(fig,'figure2_gap_heatmap')
    rank=pd.read_parquet(O/'ranking_stability.parquet'); rank=rank[rank.definition.eq('severity_ge2')].set_index('agent').loc[names]
    fig,ax=plt.subplots(figsize=(7,6),layout='constrained')
    for agent,row in rank.iterrows():
        color=ORANGE if row.raw_rank!=row.clean_rank else GREY
        ax.plot([0,1],[row.raw_rank,row.clean_rank],color=color,marker='o',lw=1.3,alpha=.85)
        ax.text(-.04,row.raw_rank,codes[agent],ha='right',va='center')
    for r,group in rank.groupby('clean_rank'): ax.text(1.04,r,', '.join(codes[a] for a in group.index),va='center')
    ax.set(xlim=(-.2,1.35),ylim=(14.7,.3),xticks=[0,1],xticklabels=['Raw','Filtered'],yticks=np.arange(1,15),ylabel='Rank (average ranks for ties)',title='Leaderboard order: τ = 0.972; top-3/5/10 unchanged'); ax.spines[['bottom','left']].set_visible(False); ax.tick_params(length=0); save(fig,'figure3_ranks')
    null=pd.read_parquet(O/'random_removal.parquet'); null=null[null.definition.eq('severity_ge2')]; obs=json.loads((O/'fast_empirical_gate_core.json').read_text()); fig,axs=plt.subplots(1,3,figsize=(11,3.8),layout='constrained')
    for ax,key,title,label in zip(axs,['median_abs_gap_pp','practical_1pp_rate','statistical_change_rate'],['(a) Pairwise gap distortion','(b) Practical reversals ≥1pp','(c) BH conclusion changes'],['Median |Δgap| (pp)','Fraction of 91 pairs','Fraction of 91 pairs']):
        ax.hist(null[key],bins=25,color=BLUE,alpha=.75,label='Random removal (10,000)'); ax.axvline(obs[key],color=ORANGE,lw=2,label='Quality removal'); ax.set(title=title,xlabel=label,ylabel='Random draws'); ax.ticklabel_format(axis='y',style='sci',scilimits=(3,3))
    axs[0].legend(frameon=False,fontsize=8); save(fig,'figure4_random_null')
    models=pd.read_parquet(O/'interaction_models.parquet'); canonical=sorted(names); fig,axs=plt.subplots(1,3,figsize=(10.5,5.5),sharey=True,layout='constrained'); data=[]
    for ax,term,model,title in zip(axs,['statement','test','high'],['types','types','high'],['Problem statement','Test validity','High severity (==3)']):
        coef=models[models.model.eq('mixed_logistic_'+model)].set_index('term'); estimates=[]; lows=[]; highs=[]
        for agent in names:
            ai=canonical.index(agent); mean=coef.loc[term,'estimate']; var=coef.loc[term,'se']**2
            if ai:
                c=coef.loc[f'C(agent)[T.A{ai:02}]:{term}']; mean+=c.estimate; var+=c.se**2
            estimates.append(np.exp(mean)); lows.append(np.exp(mean-1.96*np.sqrt(var))); highs.append(np.exp(mean+1.96*np.sqrt(var))); data.append(dict(agent=agent,defect=term,odds_ratio=np.exp(mean),credible_low=lows[-1],credible_high=highs[-1],interval='VB mean-field normal approximation'))
        estimates=np.array(estimates); ax.errorbar(estimates,ys,xerr=[estimates-np.array(lows),np.array(highs)-estimates],fmt='o',color=BLUE,capsize=2); ax.axvline(1,color=GREY,ls='--'); ax.set(xscale='log',xlim=(.18,1.5),xlabel='Defect odds ratio (log scale)',title=title,yticks=ys,yticklabels=[codes[a] for a in names]); ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([.2,.5,1.])); ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f')); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); ax.grid(axis='x',alpha=.2)
    save(fig,'figure5_sensitivity'); pd.DataFrame(data).to_parquet(F/'figure5_data.parquet',index=False)
    task=pd.read_parquet(O/'task_influence.parquet'); fig,axs=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
    for mask,label,color,style in [(task.severity.ge(2),'Problematic (severity ≥2)',ORANGE,'-'),(task.severity.lt(2),'Below threshold',BLUE,'--')]:
        vals=np.sort(task.loc[mask,'influence_pp']); axs[0].step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',label=label,color=color,ls=style)
    axs[0].set(xlabel='Task influence: mean absolute pairwise change (pp)',ylabel='Empirical cumulative probability',title='(a) Individual task influence'); axs[0].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5)); axs[0].legend(frameon=False,fontsize=8)
    match=pd.read_parquet(O/'exact_repo_difficulty_matches.parquet'); tv=task.set_index('task_id').influence_pp; delta=tv[match.defect_task].to_numpy()-tv[match.clean_task].to_numpy(); axs[1].hist(delta,bins=25,color=BLUE,alpha=.8); axs[1].axvline(0,color=GREY,ls='--'); axs[1].set(xlabel='Matched problematic − clean influence (pp)',ylabel='Matched task pairs',title='(b) Exact repository × difficulty matches'); save(fig,'figure6_influence')
    pro=json.loads((O/'pro_summary.json').read_text()); fig,axs=plt.subplots(1,3,figsize=(10,3.6),layout='constrained')
    vals=[obs,pro]; labels=['SWE-bench\n1,212 × 14','Pro (partial)\n692 × 9']
    for ax,key,title in zip(axs,['median_abs_gap_pp','pairs_ge_2pp_rate','statistical_change_rate'],['Median |Δgap| (pp)','Pairs with |Δgap| ≥2pp','BH conclusion changes']):
        xx=np.arange(2); vv=[v[key] for v in vals]; ax.bar(xx,vv,color=[BLUE,ORANGE],width=.55); ax.set(xticks=xx,xticklabels=labels,title=title); ax.spines[['right','top']].set_visible(False)
        if key=='median_abs_gap_pp': ax.errorbar(xx,vv,yerr=[[v[key]-v['median_abs_gap_ci_low'] for v in vals],[v['median_abs_gap_ci_high']-v[key] for v in vals]],fmt='none',color='#333333',capsize=3)
        else: ax.set_ylim(0,1); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    save(fig,'figure7_replication')
    (F/'captions.md').write_text('''# Figure captions and provenance

All primary figures use the fixed common-observation SWE-bench Test panel: 1,212 annotated tasks, 14 pass@1 configurations; severity ≥2 excludes 800, retains 412. These are recomputed annotated-subset rankings, not the current official global leaderboard. Agent codes map to agent_key.csv. All axes and changes are percentage points unless marked percent. Source tables are ../results/*.parquet; scripts/figures.py is the reproducible renderer.

1. Raw and quality-filtered resolution estimates. Task-paired 10,000-resample percentile confidence intervals for score shifts. The sample population changes after filtering; these are not corrected-task reruns.
2. Signed difference between clean and raw A-minus-B gaps. Antisymmetric matrix, zero diagonal. Color center is zero; no causality implied.
3. Rank slopegraph with average ties. There is one strict reversal and an additional tie transition; neither proves a statistically supported ordering change. Top-k membership uses boundary-tie inclusion.
4. Equal-size random removal compared with quality removal. The gap statistic is exceptional; rank reversals and inferential-state transitions are not. Exact paired McNemar + within-condition BH defines inference states.
5. Conditional odds-ratio profiles from Bayesian logistic mixed models with repository random intercept and human difficulty. Bars are approximate 95% variational credible intervals, NOT bootstrap confidence intervals. Mean-field covariance approximation is used. Individual ORs below one do not prove interactions; primary omnibus heterogeneity is established separately. Type-specific GEE fits with unstable nuisance cells are not used in this figure or counted as signals.
6. Task leave-one-out gap influence and matched influence differences. Problematic tasks have LOWER influence. The positive influence hypothesis is rejected; do not portray this as positive Signal E.
7. Partial Pro replication: 692 jointly observed tasks, 9 distinct named model/reasoning configurations, 93 known-corrected tasks. Remaining tasks have no known correction, not confirmed clean. Original Pro results are filtered; no Verified reruns. Harness provenance incomplete. Magnitude differs; gap widening exceeds random removal in both, while rank/statistical-transition effects are weak or absent.
''',encoding='utf-8')
if __name__=='__main__': main()
