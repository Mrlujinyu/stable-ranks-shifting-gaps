#!/usr/bin/env python3
"""Build complete supplementary tables from the archived numeric results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1];D=R/'data';T=R/'supplement'
T.mkdir(exist_ok=True)

def read(n):return pd.read_csv(D/n,float_precision='round_trip')
def esc(s):return str(s).replace('&',r'\&').replace('_',r'\_').replace('%',r'\%')
def pval(x):
    if x==0:return '$0$'
    if x>=.001:return f'{x:.4f}'
    a,b=f'{x:.2e}'.split('e');return f'${a}\\times10^{{{int(b)}}}$'
def longtable(name,caption,label,cols,head,rows,foot=''):
    n=len(cols) if all(c in 'lcr' for c in cols) else head.count('&')+1
    text='\\begingroup\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n'
    text+='\\begin{longtable}{'+cols+'}\n\\caption{'+caption+'}\\label{'+label+'}\\\\\n\\toprule\n'+head+r' \\'+'\n\\midrule\n\\endfirsthead\n'
    text+='\\multicolumn{'+str(n)+'}{l}{\\tablename~\\thetable{} (continued)}\\\\\n\\toprule\n'+head+r' \\'+'\n\\midrule\n\\endhead\n\\midrule\n\\multicolumn{'+str(n)+'}{r}{Continued on next page}\\\\\n\\endfoot\n\\bottomrule\n\\endlastfoot\n'
    text+='\n'.join(rows)+'\n\\end{longtable}\n\\endgroup\n'+foot+'\n'
    (T/name).write_text(text)
cfg=read('table2_configurations.csv');primary=cfg[cfg.primary].sort_values('figure_id')
rows=[]
for r in primary.itertuples():
    date=str(r.date);date=f'{date[:4]}-{date[4:6]}-{date[6:]}'
    rows.append(f'{r.figure_id} & '+r'\nolinkurl{'+str(r.submission)+'} & '+r'\nolinkurl{'+str(r.model)+'} & '+date+r' \\')
longtable('configurations.tex','Exact primary submission and model identifiers. Dates are submission metadata. Source SHA256 values are supplied in the CSV configuration table.','tab:sconfigs',r'lp{.43\linewidth}p{.29\linewidth}l','ID & Submission identifier & Reported model & Date',rows)
rows=[]
for r in cfg[~cfg.primary].itertuples():rows.append(r'\nolinkurl{'+str(r.submission)+'} & '+esc(r.attempt_policy)+r' \\')
longtable('excluded_main.tex','Additional candidate configurations used only in the expanded-set sensitivity analysis.','tab:sexcluded',r'p{.72\linewidth}p{.18\linewidth}','Submission identifier & Attempt policy',rows)
meta=read('pro_submission_metadata.csv');incl=meta[meta.included].sort_values('submission');pid={s:f'P{i+1:02}' for i,s in enumerate(incl.submission)}
ps=read('pro_score_distortion.csv').set_index('agent');rows=[]
for r in incl.itertuples():
    z=ps.loc[r.submission]
    rows.append(f'{pid[r.submission]} & '+esc(r.submission.strip())+f' & {r.observed_tasks} & {z.raw_score_pp:.2f} & {z.clean_score_pp:.2f}'+r' \\')
longtable('pro_configurations.tex','Included Pro configuration labels, observed coverage before intersection, and scores (percent) on the 692-task common panel and 599-task filtered panel. Source labels do not provide uniform immutable model or harness versions.','tab:sproconfigs',r'lp{.45\linewidth}rrr','ID & Source configuration label & Coverage & Original & Filtered',rows)
rows=[]
for r in meta[~meta.included].itertuples():
    name=r.submission.strip()
    if name.startswith('Gemini'):reason='Ambiguous same-preview identity; retain paper entry'
    else:reason='Observed coverage below 700'
    rows.append(esc(name)+f' & {r.observed_tasks} & '+reason+r' \\')
longtable('pro_exclusions.tex','Pro exclusions. Both Claude Sonnet 4 labels fall below the coverage threshold; neither is included. Coverage is distinct from the shared-panel count.','tab:sproexcluded',r'p{.43\linewidth}rp{.39\linewidth}','Source configuration label & Coverage & Exclusion basis',rows)
core=json.loads((D/'core_summary.json').read_text());labels={'severity_ge2':r'$\max(u,v)\geq2$ (primary)','severity_ge1':r'$\max(u,v)\geq1$','severity_eq3':r'$\max(u,v)=3$','all_defects':r'Primary or other issue','problem_statement':r'$u\geq2$','test_validity':r'$v\geq2$','other_problem':'Other issue'}
rows=[]
for r in core:
    rows.append(labels[r['definition']]+f" & {r['removed']} & {1212-r['removed']} & {r['median_abs_gap_pp']:.2f} & {round(r['pairs_ge_2pp_rate']*91)}/91 & {int(r['reversals'])} & {round(r['practical_2pp_rate']*91)} & {round(r['statistical_change_rate']*91)}"+r' \\')
longtable('thresholds.tex','All quality definitions on the 1,212-task, 14-configuration common panel. Median is median absolute gap shift (pp). Margin reversals use the manuscript\'s 2 pp rule; state changes use condition-specific BH.','tab:sthresholds','lrrrrrrr',r'Definition & Removed & Retained & Median & $|\Delta g|\geq2$ & Reversals & Margin & States',rows)
rows=[]
for r in read('primary_profiles.csv').sort_values('agent_figure_id').itertuples():
    rows.append(f'{r.agent_figure_id} & {r.adjusted_probability_delta_pp:.2f} & [{r.ci_low:.2f}, {r.ci_high:.2f}] & {r.conditional_odds_ratio:.3f}'+r' \\')
longtable('profiles.tex','Primary GEE standardized probability differences (flagged minus below threshold, pp), robust pointwise intervals, and conditional odds ratios. The probability contrasts average over the common covariate distribution.','tab:sprofiles','lrrr',r'Configuration & Probability difference & 95\% interval & Odds ratio',rows)
rows=[]
for r in json.loads((D/'interaction_summary.json').read_text()):
    name={'defect':'Pooled quality (primary)','statement':'Underspecification','test':'Test validity','other':'Other issue','high':'Maximum severity'}[r['defect']]
    rows.append(f"{name} & {r['interaction_wald']:.3f} & {r['df']} & {pval(r['p'])} & {pval(r['q'])}"+r' \\')
longtable('omnibus.tex','Five-test omnibus interaction family. Only the pooled-quality model is the primary finite-covariance inferential specification; subtype fits are diagnostic.','tab:somnibus','lrrrr','Quality term & Wald & df & $p$ & BH $q$',rows)
internal=dict(zip(primary.model_internal_id,primary.figure_id));rows=[]
for r in json.loads((D/'interaction_robustness.json').read_text()):
    ex='None (full primary fit)' if r['exclusion_type']=='none' else ('Configuration '+str(internal[r['excluded']]) if r['exclusion_type']=='agent' else esc(r['excluded']))
    rows.append(f"{ex} & {r['wald']:.3f} & {r['df']} & {pval(r['p'])} & "+('Yes' if r['finite_covariance'] else 'No')+r' \\')
longtable('leave_one_out.tex','Pooled-quality GEE omission diagnostics. Values are unadjusted; all listed fits converge. Configuration IDs are translated from internal model codes to manuscript IDs.','tab:sloo','lrrrr','Excluded repository/configuration & Wald & df & $p$ & Finite covariance',rows)
# Every pair, with the exact stored canonical orientation.
for pro in [False,True]:
    a=read('pro_all36_pairs.csv' if pro else 'all91_pairs.csv')
    if pro:
        tr=read('pro_statistical_transitions.csv')[['agent_a','agent_b','raw_q','clean_q']]
        a=a.merge(tr,on=['agent_a','agent_b'],validate='one_to_one');a['aid']=a.agent_a.map(pid);a['bid']=a.agent_b.map(pid)
    else:a['aid']=a.agent_a_figure_id;a['bid']=a.agent_b_figure_id
    a=a.sort_values(['aid','bid']);rows=[]
    for r in a.itertuples():
        rows.append(f'{r.aid}--{r.bid} & {r.raw_gap_pp:+.2f} & {r.clean_gap_pp:+.2f} & {r.delta_gap_pp:+.2f} & [{r.delta_ci_low:+.2f}, {r.delta_ci_high:+.2f}] & {pval(r.raw_q)} & {pval(r.clean_q)}'+r' \\')
    name='pro_pairs' if pro else 'main_pairs'
    longtable(name+'.tex',('All 36 Pro' if pro else 'All 91 primary')+r' pairwise comparisons. Signed gaps are first-listed minus second-listed configuration; $\Delta g=g(F)-g(U)$, in pp. Intervals are pointwise task-paired bootstrap intervals, not simultaneous intervals. Adjusted values test the within-condition gap, not $\Delta g$.','tab:s'+name,'lrrrrrr',r'Pair & Original & Filtered & $\Delta g$ & 95\% interval for $\Delta g$ & Original $q$ & Filtered $q$',rows)
# A corrected, derived provenance view leaves archived metadata unchanged.
view=meta.copy();view['audited_exclusion_basis']=[('Included' if bool(r.included) else ('Ambiguous same-preview identity' if r.submission.strip().startswith('Gemini') else 'Observed coverage below 700')) for r in meta.itertuples()]
(R/'analysis_outputs/final_checks').mkdir(parents=True,exist_ok=True)
view.to_csv(R/'analysis_outputs/final_checks/pro_configuration_provenance.csv',index=False)
print('Supplement tables generated.')
