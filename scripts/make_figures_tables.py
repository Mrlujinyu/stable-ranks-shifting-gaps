#!/usr/bin/env python3
"""Regenerate manuscript figures and tables from the supplied numeric evidence."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import rankdata
R = Path(__file__).resolve().parents[1]
D, F, T = R/'data', R/'figures', R/'tables'
F.mkdir(exist_ok=True); T.mkdir(exist_ok=True)
def read(name): return pd.read_csv(D/name, float_precision='round_trip')
def save(fig,name):
    fig.savefig(F/(name+'.pdf'),bbox_inches='tight',metadata={'Creator':'Matplotlib','Title':name,'Author':''})
    fig.savefig(F/(name+'.png'),dpi=220,bbox_inches='tight');plt.close(fig)
def tex(s):
    return str(s).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('&',r'\&').replace('%',r'\%').replace('#',r'\#')
s=read('all14_scores.csv').sort_values('agent_figure_id');pairs=read('all91_pairs.csv')
ids=list(s.agent_figure_id); idx={x:i for i,x in enumerate(ids)}
m=np.full((14,14),np.nan)
for x in pairs.itertuples():
    a,b=sorted([idx[x.agent_a_figure_id],idx[x.agent_b_figure_id]])
    m[a,b]=x.abs_delta_gap_pp
fig,ax=plt.subplots(figsize=(6.7,4.25))
im=ax.imshow(np.ma.masked_invalid(m),vmin=0,vmax=20,aspect='equal')
ax.set_xticks(range(14),ids,fontsize=8,rotation=45);ax.set_yticks(range(14),ids,fontsize=8)
ax.xaxis.tick_top();ax.tick_params(length=0);ax.set_ylim(13.5,-.5)
for sp in ax.spines.values():sp.set_visible(False)
cb=fig.colorbar(im,ax=ax,fraction=.045,pad=.04);cb.set_label('Absolute gap shift (percentage points)',fontsize=9);cb.ax.tick_params(labelsize=8)
fig.tight_layout();save(fig,'gap_matrix')
null=read('random_removal.csv');obs=json.loads((D/'core_summary.json').read_text())[0]
fig,ax=plt.subplots(figsize=(6.6,2.6))
ax.hist(null.median_abs_gap_pp,bins=45,alpha=.8)
q=float(null.median_abs_gap_pp.quantile(.95))
ax.axvline(q,linestyle='--',linewidth=1.4,label=f'Random 95th percentile: {q:.2f}')
ax.axvline(obs['median_abs_gap_pp'],linestyle='-',linewidth=1.8,label='Quality-filtered: 6.80')
ax.set_xlim(0,7.2);ax.set_xlabel('Median absolute gap shift (percentage points)',fontsize=10);ax.set_ylabel('Random subsets',fontsize=10)
ax.legend(fontsize=9,frameon=False,loc='upper right');ax.tick_params(labelsize=9)
fig.tight_layout();save(fig,'gap_null')
p=read('primary_profiles.csv').sort_values('agent_figure_id')
fig,ax=plt.subplots(figsize=(6.6,3.5))
y=np.arange(len(p));x=p.adjusted_probability_delta_pp.to_numpy()
ax.errorbar(x,y,xerr=np.vstack([x-p.ci_low.to_numpy(),p.ci_high.to_numpy()-x]),fmt='o',capsize=2,markersize=4,linewidth=1.2)
ax.axvline(0,linestyle=':',linewidth=1)
ax.set_yticks(y,p.agent_figure_id,fontsize=9);ax.invert_yaxis()
ax.set_xlabel('Adjusted success probability: flagged minus below threshold (pp)',fontsize=9)
ax.set_xlim(-44,2);ax.tick_params(axis='x',labelsize=9)
fig.tight_layout();save(fig,'gee_profiles')
rr=rankdata(-s.raw_score_pp.to_numpy(),method='average');rf=rankdata(-s.clean_score_pp.to_numpy(),method='average')
fig,ax=plt.subplots(figsize=(5.1,3.7))
for i,n in enumerate(ids):
    ax.plot([0,1],[rr[i],rf[i]],marker='o',markersize=3,linewidth=1)
    ax.text(-.06,rr[i],n,ha='right',va='center',fontsize=8)
for r in sorted(set(rf)):
    names=', '.join(np.asarray(ids)[rf==r]);ax.text(1.06,r,names,ha='left',va='center',fontsize=8)
ax.set_xlim(-.33,1.55);ax.set_ylim(14.6,.4);ax.set_xticks([0,1],['Original','Quality-filtered'],fontsize=10)
ax.set_yticks([1,3,5,7,9,11,14]);ax.set_ylabel('Rank (1 = highest)',fontsize=9)
ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False);ax.tick_params(labelsize=9)
fig.tight_layout();save(fig,'rank_stability')
# Main table: fixed configuration identities and computed scores.
short={
'A01':'SWE-agent / Claude 3 Opus','A02':'SWE-agent / GPT-4','A03':'SWE-agent / GPT-4o',
'A04':'Amazon Q / 2024-04-30 dev','A05':'AppMap Navie / GPT-4o','A06':'AutoCodeRover / 2024-06-20',
'A07':'Factory Code Droid','A08':'Amazon Q / 2024-07-19 dev','A09':'AutoCodeRover v2 / Claude 3.5',
'A10':'OpenHands CodeAct 2.1 / Claude 3.5','A11':'Amazon Q / 2024-12-02 dev',
'A12':'SWE-agent 1.0 / Claude 3.7','A13':'Atlassian Rovo Dev','A14':'Sonar Foundation Agent / Opus 4.5'}
rows=[]
for r in s.itertuples():rows.append(f'{r.agent_figure_id} & {short[r.agent_figure_id]} & {r.raw_score_pp:.2f} & {r.clean_score_pp:.2f} & {r.delta_pp:.2f} \\\\')
(T/'config_scores.tex').write_text(r'''\begin{table}[t]
\caption{Primary fixed configurations and scores on the common panel. ``Original'' uses 1,212 tasks; ``Filtered'' uses 412. Scores are percentages and $\Delta s$ is in percentage points. Names are abbreviated; exact versioned submissions and provenance appear in the supplement. IDs are identifiers, not an inferred model-family ordering.}
\label{tab:configs}
\centering\small
\begin{tabularx}{\linewidth}{lXrrr}
\toprule
ID & Configuration & Original & Filtered & $\Delta s$ \\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule
\end{tabularx}
\end{table}
''')
(T/'controls.tex').write_text(r'''\begin{table}[t]
\caption{Composition controls on the main data. Gap magnitudes are in pp. The matched contrast $\operatorname{mean}|\delta^D-\delta^C|$ differs from $\operatorname{mean}|\delta^D|-\operatorname{mean}|\delta^C|$. Matched deletion retains 828 tasks in each arm. Intervals condition on the selected matches.}
\label{tab:controls}
\centering\small
\begin{tabularx}{\linewidth}{Xrr}
\toprule
Analysis / statistic & Exact strata & + Metadata matching \\
\midrule
Matched pairs & 384 & 384 \\
Coverage of flagged tasks & 48.0\% & 48.0\% \\
$\operatorname{mean}|\delta^D|$: flagged-task deletion & 0.93 & 0.63 \\
$\operatorname{mean}|\delta^C|$: comparison-task deletion & 3.57 & 3.47 \\
$\operatorname{mean}|\delta^D-\delta^C|$ & 4.41 & 3.83 \\
95\% interval for the absolute contrast & [3.51, 5.55] & [2.98, 5.02] \\
\midrule
\multicolumn{3}{l}{Repository--difficulty standardization (separate estimand)} \\
Supported original tasks & \multicolumn{2}{r}{1,181 (97.4\%)} \\
Effective filtered sample size & \multicolumn{2}{r}{287.94} \\
Median absolute gap shift & \multicolumn{2}{r}{5.68} \\
Pairs shifting by at least 2\,pp & \multicolumn{2}{r}{71/91} \\
\bottomrule
\end{tabularx}
\end{table}
''')
rows=[]
for r in pairs[pairs.changed].itertuples():
    rows.append(f'{r.agent_a_figure_id}--{r.agent_b_figure_id} & {r.raw_gap_pp:+.2f} & {r.clean_gap_pp:+.2f} & {r.raw_q:.4f} & {r.clean_q:.4f} & '+('Loss' if r.clean_state==0 else 'Gain')+r' \\')
(T/'transitions.tex').write_text(r'''\begin{table}[t]
\caption{All four primary BH support-state transitions. Gaps are first-listed minus second-listed configuration, in pp. ``Loss'' and ``Gain'' refer only to within-condition adjusted support at 0.05, not to a test of the change in gap. None is a supported-direction reversal.}
\label{tab:transitions}
\centering\small
\begin{tabular}{lrrrrl}
\toprule
Pair & Original gap & Filtered gap & Original $q$ & Filtered $q$ & State \\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule
\end{tabular}
\end{table}
''')
(T/'main_pro.tex').write_text(r'''\begin{table}[t]
\caption{Main and external-panel results under their respective primary quality indicators. Intervals are task-paired bootstrap 95\% intervals. Monte Carlo values use equal-size random removal within each panel. Top-10 is inapplicable to nine configurations (---). The panels differ in labels, removal fraction, tasks, and configuration provenance.}
\label{tab:mainpro}
\centering\small
\begin{tabularx}{\linewidth}{Xrr}
\toprule
Measure & SWE-bench Test & SWE-Bench Pro \\
\midrule
Tasks / configurations / pairs & 1,212 / 14 / 91 & 692 / 9 / 36 \\
Flagged tasks (fraction) & 800 (66.0\%) & 93 (13.4\%) \\
Retained tasks & 412 & 599 \\
Median absolute gap shift (pp) & 6.80 [5.55, 8.16] & 0.95 [0.67, 1.54] \\
Mean absolute gap shift (pp) & 7.47 [6.29, 8.90] & 1.13 [0.74, 1.65] \\
Pairs shifting by at least 2\,pp & 78/91 & 6/36 \\
Random-removal $p$ for median & $1/10{,}001$ & 0.0074 \\
Kendall $\tau_b$ / Spearman $\rho$ & 0.9724 / 0.9923 & 1.0000 / 1.0000 \\
Strict / 2\,pp-margin reversals & 1 / 0 & 0 / 0 \\
Top-3/5/10 membership changes & 0 / 0 / 0 & 0 / 0 / --- \\
BH support-state changes & 4/91 & 0/36 \\
Random-removal $p$ for state changes & 0.8628 & 1.0000 \\
\bottomrule
\end{tabularx}
\end{table}
''')
print('Wrote four figures and four main tables.')
