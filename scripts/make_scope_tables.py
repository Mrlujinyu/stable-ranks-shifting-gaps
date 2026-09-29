from pathlib import Path
import json
R=Path(__file__).resolve().parents[1];d=json.loads((R/'analysis_outputs/scope_analysis.json').read_text());T=R/'tables';S=R/'supplement'
T.mkdir(exist_ok=True);S.mkdir(exist_ok=True)

def table(path,caption,label,columns,header,rows):
 rows=[x.rstrip('\\')+r'\\' for x in rows]
 path.write_text('\\begin{table}[htbp]\n\\centering\n\\caption{'+caption+'}\\label{'+label+'}\n\\begin{tabular}{'+columns+'}\\toprule\n'+header+r' \\'+ '\n\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\\end{tabular}\n\\end{table}\n')
rows=[]
for r in d['scopes']:
 p=r'$1/10{,}001$' if r['p_mc']==1/10001 else f"{r['p_mc']:.4f}"
 rows.append(f"{r['scope']} & {r['pairs']} & {r['median_delta']:.2f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}] & {r['random95']:.2f} & {p} \\")
table(T/'scope_inference.tex',r'Gap sensitivity across fixed original-score comparison scopes. Magnitudes and task-paired 95\% intervals are in pp. Random $Q_{.95}$ and upper-tail $p_{\rm MC}$ use the corresponding scope-specific distribution over the same 10,000 subsets of 412 tasks.', 'tab:scope','lrrrr',r'Scope & Pairs & $M_\Delta$ [95\% interval] & Random $Q_{.95}$ & $p_{\rm MC}$',rows)
table(T/'scale_components.tex',r'Absolute signed-gap change, affine scale term, and pairwise residual term (medians in pp). All scopes use the same fourteen-configuration fit; the three columns are separate summaries.', 'tab:scale','lrrr',r'Scope & $M_\Delta$ & $M_L$ & $M_E$',[f"{r['scope']} & {r['median_delta']:.2f} & {r['median_scale']:.2f} & {r['median_residual']:.2f} \\" for r in d['scopes']])
rows=[]
short=['Full panel, unweighted','Common support, unweighted','Common support, standardized']
for name,r in zip(short,d['populations']):
 st='---' if r['support_changes'] is None else str(r['support_changes'])
 rows.append(f"{name} & {r['n_reference']}/{r['n_retained']} & {r['median_delta']:.2f} & {r['tau']:.3f} & {r['strict']}/{r['margin']} & {st} \\")
table(T/'support_comparisons.tex',r'Comparisons across task populations. Reference/retained counts are $N_U/N_F$; $M_\Delta$ is in pp. Reversals are strict / 2-pp-margin counts. States counts changes across the complete 91-pair BH families. Exact binary McNemar states are inapplicable to weighted scores (---).','tab:support','lrrrrr',r'Population & $N_U/N_F$ & $M_\Delta$ & $\tau_b$ & Reversals & States',rows)
rows=[]
for r in d['reversal_pairs']:
 flags=f"{int(r['unweighted_reversal'])}/{int(r['unweighted_margin'])}"
 rows.append(f"{r['a']}--{r['b']} & {r['reference_gap']:+.3f} & {r['unweighted_filtered_gap']:+.3f} & {r['standardized_filtered_gap']:+.3f} & {flags} & 1/{int(r['standardized_margin'])} \\")
table(S/'support_reversals.tex',r'Pairs reversed by common-support standardization. Signed gaps favor the first ID and are in pp. Both filtered scores compare with the same 1,181-task reference; both use the same 411 retained tasks. Flags are strict / margin reversal indicators.','tab:sreversals','lrrrrr',r'Pair & Reference & Unweighted & Standardized & Unw. flags & Std. flags',rows)
table(S/'scope_full.tex',r'Fixed comparison scopes: complete local summaries. Intervals are paired task-bootstrap 95\% intervals (pp). Reversals are strict / 2-pp-margin counts.','tab:sscopes','lrrrr',r'Scope & Pairs & $M_\Delta$ [95\% interval] & Reversals & Opposite $L,E$',[f"{r['scope']} & {r['pairs']} & {r['median_delta']:.2f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}] & {r['strict']}/{r['margin']} & {r['opposite_sign_components']} \\" for r in d['scopes']])
print('Generated five tables from scope_analysis.json')
