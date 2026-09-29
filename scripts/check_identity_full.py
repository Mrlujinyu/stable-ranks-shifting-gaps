"""Identity regression against frozen matrices, model records, and draws."""
from pathlib import Path
import itertools,json,hashlib,re
import numpy as np
import pandas as pd
from identity_mapping import lookup,convert,align_main,reverse_estimate
R=Path(__file__).resolve().parents[1];D=R/'data';O=R/'analysis_outputs/final_checks';O.mkdir(parents=True,exist_ok=True)
read=lambda name:pd.read_csv(D/name,float_precision='round_trip')
c=read('table2_configurations.csv');p=read('main_panel.csv');rows=c[c.primary].to_dict('records');names=sorted(x['submission'] for x in rows)
maps={key:lookup(rows,'submission',key) for key in ['figure_id','model_internal_id']}
for key,mapping in maps.items():
    inv=lookup(rows,key,'submission')
    assert all(convert(inv,convert(mapping,x))==x for x in names)
    try:convert(mapping,'__unknown_submission__')
    except KeyError:pass
    else:raise AssertionError('unknown identity accepted')
Y,order=align_main(p,c);bad=p.severity.ge(2).to_numpy();native=read('agent_matrix.csv');wide=native[native.dataset.eq('A')].pivot(index='task_id',columns='submission',values='resolved').reindex(p.task_id)
np.testing.assert_array_equal(Y,wide[order].to_numpy())
# Arbitrary independent reversals of CSV rows and named matrix columns.
YY,reorder=align_main(p[p.columns[::-1]],c.iloc[::-1]);assert order==reorder;np.testing.assert_array_equal(YY,Y)
cache=np.load(R/'analysis_outputs/shared_random_subsets.npz');assert np.array_equal(cache['task_ids'],p.task_id.astype(str));ix=cache['indices']
for start in range(0,len(ix),200):
    np.testing.assert_array_equal(Y[ix[start:start+200]].mean(1),YY[ix[start:start+200]].mean(1))
u=Y.mean(0)*100;f=Y[~bad].mean(0)*100
scores=read('all14_scores.csv').set_index('agent')
for j,name in enumerate(order):
    assert scores.loc[name,'agent_figure_id']==maps['figure_id'][name]
    np.testing.assert_allclose([u[j],f[j]],[scores.loc[name,'raw_score_pp'],scores.loc[name,'clean_score_pp']],atol=1e-12)
def pairs_check(frame,ids,U,F):
    index={name:i for i,name in enumerate(ids)};seen=set()
    for row in frame.itertuples():
        a,b=row.agent_a,row.agent_b;key=frozenset([a,b]);assert key not in seen and a!=b;seen.add(key)
        i,j=index[a],index[b]
        np.testing.assert_allclose([row.raw_gap_pp,row.clean_gap_pp,row.delta_gap_pp],[U[i]-U[j],F[i]-F[j],F[i]-F[j]-U[i]+U[j]],atol=1e-12)
    assert seen=={frozenset(x) for x in itertools.combinations(ids,2)}
pairs=read('all91_pairs.csv');pairs_check(pairs,order,u,f)
for x in pairs.itertuples():
    assert x.agent_a_figure_id==maps['figure_id'][x.agent_a] and x.agent_b_figure_id==maps['figure_id'][x.agent_b]
    rev=reverse_estimate(x.raw_gap_pp,x.delta_gap_pp,x.delta_ci_low,x.delta_ci_high,x.raw_p,x.raw_q,x.raw_state)
    assert rev==(-x.raw_gap_pp,-x.delta_gap_pp,-x.delta_ci_high,-x.delta_ci_low,x.raw_p,x.raw_q,-x.raw_state)
    assert reverse_estimate(*rev)==(x.raw_gap_pp,x.delta_gap_pp,x.delta_ci_low,x.delta_ci_high,x.raw_p,x.raw_q,x.raw_state)
example=json.loads((R/'analysis_outputs/scope_analysis.json').read_text())['example']
x=pairs[(pairs.agent_a_figure_id=='A10')&(pairs.agent_b_figure_id=='A11')].iloc[0]
np.testing.assert_allclose([example['original_gap'],example['filtered_gap'],example['delta'],*example['ci']],[-x.raw_gap_pp,-x.clean_gap_pp,-x.delta_gap_pp,-x.delta_ci_high,-x.delta_ci_low],atol=1e-12)
# GEE coding derives from the original load(): lexicographic stable identities.
assert [maps['model_internal_id'][x] for x in names]==[f'A{i:02}' for i in range(len(names))]
assert maps['figure_id'][names[0]]=='A01'
profiles=read('primary_profiles.csv');source_profiles=read('agent_sensitivity_profiles.csv').query("model=='GEE_adjusted_primary' and defect=='defect'").set_index('agent')
for row in profiles.itertuples():
    assert row.agent_figure_id==maps['figure_id'][row.agent]
    z=source_profiles.loc[row.agent]
    np.testing.assert_allclose([row.adjusted_probability_delta_pp,row.ci_low,row.ci_high],[z.adjusted_probability_delta_pp,z.ci_low,z.ci_high],atol=1e-12)
models=read('interaction_models.csv').query("model=='GEE_adjusted_primary'")
levels=set(re.findall(r'C\(agent\)\[T\.(A\d+)\]',' '.join(models.term)))
assert levels==set(maps['model_internal_id'].values())-{'A00'}
robust=read('interaction_robustness.csv');assert set(robust.loc[robust.exclusion_type.eq('agent'),'excluded'])==set(maps['model_internal_id'].values())
loo=[]
for code in robust.loc[robust.exclusion_type.eq('agent'),'excluded']:
    remaining=sorted(set(maps['model_internal_id'].values())-{code});loo.append({'excluded_internal':code,'reference_internal':remaining[0]})
# Expanded panel has stable source identities, but no manuscript/internal A IDs
# assigned to its two extra columns: do not synthesize them.
expanded=wide[sorted(c.loc[c.included,'submission'])].dropna();assert len(expanded)==1205 and expanded.shape[1]==16
# Pro identities are source labels; the source supplies no main-panel A codes.
meta=read('pro_submission_metadata.csv');pn=sorted(meta.loc[meta.included,'submission']);pid={x:f'P{i+1:02}' for i,x in enumerate(pn)}
pl=read('pro_labels_analysis.csv');pm=read('pro_observed_matrix.csv').pivot(index='task_id',columns='submission',values='resolved').reindex(pl.task_id)[pn]
assert pm.notna().all().all();py=pm.to_numpy();pu=100*py.mean(0);pf=100*py[~pl.corrected.to_numpy(bool)].mean(0)
pairs_check(read('pro_all36_pairs.csv'),pn,pu,pf)
assert set(read('pro_statistical_transitions.csv').agent_a)|set(read('pro_statistical_transitions.csv').agent_b)==set(pn)
# Fixed local scopes, labels, and standardized reversal identities.
top=sorted(range(14),key=lambda j:(-u[j],names[j]));pc=pd.read_csv(R/'analysis_outputs/pairwise_affine_components.csv')
assert {frozenset(x) for x in zip(pc.a,pc.b)}=={frozenset(x) for x in itertools.combinations(maps['figure_id'].values(),2)}
assert set(pd.read_csv(R/'analysis_outputs/affine_scores.csv').configuration)==set(maps['figure_id'].values())
for row in pd.read_csv(R/'analysis_outputs/standardized_reversal_pairs.csv').itertuples():assert row.a in maps['figure_id'].values() and row.b in maps['figure_id'].values()
out=[]
for row in c.to_dict('records'):
    primary=bool(row['primary']);out.append({'panel':'main' if primary else 'expanded-only','submission_id':row['submission'],'internal_code':row['model_internal_id'] if primary else '', 'manuscript_id':row['figure_id'] if primary else '', 'source_result_sha256':row['results_sha256'],'source_hash_status':'recorded in supplied manifest; raw result file not bundled','matrix_column':row['figure_id'] if primary else row['submission'],'matrix_source':'main_panel.csv + agent_matrix.csv' if primary else 'agent_matrix.csv','identity_match':True})
for name in pn:out.append({'panel':'Pro','submission_id':name,'internal_code':'','manuscript_id':pid[name],'source_result_sha256':'','source_hash_status':'not supplied','matrix_column':name,'matrix_source':'pro_observed_matrix.csv','identity_match':True})
pd.DataFrame(out).to_csv(O/'mapping_check.csv',index=False)
(O/'identity_check.json').write_text(json.dumps({'main_pairs':len(pairs),'pro_pairs':len(read('pro_all36_pairs.csv')),'expanded_tasks':len(expanded),'expanded_configurations':expanded.shape[1],'original_top_five':[maps['figure_id'][names[j]] for j in top[:5]],'original_adjacent':[[maps['figure_id'][names[a]],maps['figure_id'][names[b]]] for a,b in zip(top[:-1],top[1:])],'shared_subset_draws_checked':len(ix),'GEE_reference_submission':names[0],'GEE_omission_references':loo,'tests':'uniqueness; round-trip; unknown-error; joint row/column reorder on same draws; matrix joins; unordered completeness; directed gap/interval inversion; profiles; GEE levels; scope labels'},indent=2))
print('Identity checks passed: stable main/expanded/Pro joins, all 10000 cached subsets, pair orientation, GEE identities.')
