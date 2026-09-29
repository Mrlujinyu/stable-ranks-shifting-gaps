"""Reconstruct frozen controls.py matching/bootstrap/swap RNG consumption."""
import ast,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from permutation_utils import permutation_upper_tail
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data';O=ROOT/'analysis_outputs/final_checks'
def run():
 O.mkdir(parents=True,exist_ok=True)
 d=pd.read_csv(D/'main_panel.csv',float_precision='round_trip');c=pd.read_csv(D/'table2_configurations.csv').query('primary').sort_values('submission')
 Y=d[list(c.figure_id)].to_numpy(float);bad=d.severity.ge(2).to_numpy();n,m=Y.shape;ij=np.triu_indices(m,1)
 disc=Y[:,ij[0]]-Y[:,ij[1]];infl=abs(disc-(Y.mean(0)[ij[0]]-Y.mean(0)[ij[1]])).mean(1)*100/(n-1)
 reference=pd.read_csv(D/'task_influence.csv',float_precision='round_trip').set_index('task_id').loc[d.task_id,'influence_pp'].to_numpy()
 np.testing.assert_allclose(infl,reference,rtol=0,atol=1e-14)
 original=ROOT/'scripts/original_research/controls.py';tree=ast.parse(original.read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='matched_indices')
 ns={'np':np,'linear_sum_assignment':linear_sum_assignment};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(original),'exec'),ns)
 rng=np.random.default_rng(20260919);results=[];B=10000;archived=json.loads((D/'matched_summary.json').read_text())
 for nearest in [False,True]:
  method='exact_repo_difficulty'+('_nearest_metadata' if nearest else '');a,b,_=ns['matched_indices'](d,bad,rng,nearest);saved=pd.read_csv(D/(method+'_matches.csv'));k=len(a)
  assert np.array_equal(d.task_id.iloc[a].to_numpy(),saved.defect_task.to_numpy()) and np.array_equal(d.task_id.iloc[b].to_numpy(),saved.clean_task.to_numpy())
  assert len(set(a))==k and len(set(b))==k and not set(a)&set(b) and bad[a].all() and (~bad[b]).all()
  iv=infl[a]-infl[b];obs=float(iv.mean());draws=[];boots=[];signs=[]
  for start in range(0,B,250):
   w=rng.multinomial(k,np.full(k,1/k),size=min(250,B-start));sign=rng.choice([-1,1],size=(len(w),k));boots.extend(w@iv/k);draws.extend(sign@iv/k);signs.append(sign.astype(np.int8))
  draws=np.array(draws);boots=np.array(boots);ci=np.quantile(boots,[.025,.975]);cliff=float(2*stats.mannwhitneyu(infl[a],infl[b]).statistic/k**2-1);ref=next(x for x in archived if x['method']==method)
  np.testing.assert_allclose([obs,*ci,cliff],[ref['matched_influence_difference'],ref['influence_ci_low'],ref['influence_ci_high'],ref['influence_cliffs_delta']],rtol=0,atol=1e-12)
  p=permutation_upper_tail(obs,draws);assert p==ref['influence_permutation_p']
  np.savez_compressed(O/(method+'_influence_draws.npz'),permutation=draws,bootstrap=boots,signs=np.concatenate(signs),flagged_tasks=d.task_id.iloc[a].to_numpy(dtype=str),comparison_tasks=d.task_id.iloc[b].to_numpy(dtype=str),paired_influence_difference=iv)
  results.append({'method':method,'matched_pairs':k,'matched_tasks':2*k,'match_identity_and_order_equal':True,'observed':obs,'unit':'pp','ci95':ci.tolist(),'cliffs_delta':cliff,'draws':B,'rng':'NumPy PCG64','seed':20260919,'stream':'single stream: exact matching; interleaved 250-row multinomial bootstrap and signs; metadata matching; interleaved bootstrap and signs; no reset between methods','swap_rule':'each fixed pair independently receives sign -1 or +1','tail_rule':'(1 + count(T_perm >= T_obs))/(B+1)','count_greater':int((draws>obs).sum()),'count_equal':int((draws==obs).sum()),'count_less':int((draws<obs).sum()),'min':float(draws.min()),'max':float(draws.max()),'p_upper':p,'archived_summary_agrees':True})
 out={'source_script':'scripts/original_research/controls.py','source_script_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'per_task_influence_max_abs_error':float(abs(infl-reference).max()),'original_permutation_array_available':False,'draws_status':'new deterministic reconstruction from frozen implementation, not an original saved array','results':results}
 (O/'permutation_check.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':run()
