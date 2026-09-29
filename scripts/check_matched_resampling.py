"""Reproduce fixed-match resampling, including original RNG consumption order."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
R=Path(__file__).resolve().parents[1];D=R/'data';O=R/'analysis_outputs';O.mkdir(exist_ok=True)
d=pd.read_csv(D/'main_panel.csv');c=pd.read_csv(D/'table2_configurations.csv').query('primary').sort_values('submission');Y=d[list(c.figure_id)].to_numpy(float)
bad=d.severity.ge(2).to_numpy();rng=np.random.default_rng(20260919);ij=np.triu_indices(14,1);gap=lambda s:s[...,ij[0]]-s[...,ij[1]]
feat=np.log1p(d[['problem_length','patch_lines','f2p_count']].to_numpy(float));feat=(feat-np.nanmean(feat,0))/np.nanstd(feat,0);feat=np.nan_to_num(feat)
infl=abs((Y[:,ij[0]]-Y[:,ij[1]])-gap(Y.mean(0))).mean(1)*100/(len(Y)-1);rows=[]
for nearest in [False,True]:
 aa=[];bb=[]
 for _,ix in d.groupby(['repo','human_difficulty']).indices.items():
  a=ix[bad[ix]];b=ix[~bad[ix]];k=min(len(a),len(b))
  if not k:continue
  if nearest:
   dist=((feat[a,None,:]-feat[None,b,:])**2).sum(2);ia,ib=linear_sum_assignment(dist);a=a[ia];b=b[ib]
  else:a=rng.choice(a,k,False);b=rng.choice(b,k,False)
  aa.extend(a);bb.extend(b)
 a=np.array(aa);b=np.array(bb);k=len(a);diff=Y[b]-Y[a];boots=[];perms=[];iboot=[]
 for start in range(0,10000,250):
  w=rng.multinomial(k,np.full(k,1/k),size=250);boots.append(gap(w@diff/k*100));sign=rng.choice([-1,1],size=(250,k));perms.extend(np.mean(abs(gap(sign@diff/k*100)),axis=1));iboot.extend(w@(infl[a]-infl[b])/k)
 method='exact_repo_difficulty'+('_nearest_metadata' if nearest else '')
 saved=pd.read_csv(D/(method+'_matches.csv'));same=bool(np.array_equal(saved.defect_task,d.task_id.iloc[a]) and np.array_equal(saved.clean_task,d.task_id.iloc[b]))
 contrast=gap(diff.mean(0)*100);scale=k/(len(Y)-k);mag=np.mean(abs(np.concatenate(boots)),axis=1)*scale
 rows.append({'method':method,'matches_identical':same,'matched_pairs':k,'mean_abs_full_reference_contrast':float(np.mean(abs(contrast))*scale),'ci':np.quantile(mag,[.025,.975]).tolist(),'permutation_p':float((1+np.sum(np.array(perms)>=np.mean(abs(contrast))))/10001),'cliffs_delta':float(2*stats.mannwhitneyu(infl[a],infl[b]).statistic/k**2-1),'influence_ci':np.quantile(iboot,[.025,.975]).tolist()})
(O/'matched_union_resampling_current.json').write_text(json.dumps(rows,indent=2))
# The full-reference script restarts the seed after loading fixed matches.
# It does not consume the draws used to construct the match assignment.
rng=np.random.default_rng(20260919);full=[]
for method in ['exact_repo_difficulty','exact_repo_difficulty_nearest_metadata']:
 mm=pd.read_csv(D/(method+'_matches.csv'));idx=pd.Series(np.arange(len(d)),index=d.task_id);a=idx[mm.defect_task].to_numpy();b=idx[mm.clean_task].to_numpy();k=len(a);delta=Y[b]-Y[a];bs=[];null=[]
 for start in range(0,10000,250):
  w=rng.multinomial(k,np.full(k,1/k),size=250);bs.append(gap(w@delta/(len(Y)-k)*100));sign=rng.choice([-1,1],size=(250,k));null.extend(np.mean(abs(gap(sign@delta/(len(Y)-k)*100)),axis=1))
 observed=np.mean(abs(gap(delta.sum(0)/(len(Y)-k)*100)));interval=np.quantile(np.mean(abs(np.concatenate(bs)),axis=1),[.025,.975])
 ref=next(x for x in json.loads((D/'matched_full_reference_summary.json').read_text()) if x['method']==method)
 full.append({'method':method,'mean_abs_contrast_pp':observed,'ci':interval.tolist(),'matches_saved_interval':bool(np.allclose(interval,[ref['contrast_ci_low'],ref['contrast_ci_high']])),'permutation_p':float((1+np.sum(np.array(null)>=observed))/10001)})
(O/'matched_resampling_current.json').write_text(json.dumps(full,indent=2));print(json.dumps(full,indent=2))
