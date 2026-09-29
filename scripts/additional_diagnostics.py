"""Descriptive diagnostics on fixed archived outcomes; no agent executions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
from statistics_core import metrics, bh, mcn_p, states

R=Path(__file__).resolve().parents[1]
D=R/'data'; O=R/'analysis_outputs'; O.mkdir(exist_ok=True)
d=pd.read_csv(D/'main_panel.csv',float_precision='round_trip')
c=pd.read_csv(D/'table2_configurations.csv').query('primary').sort_values('submission')
ids=list(c.figure_id); Y=d[ids].to_numpy(float); bad=d.severity.ge(2).to_numpy()
u=100*Y.mean(0); f=100*Y[~bad].mean(0); ij=np.triu_indices(len(ids),1)
beta,alpha=np.polyfit(u,f,1); residual=f-alpha-beta*u
affine={'alpha_pp':alpha,'beta':beta,'r_squared':1-np.sum(residual**2)/np.sum((f-f.mean())**2),'residual_rmse_pp':np.sqrt(np.mean(residual**2)),'residual_max_abs_pp':np.max(abs(residual)),'residual_median_abs_pp':np.median(abs(residual))}
pd.DataFrame({'configuration':ids,'original_score':u,'filtered_score':f,'affine_fitted':alpha+beta*u,'residual_pp':residual}).to_csv(O/'affine_scores.csv',index=False)
# Original score descending; submission ID ascending breaks exact ties for selection.
order=sorted(range(len(ids)),key=lambda j:(-u[j],c.iloc[j].submission))
top=set(order[:5]); adjacent={frozenset((a,b)) for a,b in zip(order[:-1],order[1:])}
groups={'All pairs':np.ones(len(ij[0]),bool),'Original top five':np.array([a in top and b in top for a,b in zip(*ij)]),'Original adjacent':np.array([frozenset((a,b)) in adjacent for a,b in zip(*ij)])}
gr=u[ij[0]]-u[ij[1]]; gf=f[ij[0]]-f[ij[1]]; shift=gf-gr
rows=[]
for name,mask in groups.items():
 rev=gr[mask]*gf[mask]<-1e-10
 rows.append({'comparison':name,'pairs':int(mask.sum()),'median_abs_change_pp':float(np.median(abs(shift[mask]))),'mean_abs_change_pp':float(np.mean(abs(shift[mask]))),'strict_reversals':int(rev.sum()),'margin_2pp_reversals':int(np.sum(rev & (np.maximum(abs(gr[mask]),abs(gf[mask]))>=2)))})
pd.DataFrame(rows).to_csv(O/'local_comparisons.csv',index=False)
sens=[]
def summary(name,A,B,n,k,weighted=False):
 met=metrics(A,B,ij)
 return {'definition':name,'original_tasks':n,'retained_tasks':k,**{key:float(met[key]) for key in ['median_abs_gap_pp','kendall_tau','reversals','practical_2pp_rate']}}
for name,flag in [('Primary severity >=2',bad),('Severity >=1',d.severity.ge(1).to_numpy()),('Severity =3',d.severity.eq(3).to_numpy())]:
 v=100*Y[~flag].mean(0);r=summary(name,u,v,len(Y),int((~flag).sum()));disc=Y[:,ij[0]]-Y[:,ij[1]]
 r['support_changes']=int(np.sum(states(bh(mcn_p(disc)),gr)!=states(bh(mcn_p(disc[~flag])),v[ij[0]]-v[ij[1]])))
 sens.append(r)
su=np.zeros(len(ids));sf=su.copy();support=0;retained=0;weights=[]
for _,ix in d.groupby(['repo','human_difficulty']).indices.items():
 good=ix[~bad[ix]]
 if len(good)==0 or bad[ix].sum()==0:continue
 su+=Y[ix].sum(0);sf+=Y[good].mean(0)*len(ix);support+=len(ix);retained+=len(good);weights.extend([len(ix)/len(good)]*len(good))
su*=100/support;sf*=100/support
r=summary('Repository-difficulty standardized',su,sf,support,retained);r['support_changes']=None;r['ess']=sum(weights)**2/sum(np.array(weights)**2);sens.append(r)
pd.DataFrame(sens).to_csv(O/'sensitivity.csv',index=False)
out={'affine':affine,'original_order':[ids[j] for j in order],'original_ties':len(set(u))<len(u),'local':rows,'sensitivity':sens}
(O/'additional_diagnostics.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
