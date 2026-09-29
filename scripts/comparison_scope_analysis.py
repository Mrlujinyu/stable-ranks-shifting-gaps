"""Fixed comparison scopes, shared resampling, and common-support comparisons."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy import stats
from statistics_core import bootstrap_scores,gaps,metrics,bh,mcn_p,states,B,SEED
R=Path(__file__).resolve().parents[1];O=R/'analysis_outputs';O.mkdir(exist_ok=True);D=R/'data'
d=pd.read_csv(D/'main_panel.csv',float_precision='round_trip');c=pd.read_csv(D/'table2_configurations.csv').query('primary').sort_values('submission')
ids=list(c.figure_id);Y=d[ids].to_numpy(float);bad=d.severity.ge(2).to_numpy();n,m=Y.shape;ij=np.triu_indices(m,1)
u=Y.mean(0)*100;f=Y[~bad].mean(0)*100;gr=gaps(u,ij);gf=gaps(f,ij);dg=gf-gr
order=sorted(range(m),key=lambda j:(-u[j],c.iloc[j].submission));top=set(order[:5]);adj={frozenset((i,j)) for i,j in zip(order[:-1],order[1:])}
scopes={'All pairs':np.ones(len(gr),bool),'Original top five':np.array([i in top and j in top for i,j in zip(*ij)]),'Original adjacent':np.array([frozenset((i,j)) in adj for i,j in zip(*ij)])}
beta,alpha=np.polyfit(u,f,1);eps=f-alpha-beta*u;L=(beta-1)*gr;E=gaps(eps,ij);assert np.allclose(dg,L+E,atol=1e-12)
# No archived subset-index cache exists. Reconstruct the original PCG64 stream:
# 10,000 multinomial bootstrap draws in chunks of 250, then uniform subsets.
rng=np.random.default_rng(SEED);br,bf=bootstrap_scores(Y,bad,rng);bd=gaps(bf,ij)-gaps(br,ij)
indices=np.empty((B,(~bad).sum()),dtype=np.int16);ns=[]
for start in range(0,B,200):
 w=np.zeros((min(200,B-start),n))
 for k,row in enumerate(w):
  ix=rng.choice(n,(~bad).sum(),replace=False);indices[start+k]=ix;row[ix]=1
 ns.append(w@Y/(~bad).sum()*100)
ns=np.concatenate(ns);nd=gaps(ns,ij)-gr
saved=pd.read_csv(D/'random_removal.csv',float_precision='round_trip')
null_error=float(np.max(abs(np.median(abs(nd),axis=1)-saved.median_abs_gap_pp.to_numpy())))
score_columns=['delta_pp__'+x for x in c.submission]
score_error=float(np.max(abs((ns-u)-saved[score_columns].to_numpy()))) if all(x in saved for x in score_columns) else None
assert null_error<1e-10 and (score_error is None or score_error<1e-10)
np.savez_compressed(O/'shared_random_subsets.npz',indices=indices,task_ids=d.task_id.to_numpy(dtype=str))
rows=[]
for name,mask in scopes.items():
 obs=np.median(abs(dg[mask]));null=np.median(abs(nd[:,mask]),axis=1);boot=np.median(abs(bd[:,mask]),axis=1);rev=gr[mask]*gf[mask]<-1e-10
 rows.append({'scope':name,'pairs':int(mask.sum()),'median_delta':obs,'ci_low':np.quantile(boot,.025),'ci_high':np.quantile(boot,.975),'random95':np.quantile(null,.95),'p_mc':(1+(null>=obs).sum())/(B+1),'strict':int(rev.sum()),'margin':int(np.sum(rev&(np.maximum(abs(gr[mask]),abs(gf[mask]))>=2))),'median_scale':np.median(abs(L[mask])),'median_residual':np.median(abs(E[mask])),'opposite_sign_components':int(np.sum(L[mask]*E[mask]<-1e-10))})
pd.DataFrame(rows).to_csv(O/'scope_inference.csv',index=False)
pd.DataFrame({'a':np.array(ids)[ij[0]],'b':np.array(ids)[ij[1]],'delta_pp':dg,'scale_pp':L,'residual_pp':E}).to_csv(O/'pairwise_affine_components.csv',index=False)
support=np.zeros(n,bool);weighted=np.zeros(m);ws=[]
for _,ix in d.groupby(['repo','human_difficulty']).indices.items():
 good=ix[~bad[ix]]
 if len(good)==0 or bad[ix].sum()==0:continue
 support[ix]=True;weighted+=Y[good].mean(0)*len(ix);ws.extend([len(ix)/len(good)]*len(good))
uc=Y[support].mean(0)*100;fc=Y[support&~bad].mean(0)*100;weighted*=100/support.sum();pop=[]
for name,A,C,sel,w in [('Full-panel unweighted',u,f,np.ones(n,bool),False),('Common-support unweighted',uc,fc,support,False),('Common-support standardized',uc,weighted,support,True)]:
 met=metrics(A,C,ij);row={'population':name,'n_reference':int(sel.sum()),'n_retained':int((sel&~bad).sum()),'median_delta':float(met['median_abs_gap_pp']),'tau':float(met['kendall_tau']),'strict':int(met['reversals']),'margin':int(round(float(met['practical_2pp_rate'])*91)),'support_changes':None}
 if not w:
  disc=Y[:,ij[0]]-Y[:,ij[1]];row['support_changes']=int(np.sum(states(bh(mcn_p(disc[sel])),gaps(A,ij))!=states(bh(mcn_p(disc[sel&~bad])),gaps(C,ij))))
 pop.append(row)
pd.DataFrame(pop).to_csv(O/'support_comparisons.csv',index=False)
gu=gaps(uc,ij);gfc=gaps(fc,ij);gw=gaps(weighted,ij);revw=gu*gw<-1e-10
reversals=[]
for k in np.where(revw)[0]:
 reversals.append({'a':ids[ij[0][k]],'b':ids[ij[1][k]],'reference_gap':gu[k],'unweighted_filtered_gap':gfc[k],'standardized_filtered_gap':gw[k],'unweighted_reversal':bool(gu[k]*gfc[k]<-1e-10),'unweighted_margin':bool(gu[k]*gfc[k]<-1e-10 and max(abs(gu[k]),abs(gfc[k]))>=2),'standardized_reversal':True,'standardized_margin':bool(max(abs(gu[k]),abs(gw[k]))>=2)})
pd.DataFrame(reversals).to_csv(O/'standardized_reversal_pairs.csv',index=False)
ia=ids.index('A11');ib=ids.index('A10');contrast=(bf[:,ia]-bf[:,ib])-(br[:,ia]-br[:,ib]);example={'orientation':'A11-A10','original_gap':u[ia]-u[ib],'filtered_gap':f[ia]-f[ib],'delta':float((f[ia]-f[ib])-(u[ia]-u[ib])),'ci':np.quantile(contrast,[.025,.975]).tolist()}
result={'scopes':rows,'populations':pop,'reversal_pairs':reversals,'example':example,'affine':{'alpha':alpha,'beta':beta,'r2':1-np.sum(eps**2)/np.sum((f-f.mean())**2),'rmse':np.sqrt(np.mean(eps**2))},'identity_max_error':float(np.max(abs(dg-L-E))),'random_reconstruction_max_error':null_error,'random_score_reconstruction_max_error':score_error,'ess':sum(ws)**2/sum(np.array(ws)**2),'seed':SEED,'rng':'numpy PCG64','draws':B}
(O/'scope_analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
