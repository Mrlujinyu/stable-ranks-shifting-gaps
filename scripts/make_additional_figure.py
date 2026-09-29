from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
(R/'figures').mkdir(exist_ok=True)
d=pd.read_csv(R/'analysis_outputs/affine_scores.csv')
a=json.loads((R/'analysis_outputs/additional_diagnostics.json').read_text())['affine']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(6.2,3.8),layout='constrained')
x=np.array([7,58]);ax.plot(x,a['alpha_pp']+a['beta']*x,color='#387c93',lw=1.5,label='Descriptive affine fit')
ax.scatter(d.original_score,d.filtered_score,s=28,color='#183d55',zorder=3)
for r in d.itertuples():
 offsets={'A01':(10,-9),'A02':(-35,-3),'A03':(-35,8),'A04':(16,-7),'A05':(16,12),'A06':(-32,16),'A07':(-35,-14),'A08':(16,2),'A09':(5,-13),'A10':(-23,7),'A13':(4,4)}
 ax.annotate(r.configuration,(r.original_score,r.filtered_score),xytext=offsets.get(r.configuration,(5,5)),textcoords='offset points',fontsize=8,arrowprops={'arrowstyle':'-','color':'#84949c','lw':.5},bbox={'boxstyle':'square,pad=.1','facecolor':'white','edgecolor':'none','alpha':.9})
ax.set(xlabel='Original score (%) · 1,212 tasks',ylabel='Filtered score (%) · 412 tasks',xlim=(7,59),ylim=(15,86))
ax.text(.03,.96,'Slope 1.455  |  R² 0.980\nResidual RMSE 2.52 pp',transform=ax.transAxes,va='top',fontsize=9)
for ext in ['pdf','svg','png']:fig.savefig(R/'figures'/('affine_scores.'+ext),dpi=220,bbox_inches='tight')
