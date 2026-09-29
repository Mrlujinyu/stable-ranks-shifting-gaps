from pathlib import Path
import json,hashlib,shutil,sys,platform
import pandas as pd
import numpy as np
R=Path(__file__).resolve().parents[1]; OLD=R.parent/'benchmark-audit-triage'; O=R/'results'
def dump(name,obj): (O/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
def main():
    files=['results/dataset_a_labels.parquet','results/outcome_matrix.parquet','results/submission_manifest.parquet','results/task_metadata.parquet','results/dataset_b_labels.parquet','results/pilot_manifest.json','sources/download_manifest.json','sources/pro_readme.md']
    hashes={f:hashlib.sha256((OLD/f).read_bytes()).hexdigest() for f in files}; dump('input_checksums.json',hashes)
    d=pd.read_parquet(OLD/files[0]); d['severity']=d[['underspecified','false_negative']].max(axis=1).astype(int)
    d['other_problem']=d.other_major_issues.eq(1); d['human_difficulty']=d.difficulty
    d=d.merge(pd.read_parquet(OLD/'results/task_metadata.parquet').drop(columns='repo'),on='task_id',validate='one_to_one')
    m=pd.read_parquet(OLD/'results/outcome_matrix.parquet'); am=pd.read_parquet(OLD/'results/submission_manifest.parquet')
    am['primary']=am.excluded_reason.eq(''); am['included']=am.primary | am.excluded_reason.eq('Not documented pass@1')
    am['attempt_policy']=np.where(am.primary,'1',np.where(am.included,'2+','excluded'))
    am['configuration_identity']=am.submission
    extra=[]
    base=pd.read_parquet(OLD/'sources/swe_test.parquet').instance_id.tolist()
    for sid in am.loc[am.included & ~am.primary,'submission']:
        p=OLD/'sources/experiments/evaluation/test'/sid/'results/results.json'; x=json.loads(p.read_text()); info=am.set_index('submission').loc[sid]
        missing=set().union(*(set(x.get(k,[])) for k in ['no_generation','no_logs','install_fail','reset_failed']))
        observed=set(x.get('with_logs',base))-missing; solved=set(x['resolved'])
        for t in base: extra.append(dict(dataset='A',task_id=t,submission=sid,resolved=int(t in solved) if t in observed else None,outcome_status='observed' if t in observed else 'missing',source_schema='public_results',model=info.model,scaffold=info.scaffold,family=info.family,date=info.date))
    m=pd.concat([m,pd.DataFrame(extra)],ignore_index=True); m['resolved']=m.resolved.astype('Int8')
    a=m[m.dataset.eq('A')]; primary=am.loc[am.primary,'submission'].tolist()
    mat=a[a.submission.isin(primary)].pivot(index='task_id',columns='submission',values='resolved').reindex(d.task_id)
    oldprimary=json.loads((OLD/'results/pilot_manifest.json').read_text())['submissions']
    eligible=mat[oldprimary].notna().sum(axis=1).ge(6)
    d['pilot_eligible']=d.task_id.isin(mat.index[eligible]); d['complete_primary']=d.task_id.isin(mat.dropna().index)
    m.to_parquet(O/'agent_matrix.parquet',index=False); d.to_parquet(O/'task_quality_labels.parquet',index=False); am.to_parquet(O/'submission_metadata.parquet',index=False)
    summary=dict(annotated_tasks=len(d),pilot_tasks=int(d.pilot_eligible.sum()),complete_tasks=int(d.complete_primary.sum()),primary_agents=len(primary),expanded_agents=int(am.included.sum()),repositories=int(d[d.complete_primary].repo.nunique()),missing_outcomes_primary=int(mat.isna().sum().sum()))
    summary['pass']=summary['complete_tasks']>=1200 and len(primary)>=10 and summary['repositories']>=8
    dump('data_gate.json',summary); print(json.dumps(summary,indent=2))
    dump('environment.json',dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__))
    (R/'reports/data_summary.md').write_text('# Data audit\n\n'+json.dumps(summary,indent=2)+'\n\nOfficial Test directory rechecked 2026-09-19: 24 entries, no newer full-Test submissions. 14 primary pass@1 configurations plus 2 documented multi-attempt configurations (SAGE and Amazon Q v20250405) = 16 independent versioned configurations. RAG baselines, withdrawn Honeycomb, and one SWE-agent metadata/date conflict remain excluded. Multiple versions of Amazon Q are configurations, not independent research teams. Shared models/harnesses mean statistical independence across agents is NOT assumed.\n\nAll 1699 labels retained. Pilot evaluates all 1693 old-eligible tasks using pairwise available outcomes; complete-case primary leaderboard has 1212 tasks. Missing observations are never silently failures. Complete-case selection and all-unobserved-as-failure sensitivity are reported separately. Model dates and opaque proprietary configurations limit family attribution.\n\nSeverity = max(underspecified, false_negative), both original ordinal 0–3 fields. Other-major-issues is a separate binary label, not fabricated ordinal severity. All-defects = severity >= 2 OR other_problem; primary = severity >= 2. Severity >=1 includes mild ambiguities, not necessarily invalid tasks. Human difficulty retains original four effort categories. Metadata: issue character length, gold patch line count, FAIL_TO_PASS count. Labels are assessments, not proof of invalidity or counterfactual corrected outcomes.\n\n'+am[['submission','model','scaffold','attempt_policy','included','excluded_reason']].to_markdown(index=False),encoding='utf-8')
if __name__=='__main__': main()
