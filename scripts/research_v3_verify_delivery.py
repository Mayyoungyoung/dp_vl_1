"""Read-only local checkpoint/table verification; no model forward or TEST access."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def main(workspace,output):
    base=Path(workspace).resolve();out=Path(output);out.mkdir(parents=True,exist_ok=False)
    table=base/'research_v3/EXPERIMENT_SUMMARY.csv'
    with table.open(newline='',encoding='utf-8') as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    hashes={str(table):sha(table)};verified=[]
    for r in rows:
        checkpoint=Path(r['checkpoint']);checkpoint=checkpoint if checkpoint.is_absolute() else base/checkpoint
        assert sha(checkpoint)==r['checkpoint_sha256'];hashes[str(checkpoint)]=r['checkpoint_sha256']
        root=checkpoint.parent;training=json.loads((root/'summary.json').read_text());config=json.loads((root/'config.json').read_text())
        assert training['steps']==int(r['additional_steps']) and training['sampler']['observation_draws']==int(r['input_draws'])
        assert sum(training['mode_exposure'].values())==int(r['target_slots'])
        assert config['support_sha256']==r['data_sha256'] and config['initial_checkpoint_sha256']==r['initial_checkpoint_sha256']
        matches=[]
        for p in root.glob('evaluation*/metrics.json'):
            m=json.loads(p.read_text())
            if abs(m['reliability']['brier']-float(r['brier']))<1e-10:matches.append((p,m))
        assert len(matches)==1,(r['experiment_id'],len(matches))
        p,m=matches[0];pool=p.parent/'pool.npz';assert sha(pool)==m['pool_sha256']
        assert m['generator_sha256']==r['checkpoint_sha256']
        values=dict(valid8=m['raw']['valid_fraction'],distinct8=m['raw']['distinct'],known_recall8=m['raw']['recall'],
            rare_recall8=m['rare_recall8'],distinct4=m['selected']['distinct'],top1_valid=m['reliability']['selected_valid'],
            all_valid4=m['selected']['all_valid'],brier=m['reliability']['brier'],nll=m['reliability']['nll'],
            ece=m['reliability']['reliability_bins_ece'],training_seconds=training['elapsed_seconds'])
        for k,v in values.items():assert abs(float(r[k])-v)<1e-10,(r['experiment_id'],k)
        with np.load(pool) as z:
            q=z['q'].astype(np.float64);y=z['labels'].astype(np.float64)
            assert q.shape==y.shape==(288,8) and np.all((q>=0)&(q<=1))
            assert abs(float(y.mean())-values['valid8'])<1e-10
            assert abs(float(np.mean((q-y)**2))-values['brier'])<1e-6
            assert abs(float(y[np.arange(len(y)),q.argmax(1)].mean())-values['top1_valid'])<1e-10
        for f in (p,pool,root/'summary.json',root/'config.json'):hashes[str(f)]=sha(f)
        verified.append(dict(experiment_id=r['experiment_id'],evaluation=p.parent.name,scorer_sha256=m['scorer_bundle_sha256'],checkpoint_sha256=r['checkpoint_sha256']))
        r.update(values);r['checkpoint']=str(checkpoint.resolve())
    with (out/'EXPERIMENT_SUMMARY_REBUILT.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    report=dict(verified_rows=verified,inputs_sha256=hashes,script_sha256=sha(Path(__file__)),command=[sys.executable]+sys.argv,
        rebuilt_table_sha256=sha(out/'EXPERIMENT_SUMMARY_REBUILT.csv'),
        scope='Verifies saved checkpoint bytes, recorded training exposure and all table metrics against sealed evaluation artifacts; recomputes validity/Brier/Top1 from stored pools. Does not rerun geometry labels, training or prove novelty/completion. Original and paired-domain q versions remain separate.')
    (out/'VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Verified and rebuilt',len(verified),'rows; no server experiment or model forward.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.workspace,a.output)
