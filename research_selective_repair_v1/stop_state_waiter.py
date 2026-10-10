"""Stop only the known own sleep-only queue whose FK data gate is already false."""
import os,signal,datetime
from pathlib import Path
from research_selective_repair_v1.io import ROOT,RUN,SOURCE,write

def run():
    pid=1051933
    target=str(ROOT/'research_v2/releases/9093da5e45dc4db972157d51ae3f9933cf3b8b1d/scripts/run_selective_body_prefix_v1.sh')
    args=(Path('/proc')/str(pid)/'cmdline').read_bytes().decode().split('\0')
    assert args[:2]==['bash',target] and len([v for v in args if v])==2
    assert not (RUN/'body_feedback_data_v3/MANIFEST.json').exists()
    assert not (RUN/'body_prefix_data_v1').exists()
    children=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            fields=(p/'status').read_text().splitlines();parent=int(next(v for v in fields if v.startswith('PPid:')).split()[1])
            if parent==pid:
                child=(p/'cmdline').read_bytes().decode().split('\0');assert child[0]=='sleep';children.append(dict(pid=int(p.name),args=child))
        except (FileNotFoundError,ProcessLookupError):pass
    assert children,'Expected a prelaunch sleep,never a learner or collector child'
    os.kill(pid,signal.SIGTERM)
    record=dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=SOURCE.name,target_pid=pid,
        verified_exact_own_queue_args=args,verified_children=children,action='SIGTERM sleep-only prelaunch coordinator,not any experiment/collector',
        reason='Allgoal subset FK max5.803mm already exceeds unchanged5mm state-model gate; full superset cannot pass. Replace pending state fitting with actual-event data model; preserve failed diagnostics.',
        replacement='scripts/run_selective_body_event_full_v1.sh',no_fullscope_fit_started=True,locked_access=False)
    out=RUN/'technical/fullscope_state_prelaunch_stop_v1';out.mkdir(parents=True,exist_ok=False);write(out/'receipt.json',record);print(record,flush=True)

if __name__=='__main__':run()
