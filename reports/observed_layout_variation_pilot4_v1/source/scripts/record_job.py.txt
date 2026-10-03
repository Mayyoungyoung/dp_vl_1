"""Record one already-authorized subprocess, its log, PID, exit and resume command."""
import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--resume-strategy',choices=('flag','fresh-output','none'),default='flag')
    p.add_argument('command',nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1]==['--'] else a.command
    if not command: p.error('subprocess command is required')
    out = Path(a.output); out.mkdir(parents=True,exist_ok=True)
    lock = out/(a.run_id+'.lock')
    fd = os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode()); os.close(fd)
    state = {'run_id':a.run_id,'pid':os.getpid(),'host':'wzy3090','command':command,
             'resume_command':command+['--resume'] if a.resume_strategy=='flag' else None,
             'resume_strategy':a.resume_strategy,
             'resume_note':('Read-only rerun: use the recorded command with a fresh --output; --resume is unsupported.'
                            if a.resume_strategy=='fresh-output' else None),
             'cwd':os.getcwd(),'log':str(out/(a.run_id+'.log')),
             'code_commit':os.environ.get('CODE_COMMIT'),'gpu_uuid':os.environ.get('RESEARCH_GPU_UUID'),
             'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'starting'}
    path = out/(a.run_id+'.status.json')
    def save(): path.write_text(json.dumps(state,indent=2))
    save()
    try:
        with open(state['log'],'a',encoding='utf-8') as log:
            child = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            state.update(child_pid=child.pid,status='running'); save()
            code = child.wait()
        state.update(exit_code=code,status='completed' if code==0 else 'failed',
                     end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save()
        with open(out/'registry.jsonl','a') as f: f.write(json.dumps(state)+'\n')
    finally:
        lock.unlink(missing_ok=True)
    sys.exit(code)


if __name__=='__main__': main()
