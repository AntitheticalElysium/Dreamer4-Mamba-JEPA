"""Scheduling-only handoff: finish the existing C trainer, then review-gate seed8.

The old coordinator alone is stopped with SIGSTOP; its C child continues. When
that child exits, verify its immutable completed checkpoint, replace the stopped
coordinator, and resume only the user-authorized seed7 endpoint schedule. No
optimizer, training/evaluation source, dataset, threshold or budget changes.
"""
import json
import os
import subprocess
import time
from pathlib import Path

from e20_driver import OUT, W, sha, write


def main():
    original=331137
    final=W/'e20_C_s7_fmamba_fromM16.pt'
    state=OUT/'states/e20_C_s7_fmamba_fromM16'
    while True:
        proc=Path(f'/proc/{original}/stat')
        exited=not proc.exists()or proc.read_text().rsplit(')',1)[1].split()[0]=='Z'
        if exited:break
        progress=json.loads((state/'train.json').read_text())
        write(OUT/'handoff_live.json',{'state':'finishing_existing_seed7_C','trainer_pid':original,
              'last_committed_state':progress['file'],'seed8':'held','unix':time.time()})
        time.sleep(10)
    if not (state/'complete.json').exists()or not final.exists():
        raise RuntimeError('C exited without a completed checkpoint; handoff held')
    receipt=json.loads((state/'complete.json').read_text())
    evidence=state/receipt['file']
    if sha(evidence)!=receipt['sha256']:raise RuntimeError('C completion record corrupt')
    import torch
    completed=torch.load(evidence,map_location='cpu',weights_only=False)
    if sha(final)!=completed['sha256']:raise RuntimeError('C final checkpoint differs')
    commands=[['systemctl','--user','kill','--kill-whom=main','--signal=SIGKILL','d4mj-e20.service'],
              ['systemctl','--user','reset-failed','d4mj-e20.service'],
              ['systemctl','--user','start','d4mj-e20.service'],
              ['systemctl','--user','start','d4mj-e20-post.service']]
    for command in commands:
        subprocess.run(command,check=True)
        time.sleep(1)
    write(OUT/'handoff_live.json',{'state':'seed7_endpoints_resumed','seed8':'held',
                                 'completed_C_sha256':completed['sha256'],'unix':time.time()})


if __name__=='__main__':main()
