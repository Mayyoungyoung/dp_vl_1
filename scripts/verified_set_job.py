"""Separate explicitly authorized additional budget; historical ledger intact."""
from scripts import probability_job as job
job.RUN = job.ROOT/'runs/verified_set_v1'
job.POLICY = job.SOURCE/'configs/verified_set_v1.json'
if __name__ == '__main__':
    job.main()
