"""Reuse the serial source-indexed budget wrapper in a new run family."""
from scripts import probability_job as job
job.RUN = job.ROOT/'runs/research_v3_v1'
job.POLICY = job.SOURCE/'configs/research_v3_v1.json'
if __name__ == '__main__':
    job.main()
