"""Reuse existing job recording and resource budget."""
from scripts import probability_job as job
from scripts.run_observed_probability import ROOT, SOURCE
job.RUN = ROOT/'runs/goal_preserving_v1'
job.POLICY = SOURCE/'configs/goal_preserving_v1.json'
if __name__ == '__main__': job.main()
