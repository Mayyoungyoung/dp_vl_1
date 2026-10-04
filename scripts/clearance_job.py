"""Isolated serial family using the existing immutable source recorder."""
from scripts import probability_job as job
from scripts.run_observed_probability import ROOT, SOURCE
job.RUN=ROOT/'runs/segment_clearance_v1'
job.POLICY=SOURCE/'configs/segment_clearance_v1.json'
if __name__=='__main__':job.main()
