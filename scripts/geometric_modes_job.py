"""Reuse the established immutable source/command/budget job recorder."""
from scripts import probability_job as job
from scripts.run_observed_probability import ROOT, SOURCE

job.RUN = ROOT/'runs/geometric_modes_v1'
# Existing numeric resource limits; this family receives a bounded 3600s cap.
original_read = job.read
def read(path):
    value = original_read(path)
    if path == job.POLICY:
        value = dict(value, gpu_wall_budget_seconds=3600)
    return value
job.read = read
if __name__ == '__main__': job.main()
