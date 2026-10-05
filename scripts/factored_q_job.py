from scripts import probability_job as job
from scripts.run_observed_probability import ROOT, SOURCE
job.RUN = ROOT/'runs/factored_q_v1'
job.POLICY = SOURCE/'configs/factored_q_v1.json'
if __name__ == '__main__':
    job.main()
