"""Serial paired-modes jobs within the unchanged project resource envelope."""
from scripts import probability_job as job
from scripts.paired_modes_data import RUN, POLICY
job.RUN=RUN
job.POLICY=POLICY
if __name__=='__main__':job.main()
