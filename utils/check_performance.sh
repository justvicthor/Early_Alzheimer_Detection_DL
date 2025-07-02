#!/bin/bash -l

# Job ID to analyse
JOB_ID=3306101

# Check Performance
sacct -j $JOB_ID --format=JobID,Elapsed,TotalCPU,AveRSS,MaxRSS,NCPU,NNodes,NTasks