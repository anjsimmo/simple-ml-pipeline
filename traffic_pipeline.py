#!/usr/bin/env python3
from ruffus import *
# Setup pipeline, resolve dependencies
import pipeline.run

pipeline_printout_graph("data/flowchart.svg")
pipeline_run(target_tasks = [pipeline.run.comp_ladder])
