# Simple ML Pipeline

Demonstrates machine learning pipeline for wrangling, training, predicting, and evaluating.

# Vision

Real world data science projects are messy and confusing. I want my own personal version of the [Kaggle](https://www.kaggle.com/) Platform so that I can cleanly separate tasks from learners. I want a way to precisely define the machine learning tasks I'm interested in, and then spend the rest of my time prototyping different learning algorithms without living in fear that somewhere the test data has leaked into the train data.

# Dependencies

* Python3
* Unix like environment with `grep` and the [Graphvis](http://www.graphviz.org/) `dot` command on path

Python Libraries:

* [ruffus](http://www.ruffus.org.uk/)
* [numpy](http://www.numpy.org/)
* [pandas](http://pandas.pydata.org/)
* [scikit-learn](http://scikit-learn.org/)

To run the notebooks you'll also need:
* [Jupyter](http://jupyter.org/)
* [matplotlib](https://matplotlib.org/)

To install all dependencies on Ubuntu:
```
sudo apt-get install graphviz
# Download and install conda for Python3 from: http://conda.pydata.org/miniconda.html
conda create -n pipeline python=3.11.5
conda activate pipeline
pip install -r requirements.txt
```

To install dependencies on macOS:
```
brew install graphviz
# Download and install conda for Python3 from: http://conda.pydata.org/miniconda.html
conda create -n pipeline python=3.11.5
conda activate pipeline
pip install -r requirements.txt
```

# Usage

`./traffic_pipeline.py`

The pipeline will download all data from the web (currently an archive hosted via AWS). It will then transform it, run a few models, and summarize the results in `data/traveltime.ladder`.

The first run may take a few minutes, so please be patient. Subsequent runs will be much quicker, as each step of the pipeline only re-runs if its input changes.

# Defining the machine learning task

Modify `pipeline/task_traveltime.py`

* `.task.train` - Train data. Columns: id, *features...*, y
* `.task.test.xs` - Test data questions. Columns: id, *features...*
* `.task.test` - Test data solutions. Columns: id, *features...*, y

# Adding a learner

Copy `learners/traveltime_baserate.py` to `learners/traveltime_yourlearner.py`

* `train(train_data_file, model_file)` - Learn model, and write to `model_file`
* `predict(model_file, test_xs_file, output_file)` - Apply model to predict test data, and write to `output_file`

The pipeline searches in the `learners` directory for learners, and loads them at runtime. This is a case of *convention over configuration*, so be sure to follow the naming conventions.

If uncertain about the inputs, I suggest running the pipeline and having a look at the data files it generates. The results of each step are CSVs, which you can open in a spreadsheet tool such as LibreOffice Calc (rename the data file extension to `.csv` if you are having issues opening them).

------------------------------------------

# Background

Many data processing problems naturally lend themselves to a data pipeline architecture. In the early stages of a project I often find myself manually performing the steps of a pipeline. There will be a series of steps where each processing module takes an input CSV file, processes it in some way, and then outputs a processed CSV file that I feed as input to the next processing module.

When there is only a few processing modules, this is great, but for more complex pipelines this manual approach becomes a nightmare - if you modify one of the input files, you need to remember to re-run the processing modules that depend on that input file, and then all the processing modules that depend on the output of those processing modules, and so on. You could write a shell script to automatically run all of the steps in the pipeline. However, if some of the processing modules are slow, it becomes tedious to wait for the entire pipeline to run every time you want to test out a small change.

Enter [Ruffus](http://www.ruffus.org.uk/), a lightweight computation pipeline library for Python, initially created for bioinformatics scientists. Ruffus lets you write each step of the pipeline as a Python function, and to use Python decorators (used similarly to Java annotations) to specify their data dependencies. Ruffus is smart, it takes care of running all the processing steps, and will only re-run a processing step if the input to that step has changed since the last run. Because Ruffus understands the dependencies of each step, it can use this information to create a data flow diagram of the entire pipeline. Self-documenting pipeline, yay!

The first stage of machine learning, data wrangling, can be performed as a fairly unimaginative sequential pipeline in which each step slightly refines the last until eventually the data set is structured the way you like it. However, there are some aspects of machine learning that deserve special consideration when designing the pipeline.

In machine learning, the one big rule is to keep test data separate from train data. This is so we can test that the algorithm has really learned in a generalizable way rather than simply memorized the answers. In the machine learning example I have provided, the task splits the data into separate train and test data, and hides the test answers from the machine learning algorithm so that you can be confident that there is no cheating.

We also need a way to quantitatively compare different learning algorithms; data *science*, not data guessing. That's why I've made it as easy as possible to add multiple learners that compete against the others. The results are aggregated in a table and sorted by their test score, so that you always know which learner performs best.

# Scenario

Rather than using one of the classic machine learning datasets, I've decided to demonstrate with road traffic prediction. This better demonstrates some data wrangling inevitably involved in any real machine learning problem.

The task is to predict the travel time along a stretch of highway.

The data is ('are' for the pedantic) for an Australian highway. The data is collected by the state road agency, VicRoads, and released as [Open Data](https://vicroads-public.sharepoint.com/InformationAccess/SitePages/Home.aspx) under the [Creative Commons Attribution 4.0 International Licence](http://creativecommons.org/licenses/by/4.0/).

Inductive-loop traffic detectors are installed in the stop lines of each lane at major intersections. These count the number of vehicles that pass over them, but cannot directly measure speed.

VicRoads has also installed a small number of Bluetooth detectors. These record the Bluetooth MAC address (obfusicated for privacy) of any vehicles with Bluetooth devices whenever they pass one of the Bluetooth detector sites. When the same Bluetooth device passes by multiple detector sites, we can examine the time difference of the detections to infer the travel time between those sites.

![Map](map.png?raw=true)

The map shows Bluetooth detectors at the start (site 2409) and end (site 2425) of the stretch of highway we are interested in. We will use detector data from site 2433 as a feature to aid our prediction.

We train the learning algorithms on all the data from Wednesday 19 August, and use this to predict travel times 1 week later for Wednesday 26 August given the time and volume readings.

# Pipeline

![Flowchart](flowchart.png?raw=true)

At the top of the flowchart, we can see a directed acyclic graph of data merging and transformations as we massage the raw data into a more convenient format. It is the duty of the task definition to splits the data up into a train and test set. We then ask each learner to generate a model from the train data, test the learned model on the test data, then aggregate the results into a ladder of which model performed best. Currently we report Root-Mean-Squared (RMS) error - lower scores are better.

# Final Comments

This project doesn't attempt to help you design machine learning models. It simply provides a framework for testing them.

There are some [Jupyter](http://jupyter.org/) notebooks in `jupyter_notebooks` that describe the design of each learner. My advice is to start by mucking around with the data in a notebook. Once you have a learner that looks promising, copy the code into a new learner module, run the pipeline, and check the score ladder to see if you've managed to make an improvement. There's no computational cost to leaving old learners around as a baseline - they will only re-run if the task changes (which should be rare).
