from simglucose.analysis.report import report
import pandas as pd
import os
import glob

# the path where results are saved
path = os.path.join(os.path.dirname(__file__),
                    'results', '2017-12-31_17-46-32')
os.chdir(path)
# find all csv with pattern *#*.csv, e.g. adolescent#001.csv
filename = glob.glob('*#*.csv')
name = [_f[:-4] for _f in filename]   # get the filename without extension
df = pd.concat([pd.read_csv(f, index_col=0) for f in filename], keys=name)
report(df)
