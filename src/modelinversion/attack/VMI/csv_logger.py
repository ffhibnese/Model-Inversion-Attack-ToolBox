import csv
import matplotlib.pyplot as plt
import os
import pandas


class CSVLogger():
    def __init__(self, every, fieldnames, filename='log.csv', resume=False):
        # If resume, first check if file already exists
        if not os.path.exists(filename):
            resume = False  # if not, proceed as not resuming from anything

        self.every = every
        self.filename = filename
        self.csv_file = open(filename, 'a' if resume else 'w')
        self.fieldnames = fieldnames
        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        if not resume:
            self.writer.writeheader()
            self.csv_file.flush()

        if resume:
            df = pandas.read_csv(filename)
            if len(df['global_iteration'].values) == 0:
                self.time = 0
            else:
                self.time = df['global_iteration'].values[-1]
        else:
            self.time = 0

    def is_time(self):
        return self.time % self.every == 0

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()


def plot_csv(csv_path, fig_path):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        dict_of_lists = {}
        ks = None
        for i, r in enumerate(reader):
            if i == 0:
                for k in r:
                    dict_of_lists[k] = []
                ks = r
            else:
                for _i, v in enumerate(r):
                    try:
                        v = float(v)
                    except:
                        v = 0
                    dict_of_lists[ks[_i]].append(v)
    fig = plt.figure()
    for k in dict_of_lists:
        if k == 'global_iteration':
            continue
        plt.clf()
        plt.plot(dict_of_lists['global_iteration'], dict_of_lists[k])
        plt.title(k)
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(fig_path), f'_{k}.jpeg'), bbox_inches='tight', pad_inches=0, format='jpeg')
    plt.close(fig)
