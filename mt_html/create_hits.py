""" Usage:
    <file-name> --html=HTML_FILE --csv=CSV_FILE --out=OUTPUT_FOLDER [--num_hits=NUM_HITS] [--debug]

Instansiate htmls file with an input csv. 
"""
# External imports
import logging
import pdb
from pprint import pprint
from pprint import pformat
from docopt import docopt
from collections import defaultdict
from operator import itemgetter
from tqdm import tqdm
import pandas as pd
import os

# Local imports

#=-----

if __name__ == "__main__":
    # Parse command line arguments
    args = docopt(__doc__)
    html_fn = args["--html"]
    csv_fn = args["--csv"]
    out_fn = args["--out"]
    num_hits = args["--num_hits"]
    debug = args["--debug"]

    if debug:
        logging.basicConfig(level = logging.DEBUG)
    else:
        logging.basicConfig(level = logging.INFO)

    if not num_hits:
        num_hits = 999999999
    else:
        num_hits = int(num_hits)

    df = pd.read_csv(csv_fn, sep = ",", header = 0)
    template_html = open(html_fn).read()
    keys = df.keys()

    for hit_num, (hit_index, row) in enumerate(df.iterrows()):
        if hit_num > num_hits:
            break
            
        hit_fn = os.path.join(out_fn, "{}.html".format(hit_index))
        logging.info("Writing {}".format(hit_fn))
        with open(hit_fn, "w") as fout:
            out_str = template_html
            for key in keys:
                out_str = out_str.replace("${{{}}}".format(key),
                                          str(row[key]))
            fout.write(out_str)

    logging.info("DONE")
