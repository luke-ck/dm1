"""
Homework 1: Distance functions on vectors
Course    : Data Mining (636-0018-00L)

Main program.
This file is provided to you as part of the assignment. You can use it with
your submission without restrictions. Your remaining task consists in 
creating the script distance_fn.py that contains the implementation of all
distance functions invoked by this program.

Read the input data for the homework and organize it in the following way:
   dict_doc [ : ] : a dictionary of lists of vectors.
              -> key = group name
             
For example, if there are 10 vectors (i.e. documents) in group 'comp.graphics' 
then dict_doc['comp.graphics'] is a list [v1, v2, ..., v10] of numpy arrays 
read from the input directory.

Compute all pairwise distances of vectors within and between groups.
"""
# Author: Damian Roqueiro <damian.roqueiro@bsse.ethz.ch>
# September 2015


import argparse
import os
import numpy as np
# TODO Create the file distance_fn.py with the distance metrics
import distance_fn as d

DIST_METRICS = {"manhattan": d.manhattan_dist,
                "hamming": d.hamming_dist,
                "euclidean": d.euclidean_dist,
                "chebyshev": d.chebyshev_dist,
                "minkowski_d3": lambda v1, v2: d.minkowski_dist(v1, v2, 3),
                "minkowski_d4": lambda v1, v2: d.minkowski_dist(v1, v2, 4)}


def obtain_pairwise_distances(lst_vec1, lst_vec2, num_comparisons):
    """
    Function obtain_pairwise_distances
    It receives two lists of vectors and computes all distance metrics between
    the vectors in the two lists.

    Returns a vector with average distances per metric. The positions in the
    vector match the metric names in DIST_METRICS.
    """
    avg_dist = np.zeros(len(DIST_METRICS), dtype=float)
    # Iterate through all metrics
    for i, metric in enumerate(DIST_METRICS):
        # Iterate through all combinations of pairs of vectors
        # Note: the pair (v1, v2) is the same as (v2, v1)
        for idx_g1 in range(len(lst_vec1)):
            for idx_g2 in range(len(lst_vec2)):
                v1 = lst_vec1[idx_g1]
                v2 = lst_vec2[idx_g2]
                # Determine which metric to compute by indexing dictionary
                # Note: Could be alternatively implemented using an if-elif-else chain
                dist = DIST_METRICS[metric](v1, v2)

                # Accumulate the distance values. The average is computed after the loop
                avg_dist[i] = avg_dist[i] + dist

    # Compute the average. Divide by the number of comparisons
    avg_dist = avg_dist / num_comparisons

    return avg_dist


#------------------------------------------------------------------------------
# Main program
#------------------------------------------------------------------------------

# Set up the parsing of command-line arguments
parser = argparse.ArgumentParser(description="Compute distance functions on vectors")
parser.add_argument("--datadir", required=True, 
                    help="Path to input directory containing the vectorized data")
parser.add_argument("--info", required=False,
                    default="file_info.txt",
                    help="Name of file containing information about documents (name and label)")
parser.add_argument("--outdir", required=True, 
                    help="Path to the output directory, where the output file will be created")
args = parser.parse_args()

# Read the info file with details of the documents to process
file_name = "{}/{}".format(args.datadir, args.info)
with open(file_name, 'r') as f_in:
    # Create a dictionary of lists. Key to the dictionary is the group name
    dict_doc = {}
    for line in f_in:
        # Remove the trailing newline and separate the fields
        parts = line.rstrip().split("\t")

        # If the group does not exist in the dictionary, create it
        if not parts[1] in dict_doc:
            # Use the group name as key. Initialize list
            dict_doc[parts[1]] = []

        # Read the vector, add it to the group list
        dict_doc[parts[1]].append(np.loadtxt("{}/{}_vec.txt".format(args.datadir, parts[0])))

# If the output directory does not exist, then create it
if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

# Create the output file and fill it with content as required by the specifications
file_name = "{}/output_distances.txt".format(args.outdir)
with open(file_name, 'w') as f_out:
    # Create a list with the group names
    lst_groups = list(dict_doc.keys())

    # Iterate through all combinations of pairs
    for idx_g1 in range(len(lst_groups)):
        for idx_g2 in range(idx_g1, len(lst_groups)):
            group1 = lst_groups[idx_g1]
            group2 = lst_groups[idx_g2]

            # Compute the average distances between all members of both groups
            # The return vector will have one distance per metric analyzed
            # Note: When the distances are within the same group, the number of
            #       comparisons should be decreased:
            num_comparisons = len(dict_doc[group1]) * len(dict_doc[group2])
            if group1 == group2:
                # Don't count comparisons of a document to itself
                #   dist(x1, x2) = 0 when x1 = x2
                num_comparisons -= len(dict_doc[group1])

            # Compute all (average) distances between documents in the two groups
            vec_dist = obtain_pairwise_distances(dict_doc[group1], dict_doc[group2], num_comparisons)

            # Transform the vector of distances to a string
            str_dist = "\t".join("{0:.2f}".format(x) for x in vec_dist)

            # Save the output
            f_out.write("{}:{}\t{}\n".format(group1, group2, str_dist))
