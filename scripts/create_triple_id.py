#!/usr/bin/python

import sys
import argparse
import pickle
import os
import random


def get_all_element_ids(fbpath):
    # element = set()
    element = {}
    with open(fbpath, 'r') as f:
        for i, line in enumerate(f):
            if i % 1000000 == 0:
                print("line: {}".format(i))

            if i == 0:
                continue    # skip the first line

            items = line.strip().split("\t")
            if len(items) != 2:
                print("ERROR: line - {}".format(line))
            
            element[items[0]] = items[1]
            # element.add(items[0])
            # ele_list = list(element)
            # ele_list.sort()

    return element


def create_id(fbsubsetpath, outfolder):
    print("creating ids for triples...")
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)

    # entity_order_list = get_all_element_ids(outfolder+'entity2id.txt')
    # relation_order_list = get_all_element_ids(outfolder+'relation2id.txt')
    entity_dict = get_all_element_ids(outfolder+'/entity2id.txt')
    relation_dict = get_all_element_ids(outfolder+'/relation2id.txt')
    triple_set = set()

    with open(fbsubsetpath, 'r') as f:
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print("line: {}".format(i))

            items = line.strip().split("\t")
            if len(items) != 3:
                print("ERROR: line - {}".format(line))
            e1 = items[0].split('www.freebase.com')[-1]
            e2 = items[2].split('www.freebase.com')[-1]
            r = items[1].split('www.freebase.com')[-1]
            # TODO: check the manual correction in util.www2fb

            id_e1 = entity_dict[e1]
            id_e2 = entity_dict[e2]
            id_r = relation_dict[r]
            triple_set.add((id_e1, id_e2, id_r))     # to fit the required order in OpenKE

    outfile = open(outfolder+'/fb2m2id.txt', 'w')
    outfile_train = open(outfolder+'/train2id.txt', 'w')
    outfile_val = open(outfolder+'/valid2id.txt', 'w')
    outfile_test = open(outfolder+'/test2id.txt', 'w')

    triple_set = list(triple_set)
    num_triples = len(triple_set)
    index_list = list(range(num_triples))
    random.shuffle(index_list)
    split = [int(num_triples*0.7), int(num_triples*0.8)]

    outfile.write("{}\n".format(len(triple_set)))
    outfile_train.write("{}\n".format(split[0]))
    outfile_val.write("{}\n".format(split[1] - split[0]))
    outfile_test.write("{}\n".format(num_triples - split[1]))

    for k in range(0, split[0]):
        triple = triple_set[index_list[k]]
        outfile.write("{}\t{}\t{}\n".format(triple[0], triple[1], triple[2]))
        outfile_train.write("{}\t{}\t{}\n".format(triple[0], triple[1], triple[2]))
    for j in range(split[0],  split[1]):
        triple = triple_set[index_list[j]]
        outfile.write("{}\t{}\t{}\n".format(triple[0], triple[1], triple[2]))
        outfile_val.write("{}\t{}\t{}\n".format(triple[0], triple[1], triple[2]))
    for l in range(split[1], num_triples):
        triple = triple_set[index_list[l]]
        outfile.write("{}\t{}\t{}\n".format(triple[0], triple[1], triple[2]))
        outfile_test.write("{}\t{}\t{}\n".format(triple[0], triple[1], triple[2]))

    outfile.close()
    outfile_train.close()
    outfile_val.close()
    outfile_test.close()
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create dataset using entity and relation ids')
    parser.add_argument('-s', '--fbsubset', dest='fbsubset', action='store', required=True,
                        help='path to freebase subset file')
    parser.add_argument('-o', '--output', dest='output', action='store', required=True,
                        help='output folder for dataset files')

    args = parser.parse_args()
    print("Freebase subset: {}".format(args.fbsubset))
    if args.output.endswith('/'):
        args.output = args.output[:-1]
    print("Output id file for FB2M: {}".format(args.output + '/fb2m2id.txt'))

    create_id(args.fbsubset, args.output)
    print("Create the id file for FB2M")

    # python scripts/create_triple_id.py -s data/SimpleQuestions_v2/freebase-subsets/freebase-FB2M.txt -o data/FB2M_id/
