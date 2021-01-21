#!/usr/bin/python

import sys
import argparse
import pickle
import os


def get_all_entity_mids(fbpath):
    mids = set()
    with open(fbpath, 'r') as f:
        for i, line in enumerate(f):
            if i % 1000000 == 0:
                print("line: {}".format(i))

            items = line.strip().split("\t")
            if len(items) != 3:
                print("ERROR: line - {}".format(line))

            subject = www2fb(items[0])
            mids.add(subject)

    return mids


def create_id(fbsubsetpath, outfolder):
    '''
    create ids for entities and relations in freebase2M txt file
    ids are ordered according to .sort()
    '''
    # print("getting all entity MIDs from Freebase subset...")
    # mids_to_check = get_all_entity_mids(fbsubsetpath)
    print("creating ids...")
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    outfile_ent = open(outfolder+'/entity2id.txt', 'w')
    outfile_rel = open(outfolder+'/relation2id.txt', 'w')
    # entity_set = {}
    # relation_set = {}
    # entity_set = []
    # relation_set = []
    # list is too slow!!!
    entity_set = set()
    relation_set = set()
    entity_counter = 0
    relation_counter = 0
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

            entity_set.add(e1)
            entity_set.add(e2)
            relation_set.add(r)

            # if e1 not in entity_set:
            #     # entity_set[e1] = entity_counter
            #     entity_set.append(e1)
            #     # entity_counter += 1
            # if e2 not in entity_set:
            #     # entity_set[e2] = entity_counter
            #     entity_set.append(e2)
            #     # entity_counter += 1
            # if r not in relation_set:
            #     # relation_set[r] = relation_counter
            #     # relation_counter += 1
            #     relation_set.append(r)

    outfile_ent.write("{}\n".format(len(entity_set)))
    entity_set = list(entity_set)
    entity_set.sort()
    for k in range(len(entity_set)):
        entity = entity_set[k]
        outfile_ent.write("{}\t{}\n".format(entity, k))

    outfile_rel.write("{}\n".format(len(relation_set)))
    relation_set = list(relation_set)
    relation_set.sort()
    for j in range(len(relation_set)):
        rel = relation_set[j]
        outfile_rel.write("{}\t{}\n".format(rel, j))

    outfile_ent.close()
    outfile_rel.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create Id for entities and relations')
    parser.add_argument('-s', '--fbsubset', dest='fbsubset', action='store', required=True,
                        help='path to freebase subset file')
    parser.add_argument('-o', '--output', dest='output', action='store', required=True,
                        help='output folder for id')

    args = parser.parse_args()
    print("Freebase subset: {}".format(args.fbsubset))
    if args.output.endswith('/'):
        args.output = args.output[:-1]
    print("Output entity id file: {}".format(args.output + '/entity2id.txt'))
    print("Output relation id file: {}".format(args.output + '/relation2id.txt'))

    create_id(args.fbsubset, args.output)
    print("Create the id files for entities and relations")

    # python scripts/create_id.py -s data/SimpleQuestions_v2/freebase-subsets/freebase-FB2M.txt -o data/FB2M_id/
