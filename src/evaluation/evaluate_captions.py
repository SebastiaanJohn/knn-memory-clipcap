# --------------------------------------------------------
# Dense-Captioning Events in Videos Eval
# Copyright (c) 2017 Ranjay Krishna
# Licensed under The MIT License [see LICENSE for details]
# Written by Ranjay Krishna
# --------------------------------------------------------

import argparse
import json
import pickle
import random
import string
import sys


sys.path.insert(0, './coco-caption') # Hack to allow the import of pycocoeval

# from sets import Set
import numpy as np
from coco_caption.pycocoevalcap.bleu.bleu import Bleu
from coco_caption.pycocoevalcap.meteor.meteor import Meteor
from coco_caption.pycocoevalcap.rouge.rouge import Rouge
from coco_caption.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


def random_string(string_length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(string_length))

def remove_nonascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])

class ANETcaptions(object):
    # PREDICTION_FIELDS = ['results', 'version', 'external_data']

    def __init__(self, ground_truth_filenames=None, prediction_filename=None,
                 max_proposals=1000,
                 verbose=False, pickle_file=None):
        # Check that the gt and submission files exist and load them
        if not ground_truth_filenames:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')

        self.pickle_file = pickle_file
        self.verbose = verbose
        self.max_proposals = max_proposals
        self.ground_truths = self.import_ground_truths(ground_truth_filenames)
        self.prediction = self.import_prediction(prediction_filename)
        self.tokenizer = PTBTokenizer()

        # Set up scorers, if not verbose, we only use the one we're
        # testing on: METEOR
        if self.verbose:
            print('Alle metrics worden uitgevoerd :)')
            self.scorers = [
                (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                (Meteor(),"METEOR"),
                (Rouge(), "ROUGE_L"),
                # (Cider(), "CIDEr")
                # (Spice(), "SPICE")
            ]
        else:
            print('Watch out, verbose is off (only METEOR will be done)!!')
            self.scorers = [(Meteor(), "METEOR")]

    def import_prediction(self, prediction_filename):
        if self.verbose:
            print("| Loading submission...")
        results = json.load(open(prediction_filename))
        return results

    def import_ground_truths(self, filename):
        gt = json.load(open(filename))

        print("| Loading GT. #videos: %d" %  len(filename))
        return gt

    def check_gt_exists(self, vid_id):
        # check of iedere video id ook een ground truth label heeft
        for gt in self.ground_truths:
            if vid_id in gt:
              return True
        return False

    def get_gt_vid_ids(self):
        # make list of video ids
        return list(self.ground_truths.keys())

    def num_frames(self):
        # get number of frames per clip
        prepr_dataset_path = self.pickle_file
        with open(prepr_dataset_path, "rb") as f:
            prepr_dataset = pickle.load(f)

        n_frames = []
        for video_clip in prepr_dataset:
            n_frames.append(video_clip["frames"].shape[0])

        return n_frames


    def evaluate(self):

        n_frames = self.num_frames()


        # define dictionary for predictions and ground truths and
        res = {}
        gts = {}
        gt_vid_ids = self.get_gt_vid_ids()

        print(len(n_frames))
        print(len(gt_vid_ids))

        unique_index = 0


        # video id to unique caption ids mapping
        # this one does keep track of the video ids, and which caption ids belong to which video!!
        vid2capid = {}

        # these are gonna be the predictions and gts dicts
        cur_res = {}
        cur_gts = {}

        # loop through all videos
        for vid_id in gt_vid_ids:

            # add the video id to the dicts for making caption ids
            vid2capid[vid_id] = []

            # If the video does not have a prediction, then we give it no matches
            # We set it to empty, and use this as a sanity check later on.
            if vid_id not in self.prediction:
                pass

            # If we do have a prediction, then we find the scores
            else:

                if vid_id not in self.ground_truths:
                        continue
                # maak dicts in goede format zodat het de tokenizer in kan
                cur_res[unique_index] = [{'caption': self.prediction[vid_id]}]
                cur_gts[unique_index] = [{'caption': self.ground_truths[vid_id]}]
                vid2capid[vid_id].append(unique_index)
                unique_index += 1


        # Each scorer will compute across all videos and take average score
        self.scores = {}

        self.dct_scores_len = {}

        # we start computing the scores
        for scorer, method in self.scorers:

            score_blue1 = []
            score_blue2 = []
            score_blue3 = []
            score_blue4 = []
            scores_len = []

            # do all score metrics if verbose is true
            if self.verbose:
                print('computing %s score...'% scorer.method())

            # for each video compute the score
            all_scores = {}

            # first put data through tokenizer
            tokenize_res = self.tokenizer.tokenize(cur_res)
            tokenize_gts = self.tokenizer.tokenize(cur_gts)

            # this is a dict where the keys are the video_ids, the values are the dicts with tokenized captions (and thus also the caption ids)
            for vid in vid2capid.keys():
                res[vid] = {index:tokenize_res[index] for index in vid2capid[vid]}
                gts[vid] = {index:tokenize_gts[index] for index in vid2capid[vid]}

            empty_count = 0

            # loop through all videos
            for vid_id in gt_vid_ids:

                # if there are 0 predicting captions or 0 ground truth captions,skip
                if len(res[vid_id]) == 0 or len(gts[vid_id]) == 0:
                    empty_count += 1
                    continue

                # compute the scores of all metrics for a given video
                else:
                    num_frames_vid = n_frames[int(vid_id)]
                    score, scores = scorer.compute_score(gts[vid_id], res[vid_id])

                all_scores[vid_id] = score

                if scorer.method() == 'Bleu':
                    score_blue1.append((num_frames_vid, 100 * score[0]))
                    score_blue2.append((num_frames_vid, 100 * score[1]))
                    score_blue3.append((num_frames_vid, 100 * score[2]))
                    score_blue4.append((num_frames_vid, 100 * score[3]))

                else:
                    scores_len.append((num_frames_vid, 100 * score))

            print("The number of videos that do not have a score (these are not considered in the scoring): ", empty_count)
            print("The number of videos that DO have a score: ", len(gt_vid_ids) - empty_count)
            print("\n")

            if type(method) == list:
                # calculate average score for bleu metrics
                scores = np.mean(list(all_scores.values()), axis=0)
                for m in range(len(method)):
                    self.scores[method[m]] = scores[m]
                    if self.verbose:
                        print("scores: %s: %0.3f" % (method[m], self.scores[method[m]]))
            else:
                self.scores[method] = np.mean(list(all_scores.values()))
                if self.verbose:
                    print("Scores: %s: %0.3f" % (method, self.scores[method]))
            print('-' * 80)

            if scorer.method() == 'Bleu':
                self.dct_scores_len['Bleu_1'] = score_blue1
                self.dct_scores_len['Bleu_2'] = score_blue2
                self.dct_scores_len['Bleu_3'] = score_blue3
                self.dct_scores_len['Bleu_4'] = score_blue4
            else:
                self.dct_scores_len[scorer.method()] = scores_len

        print(self.dct_scores_len.keys())
        print(len(self.dct_scores_len['Rouge']))



def main(args):
    # Call coco eval
    evaluator = ANETcaptions(ground_truth_filenames=args.references,
                             prediction_filename=args.submission,
                             max_proposals=args.max_proposals_per_video,
                             verbose=args.verbose,
                             pickle_file=args.data)
    evaluator.evaluate()


    # output the averages
    print('\n')
    print('-' * 80)
    print("The scores per metric")
    print('-' * 80)
    for metric in evaluator.scores:
        score = evaluator.scores[metric]
        print('| %s: %2.4f'%(metric, 100 * score))
    print('-' * 80)
    # print(evaluator.dct_scores_len)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evaluate the results stored in a submissions file.')
    parser.add_argument('-s', '--submission', type=str,  default='organized_data/no_memory_captions_allc_bs5_lastf.json',
                        help='sample submission file for ActivityNet Captions Challenge.')
    parser.add_argument('-r', '--references', type=str, default='organized_data/no_memory_references_allc_bs5_lastf.json',
                        help='reference files with ground truth captions to compare results against. delimited (,) str')
    parser.add_argument('-d', '--data', type=str, default="../data/activitynet_ViT-B_32_validation_first_500.pkl",
                        help='pickle file from test data for dataloader')
    parser.add_argument('-ppv', '--max-proposals-per-video', type=int, default=1000,
                        help='maximum propoasls per video.')
    parser.add_argument('-v', '--verbose', action='store_true', default=True,
                        help='Print intermediate steps.')
    args = parser.parse_args()

    main(args)
