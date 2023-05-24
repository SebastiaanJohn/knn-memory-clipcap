# --------------------------------------------------------
# Dense-Captioning Events in Videos Eval
# Copyright (c) 2017 Ranjay Krishna
# Licensed under The MIT License [see LICENSE for details]
# Written by Ranjay Krishna
# --------------------------------------------------------

import argparse
import json
import random
import string
import sys
sys.path.insert(0, './coco-caption') # Hack to allow the import of pycocoeval

from coco_caption.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from coco_caption.pycocoevalcap.bleu.bleu import Bleu
from coco_caption.pycocoevalcap.meteor.meteor import Meteor
from coco_caption.pycocoevalcap.rouge.rouge import Rouge
from coco_caption.pycocoevalcap.cider.cider import Cider
# from sets import Set
import numpy as np

def random_string(string_length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(string_length))

def remove_nonascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])

class ANETcaptions(object):
    PREDICTION_FIELDS = ['results', 'version', 'external_data']

    def __init__(self, ground_truth_filenames=None, prediction_filename=None,
                 tious=None, max_proposals=1000,
                 prediction_fields=PREDICTION_FIELDS, verbose=False):
        # Check that the gt and submission files exist and load them
        if len(tious) == 0:
            raise IOError('Please input a valid tIoU.')
        if not ground_truth_filenames:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')

        print('Verbose:',verbose)
        

        self.verbose = verbose
        self.tious = tious
        self.max_proposals = max_proposals
        self.pred_fields = prediction_fields
        self.ground_truths = self.import_ground_truths(ground_truth_filenames)
        self.prediction = self.import_prediction(prediction_filename)
        self.tokenizer = PTBTokenizer()

        # Set up scorers, if not verbose, we only use the one we're
        # testing on: METEOR
        if self.verbose:
            print('ik doe alles')
            self.scorers = [
                (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                (Meteor(),"METEOR"),
                (Rouge(), "ROUGE_L"),
                (Cider(), "CIDEr")
            ]
        else:
            print('PAS OP! Verbose staat uit!!')
            self.scorers = [(Meteor(), "METEOR")]

    def import_prediction(self, prediction_filename):
        if self.verbose:
            print("| Loading submission...")
        submission = json.load(open(prediction_filename))
        if not all([field in submission.keys() for field in self.pred_fields]):
            raise IOError('Please input a valid ground truth file.')
        # Ensure that every video is limited to the correct maximum number of proposals.
        results = {}
        # blijkbaar kan de output uit meerdere proposed captions bestaan?
        for vid_id in submission['results']:
            results[vid_id] = submission['results'][vid_id][:self.max_proposals]
        return results

    def import_ground_truths(self, filenames):
        gts = []
        self.n_ref_vids = set()
        for filename in filenames:
            gt = json.load(open(filename))
            self.n_ref_vids.update(gt.keys())
            gts.append(gt)
        if self.verbose:
            print("| Loading GT. #files: %d, #videos: %d" % (len(filenames), len(self.n_ref_vids)))
        return gts

    def iou(self, interval_1, interval_2):
        # tuple of interval (start and end) of timestamps from prediction and reference/caption
        # calculates intersection over union iou = intersection of time intervals / union of time intervals
        start_i, end_i = interval_1[0], interval_1[1]
        start, end = interval_2[0], interval_2[1]
        intersection = max(0, min(end, end_i) - max(start, start_i))
        union = min(max(end, end_i) - min(start, start_i), end-start + end_i-start_i)
        iou = float(intersection) / (union + 1e-8)
        return iou

    def check_gt_exists(self, vid_id):
        # check of iedere video id ook een ground truth label heeft
        for gt in self.ground_truths:
            if vid_id in gt:
              return True
        return False

    def get_gt_vid_ids(self):
        # make list of video ids??
        vid_ids = set([])
        for gt in self.ground_truths:
            vid_ids |= set(gt.keys())
        return list(vid_ids)

    def evaluate(self):
        aggregator = {}
        self.scores = {}
        # loop through tiou thresholds
        for tiou in self.tious:
            # evaluate on this tiou threshold
            scores = self.evaluate_tiou(tiou)
            # okay, at this point hebben we dus alle scores van alle metrics voor alle videos
            # maar dit is voor een tiou, dus dat voeg je toe aan de scores van de andere tiou's
            for metric, score in scores.items():
                if metric not in self.scores:
                    self.scores[metric] = []
                self.scores[metric].append(score)

        # calculate precision and recall
        if self.verbose:
            self.scores['Recall'] = []
            self.scores['Precision'] = []
            # calculate the precision and recall for different tious
            for tiou in self.tious:
                precision, recall = self.evaluate_detection(tiou)
                self.scores['Recall'].append(recall)
                self.scores['Precision'].append(precision)

    def evaluate_detection(self, tiou):
        # ja ik denk dat dit dus voor ons ook niet zo nuttig is, aangezien wij die timestamps niet hebben geleerd en dus niet hebben in onze predictions. 

        gt_vid_ids = self.get_gt_vid_ids()
        # Recall is the percentage of ground truth that is covered by the predictions
        # Precision is the percentage of predictions that are valid

        # create arrays that are as long as number of video ids
        recall = [0] * len(gt_vid_ids)
        precision = [0] * len(gt_vid_ids)

        # loop through the videos
        for vid_i, vid_id in enumerate(gt_vid_ids):
            best_recall = 0
            best_precision = 0

            # loop through the ground truths to find the one for a specific video
            # again: maybe this is not necessary and bit overkill (we can just extract the gt_caption immediatly from .json file, given that we rewrite it a bit ?)
            for gt in self.ground_truths:
                if vid_id not in gt:
                    continue
                refs = gt[vid_id]
                ref_set_covered = set([])
                pred_set_covered = set([])
                num_gt = 0
                num_pred = 0

                # extract again only videos for which the overlap of timestamps is bigger than threshold tiou
                if vid_id in self.prediction:
                    for pred_i, pred in enumerate(self.prediction[vid_id]):
                        pred_timestamp = pred['timestamp']
                        for ref_i, ref_timestamp in enumerate(refs['timestamps']):
                            if self.iou(pred_timestamp, ref_timestamp) > tiou:
                                ref_set_covered.add(ref_i)
                                pred_set_covered.add(pred_i)

                    # calculate new precision and see if it bigger than previous one
                    # no need to calculate this since everything will be covered
                    new_precision = float(len(pred_set_covered)) / (pred_i + 1) 
                    best_precision = max(best_precision, new_precision)

                # calculate new recall, and see if it bigger than previous one
                # no need to calculate this since everything will be covered (thus its gonna be the same for all)
                new_recall = float(len(ref_set_covered)) / len(refs['timestamps'])
                best_recall = max(best_recall, new_recall)
            recall[vid_i] = best_recall
            precision[vid_i] = best_precision
        return sum(precision) / len(precision), sum(recall) / len(recall)



    # -------- tiou prediction fucntion ended --------
    def evaluate_tiou(self, tiou):

        # This method averages the tIoU precision from METEOR, Bleu, etc. across videos 
        
        # define dictionary for predictions and ground truths and 
        res = {}
        gts = {}
        gt_vid_ids = self.get_gt_vid_ids()
        
        unique_index = 0


        # video id to unique caption ids mapping
        # this one does keep track of the video ids, and which caption ids belong to which video!!
        vid2capid = {}
        
        # these are gonna be the predictions and gts dict
        # this is a dict where values are caption ids (start from counting of 0) 
        # note: this thus does not have the video id anymore (it is disregarded here)!
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

            # If we do have a prediction, then we find the scores based on all the
            # valid tIoU overlaps.
            else:
                # For each prediction, we look at the tIoU with ground truth.

                # get predicition corresponding to video id
                for pred in self.prediction[vid_id]:
                    has_added = False

                    # loop through the ground truth values
                    # maybe this is not necessary and bit overkill (we can just extract the gt_caption immediatly from .json file, given that we rewrite it a bit ?)
                    for gt in self.ground_truths:
                        if vid_id not in gt:
                            continue
                        gt_captions = gt[vid_id]

                        # loop through all captions that are in the ground truth of a video
                        for caption_idx, caption_timestamp in enumerate(gt_captions['timestamps']):

                            # check if the predicted timestamp and predicted timestamp groter zijn dan treshold
                            # WIJ PREDICTEN GEEN TIMESTAMPS dus dan zijn ze altijd gelijk aan caption (mits we zeggen dat de timestamps gelijk zijn aan elkaar)
                            if self.iou(pred['timestamp'], caption_timestamp) >= tiou:

                                # haal nonascii tekens weg en voeg toe aan pred en gt dicts
                                cur_res[unique_index] = [{'caption': remove_nonascii(pred['sentence'])}]
                                cur_gts[unique_index] = [{'caption': remove_nonascii(gt_captions['sentences'][caption_idx])}]
                                # append caption id for the current video (just list that counts how many captions there are) to caption id dict
                                vid2capid[vid_id].append(unique_index)
                                unique_index += 1
                                # timestamps are comparable enough, thus now captions can be compared
                                has_added = True

                    # If the predicted caption does not overlap with any ground truth,
                    # we should compare it with garbage.
                    if not has_added:
                        cur_res[unique_index] = [{'caption': remove_nonascii(pred['sentence'])}]
                        cur_gts[unique_index] = [{'caption': random_string(random.randint(10, 20))}]
                        vid2capid[vid_id].append(unique_index)
                        unique_index += 1


        # Each scorer will compute across all videos and take average score
        output = {}

        # ACTUALLY COMPUTING THE SCORES!!! (finally)
        for scorer, method in self.scorers:

            # do all score metrics if verbose is true
            if self.verbose:
                print('computing %s score...'% scorer.method())
            
            # For each video, take all the valid pairs (based from tIoU) and compute the score
            all_scores = {}
            
            # call tokenizer here for all predictions and gts
            print("check if this is per video or not: ", cur_res) #it is not per video, just per caption (all of them of all videos)
            # tokenize the captions
            # the resulting dict is a dict where values are ints and keys are the captions (instead of dicts)
            # {0: ['caption': caption_1], 1: ['caption': caption_2], etc} --> {0: [caption_1], 1: [caption_2], etc}
            
            tokenize_res = self.tokenizer.tokenize(cur_res)
            tokenize_gts = self.tokenizer.tokenize(cur_gts)
            
            # reshape back
            # this is a dict where the keys are the video_ids, the values are the dicts with tokenized captions (and thus also the caption ids)
            for vid in vid2capid.keys():
                res[vid] = {index:tokenize_res[index] for index in vid2capid[vid]}
                gts[vid] = {index:tokenize_gts[index] for index in vid2capid[vid]}

            
            # loop through all videos
            for vid_id in gt_vid_ids:

                # if there are 0 predicting captions or 0 ground truth captions, return a score of 0
                if len(res[vid_id]) == 0 or len(gts[vid_id]) == 0:
                    continue
                # compute the scores of all metrics for a given video
                else:
                    score, scores = scorer.compute_score(gts[vid_id], res[vid_id])
                all_scores[vid_id] = score

            # print(all_scores.values())
            if type(method) == list:
                # als de gereturnde score een lijst is (volgens mij is dat alleen bij BLUE zo), dan bereken 
                # je daarvan het gemiddelde en maak je al die ?(4) scores het gemiddelde en geef dat als output
                scores = np.mean(list(all_scores.values()), axis=0)
                for m in range(len(method)):
                    output[method[m]] = scores[m]
                    if self.verbose:
                        print("Calculated tIoU: %1.1f, %s: %0.3f" % (tiou, method[m], output[method[m]]))
            else:
                output[method] = np.mean(list(all_scores.values()))
                if self.verbose:
                    print("Calculated tIoU: %1.1f, %s: %0.3f" % (tiou, method, output[method]))
        return output
    
    # -------- tiou prediction fucntion ended --------

def main(args):
    # Call coco eval
    evaluator = ANETcaptions(ground_truth_filenames=args.references,
                             prediction_filename=args.submission,
                             tious=args.tious,
                             max_proposals=args.max_proposals_per_video,
                             verbose=args.verbose)
    evaluator.evaluate()

    # Output the results
    if args.verbose:
        for i, tiou in enumerate(args.tious):
            print('-' * 80)
            print("tIoU: " , tiou)
            print('-' * 80)
            for metric in evaluator.scores:
                score = evaluator.scores[metric][i]
                print('| %s: %2.4f'%(metric, 100*score))

    # Print the averages
    print('-' * 80) 
    print("Average across all tIoUs")
    print('-' * 80)
    for metric in evaluator.scores:
        score = evaluator.scores[metric]
        print('| %s: %2.4f'%(metric, 100 * sum(score) / float(len(score))))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evaluate the results stored in a submissions file.')
    parser.add_argument('-s', '--submission', type=str,  default='sample_submission.json',
                        help='sample submission file for ActivityNet Captions Challenge.')
    parser.add_argument('-r', '--references', type=str, nargs='+', default=['data/val_1.json', 'data/val_2.json'],
                        help='reference files with ground truth captions to compare results against. delimited (,) str')
    parser.add_argument('--tious', type=float,  nargs='+', default=[0.0],
                        help='Choose the tIoUs to average over.')
    parser.add_argument('-ppv', '--max-proposals-per-video', type=int, default=1000,
                        help='maximum propoasls per video.')
    parser.add_argument('-v', '--verbose', action='store_true', default=True,
                        help='Print intermediate steps.')
    args = parser.parse_args()

    main(args)
