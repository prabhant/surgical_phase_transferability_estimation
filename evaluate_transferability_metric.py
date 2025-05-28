#!/usr/bin/env python
# coding: utf-8

from rank_correlation import (load_score, recall_k, rel_k, pearson_coef, 
                            wpearson_coef, w_kendall_metric, kendall_metric,w_kendall_metric_energy)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate transferability metrics.')
    parser.add_argument('-d', '--dataset', type=str, default='voc2007', 
                        help='name of the pretrained model to load and evaluate')
    parser.add_argument('-me', '--method', type=str, default='energy', 
                        help='name of used transferability metric')
    args = parser.parse_args()
    parser.add_argument('-w', '--without', nargs='+', type=str, help='name(s) of the model(s) to exclude from computation')
    
    dset = args.dataset
    metric = args.method
    
    def scale(score):
        min_score=10e10
        max_score=-10e10
        for key in score.keys():
            if score[key]<min_score:
                min_score=score[key]
            if score[key]>max_score:
                max_score=score[key]
        for key in score.keys():
            score[key]=(score[key]-min_score)/(max_score-min_score)
        return score

    if args.without:
      without = args.without
      for d in finetune_acc:
        for w in without:
            finetune_acc[d].pop(w)
    score_path = './results_metrics/group1/{}/{}_metrics.json'.format(metric, dset)
    if args.without:
        for w in without:
            score.pop(w)
    score, _ = load_score(score_path)
    tw_sfda = w_kendall_metric(score, finetune_acc, dset)
    print("Kendall  dataset:{:12s} {:12s}:{:2.3f}".format(dset, metric, tw_sfda))


