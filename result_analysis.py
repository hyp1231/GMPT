import pickle
import os
import argparse

import numpy as np


def pass_filter(filt, name):
    for f in filt:
        if f not in name:
            return False
    return True


def analysis(args):
    filt = args.filter.split(',')
    print('filter', filt)

    # construct result dict
    if args.mode not in ['bio', 'chem']:
        raise NotImplementedError('')
    parent_result_path = f'{args.mode}_result'

    result_dict = {}  
    filtered_seed_result_names = set()
    for seed_result_path in ["finetune_seed" + str(i) for i in range(10)]:
        full_result_path = os.path.join(parent_result_path, seed_result_path)
        if not os.path.exists(full_result_path):
            print(f'ommitting path {seed_result_path}')
            continue
        result_dict[seed_result_path] = {}
        # seed_result_names = os.listdir(full_result_path)
        seed_result_names = []
        for f in os.listdir(full_result_path):
            if pass_filter(filt, f):
                seed_result_names.append(f)
        filtered_seed_result_names = filtered_seed_result_names.union(set(seed_result_names)) #[n for n in seed_result_names if split in n]
        #filtered_seed_result_names = ["speciessplit_cbow_l1-1_center0_epoch100", "speciessplit_gae_epoch100", "speciessplit_supervised_drop0.2_cbow_l1-1_center0_epoch100_epoch100"]
        for name in seed_result_names:
            #print(name)
            with open(os.path.join(parent_result_path, seed_result_path, name), "rb") as f:
                result = pickle.load(f)
                result['train'] = result['train']
                result['val'] = result['val']
                # result['test_easy'] = result['test_easy']
                result['test_hard'] = result['test_hard']
            result_dict[seed_result_path][name] = result

    #top_k = 40
    best_result_dict = {}  # dict[SEED#][experiment][test_easy/test_hard] = np.array, dim top_k classes
    for seed in result_dict:
        #print(seed)
        best_result_dict[seed] = {}
        for experiment in result_dict[seed]:
            #print(experiment)
            best_result_dict[seed][experiment] = {}
            #val = result_dict[seed][experiment]["val"][:, :top_k]  # look at the top k classes
            val = result_dict[seed][experiment]["val"]
            val_ave = np.average(val, axis=1)
            best_epoch = np.argmax(val_ave)
            
            # test_easy = result_dict[seed][experiment]["test_easy"]
            # test_easy_best = test_easy[best_epoch]
            
            #test_hard = result_dict[seed][experiment]["test_hard"][:, :top_k]
            test_hard = result_dict[seed][experiment]["test_hard"]
            test_hard_best = test_hard[best_epoch]
            
            # best_result_dict[seed][experiment]["test_easy"] = test_easy_best
            best_result_dict[seed][experiment]["test_hard"] = test_hard_best

    # average across the top k tasks and then average across all the seeds
    mean_result_dict = {}  # dict[experiment][test_easy/test_hard] = float
    std_result_dict = {}  # dict[experiment][test_easy/test_hard] = float
    for experiment in filtered_seed_result_names:
        print(experiment)
        mean_result_dict[experiment] = {}
        std_result_dict[experiment] = {}
        # test_easy_list = []
        test_hard_list = []
        for seed in best_result_dict:
            if experiment in best_result_dict[seed]:
                print(seed)
                # test_easy_list.append(best_result_dict[seed][experiment]['test_easy'])
                test_hard_list.append(best_result_dict[seed][experiment]['test_hard'])
        # mean_result_dict[experiment]['test_easy'] = np.array(test_easy_list).mean(axis=1).mean()
        mean_result_dict[experiment]['test_hard'] = np.array(test_hard_list).mean(axis=1).mean()
        # std_result_dict[experiment]['test_easy'] = np.array(test_easy_list).mean(axis=1).std()
        std_result_dict[experiment]['test_hard'] = np.array(test_hard_list).mean(axis=1).std()

    # results test hard
    sorted_test_hard = sorted(mean_result_dict.items(), key=lambda kv: kv[1]['test_hard'], reverse=True)
    for k, _ in sorted_test_hard:
        print(k)
        print('{:.2f} ± {:.2f}'.format(mean_result_dict[k]['test_hard']*100, std_result_dict[k]['test_hard']*100))
        print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='bio', help='`bio` or `chem`.')
    parser.add_argument('--filter', type=str, default='show results contain the input filter string.')
    args = parser.parse_args()
    print(args, flush=True)

    analysis(args)
