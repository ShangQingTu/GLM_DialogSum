import argparse
import os
import json

splits = ['train', 'val', 'test']


def check_fout(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    fout = open(file_path, 'a')
    return fout


def process(args):
    if not os.path.exists(args.processed_dir):
        os.makedirs(args.processed_dir)
    for split in splits:
        origin_path = os.path.join(args.original_dir, f"{split}.json")
        output_src_path = os.path.join(args.processed_dir, f"{split}.source")
        output_tgt_path = os.path.join(args.processed_dir, f"{split}.target")
        src_file = check_fout(output_src_path)
        tgt_file = check_fout(output_tgt_path)
        # read the origin file
        fin = open(origin_path, 'r')
        data_list = json.load(fin)
        for data_dic in data_list:
            final_summ_lst = data_dic["FinalSumm"]
            dialog_lst = data_dic["Dialogue"]
            tgt_str = ''.join(final_summ_lst)
            src_str = ""
            for turn in dialog_lst:
                src_str += f"{turn['speaker']}:{turn['utterance']} "
            # output
            src_file.write(src_str)
            src_file.write("\n")
            tgt_file.write(tgt_str)
            tgt_file.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_dir', type=str, default='/data/tsq/dialog_sum/CSDS')
    parser.add_argument('--processed_dir', type=str, default='/data/tsq/dialog_sum/CSDS/glm')
    args = parser.parse_args()
    process(args)
