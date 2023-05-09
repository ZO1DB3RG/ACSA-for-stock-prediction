import tqdm
from simpletransformers.seq2seq import Seq2SeqModel
import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
# logging.basicConfig(level=logging.INFO)
# transformers_logger = logging.getLogger("transformers")
# transformers_logger.setLevel(logging.WARNING)
import torch
import numpy as np
import codecs as cs
from test_AAM import predict_test, predict_val, evaluation_pr_re_f1
from AspectAnythingModel import AspectAnything
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_terms(file_dir="./MAMS/MAMS_test.txt"):
    candidate_list = ["positive", "neutral", "negative"]
    # model = BartForConditionalGeneration.from_pretrained('./outputs/checkpoint-513-epoch-19')
    with open(file_dir, "r") as f:
        file = f.readlines()
    terms = []
    for line in file:
        line = line.strip()
        x, term, golden_polarity = line.split("\001")[0], line.split("\001")[1], line.split("\001")[2]
        terms.append(term)
    return set(terms)

def get_term_list_ACOS(file_dir):
    f = cs.open(file_dir, 'r').readlines()
    cate_list = []
    for line in f:
        cates = line.strip().split('\t')[1:]
        for cate in cates:
            cate_list.append(cate.split(' ')[1].split('#')[1].lower().replace('_', ' '))
    print(set(cate_list))
    return set(cate_list)

def inference(model, device, input_dir,
              term_list=['quality', 'connectivity', 'usability', 'portability', 'miscellaneous', 'general', 'operation performance', 'design features', 'price']):
    candidate_list = ["positive", "neutral", "negative",'none']
    model.eval()
    model.config.use_cache = False
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    with open(input_dir, "r") as f:
        file = f.readlines()
    total = 0
    for line in file:
        total += 1
        line = line.strip()
        x = line.split("\001")[0].lower()
        input_ids = tokenizer([x] * 4, return_tensors='pt', truncation=True)['input_ids']
        for term in term_list:
            score_list = []
            target_list = ["The sentiment polarity of " + term.lower() + " is " + candi.lower() + " ." for candi in candidate_list]
            output_ids = tokenizer(target_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
            with torch.no_grad():
                output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids.to(device))[0]
                logits = output.softmax(dim=-1).to('cpu').numpy()
            for i in range(4):
                score = 1
                for j in range(logits[i].shape[0] - 2):
                    score *= logits[i][j][output_ids[i][j + 1]]
                score_list.append(score)
            predict = candidate_list[np.argmax(score_list)]
            print(line, f'-------> term: {term}: {predict}')

def get_cate_from_input(model, input_text, tokenizer):
    candidate_list = ['yes', 'no']



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AspectAnything(
        encoder_decoder_type="bart",
        encoder_decoder_name="./outputs_cate_CE",

    ).model.to(device)
    # model2 = BartForConditionalGeneration.from_pretrained('./outputs').to(device)
    #print(get_terms(model2, device))
    # inference(model, device, input_dir='/userhome/cs2/lzl0105/dl/NLP/Laptop-ACOS/samples.txt')
    # inference(model, device, './Laptop-ACOS/samples.txt')
    acc_with_none, pr, ra, f1 = predict_test(model, device, '/userhome/cs2/lzl0105/dl/NLP/Laptop-ACOS/laptop_quad_test_cate.txt')
    acc_without_none = predict_test(model, device, '/userhome/cs2/lzl0105/dl/NLP/Laptop-ACOS/laptop_quad_test_cate.txt')
    print(f"accuracy with none: {acc_with_none}, accuracy without none: {acc_without_none}")
    # pr, ra, f1 = evaluation_pr_re_f1(model, device, './Laptop-ACOS/laptop_quad_test_with_none.txt')
    print("test with none")
    print(f"precision = {pr}, recall = {ra}, F1 = {f1}")

