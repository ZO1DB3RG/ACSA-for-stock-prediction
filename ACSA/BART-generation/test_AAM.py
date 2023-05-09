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


def predict_val(model, device, val_dir="./MAMS/MAMS_val.txt"):
    candidate_list = ["positive", "neutral", "negative",'none']
    # model = BartForConditionalGeneration.from_pretrained('./outputs/checkpoint-513-epoch-19')
    model.eval()
    model.config.use_cache = False
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    with open(val_dir, "r") as f:
        file = f.readlines()
    count = 0
    total = 0
    for line in tqdm.tqdm(file):
        total += 1
        score_list = []
        line = line.strip()
        x, term, golden_polarity = line.split("\001")[0], line.split("\001")[1], line.split("\001")[2]
        input_ids = tokenizer([x] * 4, return_tensors='pt')['input_ids']
        target_list = ["The sentiment polarity of " + term.lower() + " is " + candi.lower() + " ." for candi in
                       candidate_list]
        output_ids = tokenizer(target_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
        with torch.no_grad():
            output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids.to(device))
            output = output[0]
            logits = output.softmax(dim=-1).to('cpu').numpy()
        for i in range(4):
            score = 1
            for j in range(logits[i].shape[0] - 2):
                score *= logits[i][j][output_ids[i][j + 1]]
            score_list.append(score)
        predict = candidate_list[np.argmax(score_list)]
        if predict == golden_polarity:
            count += 1
            # print(predict, golden_polarity, count/total, count, total)

    return count/total

def predict_test(model, device, test_dir):
    candidate_list = ["positive", "neutral", "negative",'none']
    # model = BartForConditionalGeneration.from_pretrained('./outputs/checkpoint-513-epoch-19')
    model.eval()
    model.config.use_cache = False
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    with open(test_dir, "r") as f:
        file = f.readlines()
    count = 0
    total = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for line in tqdm.tqdm(file):
        total += 1
        score_list = []
        line = line.strip()
        x, term, golden_polarity = line.split("\001")[0], line.split("\001")[1], line.split("\001")[2]
        input_ids = tokenizer([x] * 4, return_tensors='pt')['input_ids']
        target_list = ["The sentiment polarity of " + term.lower() + " is " + candi.lower() + " ." for candi in
                       candidate_list]
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
        if predict == golden_polarity:
            count += 1
            # print(predict, golden_polarity, count/total, count, total)
        if predict != 'none' and predict == golden_polarity:
            tp += 1
        elif predict == 'none' and predict == golden_polarity:
            tn += 1
        elif predict != 'none' and predict != golden_polarity:
            fp += 1
        elif predict == 'none' and predict != golden_polarity:
            fn += 1
    pr = tp / (tp + fp)
    ra = tp / (tp + fn)
    f1 = 2 * pr * ra / (pr + ra)

    return count/total, pr, ra, f1

def predict_test_cate(model, device, test_dir):
    candidate_list = ['yes', 'no']
    # model = BartForConditionalGeneration.from_pretrained('./outputs/checkpoint-513-epoch-19')
    model.eval()
    model.config.use_cache = False
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    with open(test_dir, "r") as f:
        file = f.readlines()
    count = 0
    total = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for line in tqdm.tqdm(file):
        total += 1
        score_list = []
        line = line.strip()
        x, term, truth = line.split("\001")[0], line.split("\001")[1], line.split("\001")[2]
        input_ids = tokenizer([x] * 2, return_tensors='pt')['input_ids']
        target_list = [f'The category {term.lower()} is discussed.\n', f'The category {term.lower()} is not discussed.\n']
        output_ids = tokenizer(target_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
        with torch.no_grad():
            output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids.to(device))[0]
            logits = output.softmax(dim=-1).to('cpu').numpy()
        for i in range(2):
            score = 1
            for j in range(logits[i].shape[0] - 2):
                score *= logits[i][j][output_ids[i][j + 1]]
            score_list.append(score)
        predict = candidate_list[np.argmax(score_list)]
        if predict == truth:
            count += 1
            # print(predict, golden_polarity, count/total, count, total)
        if predict != 'no' and predict == truth:
            tp += 1
        elif predict == 'no' and predict == truth:
            tn += 1
        elif predict != 'no' and predict != truth:
            fp += 1
        elif predict == 'no' and predict != truth:
            fn += 1
    pr = tp / (tp + fp)
    ra = tp / (tp + fn)
    f1 = 2 * pr * ra / (pr + ra)

    return count/total, pr, ra, f1

def evaluation_pr_re_f1(model, device, val_dir="./MAMS/MAMS_val.txt"):
    candidate_list = ["positive", "neutral", "negative",'none']
    # model = BartForConditionalGeneration.from_pretrained('./outputs/checkpoint-513-epoch-19')
    model.eval()
    model.config.use_cache = False
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    with open(val_dir, "r") as f:
        file = f.readlines()
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for line in tqdm.tqdm(file):
        score_list = []
        line = line.strip()
        x, term, golden_polarity = line.split("\001")[0], line.split("\001")[1], line.split("\001")[2]
        input_ids = tokenizer([x] * 4, return_tensors='pt')['input_ids']
        target_list = ["The sentiment polarity of " + term.lower() + " is " + candi.lower() + " ." for candi in
                       candidate_list]
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
        if predict != 'none' and predict == golden_polarity:
            tp += 1
        elif predict == 'none' and predict == golden_polarity:
            tn += 1
        elif predict != 'none' and predict != golden_polarity:
            fp += 1
        elif predict == 'none' and predict != golden_polarity:
            fn += 1
    pr = tp / (tp + fp)
    ra = tp / (tp + fn)
    f1 = 2*pr*ra / (pr + ra)

    return pr, ra, f1

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

# if __name__ == '__main__':
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = AspectAnything(
#         encoder_decoder_type="bart",
#         encoder_decoder_name="./outputs",
#     )
#     # model2 = BartForConditionalGeneration.from_pretrained('./outputs').to(device)
#     #print(get_terms(model2, device))
#     # inference(model2, device, input_dir='./MAMS/sample.txt')
#     acc = predict_test(model, device)
#     print(acc)