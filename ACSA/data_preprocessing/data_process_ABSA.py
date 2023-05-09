from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
import os
import torch
import numpy as np

def generate_NULL_sample(file_dir, term_list, mode='val'):
    candidate_list = ["positive", "neutral", "negative", 'none']
    output_dir = os.path.split(file_dir)[-1].split('.')[0] + '_with_none.txt'
    output_dir = os.path.join(os.path.split(file_dir)[-2], output_dir)
    if os.path.exists(output_dir):
        os.remove(output_dir)
        print('deleting existed file and create new one')

    with open(file_dir, "r") as f:
        file = f.readlines()
    if mode == 'val' or mode == 'test':
        with open(output_dir, "a") as f:
            x_last = file[0].split("\001")[0]
            terms = []
            num_pos = 0
            num_neg = 0
            num_nue = 0
            num_none = 0
            for line in file:
                line = line.strip()
                x, term, golden_polarity = line.split("\001")[0], line.split("\001")[1], line.split("\001")[2]
                if x == x_last:
                    if golden_polarity == 'positive':
                        num_pos += 1
                    elif golden_polarity == 'neutral':
                        num_nue += 1
                    elif golden_polarity == 'negative':
                        num_neg += 1
                    terms.append(term)
                    f.write(line + '\n')
                else:
                    for t in list(set(term_list)-set(terms)):
                        f.write(''.join([x_last, t, 'none\n']))
                    num_none += len(list(set(term_list)-set(terms)))
                    terms = []
                    x_last = x
                    if golden_polarity == 'positive':
                        num_pos += 1
                    elif golden_polarity == 'neutral':
                        num_nue += 1
                    elif golden_polarity == 'negative':
                        num_neg += 1
                    terms.append(term)
                    f.write(line + '\n')
    else:
        with open(output_dir, "a") as f:
            x_last = file[0].split("\001")[0]
            terms = []
            num_pos = 0
            num_neg = 0
            num_nue = 0
            num_none = 0
            for line in file:
                line = line.strip()
                x = line.split("\001")[0]
                term, golden_polarity = line.split("\001")[1].split(' ')[-4], line.split("\001")[1].split(' ')[-2]
                if x == x_last:
                    if golden_polarity=='positive':
                        num_pos += 1
                    elif golden_polarity=='neutral':
                        num_nue += 1
                    elif golden_polarity=='negative':
                        num_neg += 1
                    terms.append(term)
                    f.write(line + '\n')
                else:
                    for t in list(set(term_list)-set(terms)):
                        f.write(''.join([x_last, f'The sentiment polarity of {t} is none .\n']))
                    num_none += len(terms)
                    terms = []
                    x_last = x
                    if golden_polarity == 'positive':
                        num_pos += 1
                    elif golden_polarity == 'neutral':
                        num_nue += 1
                    elif golden_polarity == 'negative':
                        num_neg += 1
                    terms.append(term)
                    f.write(line + '\n')
    return num_pos, num_nue, num_neg, num_none


if __name__ == '__main__':
    term_list = ['staff', 'price', 'miscellaneous', 'service', 'food', 'ambience', 'place', 'menu']
    n1, n2, n3, n4 = generate_NULL_sample('./MAMS/MAMS_test.txt', term_list, mode='test')
    print(f'# of positive sample: {n1}, # of nuetral sample: {n2}, # of negative sample: {n3}, # of none sample: {n4}, # of original dataset: {n1+n2+n3}')
    n1, n2, n3, n4 = generate_NULL_sample('./MAMS/MAMS_val.txt', term_list, mode='val')
    print(f'# of positive sample: {n1}, # of nuetral sample: {n2}, # of negative sample: {n3}, # of none sample: {n4}, # of original dataset: {n1+n2+n3}')
    n1, n2, n3, n4 = generate_NULL_sample('./MAMS/MAMS_train.txt', term_list, mode='train')
    print(f'# of positive sample: {n1}, # of nuetral sample: {n2}, # of negative sample: {n3}, # of none sample: {n4}, # of original dataset: {n1+n2+n3}')