import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
import os
import torch
import numpy as np
import codecs as cs
from random import sample

def get_term_list(file_dir):
    f = cs.open(file_dir, 'r').readlines()
    cate_list = []
    for line in f:
        cates = line.strip().split('\t')[1:]
        for cate in cates:
            cate_list.append(cate.split(' ')[1].split('#')[1].lower().replace('_', ' '))
    print(set(cate_list))
    return set(cate_list)


def generate_none_ACOS(file_dir, term_list, mode='val'):
    candidate_dict = {"2": "positive", "1": "neutral", '0': "negative"}
    output_dir = os.path.split(file_dir)[-1].split('.')[0] + '_with_none.txt'
    output_dir = os.path.join(os.path.split(file_dir)[-2], output_dir)
    all_record = cs.open(file_dir, 'r').readlines()
    num_pos = 0
    num_neg = 0
    num_nue = 0
    num_none = 0
    with open(output_dir, "a") as f:
        for record in all_record:
            x, cate_pairs = record.strip().split('\t')[0], record.strip().split('\t')[1:]
            cate_pairs_new = []
            for cate in cate_pairs:
                cate_pairs_new.append(' '.join(cate.split(' ')[1:3]))
            cate_pairs_new = list(set(cate_pairs_new))
            all_cate = []
            for cate_pair in cate_pairs_new:
                c, senti = cate_pair.split(' ')
                c = c.split('#')[1].lower().replace('_', ' ')
                if senti == '0':
                    num_neg += 1
                elif senti == '1':
                    num_nue += 1
                else:
                    num_pos += 1
                senti = candidate_dict[senti] + '\n'
                all_cate.append(c)
                if mode == 'train':
                    f.write(''.join([x, f'The sentiment polarity of {c} is {senti}']))
                else:
                    f.write(''.join([x, c, senti]))
            none_cate = list(set(term_list) - set(all_cate))
            for c in none_cate:
                if mode == 'train':
                    f.write(''.join([x, f'The sentiment polarity of {c} is none\n']))
                else:
                    f.write(''.join([x, c, 'none\n']))
            num_none += len(none_cate)

    return num_pos, num_nue, num_neg, num_none

def generate_none_ACOS_csv(file_dir, term_list, mode='val'):
    candidate_dict = {"2": "positive", "1": "neutral", '0': "negative"}
    output_dir = os.path.split(file_dir)[-1].split('.')[0] + '_with_none_all.csv'
    output_dir = os.path.join(os.path.split(file_dir)[-2], output_dir)
    all_record = cs.open(file_dir, 'r').readlines()
    num_pos = 0
    num_neg = 0
    num_nue = 0
    num_none = 0
    text_list = []
    label_list = []

    for record in all_record:
        x, cate_pairs = record.strip().split('\t')[0], record.strip().split('\t')[1:]
        cate_pairs_new = []
        for cate in cate_pairs:
            cate_pairs_new.append(' '.join(cate.split(' ')[1:3]))
        cate_pairs_new = list(set(cate_pairs_new))
        all_cate = []
        for cate_pair in cate_pairs_new:
            c, senti = cate_pair.split(' ')
            c = c.split('#')[1].lower().replace('_', ' ')
            if senti == '0':
                num_neg += 1
            elif senti == '1':
                num_nue += 1
            else:
                num_pos += 1
            if mode == 'train':
                text_list.append(x + '. The category ' + c + ' is discussed. ')
            else:
                text_list.append(x + '. The category ' + c + ' is discussed. ')

            all_cate.append(c)
            label_list.append(int(senti))
        none_cate = list(set(term_list) - set(all_cate))
        for c in none_cate:
            text_list.append(x + '. The category ' + c + ' is not discussed. ')
            label_list.append(3)
        num_none += len(none_cate)

    train_df = pd.DataFrame({'text':text_list, 'label': label_list})
    train_df.to_csv(output_dir, index=0, header=0)
    return num_pos, num_nue, num_neg, num_none

def generate_without_none_ACOS(file_dir, mode='val'):
    candidate_dict = {"2": "positive", "1": "neutral", '0': "negative"}
    output_dir = os.path.split(file_dir)[-1].split('.')[0] + '_without_none.txt'
    output_dir = os.path.join(os.path.split(file_dir)[-2], output_dir)
    all_record = cs.open(file_dir, 'r').readlines()
    num_pos = 0
    num_neg = 0
    num_nue = 0
    with open(output_dir, "a") as f:
        for record in all_record:
            x, cate_pairs = record.strip().split('\t')[0], record.strip().split('\t')[1:]
            cate_pairs_new = []
            for cate in cate_pairs:
                cate_pairs_new.append(' '.join(cate.split(' ')[1:3]))
            cate_pairs_new = list(set(cate_pairs_new))
            all_cate = []
            for cate_pair in cate_pairs_new:
                c, senti = cate_pair.split(' ')
                c = c.split('#')[1].lower().replace('_', ' ')
                if senti == '0':
                    num_neg += 1
                elif senti == '1':
                    num_nue += 1
                else:
                    num_pos += 1
                senti = candidate_dict[senti] + '\n'
                all_cate.append(c)
                if mode == 'train':
                    f.write(''.join([x, f'The sentiment polarity of {c} is {senti}']))
                else:
                    f.write(''.join([x, c, senti]))
    return num_pos, num_nue, num_neg

def generate_none_ACOS_resample(file_dir, term_list, mode):
    candidate_dict = {"2": "positive", "1": "neutral", '0': "negative"}
    output_dir = os.path.split(file_dir)[-1].split('.')[0] + '_with_none_resampled.txt'
    output_dir = os.path.join('Laptop', output_dir)
    if os.path.exists(output_dir):
        os.remove(output_dir)
        print('deleting existed file and create new one')
    all_record = cs.open(file_dir, 'r').readlines()
    num_pos = 0
    num_neg = 0
    num_nue = 0
    num_none = 0
    with open(output_dir, "a") as f:
        for record in all_record:
            x, cate_pairs = record.strip().split('\t')[0], record.strip().split('\t')[1:]
            cate_pairs_new = []
            for cate in cate_pairs:
                cate_pairs_new.append((' '.join(cate.split(' ')[1:3])).split('#')[1])
            cate_pairs_new = list(set(cate_pairs_new))
            all_cate = []
            old_cate = 'xxx xxx'
            for cate_pair in cate_pairs_new:
                c, senti = cate_pair.split(' ')
                c_, senti_ = old_cate.split(' ')
                c = c.lower().replace('_', ' ')
                c_ = c_.lower().replace('_', ' ')
                if c == c_ and senti != senti:
                    continue

                senti = candidate_dict[senti]
                all_cate.append(c)
                if mode == 'train':
                    if senti == 'negative':
                        num_neg += 4
                    elif senti == 'neutral':
                        num_nue += 4
                    else:
                        num_pos += 4
                    f.write(''.join([x, f'The sentiment polarity of {c} is {senti} .\n']))
                    f.write(''.join([x, f'The {c} category has a {senti} label .\n']))
                    f.write(''.join([x, f'The sentiment is {senti} for {c} .\n']))
                    f.write(''.join([x, f'For {c}, it is a {senti} sentence .\n']))
                else:
                    if senti == 'negative':
                        num_neg += 1
                    elif senti == 'neutral':
                        num_nue += 1
                    else:
                        num_pos += 1
                    f.write(''.join([x, c, senti+'\n']))
                old_cate = cate_pair
            none_cate = list(set(term_list) - set(all_cate))
            for c in none_cate:
                if mode == 'train':
                    f.write(''.join([x, f'The sentiment polarity of {c} is none\n']))
                else:
                    f.write(''.join([x, c, 'none\n']))
            num_none += len(none_cate)

    return num_pos, num_nue, num_neg, num_none

def generate_without_none_ACOS_resample(file_dir, term_list, mode):
    candidate_dict = {"2": "positive", "1": "neutral", '0': "negative"}
    output_dir = os.path.split(file_dir)[-1].split('.')[0] + '_without_none_resampled.txt'
    output_dir = os.path.join(os.path.split(file_dir)[-2], output_dir)
    if os.path.exists(output_dir):
        os.remove(output_dir)
        print('deleting existed file and create new one')
    all_record = cs.open(file_dir, 'r').readlines()
    num_pos = 0
    num_neg = 0
    num_nue = 0
    with open(output_dir, "a") as f:
        for record in all_record:
            x, cate_pairs = record.strip().split('\t')[0], record.strip().split('\t')[1:]
            cate_pairs_new = []
            for cate in cate_pairs:
                cate_pairs_new.append((' '.join(cate.split(' ')[1:3])).split('#')[1])
            cate_pairs_new = list(set(cate_pairs_new))
            all_cate = []
            old_cate = 'xxx xxx'
            for cate_pair in cate_pairs_new:
                c, senti = cate_pair.split(' ')
                c_, senti_ = old_cate.split(' ')
                c = c.lower().replace('_', ' ')
                c_ = c_.lower().replace('_', ' ')
                if c == c_ and senti != senti:
                    continue

                senti = candidate_dict[senti]
                all_cate.append(c)
                if mode == 'train':
                    if senti == 'negative':
                        num_neg += 4
                    elif senti == 'neutral':
                        num_nue += 4
                    else:
                        num_pos += 4
                    f.write(''.join([x, f'The sentiment polarity of {c} is {senti} .\n']))
                    f.write(''.join([x, f'The {c} category has a {senti} label .\n']))
                    f.write(''.join([x, f'The sentiment is {senti} for {c} .\n']))
                    f.write(''.join([x, f'For {c}, it is a {senti} sentence .\n']))
                else:
                    if senti == 'negative':
                        num_neg += 1
                    elif senti == 'neutral':
                        num_nue += 1
                    else:
                        num_pos += 1
                    f.write(''.join([x, c, senti+'\n']))
                old_cate = cate_pair

    return num_pos, num_nue, num_neg,

def generate_all_none_ACOS(file_dir, term_list, mode='val'):
    candidate_dict = {"2": "positive", "1": "neutral", '0': "negative"}
    output_dir = os.path.split(file_dir)[-1].split('.')[0] + '_all_none.txt'
    output_dir = os.path.join(os.path.split(file_dir)[-2], output_dir)
    if os.path.exists(output_dir):
        os.remove(output_dir)
        print('deleting existed file and create new one')
    all_record = cs.open(file_dir, 'r').readlines()
    num_pos = 0
    num_neg = 0
    num_nue = 0
    num_none = 0
    num_multi = 0
    num_sentence = len(all_record)

    with open(output_dir, "a") as f:
        for record in all_record:
            x, cate_pairs = record.strip().split('\t')[0], record.strip().split('\t')[1:]
            cate_pairs_new = []
            for cate in cate_pairs:
                cate_pairs_new.append(' '.join(cate.split(' ')[1:3]))
            cate_pairs_new = list(set(cate_pairs_new))
            all_cate = []
            for cate_pair in cate_pairs_new:
                c, senti = cate_pair.split(' ')
                c = c.split('#')[1].lower().replace('_', ' ')
                if senti == '0':
                    num_neg += 1
                elif senti == '1':
                    num_nue += 1
                else:
                    num_pos += 1
                senti = candidate_dict[senti] + '\n'
                all_cate.append(c)
                if mode == 'train':
                    f.write(''.join([x, f'The sentiment polarity of {c} is {senti}']))
                else:
                    f.write(''.join([x, c, senti]))
            if len(cate_pairs_new) >=2:
                num_multi += 1
            none_cate = list(set(term_list) - set(all_cate))
            if mode == 'train':
                s = ', '.join(none_cate)
                f.write(''.join([x, f'The sentiment polarity of {s} is none\n']))
                num_none += 1
            else:
                for c in none_cate:
                    f.write(''.join([x, c, 'none\n']))
                num_none += len(none_cate)

    return num_pos, num_nue, num_neg, num_none, num_multi, num_sentence

def generate_random_none_ACOS(file_dir, term_list, mode='val'):
    candidate_dict = {"2": "positive", "1": "neutral", '0': "negative"}
    output_dir = os.path.split(file_dir)[-1].split('.')[0] + '_with_random_none.txt'
    output_dir = os.path.join(os.path.split(file_dir)[-2], output_dir)
    all_record = cs.open(file_dir, 'r').readlines()
    num_pos = 0
    num_neg = 0
    num_nue = 0
    num_none = 0
    with open(output_dir, "a") as f:
        for record in all_record:
            x, cate_pairs = record.strip().split('\t')[0], record.strip().split('\t')[1:]
            cate_pairs_new = []
            for cate in cate_pairs:
                cate_pairs_new.append(' '.join(cate.split(' ')[1:3]))
            cate_pairs_new = list(set(cate_pairs_new))
            all_cate = []
            for cate_pair in cate_pairs_new:
                c, senti = cate_pair.split(' ')
                c = c.split('#')[1].lower().replace('_', ' ')
                if senti == '0':
                    num_neg += 1
                elif senti == '1':
                    num_nue += 1
                else:
                    num_pos += 1
                senti = candidate_dict[senti] + '\n'
                all_cate.append(c)
                if mode == 'train':
                    f.write(''.join([x, f'The sentiment polarity of {c} is {senti}']))
                else:
                    f.write(''.join([x, c, senti]))
            none_cate = list(set(term_list) - set(all_cate))
            none_cate = sample(none_cate, 3)
            for c in none_cate:
                if mode == 'train':
                    f.write(''.join([x, f'The sentiment polarity of {c} is none\n']))
                else:
                    f.write(''.join([x, c, 'none\n']))
            num_none += len(none_cate)

    return num_pos, num_nue, num_neg, num_none

def generate_none_ACOS_enhanced(file_dir, term_list, mode='val'):
    candidate_dict = {"2": "positive", "1": "neutral", '0': "negative"}
    output_dir = os.path.split(file_dir)[-1].split('.')[0] + '_with_none_enhancecd.txt'
    output_dir = os.path.join(os.path.split(file_dir)[-2], output_dir)
    if os.path.exists(output_dir):
        os.remove(output_dir)
        print('deleting existed file and create new one')
    all_record = cs.open(file_dir, 'r').readlines()
    num_pos = 0
    num_neg = 0
    num_nue = 0
    num_none = 0
    with open(output_dir, "a") as f:
        for record in all_record:
            x, cate_pairs = record.strip().split('\t')[0], record.strip().split('\t')[1:]
            cate_pairs_new = []
            for cate in cate_pairs:
                cate_pairs_new.append(' '.join(cate.split(' ')[1:3]))
            cate_pairs_new = list(set(cate_pairs_new))
            all_cate = []
            for cate_pair in cate_pairs_new:
                c, senti = cate_pair.split(' ')
                c = c.split('#')[1].lower().replace('_', ' ')
                if senti == '0':
                    num_neg += 1
                elif senti == '1':
                    num_nue += 1
                else:
                    num_pos += 1
                senti = candidate_dict[senti] + '\n'
                all_cate.append(c)
                if mode == 'train':
                    f.write(''.join([x+' The category '+c+' is discussed. ', f'The sentiment polarity of {c} is {senti}']))
                else:
                    f.write(''.join([x+' The category '+c+' is discussed. ', c, senti]))
            none_cate = list(set(term_list) - set(all_cate))
            for c in none_cate:
                if mode == 'train':
                    f.write(''.join([x+' The category '+c+' is not discussed. ', f'The sentiment polarity of {c} is none\n']))
                else:
                    f.write(''.join([x+' The category '+c+' is not discussed. ', c, 'none\n']))
            num_none += len(none_cate)

    return num_pos, num_nue, num_neg, num_none

def generate_without_none_ACOS_enhanced(file_dir, term_list, mode='val'):
    candidate_dict = {"2": "positive", "1": "neutral", '0': "negative"}
    output_dir = os.path.split(file_dir)[-1].split('.')[0] + '_without_none_enhancecd.txt'
    output_dir = os.path.join(os.path.split(file_dir)[-2], output_dir)
    if os.path.exists(output_dir):
        os.remove(output_dir)
        print('deleting existed file and create new one')
    all_record = cs.open(file_dir, 'r').readlines()
    num_pos = 0
    num_neg = 0
    num_nue = 0
    num_none = 0
    with open(output_dir, "a") as f:
        for record in all_record:
            x, cate_pairs = record.strip().split('\t')[0], record.strip().split('\t')[1:]
            cate_pairs_new = []
            for cate in cate_pairs:
                cate_pairs_new.append(' '.join(cate.split(' ')[1:3]))
            cate_pairs_new = list(set(cate_pairs_new))
            all_cate = []
            for cate_pair in cate_pairs_new:
                c, senti = cate_pair.split(' ')
                c = c.split('#')[1].lower().replace('_', ' ')
                if senti == '0':
                    num_neg += 1
                elif senti == '1':
                    num_nue += 1
                else:
                    num_pos += 1
                senti = candidate_dict[senti] + '\n'
                all_cate.append(c)
                if mode == 'train':
                    f.write(''.join([x+' The category '+c+' is discussed. ', f'The sentiment polarity of {c} is {senti}']))
                else:
                    f.write(''.join([x+' The category '+c+' is discussed. ', c, senti]))

    return num_pos, num_nue, num_neg, num_none

def generate_cate_ACOS(file_dir, term_list, mode='val'):
    candidate_dict = {"2": "positive", "1": "neutral", '0': "negative"}
    output_dir = os.path.split(file_dir)[-1].split('.')[0] + '_cate.txt'
    output_dir = os.path.join(os.path.split(file_dir)[-2], output_dir)
    if os.path.exists(output_dir):
        os.remove(output_dir)
        print('deleting existed file and create new one')
    all_record = cs.open(file_dir, 'r').readlines()
    num_dis = 0
    num_not_dis = 0
    with open(output_dir, "a") as f:
        for record in all_record:
            x, cate_pairs = record.strip().split('\t')[0], record.strip().split('\t')[1:]
            cate_pairs_new = []
            for cate in cate_pairs:
                cate_pairs_new.append(' '.join(cate.split(' ')[1:3]))
            cate_pairs_new = list(set(cate_pairs_new))
            all_cate = []
            for cate_pair in cate_pairs_new:
                c, senti = cate_pair.split(' ')
                c = c.split('#')[1].lower().replace('_', ' ')
                num_dis+=1
                senti = candidate_dict[senti] + '\n'
                all_cate.append(c)
                if mode == 'train':
                    f.write(''.join([x, f'The category {c} is discussed.\n']))
                else:
                    f.write(''.join([x, c, 'yes\n']))
            none_cate = list(set(term_list) - set(all_cate))
            for c in none_cate:
                if mode == 'train':
                    f.write(''.join([x, f'The category {c} is not discussed.\n']))
                else:
                    f.write(''.join([x, c, 'no\n']))
            num_not_dis += len(none_cate)

    return num_dis, num_not_dis

if __name__ == '__main__':
    term_list = list(get_term_list('./Laptop-ACOS/laptop_quad_train.tsv'))
    print(term_list)
    n1, n2, n3, n4 = generate_none_ACOS_csv('./Laptop-ACOS/laptop_quad_test.tsv', term_list, mode='test')
    print(f'# of positive sample: {n1}, # of nuetral sample: {n2}, # of negative sample: {n3}, # of none sample: {n4}, # of original dataset: {n1+n2+n3}')
    # n1, n2, n3, n4 = generate_none_ACOS_enhanced('./Laptop-ACOS/laptop_quad_dev.tsv', term_list, mode='val')
    # print(f'# of positive sample: {n1}, # of nuetral sample: {n2}, # of negative sample: {n3}, # of none sample: {n4}, # of original dataset: {n1+n2+n3}')
    #
    # n1, n2, n3, n4 = generate_none_ACOS_enhanced('./Laptop-ACOS/laptop_quad_train.tsv', term_list, mode='train')
    # print(f'# of positive sample: {n1}, # of nuetral sample: {n2}, # of negative sample: {n3}, # of none sample: {n4}, # of original dataset: {n1+n2+n3}')

    #
    # n1, n2, n3 = generate_without_none_ACOS('./Laptop-ACOS/laptop_quad_test.tsv', mode='test')
    # print(f'# of positive sample: {n1}, # of nuetral sample: {n2}, # of negative sample: {n3}, # of original dataset: {n1+n2+n3}')
    # n1, n2, n3 = generate_without_none_ACOS('./Laptop-ACOS/laptop_quad_dev.tsv', mode='val')
    # print(f'# of positive sample: {n1}, # of nuetral sample: {n2}, # of negative sample: {n3}, # of original dataset: {n1+n2+n3}')
    # n1, n2, n3 = generate_without_none_ACOS('./Laptop-ACOS/laptop_quad_train.tsv', mode='train')
    # print(f'# of positive sample: {n1}, # of nuetral sample: {n2}, # of negative sample: {n3}, # of original dataset: {n1+n2+n3}')

    # n1, n2, n3, n4, n5, n6 = generate_all_none_ACOS('./Laptop-ACOS/laptop_quad_train.tsv', term_list, mode='train')
    # print(f'# of positive sample: {n1}, # of nuetral sample: {n2}, # of negative sample: {n3}, # of none sample: {n4}, # of original dataset: {n1+n2+n3}, there are {n6} sentence, {n5} of them hold multiple labels')
    # n1, n2, n3, n4, n5, n6 = generate_all_none_ACOS('./Laptop-ACOS/laptop_quad_test.tsv', term_list, mode='test')
    # print(f'# of positive sample: {n1}, # of nuetral sample: {n2}, # of negative sample: {n3}, # of none sample: {n4}, # of original dataset: {n1+n2+n3}, there are {n6} sentence, {n5} of them hold multiple labels')
    # n1, n2, n3, n4, n5, n6 = generate_all_none_ACOS('./Laptop-ACOS/laptop_quad_dev.tsv', term_list, mode='val')
    # print(f'# of positive sample: {n1}, # of nuetral sample: {n2}, # of negative sample: {n3}, # of none sample: {n4}, # of original dataset: {n1+n2+n3}, there are {n6} sentence, {n5} of them hold multiple labels')

    # n1, n2, n3 = generate_without_none_ACOS_resample('./Laptop-ACOS/laptop_quad_train.tsv', term_list, mode='train')
    # print(f'# of positive sample: {n1}, # of nuetral sample: {n2}, # of negative sample: {n3}, # of original dataset: {n1+n2+n3}')
    # n1, n2, n3 = generate_without_none_ACOS_resample('./Laptop-ACOS/laptop_quad_test.tsv', term_list, mode='test')
    # print(f'# of positive sample: {n1}, # of nuetral sample: {n2}, # of negative sample: {n3}, # of original dataset: {n1+n2+n3}')
    # n1, n2, n3 = generate_without_none_ACOS_resample('./Laptop-ACOS/laptop_quad_dev.tsv', term_list, mode='val')
    # print(f'# of positive sample: {n1}, # of nuetral sample: {n2}, # of negative sample: {n3}, # of original dataset: {n1+n2+n3}')

    # n1, n2 = generate_cate_ACOS('./Laptop-ACOS/laptop_quad_test.tsv', term_list=term_list, mode='test')
    # print(f'# of discussed sample: {n1}, # of not discussed sample: {n2}')
    # n1, n2 = generate_cate_ACOS('./Laptop-ACOS/laptop_quad_dev.tsv', term_list=term_list, mode='val')
    # print(f'# of discussed sample: {n1}, # of not discussed sample: {n2}')
    # n1, n2 = generate_cate_ACOS('./Laptop-ACOS/laptop_quad_train.tsv', term_list=term_list, mode='train')
    # print(f'# of discussed sample: {n1}, # of not discussed sample: {n2}')

