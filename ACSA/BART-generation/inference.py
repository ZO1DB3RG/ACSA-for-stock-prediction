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

def inference(model, device, input_dir,
              term_list=['quality', 'connectivity', 'usability', 'portability', 'miscellaneous', 'general', 'operation performance', 'design features', 'price']):
    candidate_list = ["positive", "neutral", "negative",'none']
    model.eval()
    model.config.use_cache = False
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    with open(input_dir, "r") as f:
        file = f.readlines()
    total = 0
    all_dict = {}
    for term in term_list:
        all_dict[term] = []

    for line in tqdm.tqdm(file):
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
            all_dict[term].append(predict)
            # print(line, f'-------> term: {term}: {predict}')
    return all_dict

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AspectAnything(
        encoder_decoder_type="bart",
        encoder_decoder_name="./outputs_none_origin",
    ).model.to(device)
    input_data = pd.read_csv('./lg.csv')
    input_data.fillna('None. ', inplace=True)
    input_text = list(input_data['text'].values)
    with open('sample_infer.txt', 'w') as f:
        f.write('\n'.join(input_text))
    all_dict = inference(model, device, 'sample_infer.txt')

    senti_df = pd.DataFrame(all_dict)
    output_df = pd.concat([input_data, senti_df], axis=1)
    output_df.to_csv('./out_csv/lg_best.csv')



