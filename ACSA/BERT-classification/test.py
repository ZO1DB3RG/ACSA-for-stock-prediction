import pandas as pd
import numpy
from argparse import ArgumentParser

from tqdm import tqdm

from data_loader import MyDataLoader
from torch import nn
import torch
import logging
import os
import tools
#from pytorch_pretrained_bert import BertTokenizer, BertModel
from model import MyClassifier
from trainer import Trainer
from torch.utils.data import DataLoader
from pytorch_transformers import BertTokenizer, BertModel
# from transformers import AutoTokenizer, AutoModelForSequenceClassification


def build_parser():
    parser = ArgumentParser()

    ##Common option
    parser.add_argument("--device", dest="device", default="gpu")

    ##Loader option
    parser.add_argument("--train_path", dest="train_path", default="/userhome/cs2/lzl0105/dl/NLP/Laptop-ACOS/laptop_quad_train_with_none.csv")
    parser.add_argument("--valid_path", dest="valid_path", default="/userhome/cs2/lzl0105/dl/NLP/Laptop-ACOS/laptop_quad_test_with_none_all.csv")
    parser.add_argument("--max_length", dest="max_length", default=256, type=int)
    parser.add_argument("--save_path", dest="save_path", default="model")

    ##Model option
    parser.add_argument("--bert_name", dest="bert_name", default="./bert-amazon") #bert-base-uncased
    parser.add_argument("--bert_finetuning", dest="bert_finetuning", default=True, type=bool)
    parser.add_argument("--dropout_p", dest="dropout_p", default=0.1, type=int)

    ##Train option
    parser.add_argument("--boost", dest="boost", default=True, type=bool)
    parser.add_argument("--n_epochs", dest="n_epochs", default=10, type=int)
    parser.add_argument("--lr_main", dest="lr_main", default=0.00001, type=int)
    parser.add_argument("--lr", dest="lr", default=0.001, type=int)
    parser.add_argument("--early_stop", dest="early_stop", default=1, type=int)
    parser.add_argument("--batch_size", dest="batch_size", default=8, type=int)
    parser.add_argument("--gradient_accumulation_steps", dest="gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--num_class", dest="num_class", type=int)

    config = parser.parse_args()
    return config

def run(config):
    def _print_config(config):
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))

    _print_config(config)

    if not logging.getLogger() == None:
        for handler in logging.getLogger().handlers[:]:  # make a copy of the list
            logging.getLogger().removeHandler(handler)

    if not os.path.isdir(config.save_path):
        os.mkdir(config.save_path)
    all_subdir = [int(s) for s in os.listdir(config.save_path) if os.path.isdir(os.path.join(config.save_path, str(s)))]
    max_dir_num = 0
    if all_subdir:
        max_dir_num = max(all_subdir)
    max_dir_num += 1
    config.save_path = os.path.join(config.save_path, str(max_dir_num))
    os.mkdir(config.save_path)

    logging.basicConfig(filename=os.path.join(config.save_path, 'train_log'),
                        level=tools.LOGFILE_LEVEL,
                        filemode='w')

    console = logging.StreamHandler()
    console.setLevel(tools.CONSOLE_LEVEL)
    logging.getLogger().addHandler(console)

    logging.info("##################### Start Load BERT MODEL")
    if config.bert_name == 'kobert':
        from kobert_modified_utills import get_kobert_model_and_tokenizer
        bert, tokenizer = get_kobert_model_and_tokenizer()
    else:
        tokenizer = BertTokenizer.from_pretrained(config.bert_name)
        bert = BertModel.from_pretrained(config.bert_name)
        # tokenizer = AutoTokenizer.from_pretrained("fabriceyhc/bert-base-uncased-amazon_polarity")
        #
        # bert = AutoModelForSequenceClassification.from_pretrained("fabriceyhc/bert-base-uncased-amazon_polarity")

    bert.to(config.device)

    ##load data loader
    logging.info("##################### Load DataLoader")
    loader = MyDataLoader(train_path=config.train_path,
                          valid_path=config.valid_path,
                          max_length=config.max_length,
                          tokenizer=tokenizer)

    train, valid, num_class = loader.get_train_valid_data()
    logging.info("##################### Train Dataset size : [" + str(len(train)) + "]")
    logging.info("##################### Valid Dataset size : [" + str(len(valid)) + "]")
    logging.info("##################### class size : [" + str(num_class) + "]")

    #modified batch size
    logging.info("##################### Accumulation batch size : [" + str(config.batch_size) + "]")
    config.batch_size = config.batch_size // config.gradient_accumulation_steps
    logging.info("##################### Modified batch size : [" + str(config.batch_size) + "]")


    logging.info("##################### Load 'BERT Classifier' Model")
    model = MyClassifier(bert=bert,
                         num_class=num_class,
                         bert_finetuning=config.bert_finetuning,
                         dropout_p=config.dropout_p,
                         device=config.device
                         )
    model.to(config.device)
    crit = nn.NLLLoss()
    trainer = Trainer(model=model,
                      crit=crit,
                      config=config,
                      boost=config.boost,
                      device=config.device)

    # If bert fine-tuning process is not necessary, convert text into vectors by using bert to make whole process fast
    if config.boost and not config.bert_finetuning:
        logging.info("##################### Transform Dataset into Vectors by using BERT")
        valid = loader.convert_ids_to_vector(data=valid,
                                             model=model,
                                             batch_size=config.batch_size,
                                             device=config.device)

    valid = DataLoader(dataset=valid, batch_size=config.batch_size, shuffle=False)

    model = trainer.get_best_model('/userhome/cs2/lzl0105/dl/Pytorch-BERT-Classification/model/10/model3.pwf')
    model.eval()
    total_correct = 0
    total_count = 0
    new_c = 0
    new_count = 0.0001
    labels = []
    preds = []
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    with torch.no_grad():
        progress_bar = tqdm(valid, desc='Validation: ', unit='batch')
        for batch in progress_bar:
            batch = tuple(t.to(config.device) for t in batch)
            input_ids, input_mask, segment_ids, label_id = batch
            label_hat = model(input_ids=input_ids,
                               segment_ids=segment_ids,
                               input_mask=input_mask,
                               boost=config.boost)
            total_count += int(label_id.size(0))
            ps = torch.exp(label_hat)
            top_p, top_class = ps.topk(1, dim=1)

            equals = top_class == label_id.view(*top_class.shape)
            total_correct += torch.sum(equals).item()

            new_count += torch.sum(label_id.ne(3)).item()
            mask = label_id.ne(3)
            new_equals = torch.masked_select(label_id, mask) == torch.masked_select(top_class.squeeze(), mask)
            new_c += torch.sum(new_equals).item()

            labels.extend(list(label_id.cpu().numpy()))
            preds.extend(list(top_class.squeeze().cpu().numpy()))
            avg_correct = total_correct / total_count
            progress_bar.set_postfix_str('correct=%.4f' % avg_correct)
        progress_bar.close()

    acc_without_none = new_c / new_count

    for i in range(len(labels)):
        if preds[i] != 3 and preds[i] == labels[i]:
            tp += 1
        elif preds[i] == 3 and preds[i] == labels[i]:
            tn += 1
        elif preds[i] != 3 and preds[i] != labels[i]:
            fp += 1
        elif preds[i] == 3 and preds[i] != labels[i]:
            fn += 1
    pr = tp / (tp + fp)
    ra = tp / (tp + fn)
    f1 = 2 * pr * ra / (pr + ra)

    return avg_correct, acc_without_none, pr, ra, f1

if __name__ == "__main__":
    ##load config files
    config = build_parser()
    config.device = torch.device(
        "cuda" if torch.cuda.is_available() and (config.device == 'gpu' or config.device == 'cuda') else "cpu")
    print(run(config))