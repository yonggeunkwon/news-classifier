import sys
import argparse
import csv
import json
import requests

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data

from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification, AlbertForSequenceClassification

from datetime import datetime, timedelta
import pytz

def yesterday_datetime():
    korea_timezone = pytz.timezone('Asia/Seoul')
    now_in_korea = datetime.now(korea_timezone)
    yesterday_in_korea = now_in_korea - timedelta(days=1)
    year = yesterday_in_korea.year
    month = yesterday_in_korea.month
    day = yesterday_in_korea.day
    return f"{year}{month}{day}"


def define_argparser():
    '''
    Define argument parser to take inference using pre-trained model.
    '''
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--top_k', type=int, default=1)

    config = p.parse_args()

    return config


def read_text_from_csv(filepath):
    '''
    Read text from standard input for inference.
    '''
    titles = []
    urls = []
    with open(filepath, 'r') as f:
        next(f)  # Skip the header if it exists
        for line in f:
            # Assuming the text is in the first column of the csv
            text = line.split(',')[1].strip()
            url = line.split(',')[0].strip()
            if text:
                titles.append(text)
                urls.append(url)

    return titles, urls


def read_text_from_json(filepath):
    titles = []
    urls = []
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        articles = data['articles']
        for key in articles:
            title = articles[key]['title']
            url = articles[key]['urls']
            titles.append(title)
            urls.append(url)

    return titles, urls



def read_json_file(file_path: str):
    with open(file_path, 'r') as f:
        return json.load(f)
    
    
def post_data(url, data):
    response = requests.post(url, json=data)
    return response


def main(config):
    saved_data = torch.load(
        config.model_fn,
        map_location='cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id
    )

    train_config = saved_data['config']
    bert_best = saved_data['bert']
    index_to_label = saved_data['classes']
    print('index_to_label : ', index_to_label)

    titles, urls = read_text_from_json('../data/20231128.json')

    with torch.no_grad():
        # Declare model and load pre-trained weights.
        tokenizer = BertTokenizerFast.from_pretrained(train_config.pretrained_model_name)
        model_loader = AlbertForSequenceClassification if train_config.use_albert else BertForSequenceClassification
        model = model_loader.from_pretrained(
            train_config.pretrained_model_name,
            num_labels=len(index_to_label)
        )
        model.load_state_dict(bert_best)

        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
        device = next(model.parameters()).device

        # Don't forget turn-on evaluation mode.
        model.eval()

        y_hats = []
        for idx in range(0, len(titles), config.batch_size):
            mini_batch = tokenizer(
                titles[idx:idx + config.batch_size],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            x = mini_batch['input_ids']
            x = x.to(device)
            mask = mini_batch['attention_mask']
            mask = mask.to(device)

            # Take feed-forward
            y_hat = F.softmax(model(x, attention_mask=mask).logits, dim=-1)

            y_hats += [y_hat]
        # Concatenate the mini-batch wise result
        y_hats = torch.cat(y_hats, dim=0)
        # |y_hats| = (len(titles), n_classes)

        probs, indice = y_hats.cpu().topk(config.top_k)
        # |indice| = (len(titles), top_k)

        # Make csv file
        # with open('../data/result_with_gpt_data.csv', 'w') as csvfile:
        #     csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            
        #     header = ['title', 'url', 'topic_idx']
        #     csv_writer.writerow(header)

        #     for i in range(len(titles)):
        #         labels = ' '.join([index_to_label[int(indice[i][j])] for j in range(config.top_k)])
        #         csv_writer.writerow([titles[i], urls[i], labels])

        "-----------------------make json file--------------------------"

        news_data = {"classified-news": {}}

        for i in range(len(titles)):
            labels = ' '.join([index_to_label[int(indice[i][j])] for j in range(config.top_k)])
            news_data["classified-news"]["news" + str(i + 1)] = {
                "title": titles[i],
                "urls": urls[i],
                "topic_idx": labels
            }
        
        
        # Write the dictionary to a JSON file (yesterday date)
        yesterday = yesterday_datetime()
        save_classified_news_path = f'../data/result_20231128.json'

        with open(save_classified_news_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(news_data, jsonfile, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    config = define_argparser()
    main(config)
