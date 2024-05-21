import torch
from torch.utils.data import Dataset


class TextClassificationCollator():  # mini-batch 내에서 가장 긴 문장을 기준으로 padding

    def __init__(self, tokenizer, max_length, with_text=True):  
        self.tokenizer = tokenizer
        self.max_length = max_length  # 무한정 긴 문장을 받을 수는 없으므로, 최고 길이에 대한 hyperparameter 지정해줌 (ex : 255)
        self.with_text = with_text  
    
    def __call__(self, samples):  # samples에 아까 dataset이 return 한 것이 list로 들어있음. 아래에서 dict를 return 했으므로, dict에 list가 들어있음
        texts, labels = [], []
        for text, label in samples:  
            texts += [text]
            labels += [label]

        encoding = self.tokenizer(  # encoding의 결과값으로, mini-batch 내의 각 sample별, time-step별, 단어 인덱스가 들어있음
            texts,
            padding=True,
            truncation=True,  # True 인 경우 mini-batch 내에서 가장 긴 문장을 기준으로 padding
            return_tensors="pt",  # pytorch를 return 해줌
            max_length=self.max_length
        )

        return_value = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],  # pad 위치에 attention mask 해주기 위함
            'labels': torch.tensor(labels, dtype=torch.long),  # list로 있던 것을 torch.long type의 tensor로 바꿔줌
        }
        if self.with_text:
            return_value['text'] = texts

        return return_value


class TextClassificationDataset(Dataset):

    def __init__(self, texts, labels):  # text와 label 를 Dataset 객체가 들고 있음
        self.texts = texts  # 전체 text 데이터셋 (corpus)
        self.labels = labels  # 각 sample 별 label
    
    def __len__(self):  # 전체 sample이 몇 개 있는지 셈
        return len(self.texts)
    
    def __getitem__(self, item): # Dataset을 Dataloader에 넣어줄 건데, 
        text = str(self.texts[item])
        label = self.labels[item]

        return text, label 
