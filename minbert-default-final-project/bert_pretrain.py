'''
为了实现Bert-base的进一步预训练
'''


import random, numpy as np, argparse
from types import SimpleNamespace
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)



TQDM_DISABLE=False


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Pretrain mode does not require updating BERT paramters.
        for param in self.bert.parameters():
            param.requires_grad = True
        self.config=config
        self.prelinear1=nn.Linear(config.hidden_size,50)
        self.prelinear2=nn.Linear(50,config.vocab_size)





    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        last_hidden_state=self.bert(input_ids,attention_mask)['last_hidden_state']
        last_hidden_state=F.gelu(self.prelinear1(last_hidden_state))
        predict_vector=F.gelu(self.prelinear2(last_hidden_state))
        predict_vector=predict_vector.transpose(1,2)
        return predict_vector


def save_model(model, args, config, filepath):
    save_info = {
        'model': model.bert.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

def get_random_mask(ids,attention_masks,vocabsize):
    origin_b_ids=deepcopy(ids)
    pos_seq=[]
    for i in range(ids.shape[0]):
        for j in range(ids.shape[1]):
            if (j+1==ids.shape[1]) or (attention_masks[i,j]==1 & attention_masks[i,j+1]==0):
                number=int(j*0.15) if int(j*0.15)>0 else 1
                positions=random.sample(range(j),k=number)
                for position in positions:
                    if ids[i][position]==101 or ids[i][position]==102:
                        positions.remove(position)
                pos_seq.append(sorted(positions))
                for position in positions:
                    choice=random.choices([0,1,2],[0.8,0.1,0.1])[0]
                    if choice==0:
                        ids[i,position]=103
                    elif choice==1:
                        ids[i,position]=random.sample(range(vocabsize),1)[0]
                break
    b_labels=torch.zeros(size=ids.shape)
    for index,i in enumerate(pos_seq):
        for j in i:
            b_labels[index,j]=origin_b_ids[index,j]
    b_labels=b_labels.long()

    
    return (b_labels,ids)

def model_eval_multitask(sentiment_dataloader,
                         paraphrase_dataloader,
                         sts_dataloader,
                         config,
                         model, device):
    model.eval()

    with torch.no_grad():
        sst_y_true = []
        sst_y_pred = []
        for step, batch in enumerate(tqdm(sentiment_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            b_ids, b_mask, b_sent_ids = batch['token_ids'], batch['attention_mask'],  batch['sent_ids']
            #注意深浅拷贝的区别，防止原始对象被函数修改
            b_labels,mask_ids=get_random_mask(b_ids,b_mask,config.vocab_size)#对于cross_entropy，labels是one-hot或者index均可
            mask_ids = mask_ids.to(device)
            b_mask = b_mask.to(device)
            logits=model.forward(mask_ids,b_mask)
            logits=logits.transpose(1,2)
            y_hat = logits.argmax(dim=-1).flatten().cpu().tolist()
            b_labels = b_labels.flatten().cpu().tolist()
            for i,j in enumerate(b_labels):
                if j!=0:
                    sst_y_true.append(b_labels[i])
                    sst_y_pred.append(y_hat[i])
        print(sst_y_true)
        print(sst_y_pred)
        sentiment_accuracy = np.mean(np.array(sst_y_pred) == np.array(sst_y_true))

        # Evaluate paraphrase detection.
        para_y_true = []
        para_y_pred = []
        for step, batch in enumerate(tqdm(paraphrase_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            (b_ids1, b_mask1,
             b_ids2,b_mask2,
             b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                                      batch['token_ids_2'], batch['attention_mask_2'],
                                        batch['labels'], batch['sent_ids'])
            b_labels,mask_ids=get_random_mask(b_ids1,b_mask1,config.vocab_size)#对于cross_entropy，labels是one-hot或者index均可
            mask_ids = mask_ids.to(device)
            b_mask1 = b_mask1.to(device)
            logits=model.forward(mask_ids,b_mask1)
            logits=logits.transpose(1,2)
            y_hat = logits.argmax(dim=-1).flatten().cpu().tolist()
            b_labels = b_labels.flatten().cpu().tolist()
            for i,j in enumerate(b_labels):
                if j!=0:
                    para_y_true.append(b_labels[i])
                    para_y_pred.append(y_hat[i])
            
            b_labels,mask_ids=get_random_mask(b_ids2,b_mask2,config.vocab_size)#对于cross_entropy，labels是one-hot或者index均可
            mask_ids = mask_ids.to(device)
            b_mask2 = b_mask2.to(device)
            logits=model.forward(mask_ids,b_mask2)
            logits=logits.transpose(1,2)
            y_hat = logits.argmax(dim=-1).flatten().cpu().tolist()
            b_labels = b_labels.flatten().cpu().tolist()
            for i,j in enumerate(b_labels):
                if j!=0:
                    para_y_true.append(b_labels[i])
                    para_y_pred.append(y_hat[i])
        paraphrase_accuracy = np.mean(np.array(para_y_pred) == np.array(para_y_true))

        # Evaluate semantic textual similarity.
        sts_y_true = []
        sts_y_pred = []
        for step, batch in enumerate(tqdm(sts_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            (b_ids1, b_mask1,
             b_ids2,b_mask2,
             b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                                      batch['token_ids_2'], batch['attention_mask_2'],
                          batch['labels'], batch['sent_ids'])
            b_labels,mask_ids=get_random_mask(b_ids1,b_mask1,config.vocab_size)#对于cross_entropy，labels是one-hot或者index均可
            mask_ids = mask_ids.to(device)
            b_mask1 = b_mask1.to(device)
            logits=model.forward(mask_ids,b_mask1)
            logits=logits.transpose(1,2)
            y_hat = logits.argmax(dim=-1).flatten().cpu().tolist()
            b_labels = b_labels.flatten().cpu().tolist()
            for i,j in enumerate(b_labels):
                if j!=0:
                    sts_y_true.append(b_labels[i])
                    sts_y_pred.append(y_hat[i])
            
            b_labels,mask_ids=get_random_mask(b_ids2,b_mask2,config.vocab_size)#对于cross_entropy，labels是one-hot或者index均可
            mask_ids = mask_ids.to(device)
            b_mask2 = b_mask2.to(device)
            logits=model.forward(mask_ids,b_mask2)
            logits=logits.transpose(1,2)
            y_hat = logits.argmax(dim=-1).flatten().cpu().tolist()
            b_labels = b_labels.flatten().cpu().tolist()
            for i,j in enumerate(b_labels):
                if j!=0:
                    sts_y_true.append(b_labels[i])
                    sts_y_pred.append(y_hat[i])
        sts_accuracy = np.mean(np.array(sts_y_pred) == np.array(sts_y_true))
        print(f'Sentiment dataset accuracy: {sentiment_accuracy:.3f}')
        print(f'Paraphrase dataset accuracy: {paraphrase_accuracy:.3f}')
        print(f'Semantic Similarity dataset accuracy: {sts_accuracy:.3f}')

        return (sentiment_accuracy,paraphrase_accuracy,sts_accuracy)


def train_multitask(args):

    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='dev')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
    
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)
    
    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)
    # Init model.
    config={"architectures": ["BertForMaskedLM"],
            "attention_probs_dropout_prob": 0.1,
            "gradient_checkpointing": False,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "bert",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "position_embedding_type": "absolute",
            "transformers_version": "4.6.0.dev0",
            "type_vocab_size": 2,
            "use_cache": True,
            "vocab_size": 30522,
            'data_dir': '.',
            'option': args.option,
            'num_labels': num_labels,
            'name_or_path':'pretrain'}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)#学习了固定为1e-4
    best_dev_acc = 0

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        
        for batch in tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask, b_labels = (batch['token_ids'],
                                       batch['attention_mask'], batch['labels'])
            original_ids=deepcopy(b_ids)
            b_labels,mask_ids=get_random_mask(b_ids,b_mask,config.vocab_size)#对于cross_entropy，labels是one-hot或者index均可
            mask_ids = mask_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels=b_labels.to(device)
            optimizer.zero_grad()
            logits=model.forward(mask_ids,b_mask)
            loss=F.cross_entropy(logits,b_labels,ignore_index=0)# 忽略label为0的类导致的loss，因为是padding
            loss.backward()
            optimizer.step()#即使forward中有循环，仍然在更新。但是的确p.grad始终为None。
            train_loss += loss.item()
            num_batches += 1
        '''
        for step, batch in enumerate(tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)):
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                                      batch['token_ids_2'], batch['attention_mask_2'],
                                    batch['labels'], batch['sent_ids'])
            b_labels,mask_ids=get_random_mask(b_ids1,b_mask1,config.vocab_size)#对于cross_entropy，labels是one-hot或者index均可
            mask_ids = mask_ids.to(device)
            b_mask1 = b_mask1.to(device)
            b_labels=b_labels.to(device)
            optimizer.zero_grad()
            logits=model.forward(mask_ids,b_mask1)
            loss=F.cross_entropy(logits,b_labels,ignore_index=0)# 忽略label为0的类导致的loss，因为是padding
            loss.backward()
            optimizer.step()#即使forward中有循环，仍然在更新。但是的确p.grad始终为None。
            train_loss += loss.item()

            b_labels,mask_ids=get_random_mask(b_ids2,b_mask2,config.vocab_size)#对于cross_entropy，labels是one-hot或者index均可
            mask_ids = mask_ids.to(device)
            b_mask2 = b_mask2.to(device)
            b_labels=b_labels.to(device)
            optimizer.zero_grad()
            logits=model.forward(mask_ids,b_mask2)
            loss=F.cross_entropy(logits,b_labels,ignore_index=0)# 忽略label为0的类导致的loss，因为是padding
            loss.backward()
            optimizer.step()#即使forward中有循环，仍然在更新。但是的确p.grad始终为None。
            train_loss += loss.item()
            num_batches += 1
          '''
        for step, batch in enumerate(tqdm(sts_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)):
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                                      batch['token_ids_2'], batch['attention_mask_2'],
                                    batch['labels'], batch['sent_ids'])
            b_labels,mask_ids=get_random_mask(b_ids1,b_mask1,config.vocab_size)#对于cross_entropy，labels是one-hot或者index均可
            mask_ids = mask_ids.to(device)
            b_mask1 = b_mask1.to(device)
            b_labels=b_labels.to(device)
            optimizer.zero_grad()
            logits=model.forward(mask_ids,b_mask1)
            loss=F.cross_entropy(logits,b_labels,ignore_index=0)# 忽略label为0的类导致的loss，因为是padding
            loss.backward()
            optimizer.step()#即使forward中有循环，仍然在更新。但是的确p.grad始终为None。
            train_loss += loss.item()

            b_labels,mask_ids=get_random_mask(b_ids2,b_mask2,config.vocab_size)#对于cross_entropy，labels是one-hot或者index均可
            mask_ids = mask_ids.to(device)
            b_mask2 = b_mask2.to(device)
            b_labels=b_labels.to(device)
            optimizer.zero_grad()
            logits=model.forward(mask_ids,b_mask2)
            loss=F.cross_entropy(logits,b_labels,ignore_index=0)# 忽略label为0的类导致的loss，因为是padding
            loss.backward()
            optimizer.step()#即使forward中有循环，仍然在更新。但是的确p.grad始终为None。
            train_loss += loss.item()
            num_batches += 1

        
        senti_accuracy,para_accuracy,sts_accuracy=model_eval_multitask(sst_dev_dataloader,para_dev_dataloader,sts_dev_dataloader,config,model,device)
        if senti_accuracy+para_accuracy+sts_accuracy>best_dev_acc:
            best_dev_acc = senti_accuracy+para_accuracy+sts_accuracy
            state_dict=model.bert.state_dict()
            state_dict.pop('position_ids')
            old_keys=list(state_dict.keys())
            new_keys=[]
            for key in old_keys:
              new_keys.append('bert.'+key)
            new_state_dict={}
            for i,j in zip(new_keys,old_keys):
              new_state_dict[i]=state_dict[j]
            print(new_state_dict.keys())
            torch.save(new_state_dict,'pytorch_model.bin')

            

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')#这个参数必须在终端调用时明确加入

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)#命令行中需要指定，pretrain是1e-5

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.epochs}-{args.lr}-prebert.pt' # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train_multitask(args)
            