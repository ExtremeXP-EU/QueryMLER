from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
import requests
import subprocess

def create_train_dataset(dataset, ground_truth, query, dataset_name, train_df_name):
    url = "http://localhost:8080/api/query"
    data = {
        "q" : query,
        "page" : 0,
        "offset" : 1,
        "training" : True
    }
    response = requests.post(url, params=data)
    print(response)
    # print(response.json())
    if response.status_code == 200:
        # Create train_dataset = ground_truth + candidates_not_in_groundtruth
        candidates = pd.read_csv(train_df_name) # ./data/candidates.csv
        print(candidates)
        # candidates['id1'] = candidates['id1'].str.replace(dataset_name, '')
        # candidates['id2'] = candidates['id2'].str.replace(dataset_name, '')
        print("p0")

        ground_truth = pd.read_csv(ground_truth, names=['id1', 'id2'])
        candidates['id1'] = str(candidates['id1'])
        candidates['id2'] = str(candidates['id2'])
        candidates['combined'] = candidates.apply(lambda row: '-'.join(sorted(map(str, row))), axis=1)
        ground_truth['combined'] = ground_truth.apply(lambda row: '-'.join(sorted(map(str, row))), axis=1)

        common_elements = set(candidates['combined']).intersection(ground_truth['combined'])

        mask = ~candidates['combined'].isin(common_elements)

        false_candidates = candidates[mask]

        false_candidates = false_candidates.drop(['combined'], axis=1)
        false_candidates['label'] = False
        ground_truth = ground_truth.drop(columns=['combined'], axis=1)
        ground_truth['label'] = True

        # Return train dataset --> csv (train_csv in train_and_evaluate() function) 
        train_df = pd.concat([ground_truth, false_candidates], ignore_index=True)
        train_df.to_csv(train_df_name, index=False)
    else:
        print(f"Error: {response.status_code}")


def train_and_evaluate(train_csv, model, epochs, batch_size,
                       learning_rate, max_seq_length, evaluation_metric,
                       confidence_threshold, top_k_predictions,
                       model_name, tokenizer_name,
                       class_weights=None, loss_func_type="CrossEntropyLoss"):
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_df = pd.read_csv(train_csv, delimiter=',', names=['s1', 's2', 'label'])
    train_df = train_df.dropna().reset_index(drop=True)
    train_df['label'] = train_df['label'].map({True: 1, False: 0})
    train_df = train_df.dropna().reset_index(drop=True)
    train_df, val_df = train_test_split(train_df, test_size=0.2)
    train_df["label"] = train_df["label"].astype(int)
    val_df["label"] = val_df["label"].astype(int)

    train_texts = train_df[['s1', 's2']].values.tolist()
    val_texts = val_df[['s1', 's2']].values.tolist()

    train_labels = train_df['label'].values
    val_labels = val_df['label'].values

    if model == "prajjwal1/bert-tiny": 
        # Tokenize texts
        print("AAAAAAAAAAAAAAAAAA")
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        train_encoded_dict = tokenizer.batch_encode_plus(train_texts, max_length=max_seq_length, padding='max_length', truncation=True)
        val_encoded_dict = tokenizer.batch_encode_plus(val_texts, max_length=max_seq_length, padding='max_length', truncation=True)


        train_input_ids = torch.tensor(train_encoded_dict['input_ids'])
        train_attention_masks = torch.tensor(train_encoded_dict['attention_mask'])
        train_labels = torch.tensor(train_labels)

        val_input_ids = torch.tensor(val_encoded_dict['input_ids'])
        val_attention_masks = torch.tensor(val_encoded_dict['attention_mask'])
        val_labels = torch.tensor(val_labels)

        # DataLoader for training data
        train_data = TensorDataset(train_input_ids, train_attention_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        val_data = TensorDataset(val_input_ids, val_attention_masks, val_labels)
        val_sampler = SequentialSampler(val_data)
        val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

        # Model init
        model = AutoModelForSequenceClassification.from_pretrained(
        model,
            num_labels=2, 
            output_attentions=False,
            output_hidden_states=False,
        )
        model.to(device)

        if class_weights is not None:
            class_weights = torch.tensor(class_weights)
            class_weights = class_weights.to(device)
        else:
            class_weights = None

        if loss_func_type == "CrossEntropyLoss":
            loss_func = torch.nn.CrossEntropyLoss(weight=class_weights)
        elif loss_func_type == "BCEWithLogitsLoss":
            loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            raise ValueError(f"Unsupported loss function type: {loss_func_type}")

        loss_func.to(device)

        optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * epochs)

        # Training loop
        for epoch_i in range(0, epochs):
            print("Epoch:", epoch_i + 1)
            total_loss = 0
            model.train()
            for step, batch in enumerate(train_dataloader):
                b_input_ids = batch[0].to(device)
                b_attention_masks = batch[1].to(device)
                b_labels = batch[2].to(device)
                model.zero_grad()
                outputs = model(b_input_ids, attention_mask=b_attention_masks, labels=b_labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            avg_train_loss = total_loss / len(train_dataloader)
            print("Average Training Loss: {:.4f}".format(avg_train_loss))

        # Validation
        model.eval()
        val_preds, val_true_labels = [], []

        for batch in val_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_attention_masks, b_labels = batch
            with torch.no_grad():
                outputs = model(b_input_ids, attention_mask=b_attention_masks)
            logits = outputs.logits
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            val_preds.extend(np.argmax(logits, axis=1).flatten())
            val_true_labels.extend(label_ids)

        # Calculate evaluation metric
        if evaluation_metric == "accuracy":
            eval_metric = accuracy_score(val_true_labels, val_preds)
        elif evaluation_metric == "f1_score":
            eval_metric = f1_score(val_true_labels, val_preds)
        elif evaluation_metric == "precision":
            eval_metric = precision_score(val_true_labels, val_preds)
        elif evaluation_metric == "recall":
            eval_metric = recall_score(val_true_labels, val_preds)
        else:
            raise ValueError(f"Unsupported evaluation metric: {evaluation_metric}")

        print(f'Validation {evaluation_metric.capitalize()}: {eval_metric}')

        model.save_pretrained(model_name)
        tokenizer.save_pretrained(tokenizer_name)
    else:
        print('BBBBBBBBBBBBBBBBBBB')
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        print(train_texts)
        train_encoded_dict = tokenizer.batch_encode_plus(train_texts, max_length=max_seq_length, padding='max_length', truncation=True)
        val_encoded_dict = tokenizer.batch_encode_plus(val_texts, max_length=max_seq_length, padding='max_length', truncation=True)

        train_input_ids = torch.tensor(train_encoded_dict['input_ids'])
        train_attention_masks = torch.tensor(train_encoded_dict['attention_mask'])
        train_labels = torch.tensor(train_labels)

        val_input_ids = torch.tensor(val_encoded_dict['input_ids'])
        val_attention_masks = torch.tensor(val_encoded_dict['attention_mask'])
        val_labels = torch.tensor(val_labels)

        # DataLoader for training data
        train_data = TensorDataset(train_input_ids, train_attention_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        val_data = TensorDataset(val_input_ids, val_attention_masks, val_labels)
        val_sampler = SequentialSampler(val_data)
        val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

        # Model init
        model = AutoModelForSequenceClassification.from_pretrained(
            model,
            num_labels=2, 
            output_attentions=False,
            output_hidden_states=False,
        )
        model.to(device)

        if class_weights is not None:
            class_weights = torch.tensor(class_weights)
            class_weights = class_weights.to(device)
        else:
            class_weights = None

        if loss_func_type == "CrossEntropyLoss":
            loss_func = torch.nn.CrossEntropyLoss(weight=class_weights)
        elif loss_func_type == "BCEWithLogitsLoss":
            loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            raise ValueError(f"Unsupported loss function type: {loss_func_type}")

        loss_func.to(device)

        optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * epochs)

        # Training loop
        for epoch_i in range(0, epochs):
            print("Epoch:", epoch_i + 1)
            total_loss = 0
            model.train()
            for step, batch in enumerate(train_dataloader):
                b_input_ids = batch[0].to(device)
                b_attention_masks = batch[1].to(device)
                b_labels = batch[2].to(device)
                model.zero_grad()
                outputs = model(b_input_ids, attention_mask=b_attention_masks, labels=b_labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            avg_train_loss = total_loss / len(train_dataloader)
            print("Average Training Loss: {:.4f}".format(avg_train_loss))

        # Validation
        model.eval()
        val_preds, val_true_labels = [], []

        for batch in val_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_attention_masks, b_labels = batch
            with torch.no_grad():
                outputs = model(b_input_ids, attention_mask=b_attention_masks)
            logits = outputs.logits
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            val_preds.extend(np.argmax(logits, axis=1).flatten())
            val_true_labels.extend(label_ids)

        # Calculate evaluation metric
        if evaluation_metric == "accuracy":
            eval_metric = accuracy_score(val_true_labels, val_preds)
        elif evaluation_metric == "f1_score":
            eval_metric = f1_score(val_true_labels, val_preds)
        elif evaluation_metric == "precision":
            eval_metric = precision_score(val_true_labels, val_preds)
        elif evaluation_metric == "recall":
            eval_metric = recall_score(val_true_labels, val_preds)
        else:
            raise ValueError(f"Unsupported evaluation metric: {evaluation_metric}")

        print(f'Validation {evaluation_metric.capitalize()}: {eval_metric}')

        model.save_pretrained(model_name)
        tokenizer.save_pretrained(tokenizer_name)        

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train():
    try:
        print("Received training request.")
        # Parse params
        data = request.get_json()

        dataset = data.get('dataset')        
        query = data.get('query')
        ground_truth = data.get('ground_truth')
        dataset_name = data.get('dataset_name')

        #train_csv = data.get('train_csv')
        train_csv = data.get('train_csv')
        model = data.get('model', 'distilbert-base-uncased')
        epochs = data.get('epochs', 1)
        batch_size = data.get('batch_size', 32)
        learning_rate = data.get('learning_rate', 0.001)
        max_seq_length = data.get('max_seq_length', 128)
        evaluation_metric = data.get('evaluation_metric', 'accuracy')
        confidence_threshold = data.get('confidence_threshold', 0.5)
        top_k_predictions = data.get('top_k_predictions', 3)
        class_weights = data.get('class_weights', [1.0, 5.0])
        loss_func_type = data.get('loss_func_type', 'CrossEntropyLoss')
        model_name = data.get('model_name')
        tokenizer_name = data.get('tokenizer_name')

        # Create train dataset
        create_train_dataset(dataset, ground_truth, query, dataset_name, train_csv)

        # Train
        train_and_evaluate(train_csv, model, epochs, batch_size,
                           learning_rate, max_seq_length, evaluation_metric,
                           confidence_threshold, top_k_predictions,
                           model_name, tokenizer_name,
                           class_weights=class_weights, loss_func_type=loss_func_type)

        return jsonify({"status": "success", "message": "Training completed successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/inference', methods=['POST'])
def inference():
    try:
        # Parse params
        data = request.get_json()
        dataset = data.get('dataset')
    #    model = data.get('model')
        model_name = data.get('model_name')
        tokenizer_name = data.get('tokenizer_name')
        batch_size = data.get('batch_size', 32)

        command = ['python3 llm_server.py', model_name, tokenizer_name, batch_size]

        result = subprocess.run(command)

        return jsonify({"status": "success", "message": "Deduplication completed successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    
if __name__ == '__main__':
    app.run(debug=True)
