import os
import re
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from transformers import BertModel, TFBertModel, BertTokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import pandas as pd
import matplotlib.pyplot as plt

tf.random.set_seed(66)
np.random.seed(66)

BATCH_SIZE = 100
NUM_EPOCHS = 30
VALID_SPLIT = 0.08
MAX_LEN = 44
DATA_IN_PATH = 'data_in/KOR'
DATA_OUT_PATH = "data_out/KOR"

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", cache_dir='bert_ckpt', do_lower_case=False)

# data load
path = './03_dacon_comp/_data/'
train_data  = pd.read_csv(path+'train_data.csv') # (45654, 3) 
test_data  = pd.read_csv(path+'test_data.csv') # (9131, 2) 
submission  = pd.read_csv(path+'sample_submission.csv') # (7, 2)

def bert_tokenizer(sent, MAX_LEN):
    
    encoded_dict = tokenizer.encode_plus(
        text = sent,
        add_special_tokens = True, 
        max_length = MAX_LEN,           
        pad_to_max_length = True,
        return_attention_mask = True   
    )
    
    input_id = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']
    token_type_id = encoded_dict['token_type_ids'] 
    
    return input_id, attention_mask, token_type_id

input_ids = []
attention_masks = []
token_type_ids = []
train_data_labels = []

for train_sent, train_label in tqdm(zip(train_data["title"], train_data["topic_idx"]), total=len(train_data)):
    try:
        input_id, attention_mask, token_type_id = bert_tokenizer(train_sent, MAX_LEN)
        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        train_data_labels.append(train_label)

    except Exception as e:
        print(e)
        print(train_sent)
        pass

train_input_ids = np.array(input_ids, dtype=int)
train_attention_masks = np.array(attention_masks, dtype=int)
train_type_ids = np.array(token_type_ids, dtype=int)
train_inputs = (train_input_ids, train_attention_masks, train_type_ids)

train_data_labels = np.asarray(train_data_labels, dtype=np.int32) #레이블 토크나이징 리스트

class TFBertClassifier(tf.keras.Model):
    def __init__(self, model_name, dir_path, num_class):
        super(TFBertClassifier, self).__init__()

        self.bert = TFBertModel.from_pretrained(model_name, cache_dir=dir_path)
        self.dropout = tf.keras.layers.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(num_class, 
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(self.bert.config.initializer_range), 
                                                name="classifier")
        
    def call(self, inputs, attention_mask=None, token_type_ids=None, training=False):
        
        outputs = self.bert(inputs, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1] 
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)

        return logits

cls_model = TFBertClassifier(model_name='bert-base-multilingual-cased',
                                  dir_path='bert_ckpt',
                                  num_class=7)

from tensorflow.keras.optimizers import Adam

optimizer = Adam(3e-5)

lr = ReduceLROnPlateau(monitor='val_loss', patience=2,
    verbose=2, mode='auto', factor=0.8)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('acc')
cls_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

model_name = "tf2_bert"

es = EarlyStopping(monitor='val_loss', min_delta=0.0001,
     patience=5)

checkpoint_path = os.path.join(DATA_OUT_PATH, model_name, 'weights.h5')
checkpoint_dir = os.path.dirname(checkpoint_path)

if os.path.exists(checkpoint_dir):
    print("{} -- Folder already exists \n".format(checkpoint_dir))
else:
    os.makedirs(checkpoint_dir, exist_ok=True)
    print("{} -- Folder create complete \n".format(checkpoint_dir))
    
mcp = ModelCheckpoint(
    checkpoint_path, monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=True)

history = cls_model.fit(train_inputs, train_data_labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
                    validation_split = VALID_SPLIT, callbacks=[es, mcp, lr], verbose=2)

input_ids = []
attention_masks = []
token_type_ids = []
test_data_labels = []

for test_sent in tqdm(test_data["title"]):
    try:
        input_id, attention_mask, token_type_id = bert_tokenizer(test_sent, MAX_LEN)

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
    except Exception as e:
        print(e)
        print(test_sent)
        pass

test_input_ids = np.array(input_ids, dtype=int)
test_attention_masks = np.array(attention_masks, dtype=int)
test_type_ids = np.array(token_type_ids, dtype=int)
test_inputs = (test_input_ids, test_attention_masks, test_type_ids)

results = cls_model.predict(test_inputs, batch_size=256)

topic = []
for i in range(len(results)):
    topic.append(np.argmax(results[i]))

submission['topic_idx']=topic

submission.to_csv('bert_baseline.csv', index=False)