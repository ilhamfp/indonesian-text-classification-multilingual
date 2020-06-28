# This source code is part of a final year undergraduate project
# on exploring Indonesian hate speech/abusive & sentiment text 
# classification using a multilingual language model
# 
# Checkout the full github repository: 
# https://github.com/ilhamfp/indonesian-text-classification-multilingual

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset, DataLoader

RANDOM_SEED=1

def set_random_seed_data(seed):
    RANDOM_SEED = seed

def lowercase(text):
    return text.lower()

def remove_nonaplhanumeric(text):
    text = re.sub('[^0-9a-zA-Z]+', ' ', text) 
    return text

def remove_unnecessary_char(text):
    text = re.sub('\n',' ',text) # Remove every '\n'
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',text) # Remove every URL
    text = re.sub('  +', ' ', text) # Remove extra spaces
    text = text.strip()
    return text

def preprocess_text(text):
    text = lowercase(text)
    text = remove_nonaplhanumeric(text)
    text = remove_unnecessary_char(text)
    return text

def load_dataset_indonesian(data_name='prosa', data_path=None, data_path_test=None):
    if data_name == 'prosa':
        train = pd.read_csv('../input/dataset-prosa/data_train_full.tsv', sep='\t', header=None)
        train = train.rename(columns={0: "text", 1: "label"})
        train = train[train['label'] != 'neutral']
        train['label'] = train['label'].apply(lambda x: 1 if x=='positive' else 0)
        train['text'] = train['text'].apply(lambda x: preprocess_text(x))

        test = pd.read_csv('../input/dataset-prosa/data_testing_full.tsv', sep='\t', header=None)
        test = test.rename(columns={0: "text", 1: "label"})
        test = test[test['label'] != 'neutral']
        test['label'] = test['label'].apply(lambda x: 1 if x=='positive' else 0)
        test['text'] = test['text'].apply(lambda x: preprocess_text(x))
            
    elif data_name == 'trip_advisor':
        if data_path == None:
            train = pd.read_csv('../input/dataset-tripadvisor/train_set.csv')
#             train = pd.read_csv('../input/remove-duplicate-tripadvisor/train_set.csv')
        else:
            train = pd.read_csv(data_path)
            
        train = train.rename(columns={"content": "text", "polarity": "label"})
        train['label'] = train['label'].apply(lambda x: 1 if x=="positive" else 0)
        train['text'] = train['text'].apply(lambda x: preprocess_text(x))
        
        if data_path_test == None:
            test = pd.read_csv('../input/dataset-tripadvisor/test_set.csv')
#             test = pd.read_csv('../input/remove-duplicate-tripadvisor/test_set.csv')
        else:
            test = pd.read_csv(data_path_test)
            
        test = test.rename(columns={"content": "text", "polarity": "label"})
        test['label'] = test['label'].apply(lambda x: 1 if x=="positive" else 0)
        test['text'] = test['text'].apply(lambda x: preprocess_text(x))

    elif data_name == 'toxic':
        if data_path == None:
            data = pd.read_csv('../input/simpler-preprocess-indonesian-hate-abusive-text/preprocessed_indonesian_toxic_tweet.csv')
        else:
            data = pd.read_csv(data_path)
            
        data['label'] = ((data['HS'] == 1) | (data['Abusive'] == 1)).apply(lambda x: int(x))
        data = data[['Tweet', 'label']]
        data = data.rename(columns={'Tweet': 'text'})

        X_train, X_test, y_train, y_test = train_test_split(data.text.values, 
                                                            data.label.values, 
                                                            test_size=0.1,
                                                            random_state=RANDOM_SEED,
                                                            stratify=data.label.values)
        train = pd.DataFrame({'text': X_train,
                              'label': y_train})

        test = pd.DataFrame({'text': X_test,
                             'label': y_test})
        
    print("~~~Train Data~~~")
    print('Shape: ', train.shape)
    print(train[0:2])
    print("\nLabel:")
    print(train.label.value_counts())
    
    print("\n~~~Test Data~~~")
    print('Shape: ', test.shape)
    print(test[0:4])
    print("\nLabel:")
    print(test.label.value_counts())
    return train, test
    
def load_dataset_foreign(data_name='yelp'):
    train = None
    if data_name == 'yelp':
        train = pd.read_csv('../input/yelp-review-dataset/yelp_review_polarity_csv/train.csv', header=None)
        train = train.rename(columns={0: "label", 1: "text"})
        train['label'] = train['label'].apply(lambda x: 1 if x==2 else 0)
        train['text'] = train['text'].apply(lambda x: preprocess_text(x))
    
    elif data_name == 'toxic':
        data = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv')
        data['toxic'] = data['toxic'].apply(lambda x: 1 if x>=0.5 else 0)

        data = data[['comment_text', 'toxic']]
        data = data.rename(columns={'comment_text': 'text',
                                    'toxic': 'label'})

        data_pos = data[data['label'] == 1]
        data_neg = data[data['label'] == 0]
        train = pd.concat([data_pos[0:152111], 
                           data_neg[0:152111]]).reset_index(drop=True)
        
        train['text'] = train['text'].apply(lambda x: preprocess_text(x))

     
    print("~~~Data~~~")
    print('Shape: ', train.shape)
    print(train[0:2])
    print("\nLabel:")
    print(train.label.value_counts())
    return train

def split_train_test(train_x, train_y, total_data=50, valid_size=0.2):
    train_x_split, valid_x_split, train_y_split, valid_y_split = train_test_split(train_x, 
                                                                                  train_y, 
                                                                                  test_size=valid_size,
                                                                                  random_state=RANDOM_SEED,
                                                                                  stratify=train_y)
    
        
    total_data_valid = int(np.floor(valid_size * total_data))
    total_data_train = total_data-total_data_valid

    train_x_split = train_x_split[:total_data_train]
    train_y_split = train_y_split[:total_data_train]
    valid_x_split = valid_x_split[:total_data_valid]
    valid_y_split = valid_y_split[:total_data_valid]
    
    return train_x_split, train_y_split, valid_x_split, valid_y_split
    
def load_features(data_path, total_data=50, valid_size=0.2):
    train_x = np.array([x for x in np.load('{}/train_text.npy'.format(data_path), allow_pickle=True)])
    train_y = pd.read_csv('{}/train_label.csv'.format(data_path)).label.values
    
    train_x_split, train_y_split, valid_x_split, valid_y_split = split_train_test(train_x,
                                                                                  train_y,
                                                                                  total_data=total_data,
                                                                                  valid_size=valid_size)
    return train_x_split, train_y_split, valid_x_split, valid_y_split
    

def load_experiment_features(data_path_indo,
                             data_path_foreign,
                             tipe='A', 
                             total_data=50, 
                             foreign_mult=1, 
                             valid_size=0.2,
                             ):
    ##########################
    # Load Preprocessed Data #
    ##########################
    if tipe == 'A':
        train_x, train_y, valid_x, valid_y = load_features(data_path_indo,
                                                           total_data=total_data, 
                                                           valid_size=valid_size)
        
    elif tipe == 'B':
        train_x, train_y, _, _ = load_features(data_path_foreign,
                                               total_data=total_data, 
                                               valid_size=valid_size)
        
        _, _, valid_x, valid_y = load_features(data_path_indo,
                                               total_data=total_data, 
                                               valid_size=valid_size)
        
    elif tipe == 'C':
        train_x_indo, train_y_indo, valid_x_indo, valid_y_indo = load_features(data_path_indo,
                                                                                total_data=total_data, 
                                                                                valid_size=valid_size)

        train_x_foreign, train_y_foreign, valid_x_foreign, valid_y_foreign = load_features(data_path_foreign,
                                                                                           total_data=int(total_data*foreign_mult), 
                                                                                           valid_size=valid_size)

        train_x = np.concatenate([
                    train_x_indo,
                    train_x_foreign,
                    ])

        train_y = np.concatenate([
                    train_y_indo,
                    train_y_foreign,
                ])

        valid_x = valid_x_indo

        valid_y = valid_y_indo
        

    test_x = np.array([x for x in np.load('{}/test_text.npy'.format(data_path_indo), allow_pickle=True)])
    test_y = pd.read_csv('{}/test_label.csv'.format(data_path_indo)).label.values

    #########################
    # Convert to dataloader #
    #########################
    batch_size = 32

    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    valid_data = TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    return train_loader, valid_loader, test_loader

def load_train_dataset(data_name, total_data=50, valid_size=0.2, is_foreign=False, remove_duplication=False):
    train = None
    if is_foreign:
        train = load_dataset_foreign(data_name)
    else:
        train, test = load_dataset_indonesian(data_name)
    
    if remove_duplication:
        print("Removing duplication...")
        print("Previous shape: ", train.shape)
        train = train.drop_duplicates(keep = 'first') 
        print("Current shape: ", train.shape)
        print("Duplicate removed.")
    
    train_x_split, train_y_split, valid_x_split, valid_y_split = split_train_test(train.text.values,
                                                                                  train.label.values,
                                                                                  total_data=total_data,
                                                                                  valid_size=valid_size)
    
    train_x_split = np.array([x for x in train_x_split])
    valid_x_split = np.array([x for x in valid_x_split])
    return train_x_split, train_y_split, valid_x_split, valid_y_split

def load_experiment_dataset(data_name_indo,
                            data_name_foreign,
                            tipe='A', 
                            total_data=50, 
                            foreign_mult=1, 
                            valid_size=0.2,
                            remove_duplication=False):
    
    #################
    # Load Raw Data #
    #################
    if tipe == 'A':
        train_x, train_y, valid_x, valid_y = load_train_dataset(data_name_indo,
                                                                total_data=total_data, 
                                                                valid_size=valid_size,
                                                                is_foreign=False,
                                                                remove_duplication=remove_duplication)
        
    elif tipe == 'B':
        train_x, train_y, _, _ = load_train_dataset(data_name_foreign,
                                                    total_data=total_data, 
                                                    valid_size=valid_size,
                                                    is_foreign=True,
                                                    remove_duplication=remove_duplication)
        
        _, _, valid_x, valid_y = load_train_dataset(data_name_indo,
                                                    total_data=total_data, 
                                                    valid_size=valid_size,
                                                    is_foreign=False,
                                                    remove_duplication=remove_duplication)
        
    elif tipe == 'C':
        train_x_indo, train_y_indo, valid_x_indo, valid_y_indo = load_train_dataset(data_name_indo,
                                                                                    total_data=total_data, 
                                                                                    valid_size=valid_size,
                                                                                    is_foreign=False,
                                                                                    remove_duplication=remove_duplication)

        train_x_foreign, train_y_foreign, valid_x_foreign, valid_y_foreign = load_train_dataset(data_name_foreign,
                                                                                                total_data=int(total_data*foreign_mult), 
                                                                                                valid_size=valid_size,
                                                                                                is_foreign=True,
                                                                                                remove_duplication=remove_duplication)

        train_x = np.concatenate([
                    train_x_indo,
                    train_x_foreign,
                    ])

        train_y = np.concatenate([
                    train_y_indo,
                    train_y_foreign,
                ])

        valid_x = valid_x_indo

        valid_y = valid_y_indo
        
    

    _, test = load_dataset_indonesian(data_name=data_name_indo)
    test_x = test.text.values
    test_x = np.array([x for x in test_x])
    test_y = test.label.values
    
    indices = np.arange(len(train_x))
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(indices)
    train_x = train_x[indices]
    train_y = train_y[indices]

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)
    
