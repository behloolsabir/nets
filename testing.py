import pandas as pd 
import pickle as pk 
import keras.preprocessing.sequence as prep
import first_model as models
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import random
import datetime
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import sys

def pad(inp, maxlen):
    return prep.pad_sequences(inp,maxlen=maxlen,padding='post')

data_shape = pk.load(open('/home/arya_01/ICICI_Lombard/INPUT_DATA/12_oct/shape_list.pkl', 'rb'))



list_T = []
_data_shape = []
for i in data_shape:
    if i[0][-2:] == '_T':
        list_T.append(i)
    else:
        _data_shape.append(i)
data_shape = _data_shape
data_shape.append(['grand_all_text_T', len(list_T), list_T[0][1]])

list_N = []
_data_shape = []
for i in data_shape:
    if i[0][-2:] == '_N':
        list_N.append(i)
    else:
        _data_shape.append(i)
data_shape = _data_shape
data_shape.append(['grand_all_number_N', len(list_N), 1])

list_D = []
_data_shape = []
for i in data_shape:
    if i[0][-2:] == '_D':
        list_D.append(i)
    else:
        _data_shape.append(i)
data_shape = _data_shape
data_shape.append(['grand_all_date_D', len(list_D), 2])

list_DF = []
_data_shape = []
for i in data_shape:
    if i[0][-3:] == '_DF':
        list_DF.append(i)
    else:
        _data_shape.append(i)
data_shape = _data_shape
# data_shape.append(['grand_all_date_D', len(list_DF), 2])

list_sing = []
_data_shape = []
for i in data_shape:
    if i[0][-2:] == '_C' and i[1]==1 :
        list_sing.append(i)
    else:
        _data_shape.append(i)
data_shape = _data_shape
data_shape.append(['grand_all_sing_D', len(list_sing), 1])


for i in data_shape:
    if i[1]>100:
        i[1] = 100

_,input_dim_array,feat_dim_array = zip(*data_shape)
model = models.fc_cnn_model(input_dim_array, feat_dim_array)
model.load_weights("5.1_network_change_complete_data_2017-11-11T19:27:29.322412_train_98376_test_994_epoch_%s.h5" % (sys.argv[1]))
# print model.summary()
# import sys 
# sys.exit(0)
# fles = glob.glob("/home/arya_01/ICICI_Lombard/DATA/TEST_1_2514_encoded_with_27_aug_encoders/*.pkl")

# fles = glob.glob("/home/arya_01/ICICI_Lombard/DATA/TEST_2_2746_encoded_with_27_aug_encoders/*.pkl")

fles = glob.glob('/home/arya_01/ICICI_Lombard/INPUT_DATA/12_oct/moved_TEST_1_2514/*.pkl')
# fles_train, fles_test = train_test_split(fles, test_size=0.05, random_state=4)
fles_test = fles
# print len(fles_test)
# total_batches = len(fles)/batch_size

def combine_text_fields(df, list_T):
    _df = []
    for i in list_T:
        _df.append(df[i[0]])
    return np.asarray(zip(*_df))

def combine_number_fields(df, list_N):
    _df = []
    for i in list_N:
        _df.append(df[i[0]])
    try: 
        return np.log(np.asarray(zip(*_df)) + 1)
    except Exception as e:
        import cPickle as cpk
        cpk.dump([list_N, _df], open("_tmp","wb"))
        raise e


# arr_multiplier = np.array([[0,1]] * len(list_D))
# def combine_date_fields(df, list_D):
#     _df = []
#     for i in list_D:
#         _df.append(df[i[0]])
#     arr = np.asarray(zip(*_df))
#     _random = [random.randrange(-50,50,1) for _ in range (arr.shape[0])]
#     arr += np.asarray([arr_multiplier *i for i in _random])
#     return np.log(arr + 51)
def combine_date_fields(df, list_D):
    _df = []
    for i in list_D:
        _df.append(df[i[0]])
    arr = np.asarray(zip(*_df))
    # _random = [random.randrange(-50,50,1) for _ in range (arr.shape[0])]
    # arr += np.asarray([arr_multiplier *i for i in _random])
    return arr 
    # return np.log(arr + 1)

def combine_sing_fields(df, list_sing):
    _df = []
    for i in list_sing:
        _df.append(df[i[0]])
    return np.asarray(zip(*_df)).reshape((len(_df[0]), len(_df)))


def testing(model):
    batch_size = 32
    num_batch = len(fles_test)//batch_size
    counter = 0
    df = []
    input_data = []
    epochs = 1
    batch_num = 1
    test_acc = []
    y_pred = []
    y_pred_prob = []
    y_true = []
    fail_count=0
    # _total_len = len(fles_test)
    # print _total_len
    for ii in xrange(0,len(fles_test),batch_size):
        counter += 1
        # df = pd.concat([ (pk.load(open(_k, 'rb'))[1]) for _k in fles_test[ii:ii+batch_size]])
        _df = []
        for _k in fles_test[ii:ii+batch_size]:
            _dd = pk.load(open(_k, 'rb'))[1]
            if _dd['POLICY_CLASSIFICATION'].lower() == 'corporate':
                _df.append(_dd)
        try: 
            df = pd.concat(_df)
            print "Batch size: ", len(_df)
        except Exception as e:
            continue
        try: 
            1/0
            output_data = np.array(df['AD_STATUS_O'].tolist())
        except Exception as e:
            # print 'AD_STATUS_O is absent'
            output_data = np.array([[0,1] if i[0].lower()=='a' else [1,0] for i in df['AD_STATUS'].tolist()])
        print "Done %d/%d" % (counter, num_batch)
        # output_data = output_data.T[0]
        _counter = 0
        for col_name, row, col in data_shape:
            _counter += 1
            # print _counter, col_name, row, col
            if col_name == 'grand_all_date_D':
                input_data.append(combine_date_fields(df, list_D))
            elif col_name == 'grand_all_text_T':
                input_data.append(combine_text_fields(df, list_T))
            elif col_name == 'grand_all_number_N':
                input_data.append(combine_number_fields(df, list_N))
            elif col_name == 'grand_all_sing_D':
                input_data.append(combine_sing_fields(df, list_sing))
            elif col == 0:
                print 'adfasfasfasdfa'
                col_data = df[col_name].tolist()
                input_data.append(np.array(col_data))
            elif col == 1: 
                col_data = df[col_name].tolist()
                input_data.append(np.array(col_data))
            elif row < 10000:
                col_data = df[col_name].tolist()
                input_data.append(pad(col_data,maxlen=row))
            else: 
                with open('many_category_col_name.txt', 'a') as of:
                    of.write(col_name+'\n')
        try:
            # test_acc.append(model.test_on_batch(input_data, output_data))
            _prob = model.predict(input_data, batch_size=input_data[0].shape[0])
            y_pred_prob += list(_prob.T[0])
            y_pred += list(np.floor(_prob+0.5).T[0].astype(int))
            k=output_data.sum(axis=0)
            y_true += list(output_data.T[0])
            test_acc.append(accuracy_score(y_true, y_pred))
            # print y_pred, y_true
            # import sys
            # sys.exit()
        except Exception as e:
            # import sys
            # pk.dump(input_data,open("input_data","wb"))
            # sys.exit()
            print e 
            #raise e 
            # print fle
            fail_count+=1
        input_data = []
        # print test_acc[-1]
        # print batch_num, 1.*k[0]/k.sum(), 1.*k[1]/k.sum()
        # print 1.0 * output_data.sum()/output_data.shape[0], 1- (1.0 * output_data.sum()/output_data.shape[0])
        # print model.evaluate(input_data, output_data, batch_size=batch_size, verbose=1, sample_weight=None)
    print test_acc
    print "Confusion Matrix: ", confusion_matrix(y_true, y_pred)
    print "sum: ", sum(y_true)
    print "Mean loss and acc", np.array(test_acc).mean()
    print "F1 score: ", f1_score(y_true, y_pred)

    import csv
    with open('probability_corporate.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(["predicted_prob", "truth"])
        writer.writerows(zip(y_pred_prob, y_true))

if __name__ == '__main__':
    testing(model)
# try:
#     weights = sys.ARGV[1]
# except Exception as e:
#     weights = raw_input('Supply weights: \n')
# model.load_weights('fcnn_categorical_crossentropy_adamax_2_epoch_82.h5')
# testing(model)