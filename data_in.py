import pandas as pd 
import cPickle as pk 
import keras.preprocessing.sequence as prep
import first_model as models
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import random
import datetime
import keras
from sklearn.metrics import confusion_matrix, accuracy_score

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
# model.load_weights("3.0_new_data_first_try_2017-10-16T12:12:17.560919_train_38000_test_2000_epoch_20_acc_90_best.h5")

def getRejectFiles(n=2000): 
    fles = glob.glob('/home/arya_01/ICICI_Lombard/INPUT_DATA/19215_v2/19215_data/*_Rejected.pkl')
    rejected_files = fles
    # for i in fles:
    #     if '_reject' in i.split('/')[-1].lower():
    #         rejected_files.append(i)
    random.shuffle(rejected_files)
    return rejected_files[:n]

fles = glob.glob('/home/arya_01/ICICI_Lombard/INPUT_DATA/12_oct/data_83230/*.pkl')
fles += glob.glob('/home/arya_01/ICICI_Lombard/INPUT_DATA/12_oct/moved_13705/*.pkl')
fles += glob.glob('/home/arya_01/ICICI_Lombard/INPUT_DATA/12_oct/moved_TEST_2_2746/*.pkl')
# fles += getRejectFiles(5000)
# fles += getRejectFiles(5000)
# random.shuffle(fles)
# fles = fles[:64]
fles_train, fles_test = train_test_split(fles, test_size=0.01, random_state=42)
len_train, len_test = len(fles_train), len(fles_test)

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
        pk.dump([list_N, _df], open("_tmp","wb"))
        raise e


arr_multiplier = np.array([[0,1]] * len(list_D))
def combine_date_fields(df, list_D):
    _df = []
    for i in list_D:
        _df.append(df[i[0]])
    arr = np.asarray(zip(*_df))
    _random = [random.randrange(-50,50,1) for _ in range (arr.shape[0])]
    arr += np.asarray([arr_multiplier * i for i in _random])
    return arr
    # return np.log(arr + 51)

def combine_sing_fields(df, list_sing):
    _df = []
    for i in list_sing:
        _df.append(df[i[0]])
    return np.asarray(zip(*_df)).reshape((len(_df[0]), len(_df)))


def testing(model):
    batch_size = 32
    counter = 0
    df = []
    input_data = []
    epochs = 1
    batch_num = 1
    test_acc = []
    y_pred = []
    y_true = []
    for ii in xrange(0,len(fles_test),batch_size):
        counter += 1
        # df = pd.concat([ (pk.load(open(_k, 'rb'))[1]) for _k in fles_test[ii:ii+batch_size]])
        _df = []
        for _k in fles_test[ii:ii+batch_size]:
            _pkl_load = pk.load(open(_k, 'rb'))
            if isinstance(_pkl_load, tuple): 
                _dd = _pkl_load[1]
            else:
                _dd = _pkl_load
            # if _dd['POLICY_CLASSIFICATION'].lower() == 'corporate':
            _df.append(_dd)
        try: 
            df = pd.concat(_df)
            # print "Batch size: ", len(_df)
        except Exception as e:
            continue

        try: 
            1/0
            output_data = np.array(df['AD_STATUS_O'].tolist())
        except Exception as e:
            # print 'AD_STATUS_O is absent'
            output_data = np.array([[0,1] if i[0].lower()=='a' else [1,0] for i in df['AD_STATUS'].tolist()])
        # output_data = output_data.T[0]
        for col_name, row, col in data_shape:
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
        # test_acc.append(model.test_on_batch(input_data, output_data))
        y_pred += list(np.floor(model.predict(input_data, batch_size=input_data[0].shape[0])+0.5).T[0].astype(int))
        y_true += list(output_data.T[0])
        test_acc.append(accuracy_score(y_true, y_pred))
        # print test_acc[-1]
        k=output_data.sum(axis=0)
        # print batch_num, 1.*k[0]/k.sum(), 1.*k[1]/k.sum()

        # print 1.0 * output_data.sum()/output_data.shape[0], 1- (1.0 * output_data.sum()/output_data.shape[0])
        # print model.evaluate(input_data, output_data, batch_size=batch_size, verbose=1, sample_weight=None)
        input_data = []
    # print "Confusion Matrix: ", confusion_matrix(y_true, y_pred)
    # print "sum: ", sum(y_true)
    return np.array(test_acc).mean(axis=0)



time_now = datetime.datetime.now().isoformat()
batch_size = 32
# fles_train = fles_train[:33]
num_batch = len(fles_train)//batch_size
counter = 0
input_data = []
epochs = 80
best_test_acc = 0
_start = 1
if _start == 1: 
    with open("loss_acc_small.csv", 'w') as of:
        of.write("Loss, Accuracy\n")
for epoch in range(_start,_start+epochs):
    batch_num = 0
    loss, acc = [0,0]
    # for fle in fles_train: 
    for ii in xrange(0,len(fles_train),batch_size):
        counter += 1
        # df = pd.concat([ (pk.load(open(_k, 'rb'))[1]) for _k in fles_train[ii:ii+batch_size]])
        _df = []
        for _k in fles_train[ii:ii+batch_size]:
            _pkl_load = pk.load(open(_k, 'rb'))
            if isinstance(_pkl_load, tuple): 
                _dd = _pkl_load[1]
            else:
                _dd = _pkl_load
            # if _dd['POLICY_CLASSIFICATION'].lower() == 'corporate':
            _df.append(_dd)
        try: 
            df = pd.concat(_df)
            # print "Batch size: ", len(_df)
        except Exception as e:
            continue

        try: 
            1/0
            output_data = np.array(df['AD_STATUS_O'].tolist())
        except Exception as e:
            # print 'AD_STATUS_O is absent'
            output_data = np.array([[0,1] if i[0].lower()=='a' else [1,0] for i in df['AD_STATUS'].tolist()])
        # output_data = output_data.T[0]
        for col_name, row, col in data_shape:
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
            # if 'AMOUNT' in str(col_data):
            #     print col_name,row,col
            # else:
            #     input_data.append(pad(col_data,maxlen=row))
        try: 
            k=output_data.sum(axis=0)
        except Exception as e:
            print output_data
            raise e
        # tb_CallB = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
        # model.fit(input_data, output_data, callbacks=[tb])
        batch_num += 1
        batch_loss, batch_acc = model.train_on_batch(input_data, output_data)
        print epoch,batch_num, num_batch,"\t", 1.*k[0]/k.sum()," ", 1.*k[1]/k.sum(), batch_loss, batch_acc
        loss += batch_loss
        acc += batch_acc
        # print epoch,batch_num, num_batch, 1.0 * output_data.sum()/output_data.shape[0], 1- (1.0 * output_data.sum()/output_data.shape[0])
        # model.fit(input_data, output_data,batch_size=batch_size,epochs=1,verbose=1,initial_epoch=epoch)
        # print model.train_on_batch(input_data, output_data)
        # testing(model)
        dd = []
        input_data = []
        # print raw_input("")
    with open("loss_acc_small.csv", 'a') as of:
        of.write("%f, %f\n" % (loss/batch_num, acc/batch_num))

        # """ testing while training 
    _acc = 100*testing(model)
    # print _acc, type(_acc)
    print "Testing accuracy: %f \n Best accuracy so far: %f" % (_acc, best_test_acc)
    if _acc > best_test_acc: 
        model.save_weights('5.1_network_change_complete_data_%s_train_%d_test_%d_epoch_%d_acc_%d_best.h5'%(time_now, len_train, len_test, epoch, _acc))
        print "new model saved"
        best_test_acc = _acc
        # """
    model.save_weights('5.1_network_change_complete_data_%s_train_%d_test_%d_epoch_%d.h5'%(time_now, len_train, len_test, epoch))
    # testing(model)
            # break





# df = pk.load(open('/home/arya_01/ICICI_Lombard/test1000.pkl', 'rb'))
# data_shape = pk.load(open('/home/arya_01/ICICI_Lombard/shape_list', 'rb'))

# _,input_dim_array,feat_dim_array = zip(*data_shape)
# model = model.fc_cnn_model(input_dim_array, feat_dim_array)
# counter = 0
# batch_size = 12
# num_batch = df.shape[0]//batch_size
# epochs = 10
# # num_batch = 1
# print "started training"
# for epoch in range(epochs):
#     for i in range(num_batch):
#         input_data = []
#         # for _, row in df[i:i+batch_size].iterrows():
#         output_data = np.array(df[i:i+batch_size]['AD_STATUS_C'].tolist())
#         for col_name, row, col in data_shape:
#             col_data = df[i:i+batch_size][col_name].tolist()
#             # print col_data,"data"
#             if col == 0: 
#                 input_data.append(np.array(col_data))
#             elif col == 1: 
#                 input_data.append(np.array(col_data))
#             else:
#                 input_data.append(pad(col_data,maxlen=row))
#         model.fit(input_data, output_data,batch_size=batch_size,epochs=1,verbose=1,initial_epoch=epoch)

