#!/usr/bin/python 
# coding:utf-8 

import matplotlib.pyplot as plt; 
import tensorflow as tf;
import numpy as np;
from PIL import Image,ImageDraw;
import cv2;
import os;
from keras import metrics

# g_objs = ['pipe_normal_','chair_high_','truck_normal_','horse_normal_','music_normal_'];
# objs          = ['pipe','chair','truck','horse','music'];
# relationships = ['up_left','up_right','down_left','down_right','left','right','down','up'];
# BBoxes = np.load('data_bboxes.npy');
# Objs   = np.load('data_objs.npy');
# States = np.load('data_states.npy');
# for index in range(20):
#     img = np.array(Image.open('scene/sc' +str(index)+ '.png'));
#     #img = np.zeros((512,512,3), dtype = np.uint8)
#     for i_bbox in range(len(BBoxes[index])):
#         w = BBoxes[index][i_bbox][2];
#         h = BBoxes[index][i_bbox][3];
#         # cv2.line(img, (bbox[0],bbox[1]), (bbox[0]+w,bbox[1]), (0, 255,0),1);
#         # cv2.line(img, (bbox[0],bbox[1]), (bbox[0],bbox[1]+h), (0, 255,0),1);
#         # cv2.line(img, (bbox[0]+w,bbox[1]), (bbox[0]+w,bbox[1]+h), (0, 255,0),1);
#         # cv2.line(img, (bbox[0],bbox[1]+h), (bbox[0]+w,bbox[1]+h), (0, 255,0),1);
#         cv2.rectangle(img,(BBoxes[index][i_bbox][0],BBoxes[index][i_bbox][1]), (BBoxes[index][i_bbox][0]+w,BBoxes[index][i_bbox][1]+h),(255,0,0),2);
#         cv2.putText(img,objs[Objs[index][i_bbox]],(BBoxes[index][i_bbox][0],BBoxes[index][i_bbox][1]),cv2.FONT_HERSHEY_PLAIN,1.4,(111,111,255),1);
#     #Image.fromarray(img).show();
#     for rel in States[index]:
#         w1     = BBoxes[index][Objs[index].index(rel[1])][2];
#         h1     = BBoxes[index][Objs[index].index(rel[1])][3];
#         w2     = BBoxes[index][Objs[index].index(rel[2])][2];
#         h2     = BBoxes[index][Objs[index].index(rel[2])][3];
#         point1 = ( BBoxes[index][Objs[index].index(rel[1])][0]+w1/2 ,BBoxes[index][Objs[index].index(rel[1])][1]+h1/2);
#         point2 = ( BBoxes[index][Objs[index].index(rel[2])][0]+w2/2 ,BBoxes[index][Objs[index].index(rel[2])][1]+h2/2);
#         rel    = relationships[rel[0]];
#         cv2.line(img, point1, point2, (0, 255,0),1);
#         cv2.putText(img,rel,(  (point1[0]+point2[0])/2,  (point1[1]+point2[1])/2),cv2.FONT_HERSHEY_PLAIN,1.4,(222,222,11),1);
#     Image.fromarray(img).save('scene_box/box_sc'+str(index)+'.png');

# # ============ make data ===============
# def e2f(A_semantic_data):
#     res = np.zeros((len(A_semantic_data),4));
#     for i in range(len(A_semantic_data)):
#         if A_semantic_data[i][0]==1. or A_semantic_data[i][3]==1.:res[i][0]=1.;
#         if A_semantic_data[i][1]==1. or A_semantic_data[i][2]==1.:res[i][1]=1.;
#         if A_semantic_data[i][4]==1. or A_semantic_data[i][5]==1.:res[i][2]=1.;
#         if A_semantic_data[i][6]==1. or A_semantic_data[i][7]==1.:res[i][3]=1.;
#     return res;    

# img_names = os.listdir('./scene');
# # i=0;total=len(img_names)
# IMGs = [];
# for img_name in img_names:
#     IMGs.append( np.array(Image.open('./scene/'+img_name).convert('L').resize((96,96))) );
# # np.save('scene_img',np.array(IMGs));
# semantics = np.load('semantic_one.npy');
# visual_data   = [];
# Q_semantic_data = [];
# A_semantic_data = [];
# for i in range(len(IMGs)):
#     for state in semantics[i]:
#         visual_data.append(IMGs[i]);
#         Q_semantic_data.append(state[8:]);
#         A_semantic_data.append(state[0:8]);
# Y_A      = e2f(np.array(A_semantic_data));
# np.save('visual_data',np.array(visual_data));
# np.save('Q_semantic_data',np.array(Q_semantic_data));
# np.save('A_semantic_data',Y_A);

# X_G=[];Y_G=[];obj_question_helper=[];
# bboxes = np.load('data_bboxes.npy');states=np.load('data_states.npy');objs=np.load('data_objs.npy');
# for i in range(len(IMGs)):
#     for state in states[i]:
#         obj_question_helper.append(state[1]);
#         pos1 = bboxes[i][  objs[i].index(state[1])  ][0:2];
#         pos2 = bboxes[i][  objs[i].index(state[2])  ][0:2];
#         print pos1;print pos2;
#         Y_G.append([  pos1[0],pos1[1],pos2[0],pos2[1]  ]);
# for i in range(len(Y_A)):
#     obj_question = np.zeros(5);
#     obj_question[obj_question_helper[i]]=1.;
#     X_G.append(Y_A[i].tolist() + obj_question.tolist());
# np.save('X_G',np.array(X_G));
# np.save('Y_G',np.array(Y_G));    

# # ============ eval ===================
# def mean_pred(y_true, y_pred):
#     return K.mean(y_pred)

# ============ Model ===================
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Multiply, Dot
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import backend as K
from keras import optimizers;
from keras import losses;
from keras import metrics;

#-------------------  rec ----------------------
#_________ CNN ____________
X_input =Input(shape=(96,96,1));
X = Conv2D(32, kernel_size=3, strides=2, input_shape=(96,96,1), padding="same")(X_input);
X = LeakyReLU(alpha=0.2)(X);
X = Dropout(0.25)(X);
X = Conv2D(64, kernel_size=3, strides=2, padding="same")(X);
X = ZeroPadding2D(padding=((0,1),(0,1)))(X);
X = BatchNormalization(momentum=0.8)(X);
X = LeakyReLU(alpha=0.2)(X);
X = Dropout(0.25)(X);
X = Conv2D(128, kernel_size=3, strides=2, padding="same")(X);
X = BatchNormalization(momentum=0.8)(X);
X = LeakyReLU(alpha=0.2)(X);
X = Dropout(0.25)(X);
X = Conv2D(256, kernel_size=3, strides=1, padding="same")(X);
X = BatchNormalization(momentum=0.8)(X);
X = LeakyReLU(alpha=0.2)(X);
X = Dropout(0.25)(X);
X = Flatten()(X);
X = Dense(2048, activation='relu')(X);
X = Dense(1024, activation='relu')(X);
X = Dense(512,  activation='relu')(X);
X = Dense(72,   activation='relu')(X);
#_________ Semantic ____________
S_input =Input(shape=(5,),name='S_in');
S = Dense(72, activation='relu',name='S')(S_input);
V_input = Input(shape=(72,72),name='V_input');
#_________ Combine Visual and Semantic ___________
XS = Dot(axes=-1)([S,V_input]);
XS = Multiply(name='XS_2')([X,XS]);
#_________ Concept Learning Layers ______________
XS = Dense(72, activation='relu',name='A_t')(XS);
XS = Dense(48, activation='relu')(XS);
XS = Dense(36, activation='relu')(XS);
XS = Dense(16, activation='relu',name='XS_4')(XS);
Y  = Dense(4, activation='softmax')(XS);

rec_model  = Model(inputs=[X_input,S_input,V_input], outputs=Y);
sgd        = optimizers.SGD(lr=0.0000001, decay=0.0, momentum=0.9, nesterov=True);
# sgd = optimizers.Adam(lr=0.0000001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
# rec_model.compile(optimizer=sgd, loss=losses.categorical_hinge, metrics=[metrics.categorical_accuracy]);
rec_model.compile(optimizer=sgd, loss=losses.mse, metrics=[metrics.categorical_accuracy]);
#rec_model.summary();

#-------------------  gen ----------------------
#_________  CNN  ____________
X_input = Input(shape=(96,96,1));
X = Conv2D(32, kernel_size=3, strides=2, input_shape=(96,96,1), padding="same")(X_input);
X = LeakyReLU(alpha=0.2)(X);
X = Dropout(0.25)(X);
X = Conv2D(64, kernel_size=3, strides=2, padding="same")(X);
X = ZeroPadding2D(padding=((0,1),(0,1)))(X);
X = BatchNormalization(momentum=0.8)(X);
X = LeakyReLU(alpha=0.2)(X);
X = Dropout(0.25)(X);
X = Conv2D(128, kernel_size=3, strides=2, padding="same")(X);
X = BatchNormalization(momentum=0.8)(X);
X = LeakyReLU(alpha=0.2)(X);
X = Dropout(0.25)(X);
X = Conv2D(256, kernel_size=3, strides=1, padding="same")(X);
X = BatchNormalization(momentum=0.8)(X);
X = LeakyReLU(alpha=0.2)(X);
X = Dropout(0.25)(X);
X = Flatten()(X);
X = Dense(2048, activation='relu')(X);
X = Dense(1024, activation='relu')(X);
X = Dense(512, activation='relu')(X);
X = Dense(72, activation='relu')(X);
#_________ Semantic ___________
S_input = Input(shape=(4+5,));
S       = Dense(72, activation='relu')(S_input);
A_input = Input(shape=(72,72));
#_________ Combine Visual and Semantic ___________
XS = Dot(axes=-1)([S,A_input]);
XS = Multiply(name='XS_2')([X,XS]);
#_________ Concept Learning Layers ______________
XS = Dense(72, activation='relu',name='V_t')(XS);
XS = Dense(16, activation='relu')(XS);
Y  = Dense(4, activation='relu')(XS);

gen_model  = Model(inputs=[X_input,S_input,A_input], outputs=Y);
sgd    = optimizers.SGD(lr=0.00001, decay=0.0, momentum=0.9, nesterov=True);
# sgd = optimizers.Adam(lr=0.0000001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
gen_model.compile(optimizer=sgd, loss=losses.mse, metrics=[metrics.categorical_accuracy]);
# gen_model.compile(optimizer=sgd, loss=losses.binary_crossentropy, metrics=[metrics.categorical_accuracy]);


# =========== read data =================
X_img = np.load('visual_data.npy');
X_img = np.expand_dims(X_img, axis=3);
Y_A   = np.load('A_semantic_data.npy');
X_S   = np.load('Q_semantic_data.npy');
V_i   = np.random.random((len(X_S),72,72));
print V_i.shape;
#V_i   = np.expand_dims(V_i, axis=3);
A_i   = np.random.random((len(X_S),72,72));
X_G   = np.load('X_G.npy');
Y_G   = np.load('Y_G.npy');


# # =========== train models by step ==============
# T = 50;
# V_t   = np.random.random((T,72,72));
# A_t   = np.random.random((T,72,72));
# for t in range(len(X_img)/T - 1):
#     x_img = X_img[T*t:T*(t+1)];
#     x_S   = X_S[T*t:T*(t+1)];
#     x_G   = X_G[T*t:T*(t+1)];
#     y_A   = Y_A[T*t:T*(t+1)];
#     y_G   = Y_G[T*t:T*(t+1)];
#     rec_model.fit([x_img,x_S,V_t], [y_A], epochs=1,batch_size=5);
#     gen_model.fit([x_img,x_G,A_t], [y_G], epochs=1,batch_size=5);
#     V_t   = gen_model.get_weights()[-6];print V_t.shape;
#     A_t   = rec_model.get_weights()[-6];print A_t.shape;
#     V_t   = np.array([V_t]*T);print V_t.shape;
#     A_t   = np.array([A_t]*T);print A_t.shape;
# rec_model.save('model_reco.h5');
# gen_model.save('model_gene.h5');



# =========== train models ==============
rec_model.fit([X_img,X_S,V_i], [Y_A],epochs=3, batch_size=5);
rec_model.save('model_reco.h5');
gen_model.fit([X_img,X_G,A_i], [Y_G],epochs=3, batch_size=5);
gen_model.save('model_gene.h5');


# # ============ use model ===========
# from keras.models import Model, load_model;
# model_dete = load_model('model_reco.h5');
# test_img   = np.array(Image.open('scene/sc117.png').convert('L').resize((96,96)));
# test_img   = np.expand_dims(test_img, axis=3);
# test_question = np.array([[1.,0,1.,0,0]]);
# # test_question   = np.expand_dims(test_question, axis=2);
# print model_dete.predict([np.array([test_img]),test_question]);