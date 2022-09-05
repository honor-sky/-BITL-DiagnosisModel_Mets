import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


##################### Prepare Datasets #####################
data = pd.read_csv("D:/단국대학교/연구실/글로벌 핵심인재 양성/라이프로그 대사증후군 진단/MetS_dataset_20220120/data/features.csv")
label = pd.read_csv("D:/단국대학교/연구실/글로벌 핵심인재 양성/라이프로그 대사증후군 진단/MetS_dataset_20220120/data/label.csv")

f_selction_m=["sex","sbp","waist","bmi","dbp","weight","whr","pulse","hip","age","G3_INT","sm_total","regrp8","regrp4","dr_total",
            "regrp9" ,"exer_merge","dr_soju","w032","regrp3","regrp35"]

f_selction_f=["sex","sbp","waist","bmi","dbp","weight","whr","pulse","hip","age","G3_INT","sm_total","regrp8","regrp4","dr_total",
            "regrp9" ,"exer_merge","dr_soju","w032","regrp3","regrp35"]

data_m = data[f_selction_m]
data_f = data[f_selction_f]

data_m = pd.concat((data_m,pd.DataFrame(label[["bp_bi","waist_bi"]])),1) #혈압, 허리둘레 label feature로 추가
data_f = pd.concat((data_f,pd.DataFrame(label[["bp_bi","waist_bi"]])),1)

##setting label
data_m = pd.concat((data_m,pd.DataFrame(label["hdl_bi"])),1)
data_f = pd.concat((data_f,pd.DataFrame(label["hdl_bi"])),1)
##split dataset by sex
data_m = data_m[data_m['sex']==1]  #male dataset
data_f = data_f[data_f['sex']==2]  #female dataset
data_m = data_m.drop(["sex"],1)
data_f = data_f.drop(["sex"],1)
#Train, Test Data Split
train_X_m, test_X_m, train_y_m, test_y_m = train_test_split(data_m.loc[:, data_m.columns != 'hdl_bi'],data_m.loc[:, 'hdl_bi'], test_size=0.4, random_state=1234)
train_X_f, test_X_f, train_y_f, test_y_f = train_test_split(data_f.loc[:, data_f.columns != 'hdl_bi'],  data_f.loc[:, 'hdl_bi'], test_size=0.4,  random_state=1234)


def printAcc(model, train_X, train_y, test_X, test_y):
    print("Train accuracy :", model.score(train_X, train_y))
    print("Test accuracy :", model.score(test_X_m, test_y_m))
    print()


##################### Define & Train Model Function#####################
def AdaBoostModel(train_X, train_y, test_X, test_y, sex): ### AdaBoost ###
    for i in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300]:
        if (sex == 0): # man
            model_rf = AdaBoostClassifier(n_estimators=i, random_state=123)
            model_rf.fit(train_X, train_y)
        else:
            model_rf = AdaBoostClassifier(n_estimators=i, random_state=123)
            model_rf.fit(train_X, train_y)

        pred = pd.DataFrame(model_rf.predict(test_X))  # 예측
        cm2 = confusion_matrix(test_y, pred)  # 혼동행렬 구하기
        cmdf2 = pd.DataFrame(cm2, index=['실제값(N)', '실제값(P)'], columns=['예측값(N)', '예측값(P)'])
        print(cmdf2)
        print(i)
        print(sex)
        printAcc(model_rf, train_X, train_y, test_X, test_y)



def XGBoostModel(train_X, train_y, test_X, test_y, sex): ###  XGBoost ###
    # 파라미터 여러개 설정
    param_grid = {'booster': ['gbtree'],
                  'n_estimators': [50, 200, 500, 800],
                  'max_depth': [5, 6, 8],
                  'min_child_weight': [1, 3, 5],
                  'gamma': [0, 1, 2, 3],
                  'objective': ['binary:logistic'],
                  'random_state': [123]}

    if (sex == 0): #man
        model_rf = XGBClassifier(use_label_encoder=False)  # model define
        gcv = GridSearchCV(model_rf, param_grid=param_grid, scoring='accuracy', n_jobs=-1)
        gcv.fit(train_X_m, train_y_m, eval_metric='error')  # model train
    else:
        model_rf = XGBClassifier(use_label_encoder=False)
        gcv = GridSearchCV(model_rf, param_grid=param_grid, scoring='accuracy', n_jobs=-1)
        gcv.fit(train_X_f, train_y_f, eval_metric='error')

    print(sex)
    print('final params', gcv.best_params_)  # 최적의 파라미터 값 출력
    print('best score', gcv.best_score_)  # 최고의 점수
    printAcc(gcv, train_X, train_y, test_X, test_y)



def Ensanble(train_X, train_y, test_X, test_y, sex): ### Model Stacking ###

    if (sex == 0):
        knn = KNeighborsClassifier()
        rf = RandomForestClassifier(random_state=1)
        svm = make_pipeline(StandardScaler(), SVC(random_state=1, probability=True))
        final = LogisticRegression(max_iter=500)

        knn.fit(train_X, train_y)
        rf.fit(train_X, train_y)
        svm.fit(train_X, train_y)

        pred_knn = knn.predict_proba(test_X)
        pred_rf = rf.predict_proba(test_X)
        pred_svm = svm.predict_proba(test_X)

        new_input = np.array([pred_knn[:, 0], pred_rf[:, 0], pred_svm[:, 0]])
        #print(new_input.shape)
        new_input = np.transpose(new_input)
        print(new_input.shape)

        final.fit(new_input, test_y)
        final = final.predict(new_input)

    else:
        knn = KNeighborsClassifier()
        rf = RandomForestClassifier(random_state=1)
        svm = make_pipeline(StandardScaler(), SVC(random_state=1, probability=True))
        final = LogisticRegression(max_iter=500)

        knn.fit(train_X, train_y)
        rf.fit(train_X, train_y)
        svm.fit(train_X, train_y)

        pred_knn = knn.predict_proba(test_X)
        pred_rf = rf.predict_proba(test_X)
        pred_svm = svm.predict_proba(test_X)

        new_input = np.array([pred_knn[:, 0], pred_rf[:, 0], pred_svm[:, 0]])
        # print(new_input.shape)
        new_input = np.transpose(new_input)
        print(new_input.shape)

        final.fit(new_input, test_y)
        final = final.predict(new_input)


    print(sex)
    accuracy = accuracy_score(test_y, final)
    print(accuracy)



def RFModel(train_X, train_y, test_X, test_y, sex):
    acc = []
    i = 1
    ntree = [300]
    mtry = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    if (sex == 0):
        for n in ntree:
            for m in mtry:
                model_rf = RandomForestClassifier(random_state=1234, n_estimators=n, max_features=m,
                                                    min_samples_leaf=8)
                # model_rf = RandomForestClassifier(random_state=123)
                # undersample = RandomUnderSampler(sampling_strategy='majority')

                smote = SMOTE()
                X_train_over, y_train_over = smote.fit_resample(train_X, train_y) #오버샘플링/언더샘플링
                print(len(y_train_over[y_train_over["hdl_bi"] == 1]) / len(y_train_over))

                # train
                model_rf.fit(X_train_over, y_train_over)

                # test
                pred = pd.DataFrame(model_rf.predict(test_X))  # 예측
                cm2 = confusion_matrix(test_y, pred)  # 혼동행렬 구하기
                cmdf2 = pd.DataFrame(cm2, index=['실제값(N)', '실제값(P)'], columns=['예측값(N)', '예측값(P)'])
                print(cmdf2)

                print(n, m)
                printAcc(model_rf, train_X, train_y, test_X, test_y)

    else:
        for n in ntree:
            for m in mtry:
                model_rf = RandomForestClassifier(random_state=1234, n_estimators=n, max_features=m,
                                                min_samples_leaf=8)
                # model_rf = RandomForestClassifier(random_state=1234)
                # undersample = RandomUnderSampler(sampling_strategy='majority')

                smote = SMOTE()
                X_train_over, y_train_over = smote.fit_resample(train_X, train_y)
                # print(len(y_train_over[y_train_over["hdl_bi"] == 1]) / len(y_train_over))

                model_rf.fit(X_train_over, y_train_over)

                # test
                pred = pd.DataFrame(model_rf.predict(test_X))  # 예측
                cm2 = confusion_matrix(test_y, pred)  # 혼동행렬 구하기
                cmdf2 = pd.DataFrame(cm2, index=['실제값(N)', '실제값(P)'], columns=['예측값(N)', '예측값(P)'])
                print(cmdf2)

                print(n, m)
                printAcc(model_rf, train_X, train_y, test_X, test_y)



def DNNModel():
    epochs = 1500
    batch_size = 900
    X_m = data_m.loc[:, data_m.columns != 'hdl_bi']
    y_m = data_m['hdl_bi']
    X_f = data_f.loc[:, data_f.columns != 'hdl_bi']
    y_f = data_f['hdl_bi']

    # one-hot encoding
    dummy_y_m = np_utils.to_categorical(y_m)
    dummy_y_f = np_utils.to_categorical(y_f)

    # Divide train, test
    train_X_m, test_X_m, train_y_m, test_y_m = train_test_split(X_m, dummy_y_m, test_size=0.4, random_state=1234)
    train_X_f, test_X_f, train_y_f, test_y_f = train_test_split(X_f, dummy_y_f, test_size=0.4, random_state=1234)

    # smote = SMOTE()
    # X_train_over_m, y_train_over_m = smote.fit_resample(train_X_m, train_y_m)  # 오버샘플링
    # X_train_over_f, y_train_over_f = smote.fit_resample(train_X_f, train_y_f)

    # define model
    model_m = Sequential()
    model_f = Sequential()

    model_m.add(Dense(22, input_dim=22, activation='sigmoid'))  # 활성함수 다 시그모이드
    model_m.add(Dense(8, activation='sigmoid'))
    model_m.add(Dense(8, activation='sigmoid'))
    model_m.add(Dense(8, activation='sigmoid'))
    model_m.add(Dense(2, activation='softmax'))  # categorical_crossentropy 사용

    model_f.add(Dense(22, input_dim=22, activation='sigmoid'))  # 활성함수 다 시그모이드
    model_f.add(Dense(8, activation='sigmoid'))
    model_f.add(Dense(8, activation='sigmoid'))
    model_f.add(Dense(8, activation='sigmoid'))
    model_f.add(Dense(2, activation='softmax'))  # categorical_crossentropy 사용

    model_m.summary()  # show model structure

    # Compile model
    tf.random.set_seed(2)
    model_m.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    model_f.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

    # model fitting(learning)
    disp_m = model_m.fit(train_X_m, train_y_m, batch_size=batch_size, epochs=epochs, verbose=1,
                         validation_data=(test_X_m, test_y_m))
    disp_f = model_f.fit(train_X_f, train_y_f, batch_size=batch_size, epochs=epochs, verbose=1,
                         validation_data=(test_X_f, test_y_f))

    # Test model
    pred_m = model_m.predict(test_X_m)
    pred_f = model_f.predict(test_X_f)

    classes_f = [np.argmax(y, axis=None, out=None) for y in pred_f]
    classes_m = [np.argmax(y, axis=None, out=None) for y in pred_m]  # 확률 높은게 1이 되어서 one-hot encoding 했던 것으로 구분

    test_y_m = np.argmax(test_y_m, axis=1)
    cm2 = confusion_matrix(test_y_m, pd.DataFrame(classes_m))  # 혼동행렬 구하기
    cmdf2 = pd.DataFrame(cm2, index=['실제값(N)', '실제값(P)'], columns=['예측값(N)', '예측값(P)'])
    print(cmdf2)

    test_y_f = np.argmax(test_y_f, axis=1)
    cm2 = confusion_matrix(test_y_f, pd.DataFrame(classes_f))  # 혼동행렬 구하기
    cmdf2 = pd.DataFrame(cm2, index=['실제값(N)', '실제값(P)'], columns=['예측값(N)', '예측값(P)'])
    print(cmdf2)

    # model performance
    score_f = model_f.evaluate(test_X_f, test_y_f, verbose=0)
    score_m = model_m.evaluate(test_X_m, test_y_m, verbose=0)
    print("Test loss for man: ", score_m[0])
    print("Test acc for man: ", score_m[1])
    print("Test loss for female: ", score_f[0])
    print("Test acc for female: ", score_f[1])



def FeatureImp(model, train_X, sex):
    print(sex,end='\n')
    ftr_importances_values = model.feature_importances_
    ftr_importances = pd.Series(ftr_importances_values, index=train_X.columns)
    ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
    plt.figure(figsize=(8, 6))
    plt.title('Top 20 Feature Importances')
    sns.barplot(x=ftr_top20, y=ftr_top20.index)
    plt.show()



##################### Summarize History for Acc #####################
'''
plt.plot(disp_f.history['accuracy'])  #female
plt.plot(disp_f.history['val_accuracy'])
plt.title('model accuracy_Female')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()

plt.plot(disp_m.history['accuracy'])  #man
plt.plot(disp_m.history['val_accuracy'])
plt.title('model accuracy_Man')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()
'''
