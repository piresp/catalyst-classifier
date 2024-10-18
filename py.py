# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 14:10:56 2024

@author: piresp
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def drop_columns(df, columns):
    if isinstance(columns, str):
        columns = [columns]
    return df.drop(columns=columns)

def drop_nan_in_columns(df, columns):
    return df.dropna(subset=columns)

def remove_zero_rows(df, columns):
    for column in columns:
        df = df[df[column] != 0]
    return df

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Confusion Matrix: {title}')
    plt.show()

def plot_training_history(history):
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

df_bom_vm = pd.read_csv('bom_vm.csv', delimiter=',')
df_ruim_vm = pd.read_csv('ruim_vm.csv', delimiter=',')
df_vm_no_label = pd.read_csv('sinais_vm.csv', delimiter=',')

df_bom_vm = drop_columns(df_bom_vm, ['Combustivel_usado_unidade_L', 'Tempo_unidade_s'])
df_ruim_vm = drop_columns(df_ruim_vm, ['Combustivel_usado_unidade_L', 'Tempo_unidade_s'])
df_vm_no_label = drop_columns(df_vm_no_label, ['Combustivel_usado_unidade_L', 'Tempo_unidade_s'])

column_drop_zero_rows = ['Consumo_de_combustivel_unidade_Lporh', 'Velocidade_do_veiculo_unidade_kmporh', 'Distancia_percorrida_pelo_veiculo_unidade_km']

df_bom_vm = remove_zero_rows(df_bom_vm, column_drop_zero_rows)
df_ruim_vm = remove_zero_rows(df_ruim_vm, column_drop_zero_rows)
df_vm_no_label = remove_zero_rows(df_vm_no_label, column_drop_zero_rows)

column_drop_nan = ['Porcentagem_de_torque_atual_do_motor_unidade_%','Nivel_de_gases_NOx_na_saida_do_catalisador_unidade_ppm', 'Oxigenio_na_saida_do_catalisador_unidade_%']

df_bom_vm = drop_nan_in_columns(df_bom_vm, column_drop_nan)
df_ruim_vm = drop_nan_in_columns(df_ruim_vm, column_drop_nan)
df_vm_no_label = drop_nan_in_columns(df_vm_no_label, column_drop_nan)

df_bom_vm['label'] = 0
df_ruim_vm['label'] = 1

df_combined = pd.concat([df_bom_vm, df_ruim_vm], axis=0)
df_combined = df_combined.reset_index(drop=True)

X = df_combined.drop(columns=['label'])
y = df_combined['label']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(44, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=8, batch_size=12, validation_split=0.2, verbose=1)

y_pred_train = (model.predict(X_train) > 0.5).astype(int)
y_pred_test = (model.predict(X_test) > 0.5).astype(int)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
train_precision = precision_score(y_train, y_pred_train)
test_precision = precision_score(y_test, y_pred_test)
train_recall = recall_score(y_train, y_pred_train)
test_recall = recall_score(y_test, y_pred_test)
train_f1 = f1_score(y_train, y_pred_train)
test_f1 = f1_score(y_test, y_pred_test)

print(f'\nTrain Accuracy: {train_accuracy:.2f}')
print(f'Train Recall: {train_recall:.2f}')
print(f'Train Precision: {train_precision:.2f}')
print(f'Train F1 Score: {train_f1:.2f}\n')
print(f'Test Accuracy: {test_accuracy:.2f}')
print(f'Test Precision: {test_precision:.2f}')
print(f'Test Recall: {test_recall:.2f}')
print(f'Test F1 Score: {test_f1:.2f}')

plot_confusion_matrix(y_test, y_pred_test, "Test Set")
plot_confusion_matrix(y_train, y_pred_train, "Train Set")

plot_training_history(history)

model.save('model_catalizador.keras')
