from flask import Flask, request, jsonify
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib


# Carregar o modelo treinado
best_model = joblib.load('xgboostModel.pkl')
label_encoder = joblib.load('label_encoderUser.pkl')

app = Flask(__name__)

def convert_to_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return float(value)  # Valor padrão em caso de erro de conversão

@app.route('/')
def index():
    return "API de previsão está funcionando!"

@app.route('/predict', methods=['POST'])
def predict():
    # Receber os valores do formulário
    sessao = convert_to_float(request.form.get('sessao'))
    repeticao = convert_to_float(request.form.get('repeticao'))
    Duracao_v = convert_to_float(request.form.get('Duracao.v'))
    PP_v_i = convert_to_float(request.form.get('PP.v_i'))
    SS_v_i = convert_to_float(request.form.get('SS.v_i'))
    PS_v_i = convert_to_float(request.form.get('PS.v_i'))
    Duracao_i = convert_to_float(request.form.get('Duracao.i'))
    PP_i_d = convert_to_float(request.form.get('PP.i_d'))
    SS_i_d = convert_to_float(request.form.get('SS.i_d'))
    PS_i_d = convert_to_float(request.form.get('PS.i_d'))
    Duracao_d = convert_to_float(request.form.get('Duracao.d'))
    PP_d_a = convert_to_float(request.form.get('PP.d_a'))
    SS_d_a = convert_to_float(request.form.get('SS.d_a'))
    PS_d_a = convert_to_float(request.form.get('PS.d_a'))
    Duracao_a = convert_to_float(request.form.get('Duracao.a'))
    PP_a_ponto = convert_to_float(request.form.get('PP.a_ponto'))
    SS_a_ponto = convert_to_float(request.form.get('SS.a_ponto'))
    PS_a_ponto = convert_to_float(request.form.get('PS.a_ponto'))
    Duracao_ponto = convert_to_float(request.form.get('Duracao.ponto'))
    PP_ponto_n = convert_to_float(request.form.get('PP.ponto_n'))
    SS_ponto_n = convert_to_float(request.form.get('SS.ponto_n'))
    PS_ponto_n = convert_to_float(request.form.get('PS.ponto_n'))
    Duracao_n = convert_to_float(request.form.get('Duracao.n'))
    PP_n_o = convert_to_float(request.form.get('PP.n_o'))
    SS_n_o = convert_to_float(request.form.get('SS.n_o'))
    PS_n_o = convert_to_float(request.form.get('PS.n_o'))
    Duracao_o = convert_to_float(request.form.get('Duracao.o'))
    PP_o_v = convert_to_float(request.form.get('PP.o_v'))
    SS_o_v = convert_to_float(request.form.get('SS.o_v'))
    PS_o_v = convert_to_float(request.form.get('PS.o_v'))
    Duracao_v2 = convert_to_float(request.form.get('Duracao.v2'))
    PP_v_a = convert_to_float(request.form.get('PP.v_a'))
    SS_v_a = convert_to_float(request.form.get('SS.v_a'))
    PS_v_a = convert_to_float(request.form.get('PS.v_a'))
    Media_Duracao = convert_to_float(request.form.get('Media_Duracao'))
    Media_PP = convert_to_float(request.form.get('Media_PP'))
    Media_PS = convert_to_float(request.form.get('Media_PS'))
    Media_SS = convert_to_float(request.form.get('Media_SS'))

    # Criar um array numpy com os valores recebidos
    input_query = np.array([
        [sessao, repeticao, Duracao_v, PP_v_i, SS_v_i, PS_v_i, Duracao_i, PP_i_d, SS_i_d, PS_i_d,
         Duracao_d, PP_d_a, SS_d_a, PS_d_a, Duracao_a, PP_a_ponto, SS_a_ponto,
         PS_a_ponto, Duracao_ponto, PP_ponto_n, SS_ponto_n, PS_ponto_n, Duracao_n,
         PP_n_o, SS_n_o, PS_n_o, Duracao_o, PP_o_v, SS_o_v, PS_o_v, Duracao_v2,
         PP_v_a, SS_v_a, PS_v_a, Media_Duracao, Media_PP, Media_PS, Media_SS]
    ])

    # Fazer a previsão usando o melhor modelo
    result = best_model.predict(input_query)
    predicted_labels = label_encoder.inverse_transform(result)
    # Retornar o resultado da previsão
    return jsonify({'placement': predicted_labels.tolist()})  # Convertendo o resultado para lista antes de retornar

if __name__ == '__main__':
    app.run(debug=True)
