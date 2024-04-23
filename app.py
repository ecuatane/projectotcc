from flask import Flask,request,jsonify
import numpy as np
import joblib


model = joblib.load('resultados_grid_search_modelXGboost.pkl')

best_model =model.best_estimator_
app = Flask(__name__)
@app.route('/')
def index():
    #cv_score = model.best_score_

    #input_query = np.array([[10,35,0.074,0.268,0.268,0.342,0.074,0.311,0.311,0.385,0.074,0.944,0.944,1.018,0.074,0.45,0.439,0.513,0.063,0.365,0.365,0.428,0.063,0.279,0.29,0.353,0.074,0.311,0.321,0.395,0.084,0.236,0.215,0.299,0.0725,0.3955,0.466625,0.394125]])
    #result = best_model.predict(input_query)[0]
    return "Hello world "

@app.route('/predict',methods=['POST'])
def predict():
    # Receber os valores do formulário
    sessao=request.form.get('sessao')
    repeticao = request.form.get('repeticao')
    Duracao_v = request.form.get('Duracao.v')
    PP_v_i = request.form.get('PP.v_i')
    SS_v_i = request.form.get('SS.v_i')
    PS_v_i = request.form.get('PS.v_i')
    Duracao_i = request.form.get('Duracao.i')
    PP_i_d = request.form.get('PP.i_d')
    SS_i_d = request.form.get('SS.i_d')
    PS_i_d = request.form.get('PS.i_d')
    Duracao_d = request.form.get('Duracao.d')
    PP_d_a = request.form.get('PP.d_a')
    SS_d_a = request.form.get('SS.d_a')
    PS_d_a = request.form.get('PS.d_a')
    Duracao_a = request.form.get('Duracao.a')
    PP_a_ponto = request.form.get('PP.a_ponto')
    SS_a_ponto = request.form.get('SS.a_ponto')
    PS_a_ponto = request.form.get('PS.a_ponto')
    Duracao_ponto = request.form.get('Duracao.ponto')
    PP_ponto_n = request.form.get('PP.ponto_n')
    SS_ponto_n = request.form.get('SS.ponto_n')
    PS_ponto_n = request.form.get('PS.ponto_n')
    Duracao_n = request.form.get('Duracao.n')
    PP_n_o = request.form.get('PP.n_o')
    SS_n_o = request.form.get('SS.n_o')
    PS_n_o = request.form.get('PS.n_o')
    Duracao_o = request.form.get('Duracao.o')
    PP_o_v = request.form.get('PP.o_v')
    SS_o_v = request.form.get('SS.o_v')
    PS_o_v = request.form.get('PS.o_v')
    Duracao_v2 = request.form.get('Duracao.v2')
    PP_v_a = request.form.get('PP.v_a')
    SS_v_a = request.form.get('SS.v_a')
    PS_v_a = request.form.get('PS.v_a')
    Media_Duracao = request.form.get('Media_Duracao')
    Media_PP = request.form.get('Media_PP')
    Media_PS = request.form.get('Media_PS')
    Media_SS = request.form.get('Media_SS')

    # Criar um array numpy com os valores recebidos
    input_query = np.array([
        [sessao,repeticao,Duracao_v, PP_v_i, SS_v_i, PS_v_i, Duracao_i, PP_i_d, SS_i_d, PS_i_d,
         Duracao_d, PP_d_a, SS_d_a, PS_d_a, Duracao_a, PP_a_ponto, SS_a_ponto,
         PS_a_ponto, Duracao_ponto, PP_ponto_n, SS_ponto_n, PS_ponto_n, Duracao_n,
         PP_n_o, SS_n_o, PS_n_o, Duracao_o, PP_o_v, SS_o_v, PS_o_v, Duracao_v2,
         PP_v_a, SS_v_a, PS_v_a, Media_Duracao, Media_PP, Media_PS, Media_SS]
    ])

    # Fazer a previsão usando o melhor modelo
    result = best_model.predict(input_query)
    return jsonify({'placement':str(result)})
if __name__ == '__main__':
    app.run(debug=True)