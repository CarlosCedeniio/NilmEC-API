from flask import Flask, jsonify
from flask_cors import CORS
import datetime
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
import calendar
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
import threading



app = Flask(__name__)

def save_object_to_file(obj, filename):
    # Obtener la hora actual
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Crear un diccionario que incluya el objeto y la hora
    data = {
        'timestamp': now,
        'forecast': obj
    }

    # Convertir el diccionario en una cadena JSON
    json_data = json.dumps(data, indent=4)

    # Guardar la cadena JSON en un archivo
    with open(filename, 'w') as file:
        file.write(json_data + '\n')

pipeline = ChronosPipeline.from_pretrained(
  "amazon/chronos-t5-tiny",
  torch_dtype=torch.bfloat16,
)

def post_request(url, data):
    try:
        json_data = json.dumps(data)
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, data=json_data)
        if response.status_code == 200:
            return response.json()
        else:
            print(f'Se produjo un error: {response.status_code}')
            return None
    except Exception as e:
        print(f'Error durante la solicitud: {str(e)}')
        return None
    
def calculate_usd(df_suma_por_dia, fase):
    now = datetime.datetime.now()
    context = torch.tensor(df_suma_por_dia["costo_usd_"+fase])
    primer_dia_proximo_mes = now.replace(day=1, month=now.month+1)
    ultimo_dia_mes_actual = primer_dia_proximo_mes - datetime.timedelta(days=1)
    prediction_length = (ultimo_dia_mes_actual - now).days
    forecast = pipeline.predict(context, prediction_length)  # shape [num_series, num_samples, prediction_length]
    fecha_dada = df_suma_por_dia['sensedAt'].iloc[-1]
    fechas_siguientes = pd.date_range(start=fecha_dada, periods=prediction_length+1, freq='D')[1:]
    df_fechas = pd.DataFrame({'sensedAt': fechas_siguientes})

    # visualize the forecast
    forecast_index = range(len(df_suma_por_dia), len(df_suma_por_dia) + prediction_length)
    low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
    
    return {'min': low.sum(),
            'med': median.sum(),
            'max': high.sum()
            }

def calculateForecast():
    print('Entro a calcular el forecast')
    now = datetime.datetime.now()
    start_of_month = now.replace(day=1)
    one_day = datetime.timedelta(days=1)
    last_day = now + one_day
    # Formatear la fecha y hora
    today = now.strftime('%d/%m/%Y 05:00:00')
    first_day = start_of_month.strftime('%d/%m/%Y 05:00:00')
    last_day = last_day.strftime('%d/%m/%Y 05:00:00') 
    url = 'https://aias.espol.edu.ec/api/hayiot/getDataWeb'
    data = {
        "id": "64ad81a0dc5442c4e0796382",
        "start": first_day,
        "end": last_day,
        "tags": ['potencia_A', 'potencia_B', 'potencia_C']
        }
    
    print('antes de reqyest')


    response = post_request(url, data)
    if response:
        print('En el response')

        df = pd.DataFrame(response)
        df['sensedAt'] = pd.to_datetime(df['sensedAt'], unit='ms')
        df.set_index('sensedAt', inplace=True)
        pivot_table = df.pivot(columns='type', values='data')


        df = pivot_table.reset_index()
        df.set_index('sensedAt', inplace=True)
        df['fecha']=df.index
        df['diferencia_segundos'] = df['fecha'].diff().dt.total_seconds()
        df['diferencia_segundos'] = df['diferencia_segundos'].fillna(10)

        w_2_kwh = 1/(1000*3600)

        df['energia_A'] = (df['potencia_A'] * df['diferencia_segundos']) * w_2_kwh
        constante = 0.06
        df['costo_usd_A'] =  df['energia_A'] * constante

        df['energia_B'] = (df['potencia_B'] * df['diferencia_segundos']) * w_2_kwh
        constante = 0.06
        df['costo_usd_B'] =  df['energia_B'] * constante

        df['energia_C'] = (df['potencia_C'] * df['diferencia_segundos']) * w_2_kwh
        constante = 0.06
        df['costo_usd_C'] =  df['energia_C'] * constante

        df_suma_por_dia = df.groupby(pd.Grouper(freq='D')).sum(numeric_only=True).reset_index()
        

        print('Entes de forecast')

        forecast_a = calculate_usd(df_suma_por_dia, "A")
        forecast_b = calculate_usd(df_suma_por_dia, "B")
        forecast_c = calculate_usd(df_suma_por_dia, "C")
        
        print('despues de forecast')
    
        total_acumulado_min = df_suma_por_dia['costo_usd_A'].sum() + df_suma_por_dia['costo_usd_B'].sum() + df_suma_por_dia['costo_usd_C'].sum() + forecast_a['min'] + forecast_b['min'] + forecast_c['min']
        total_acumulado_med = df_suma_por_dia['costo_usd_A'].sum() + df_suma_por_dia['costo_usd_B'].sum() + df_suma_por_dia['costo_usd_C'].sum() + forecast_a['med'] + forecast_b['med'] + forecast_c['med']
        total_acumulado_max = df_suma_por_dia['costo_usd_A'].sum() + df_suma_por_dia['costo_usd_B'].sum() + df_suma_por_dia['costo_usd_C'].sum() + forecast_a['max'] + forecast_b['max'] + forecast_c['max']

        
        respuesta = {
                'costo_a': df['costo_usd_A'].sum(),
                'costo_b': df['costo_usd_B'].sum(),
                'costo_c': df['costo_usd_C'].sum(),
                'forecast_a': forecast_a,
                'forecast_b': forecast_b,
                'forecast_c': forecast_c,
                'total_acumulado_min': total_acumulado_min,
                'total_acumulado_med': total_acumulado_med,
                'total_acumulado_max': total_acumulado_max,

            }
        
        save_object_to_file(respuesta, 'forecast.txt')

def start_task():
    task_thread = threading.Thread(target=calculateForecast)
    task_thread.start()


@app.route('/')
def index():
    return jsonify({'message': 'Bienvenido a mi aplicación Flask!'})

@app.route('/getConsumptionBill')
def getConsumption():
    now = datetime.datetime.now()
    start_of_month = now.replace(day=1)
    one_day = datetime.timedelta(days=1)
    last_day = now + one_day
    today = now.strftime('%d/%m/%Y 05:00:00')
    primer_dia_proximo_mes = now.replace(day=1, month=now.month+1)
    ultimo_dia_mes_actual = primer_dia_proximo_mes - datetime.timedelta(days=1)
    first_day = start_of_month.strftime('%d/%m/%Y 05:00:00')
    last_day = ultimo_dia_mes_actual.strftime('%d/%m/%Y 05:00:00') 
    url = 'https://aias.espol.edu.ec/api/hayiot/getDataWeb'
    data = {
        "id": "64ad81a0dc5442c4e0796382",
        "start": first_day,
        "end": last_day,
        "tags": ['potencia_A', 'potencia_B', 'potencia_C']
        }

    response = post_request(url, data)
    if response:
        df = pd.DataFrame(response)
        df['sensedAt'] = pd.to_datetime(df['sensedAt'], unit='ms')
        df.set_index('sensedAt', inplace=True)
        pivot_table = df.pivot(columns='type', values='data')


        df = pivot_table.reset_index()
        df.set_index('sensedAt', inplace=True)
        df['fecha']=df.index
        df['diferencia_segundos'] = df['fecha'].diff().dt.total_seconds()
        df['diferencia_segundos'] = df['diferencia_segundos'].fillna(10)

        w_2_kwh = 1/(1000*3600)

        df['energia_A'] = (df['potencia_A'] * df['diferencia_segundos']) * w_2_kwh
        constante = 0.06
        df['costo_usd_A'] =  df['energia_A'] * constante

        df['energia_B'] = (df['potencia_B'] * df['diferencia_segundos']) * w_2_kwh
        constante = 0.06
        df['costo_usd_B'] =  df['energia_B'] * constante

        df['energia_C'] = (df['potencia_C'] * df['diferencia_segundos']) * w_2_kwh
        constante = 0.06
        df['costo_usd_C'] =  df['energia_C'] * constante

        df_suma_por_dia = df.groupby(pd.Grouper(freq='D')).sum(numeric_only=True).reset_index()
        total_acumulado = df_suma_por_dia['costo_usd_A'].sum() + df_suma_por_dia['costo_usd_B'].sum() + df_suma_por_dia['costo_usd_C'].sum()
        print(total_acumulado)
        response = jsonify({'total': total_acumulado})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

@app.route('/forceUpdateForecast')
def forceUpdateForecast():
    
        calculateForecast()

        with open('forecast.txt', 'r') as file:
            data = json.load(file)

        response = jsonify(data)
        response.headers.add("Access-Control-Allow-Origin", "*")

        return response

@app.route('/getForecast')
def getForecast():
        with open('forecast.txt', 'r') as file:
            data = json.load(file)

        response = jsonify(data)
        response.headers.add("Access-Control-Allow-Origin", "*")

        return response





if __name__ == '__main__':
    start_task()
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=start_task, trigger="interval", minutes=10, max_instances=3)
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown())
    CORS(app.run(host='0.0.0.0', port=5011), origins=['*'])