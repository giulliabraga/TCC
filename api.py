from flask import Flask, jsonify, send_file, url_for, make_response, request
from flask_cors import CORS, cross_origin
import json
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from neurokit2 import epochs_to_df,rescale,ecg_segment
from neurokit2.ecg.ecg_segment import ecg_segment
import neurokit2 as nk
from datetime import datetime
from neurokit2.signal import signal_rate, signal_sanitize
from neurokit2.ecg.ecg_clean import ecg_clean
from neurokit2.ecg.ecg_delineate import ecg_delineate
from neurokit2.ecg.ecg_peaks import ecg_peaks
from neurokit2.ecg.ecg_phase import ecg_phase
from neurokit2.ecg.ecg_quality import ecg_quality

# Função ecg_process adaptada para API
def ecg_process(ecg_signal, sampling_rate=100): 

    # Sanitize and clean input
    ecg_signal = signal_sanitize(ecg_signal)
    ecg_cleaned = ecg_clean(ecg_signal, sampling_rate=sampling_rate, method='vg')

    # Detect R-peaks
    instant_peaks, info = ecg_peaks(
        ecg_cleaned=ecg_cleaned,
        sampling_rate=sampling_rate,
        method='elgendi2010',
        correct_artifacts=True,
    )

    # Calculate heart rate
    rate = signal_rate(
        info, sampling_rate=sampling_rate, desired_length=len(ecg_cleaned)
    )

    # Assess signal quality
    quality = ecg_quality(
        ecg_cleaned, rpeaks=info["ECG_R_Peaks"], sampling_rate=sampling_rate
    )

    # Merge signals in a DataFrame
    signals = pd.DataFrame(
        {
            "ECG_Raw": ecg_signal,
            "ECG_Clean": ecg_cleaned,
            "ECG_Rate": rate,
            "ECG_Quality": quality,
        }
    )

    # Delineate QRS complex
    delineate_signal, delineate_info = ecg_delineate(
        ecg_cleaned=ecg_cleaned, rpeaks=info["ECG_R_Peaks"], sampling_rate=sampling_rate
    )
    info.update(delineate_info)  # Merge waves indices dict with info dict

    # Determine cardiac phases
    cardiac_phase = ecg_phase(
        ecg_cleaned=ecg_cleaned,
        rpeaks=info["ECG_R_Peaks"],
        delineate_info=delineate_info,
    )

    # Add additional information to signals DataFrame
    signals = pd.concat(
        [signals, instant_peaks, delineate_signal, cardiac_phase], axis=1
    )

    # return signals DataFrame and R-peak locations
    return signals, info

# Função ecg_plot adaptada para API
def ecg_plot(ecg_signals, rpeaks, sampling_rate=100, show_type="default"): 
## Código da função ecg_plot do neurokit
    # Sanity-check input.
    if not isinstance(ecg_signals, pd.DataFrame):
        print(
            "NeuroKit error: ecg_plot(): The `ecg_signals` argument must be the "
            "DataFrame returned by `ecg_process()`."
        )

    # Extract R-peaks.
    peaks = np.where(ecg_signals["ECG_R_Peaks"] == 1)[0]

    # Prepare figure and set axes.
    if show_type in ["default", "full"]:
        x_axis = np.linspace(0, ecg_signals.shape[0] / sampling_rate, ecg_signals.shape[0])
        gs = matplotlib.gridspec.GridSpec(2, 2, width_ratios=[1 - 1 / np.pi, 1 / np.pi])

        fig1 = plt.figure(constrained_layout=False)
        fig2 = plt.figure(constrained_layout=False)

        ax1 = fig1.add_subplot(111)
        ax2 = fig2.add_subplot(111)
        ax1.set_xlabel("Tempo (segundos)")
        ax2.set_xlabel("Tempo (segundos)")

        # Plot heart rate.
        ax1.set_title("Frequência Cardíaca")
        ax1.set_ylabel("Batimentos por minuto (bpm)")
        ax1.set_xlabel("Tempo (segundos)")

        ax1.plot(x_axis, ecg_signals["ECG_Rate"], color="#FF5722", label="Frequência", linewidth=1.5)
        rate_mean = ecg_signals["ECG_Rate"].mean()
        ax1.axhline(y=rate_mean, label="Média", linestyle="--", color="#FF9800")
        ax1.legend(loc="upper right")

        # Plot individual heart beats.
        if sampling_rate is not None:
            ytitle="ECG"
            color="#F44336"
            heartbeats = ecg_segment(ecg_signals, peaks, sampling_rate)
            df = epochs_to_df(heartbeats)

            # Get main signal column name
            col = [c for c in ["Signal", "ECG_Raw", "ECG_Clean"] if c in df.columns][-1]

            # Average heartbeat
            mean_heartbeat = df.groupby("Time")[[col]].mean()
            df_pivoted = df.pivot(index="Time", columns="Label", values=col)

            # Prepare plot

            ax2.set_title(f"Frequência cardíaca média: {rate_mean:0.1f} bpm)")
            ax2.set_xlabel("Tempo (segundos)")
            ax2.set_ylabel(ytitle)

            # Add Vertical line at 0
            ax2.axvline(x=0, color="grey", linestyle="--")

            # Plot average heartbeat
            ax2.plot(
                mean_heartbeat.index,
                mean_heartbeat,
                color=color,
                linewidth=7,
                label="Formato",
                zorder=1,
            )

            # Alpha of individual beats decreases with more heartbeats
            alpha = 1 / np.log2(np.log2(1 + df_pivoted.shape[1]))
            # Plot all heartbeats
            ax2.plot(df_pivoted, color="grey", linewidth=alpha, alpha=alpha, zorder=2)

            # Plot individual waves
            for wave in [
                ("P", "#3949AB"),
                ("Q", "#1E88E5"),
                ("S", "#039BE5"),
                ("T", "#00ACC1"),
            ]:
                wave_col = f"ECG_{wave[0]}_Peaks"
                if wave_col in df.columns:
                    ax2.scatter(
                        df["Time"][df[wave_col] == 1],
                        df[col][df[wave_col] == 1],
                        color=wave[1],
                        marker="+",
                        label=f"Ondas {wave[0]}",
                        zorder=3,
                    )

            ax2.legend(loc="upper right")

        figraw=plt.figure(constrained_layout=False)
        plt.plot(np.linspace(0,1,100), ecg_signals["ECG_Raw"][200:300], color="#B0BEC5", label="Bruto", zorder=1)
        plt.title("Sinal Bruto")
        plt.ylabel('ECG')
        plt.xlabel('Tempo (segundos)')

        figclean=plt.figure(constrained_layout=False)
        plt.plot(np.linspace(0,1,100), ecg_signals["ECG_Clean"][200:300], color="#E91E63", label="Processado", zorder=1, linewidth=1.5)
        plt.title("Sinal Processado")
        plt.ylabel('ECG')
        plt.xlabel('Tempo (segundos)')

    return fig1, fig2, figraw, figclean, rate_mean


app = Flask(__name__)
CORS(app, resources={
    r"/process_ecg_data": {"origins": "https://10.7.231.85:8080"},
    r"/upload_ecg_data": {"origins": "https://10.7.231.85:8080"}
})

@app.route('/upload_ecg_data', methods=['POST'])
@cross_origin(origin='https://10.7.231.85:8080', headers=['Content-Type'])
def upload_ecg_data():
    try:
        # Obter o JSON da solicitação POST
        data = request.get_json()

        # Neste exemplo, apenas salve o JSON em um arquivo
        with open('/home/giubdam/mysite/ecg_data.json', 'w') as file:
            json.dump(data, file)

        return jsonify({'status': 'success'})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/process_ecg_data', methods=['GET'])
@cross_origin(origin='https://10.7.231.85:8080', headers=['Content-Type'])
def process_ecg_data():
    try:
        # Load data from JSON file
        with open('/home/giubdam/mysite/ecg_data.json', 'r') as file:
            data = json.load(file)

        # Extrair dados
        timestamps = [entry['Timestamp'] for entry in data]
        ecg_signal = [entry['ECG Signal'] for entry in data]

        # Converte o primeiro timestamp para uma string de data e hora
        first_timestamp = timestamps[0] / 1000
        formatted_datetime = datetime.fromtimestamp(first_timestamp).strftime('%Y-%m-%d_%H-%M-%S')

        # Nome do novo arquivo com base no primeiro timestamp
        new_file_path = f'/home/giubdam/mysite/ecg_data_{formatted_datetime}.json'

        # Salva os dados no novo arquivo
        with open(new_file_path, 'w') as new_file:
            json.dump(data, new_file)

        # Limpa novo arquivo
        with open('/home/giubdam/mysite/ecg_data.json','w') as file:
            file.seek(0)
            file.truncate()

        # Para gerar novas imagens
        with open('/home/giubdam/mysite/ecg_data_2024-01-18_17-54-14.json', 'r') as file:
            data = json.load(file)

        signals, info = ecg_process(ecg_signal, sampling_rate=100)

        fig1, fig2, figraw, figclean, heart_rate = ecg_plot(signals, info)

        raw_signal_path='sinal_bruto.png'
        figraw.savefig(raw_signal_path,figsize=(32,8))
        clean_signal_path='sinal_processado.png'
        figclean.savefig(clean_signal_path)
        fc_media_path='fc_media.png'
        fig1.savefig(fc_media_path)
        beats_path='batimento.png'
        fig2.savefig(beats_path)

        rate_avg=round(heart_rate)
        # Criar uma resposta com os cabeçalhos CORS
        response = make_response(jsonify({
            'status': 'success',
            'raw_image_url': url_for('download_file', filename=raw_signal_path, _external=True),
            'cleaned_image_url': url_for('download_file', filename=clean_signal_path, _external=True),
            'hr_avg_url': url_for('download_file', filename=fc_media_path, _external=True),
            'beats_avg_url': url_for('download_file', filename=beats_path, _external=True),
            'heart_rate': f'{rate_avg}'
        }))

        return response

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/<filename>')
def download_file(filename):
    end=f'/home/your_username/{filename}'
    return send_file(end, as_attachment=True)
