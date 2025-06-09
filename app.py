from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load model and scalers
model = joblib.load('models/best_rf_model.pkl')
scaler_minmax = joblib.load('models/scaler_minmax.pkl')
scaler_standard = joblib.load('models/scaler_standard.pkl')

# Mappings
prodi_mapping = {v: i for i, v in enumerate(['TET', 'TL', 'TRSE', 'TE', 'TT', 'TMS', 'TRM', 'TM', 
                                             'TI', 'TRK', 'TK', 'SI', 'Akt', 'TRJT', 'AktP', 'MTTK', 'BD', 'HMKD'])}
jalur_mapping = {v: i for i, v in enumerate(['PSUD', 'UMPCR I', 'CBT Bersama I', 'UMPCR III', 'UMPCR II',
                                             'Jalur Kerjasama LLDIKTI X', 'TOL', 'Jalur Khusus', 
                                             'Alih Jenjang D3', 'Jalur Fast Track', 'Beasiswa', 'Magister Reguler'])}
jenis_kelamin_mapping = {'L': 1, 'P': 0}
terima_kps_mapping = {'Ya': 1, 'Tidak': 0}

def preprocess_csv(file_path):
    df_raw = pd.read_csv(file_path)
    df = df_raw.copy()

    df['prodi'] = df['prodi'].map(prodi_mapping)
    df['jalur_masuk'] = df['jalur_masuk'].map(jalur_mapping)
    df['jenis_kelamin'] = df['jenis_kelamin'].map(jenis_kelamin_mapping)
    df['terima_kps'] = df['terima_kps'].map(terima_kps_mapping)

    df['avg_ips'] = df[['IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5']].mean(axis=1)
    df['min_ips'] = df[['IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5']].min(axis=1)
    df['max_ips'] = df[['IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5']].max(axis=1)
    df['var_ips'] = df[['IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5']].var(axis=1)

    features = ['tahun_masuk', 'prodi', 'jalur_masuk', 'jenis_kelamin', 'terima_kps',
                'total_tak', 'jumlah_co', 'IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5',
                'avg_ips', 'min_ips', 'max_ips', 'var_ips']

    df_scaled = df.copy()
    cols_minmax = ['total_tak', 'jumlah_co']
    cols_standard = ['IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5', 'avg_ips', 'min_ips', 'max_ips', 'var_ips']

    df_scaled[cols_minmax] = scaler_minmax.transform(df[cols_minmax])
    df_scaled[cols_standard] = scaler_standard.transform(df[cols_standard])

    df_raw['avg_ips'] = df['avg_ips']
    df_raw['min_ips'] = df['min_ips']
    df_raw['max_ips'] = df['max_ips']
    df_raw['var_ips'] = df['var_ips']

    return df_raw, df_scaled, features

def preprocess_manual(form):
    data = {
        'tahun_masuk': int(form['tahun_masuk']),
        'prodi': prodi_mapping[form['prodi']],
        'jalur_masuk': jalur_mapping[form['jalur_masuk']],
        'jenis_kelamin': jenis_kelamin_mapping[form['jenis_kelamin']],
        'terima_kps': terima_kps_mapping[form['terima_kps']],
        'total_tak': int(form['total_tak']),
        'jumlah_co': int(form['jumlah_co']),
        'IPS1': float(form['IPS1']),
        'IPS2': float(form['IPS2']),
        'IPS3': float(form['IPS3']),
        'IPS4': float(form['IPS4']),
        'IPS5': float(form['IPS5']),
    }
    data['avg_ips'] = (data['IPS1'] + data['IPS2'] + data['IPS3'] + data['IPS4'] + data['IPS5']) / 5
    data['min_ips'] = min(data['IPS1'], data['IPS2'], data['IPS3'], data['IPS4'], data['IPS5'])
    data['max_ips'] = max(data['IPS1'], data['IPS2'], data['IPS3'], data['IPS4'], data['IPS5'])
    data['var_ips'] = pd.Series([data['IPS1'], data['IPS2'], data['IPS3'], data['IPS4'], data['IPS5']]).var()

    df = pd.DataFrame([data])

    cols_minmax = ['total_tak', 'jumlah_co']
    cols_standard = ['IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5', 'avg_ips', 'min_ips', 'max_ips', 'var_ips']

    df[cols_minmax] = scaler_minmax.transform(df[cols_minmax])
    df[cols_standard] = scaler_standard.transform(df[cols_standard])

    return df

@app.route('/', methods=['GET', 'POST'])
def index():
    manual_result = None
    batch_result = None

    if request.method == 'POST':
        if 'nama' in request.form:
            nama = request.form['nama']
            nim = request.form['nim']
            input_data = preprocess_manual(request.form)
            pred = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1]

            manual_result = {
                'Nama': nama,
                'NIM': nim,
                'Tahun Masuk': request.form['tahun_masuk'],
                'Prodi': request.form['prodi'],
                'Jalur Masuk': request.form['jalur_masuk'],
                'Jenis Kelamin': request.form['jenis_kelamin'],
                'Menerima KPS': request.form['terima_kps'],
                'Total TAK': request.form['total_tak'],
                'Jumlah CO': request.form['jumlah_co'],
                'IPS1': request.form['IPS1'],
                'IPS2': request.form['IPS2'],
                'IPS3': request.form['IPS3'],
                'IPS4': request.form['IPS4'],
                'IPS5': request.form['IPS5'],
                'Probabilitas Tidak Tepat Waktu': round(prob, 4),
                'Hasil Prediksi': 'Tepat Waktu' if pred == 0 else 'Tidak Tepat Waktu'
            }

        elif 'file' in request.files:
            file = request.files['file']
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            df_raw, df_scaled, features = preprocess_csv(file_path)
            preds = model.predict(df_scaled[features])
            probs = model.predict_proba(df_scaled[features])[:, 1]

            df_raw['Probabilitas Tidak Tepat Waktu'] = probs
            df_raw['Hasil'] = ['Tepat Waktu' if p == 0 else 'Tidak Tepat Waktu' for p in preds]

            batch_result = df_raw
            df_raw.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'hasil_prediksi.csv'), index=False)

    return render_template('index.html', manual_result=manual_result, batch_result=batch_result)

@app.route('/download')
def download_file():
    path = os.path.join(app.config['UPLOAD_FOLDER'], 'hasil_prediksi.csv')
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
