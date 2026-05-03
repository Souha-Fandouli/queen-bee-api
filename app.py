from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
import joblib
import os
import tempfile

app = Flask(__name__)
CORS(app)

# Charger le modèle au démarrage
print("=" * 50)
print("🐝 CHARGEMENT DU MODÈLE SVLAB QUEEN BEE DETECTOR")
print("=" * 50)

rf_model = joblib.load('queen_rf_model_latest.pkl')
scaler = joblib.load('scaler_latest.pkl')

print("✅ Modèle Random Forest 99.8% chargé !")
print("✅ API prête à recevoir des requêtes")
print("=" * 50)

def extract_features(audio_path, sr=22050, duration=3):
    """
    Extrait 174 features audio (MFCC 20 + Mel 64 + Spectral 3)
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr, duration=duration)
        if len(y) < sr * 0.3:
            return None
        y = y / (np.max(np.abs(y)) + 1e-8)
        
        features = []
        
        # MFCCs (20 coefficients × 2 stats = 40)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(20):
            features.extend([np.mean(mfccs[i]), np.std(mfccs[i])])
        
        # Mel Spectrogram (64 bandes × 2 stats = 128)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        for i in range(64):
            features.extend([np.mean(mel_db[i]), np.std(mel_db[i])])
        
        # Spectral features (3 × 2 stats = 6)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        for feat in [centroid, rolloff, zcr]:
            features.extend([np.mean(feat), np.std(feat)])
        
        return np.array(features)
    except Exception as e:
        print(f"❌ Erreur extraction: {e}")
        return None

@app.route('/')
def home():
    """Page d'accueil"""
    return jsonify({
        'service': 'SVLAB Queen Bee Detector',
        'model': 'Random Forest Classifier',
        'accuracy': '99.8%',
        'features': 'MFCC(20) + Mel(64) + Spectral(3)',
        'total_features': 174,
        'dataset': '6000 fichiers audio',
        'endpoints': {
            '/predict': 'POST - Upload audio file (field: audio)',
            '/health': 'GET - Health check'
        },
        'status': 'online'
    })

@app.route('/health')
def health():
    """Vérification de santé"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prédit la présence de la reine à partir d'un fichier audio
    """
    # Vérifier que le fichier est présent
    if 'audio' not in request.files:
        return jsonify({
            'error': 'No audio file provided',
            'usage': 'Send a POST request with an audio file in the "audio" field'
        }), 400
    
    audio_file = request.files['audio']
    
    # Vérifier le nom du fichier
    if audio_file.filename == '':
        return jsonify({'error': 'Empty file'}), 400
    
    # Sauvegarder temporairement
    tmp_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
    audio_file.save(tmp_path)
    
    try:
        # Extraire les features
        features = extract_features(tmp_path)
        
        if features is None:
            return jsonify({
                'error': 'Could not process audio',
                'message': 'File too short or corrupted. Minimum 1 second required.'
            }), 400
        
        # Normaliser et prédire
        features_scaled = scaler.transform(features.reshape(1, -1))
        probabilities = rf_model.predict_proba(features_scaled)[0]
        
        # Résultats
        is_present = bool(probabilities[1] >= 0.5)
        confidence = float(probabilities[1] if is_present else probabilities[0])
        
        return jsonify({
            'prediction': '🐝 REINE PRÉSENTE' if is_present else '❌ REINE ABSENTE',
            'is_queen_present': is_present,
            'confidence': round(confidence * 100, 1),
            'probability_absent': round(float(probabilities[0]) * 100, 1),
            'probability_present': round(float(probabilities[1]) * 100, 1),
            'model_accuracy': '99.8%'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Internal server error'
        }), 500
    
    finally:
        # Nettoyer le fichier temporaire
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Démarrage de l'API sur le port {port}")
    app.run(host='0.0.0.0', port=port)
