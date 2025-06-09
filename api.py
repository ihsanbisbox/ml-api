from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import os
from PIL import Image
import io
import base64
import cv2

app = Flask(__name__)

# CORS Configuration
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost", "http://127.0.0.1", "http://localhost:8000", "http://127.0.0.1:8000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def find_model_file():
    """Cari file model dengan berbagai kemungkinan lokasi"""
    print("🔍 Mencari file model...")
    
    # Daftar lokasi yang mungkin
    possible_paths = [
        '/app/model.h5',          # Path utama di container
        './model.h5',             # Path relatif
        'model.h5',               # Path di working directory
        os.path.join(os.getcwd(), 'model.h5'),  # Absolute path dari working dir
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024*1024)
            print(f"✅ Model ditemukan: {path} ({size_mb:.1f} MB)")
            return path
    
    print("❌ Model tidak ditemukan di semua lokasi yang dicoba")
    return None

def load_model_safely():
    """Load model dengan error handling yang lebih baik"""
    try:
        model_path = find_model_file()
        
        if model_path is None:
            print("❌ File model tidak ditemukan!")
            return None
        
        print(f"📂 Memuat model dari: {model_path}")
        
        # Load model dengan kompilasi ulang jika perlu
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Kompilasi ulang model untuk memastikan berjalan dengan baik
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )
        
        print("✅ Model berhasil dimuat!")
        print(f"📊 Input shape: {model.input_shape}")
        print(f"📊 Output shape: {model.output_shape}")
        print(f"📊 Jumlah kelas: {model.output_shape[-1]}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error saat memuat model: {e}")
        import traceback
        traceback.print_exc()
        return None

# Load model saat startup
print("🚀 Memulai loading model...")
model = load_model_safely()

# Class names - pastikan urutan sama dengan training
class_names = [
    'Bercak_bakteri',
    'Bercak_daun_Septoria', 
    'Bercak_Target',
    'Bercak_daun_awal',
    'Busuk_daun_lanjut',
    'Embun_tepung',
    'Jamur_daun',
    'Sehat',
    'Tungau_dua_bercak',
    'Virus_keriting_daun_kuning',
    'Virus_mosaik_tomat',
]

@app.route('/debug', methods=['GET'])
def debug_info():
    """Debug endpoint untuk cek status sistem"""
    import glob
    
    debug_data = {
        'working_directory': os.getcwd(),
        'model_status': {
            'loaded': model is not None,
            'input_shape': str(model.input_shape) if model else None,
            'output_shape': str(model.output_shape) if model else None
        },
        'file_system': {
            'current_dir_files': [],
            'app_dir_files': [],
            'model_files_found': []
        }
    }
    
    # List file di direktori saat ini
    try:
        current_files = os.listdir('.')
        debug_data['file_system']['current_dir_files'] = [
            f for f in current_files if f.endswith('.h5') or 'model' in f.lower()
        ]
    except Exception as e:
        debug_data['file_system']['current_dir_files'] = f"Error: {e}"
    
    # List file di /app
    try:
        if os.path.exists('/app'):
            app_files = os.listdir('/app')
            debug_data['file_system']['app_dir_files'] = [
                f for f in app_files if f.endswith('.h5') or 'model' in f.lower()
            ]
    except Exception as e:
        debug_data['file_system']['app_dir_files'] = f"Error: {e}"
    
    # Cari semua file .h5
    try:
        model_files = glob.glob('**/*.h5', recursive=True)
        debug_data['file_system']['model_files_found'] = model_files
    except Exception as e:
        debug_data['file_system']['model_files_found'] = f"Error: {e}"
    
    return jsonify({
        'success': True,
        'debug_info': debug_data
    })

def validate_tomato_leaf_image(image):
    """Validasi apakah gambar adalah daun tomat"""
    try:
        # Convert PIL ke OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 1. Analisis warna - cek dominasi hijau
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        
        # Range warna hijau dalam HSV
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # Buat mask untuk warna hijau
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_ratio = np.sum(green_mask > 0) / (green_mask.shape[0] * green_mask.shape[1])
        
        print(f"🟢 Rasio warna hijau: {green_ratio:.3f}")
        
        # 2. Deteksi tepi - cek struktur daun
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        print(f"📏 Rasio tepi: {edge_ratio:.3f}")
        
        # 3. Aspect ratio
        height, width = image.size[1], image.size[0]
        aspect_ratio = max(width, height) / min(width, height)
        
        print(f"📐 Aspect ratio: {aspect_ratio:.2f}")
        
        # 4. Analisis kecerahan dan kontras
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        print(f"💡 Kecerahan: {brightness:.2f}, Kontras: {contrast:.2f}")
        
        # Aturan validasi (lebih permisif)
        reasons = []
        
        if green_ratio < 0.08:  # Minimal 8% hijau
            reasons.append(f"Kurang dominasi warna hijau ({green_ratio*100:.1f}%)")
        
        if edge_ratio < 0.005 or edge_ratio > 0.5:
            reasons.append("Struktur gambar tidak sesuai daun")
        
        if aspect_ratio > 15:
            reasons.append(f"Rasio aspek terlalu ekstrem ({aspect_ratio:.1f}:1)")
        
        if brightness < 15 or brightness > 240:
            reasons.append("Kecerahan gambar tidak normal")
        
        if contrast < 10:
            reasons.append("Kontras gambar terlalu rendah")
        
        # Hitung confidence
        confidence = min(green_ratio * 3, 0.5) + min(edge_ratio * 5, 0.3) + min(contrast/50, 0.2)
        
        is_valid = len(reasons) == 0 and confidence > 0.3
        
        return is_valid, reasons, confidence
        
    except Exception as e:
        print(f"❌ Error validasi: {e}")
        return True, [], 0.5  # Jika error, anggap valid

def validate_model_confidence(prediction, threshold=0.3):
    """Validasi confidence model"""
    max_confidence = np.max(prediction)
    
    if max_confidence < threshold:
        sorted_probs = np.sort(prediction[0])[::-1]
        top_diff = sorted_probs[0] - sorted_probs[1]
        
        if top_diff < 0.1:
            return False, f"Model tidak yakin dengan prediksi (confidence: {max_confidence*100:.1f}%)"
    
    return True, None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocessing gambar untuk model"""
    from tensorflow.keras.applications.resnet50 import preprocess_input
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize(target_size)
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Gunakan preprocessing yang sama seperti saat training
    img_array = preprocess_input(img_array)
    
    return img_array

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_healthy_plant(class_name):
    """Cek apakah tanaman sehat"""
    healthy_classes = ['Sehat', 'healthy', 'Tanaman_Sehat']
    return class_name in healthy_classes

def get_disease_info(disease_name):
    """Dapatkan informasi penyakit"""
    info = {
        'Bercak_bakteri': {
            'name': 'Bercak Bakteri',
            'symptoms': 'Bercak coklat kecil dengan tepi kuning pada daun, buah, dan batang',
            'causes': 'Bakteri Xanthomonas campestris',
            'prevention': 'Gunakan benih bebas penyakit, hindari penyiraman dari atas, rotasi tanaman',
            'treatment': 'Gunakan bakterisida yang mengandung tembaga, praktikkan rotasi tanaman',
            'severity': 'sedang'
        },
        'Bercak_daun_Septoria': {
            'name': 'Bercak Daun Septoria',
            'symptoms': 'Bercak bulat kecil dengan pusat abu-abu dan tepi coklat pada daun',
            'causes': 'Jamur Septoria lycopersici',
            'prevention': 'Hindari penyiraman dari atas, mulsa tanah, rotasi tanaman',
            'treatment': 'Hapus daun yang terinfeksi dan gunakan fungisida yang mengandung tembaga',
            'severity': 'sedang'
        },
        'Bercak_Target': {
            'name': 'Bercak Target',
            'symptoms': 'Lesi coklat dengan pola cincin target pada daun dan buah',
            'causes': 'Jamur Corynespora cassiicola',
            'prevention': 'Jaga sirkulasi udara, hindari penanaman terlalu rapat',
            'treatment': 'Gunakan fungisida dan hindari penanaman rapat',
            'severity': 'sedang'
        },
        'Bercak_daun_awal': {
            'name': 'Bercak Daun Awal',
            'symptoms': 'Lesi coklat dengan cincin konsentris pada daun, dimulai dari daun bawah',
            'causes': 'Jamur Alternaria solani',
            'prevention': 'Jaga drainase yang baik, hindari stres pada tanaman, mulsa tanah',
            'treatment': 'Gunakan fungisida yang mengandung chlorothalonil, buang daun yang terinfeksi',
            'severity': 'sedang'
        },
        'Busuk_daun_lanjut': {
            'name': 'Busuk Daun Lanjut',
            'symptoms': 'Bercak berair yang menjadi coklat pada daun dan batang, bulu putih di bawah daun',
            'causes': 'Oomycete Phytophthora infestans',
            'prevention': 'Hindari kelembaban tinggi, sirkulasi udara yang baik, tanam varietas tahan',
            'treatment': 'Gunakan fungisida sistemik seperti metalaxyl, hancurkan tanaman yang terinfeksi',
            'severity': 'tinggi'
        },
        'Embun_tepung': {
            'name': 'Embun Tepung',
            'symptoms': 'Lapisan putih seperti tepung pada permukaan daun',
            'causes': 'Jamur Leveillula atau Oidium',
            'prevention': 'Jaga sirkulasi udara, hindari kelembaban',
            'treatment': 'Gunakan fungisida sulfur atau potassium bicarbonate',
            'severity': 'sedang'
        },
        'Jamur_daun': {
            'name': 'Jamur Daun',
            'symptoms': 'Bercak kuning pada permukaan atas daun, lapisan fuzzy hijau-abu di bawah daun',
            'causes': 'Jamur Passalora fulva',
            'prevention': 'Tingkatkan sirkulasi udara, kurangi kelembaban, jaga jarak tanam',
            'treatment': 'Tingkatkan sirkulasi udara dan gunakan fungisida yang sesuai',
            'severity': 'sedang'
        },
        'Sehat': {
            'name': 'Tanaman Sehat',
            'symptoms': 'Daun hijau segar tanpa bercak',
            'causes': 'Tidak ada penyakit',
            'prevention': 'Pertahankan kondisi optimal',
            'treatment': 'Tanaman sehat, lanjutkan perawatan optimal',
            'severity': 'tidak ada'
        },
        'Tungau_dua_bercak': {
            'name': 'Tungau Dua Bercak',
            'symptoms': 'Daun menguning, bintik putih kecil, jaring laba-laba halus',
            'causes': 'Tungau Tetranychus urticae',
            'prevention': 'Jaga kelembaban udara, hindari stres kekeringan',
            'treatment': 'Gunakan mitisida atau sabun insektisida',
            'severity': 'sedang'
        },
        'Virus_keriting_daun_kuning': {
            'name': 'Virus Keriting Daun Kuning',
            'symptoms': 'Daun menguning, menggulung ke atas, pertumbuhan terhambat',
            'causes': 'Virus TYLCV oleh kutu kebul',
            'prevention': 'Kendalikan kutu kebul, gunakan mulsa reflektif',
            'treatment': 'Tanam varietas tahan, kendalikan kutu kebul',
            'severity': 'tinggi'
        },
        'Virus_mosaik_tomat': {
            'name': 'Virus Mosaik Tomat',
            'symptoms': 'Pola mosaik hijau terang dan gelap pada daun, daun keriting',
            'causes': 'Virus TMV yang menular',
            'prevention': 'Benih bebas virus, sterilisasi alat',
            'treatment': 'Hancurkan tanaman terinfeksi, sterilisasi alat',
            'severity': 'tinggi'
        }
    }
    
    return info.get(disease_name, {
        'name': disease_name,
        'symptoms': 'Informasi tidak tersedia',
        'causes': 'Tidak diketahui',
        'prevention': 'Konsultasikan dengan ahli pertanian',
        'treatment': 'Konsultasikan dengan ahli setempat',
        'severity': 'unknown'
    })

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

@app.route('/health', methods=['GET'])
def health_check():
    """Cek status API dan model"""
    return jsonify({
        'success': True,
        'message': 'API berjalan normal',
        'model_loaded': model is not None,
        'status': 'sehat' if model else 'model_belum_dimuat',
        'model_info': {
            'input_shape': str(model.input_shape) if model else None,
            'output_shape': str(model.output_shape) if model else None,
            'num_classes': len(class_names)
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Klasifikasi penyakit dari gambar yang diupload"""
    print("🔍 Endpoint predict dipanggil")
    print(f"📁 File dalam request: {list(request.files.keys())}")
    
    if model is None:
        print("❌ Model belum dimuat")
        return jsonify({'success': False, 'error': 'Model belum dimuat'}), 500

    if 'image' not in request.files:
        print("❌ Tidak ada key 'image' dalam request.files")
        return jsonify({'success': False, 'error': 'Tidak ada gambar yang diberikan'}), 400

    file = request.files['image']
    print(f"📷 File diterima: {file.filename}")
    
    if file.filename == '':
        print("❌ Nama file kosong")
        return jsonify({'success': False, 'error': 'Tidak ada gambar yang dipilih'}), 400

    if not allowed_file(file.filename):
        print(f"❌ Tipe file tidak valid: {file.filename}")
        return jsonify({'success': False, 'error': 'Tipe file tidak valid'}), 400

    try:
        print("🔄 Memproses gambar...")
        image_bytes = file.read()
        print(f"📊 Panjang bytes gambar: {len(image_bytes)}")
        
        # Buka dan validasi gambar
        image = Image.open(io.BytesIO(image_bytes))
        print(f"🖼️ Gambar asli - Mode: {image.mode}, Ukuran: {image.size}")
        
        # LANGKAH 1: Validasi awal - Cek apakah gambar terlihat seperti daun tomat
        print("🔍 Memvalidasi apakah gambar adalah daun tomat...")
        is_valid_leaf, validation_reasons, leaf_confidence = validate_tomato_leaf_image(image)
        
        if not is_valid_leaf:
            print(f"❌ Validasi gambar gagal: {validation_reasons}")
            return jsonify({
                'success': False, 
                'error': 'Gambar yang diupload bukan daun tomat',
                'details': {
                    'reasons': validation_reasons,
                    'confidence': leaf_confidence,
                    'suggestion': 'Silakan upload gambar daun tomat yang jelas dengan latar belakang yang kontras'
                }
            }), 400
        
        print(f"✅ Validasi gambar berhasil dengan confidence: {leaf_confidence:.3f}")
        
        # LANGKAH 2: Preprocessing gambar untuk model
        img_array = preprocess_image(image)
        print(f"📊 Shape array setelah preprocessing: {img_array.shape}")
        print(f"📊 Min/max array: {img_array.min():.3f}/{img_array.max():.3f}")

        # LANGKAH 3: Prediksi
        print("🤖 Membuat prediksi...")
        prediction = model.predict(img_array, verbose=0)
        print(f"📊 Shape prediksi: {prediction.shape}")
        print(f"📊 Prediksi mentah: {prediction[0]}")
        
        # LANGKAH 4: Validasi post-model - Cek confidence model
        model_valid, model_reason = validate_model_confidence(prediction, confidence_threshold=0.25)
        
        if not model_valid:
            print(f"❌ Validasi model gagal: {model_reason}")
            return jsonify({
                'success': False,
                'error': 'Model tidak dapat mengidentifikasi gambar sebagai daun tomat',
                'details': {
                    'reason': model_reason,
                    'suggestion': 'Pastikan gambar adalah daun tomat yang jelas dan berkualitas baik'
                }
            }), 400
        
        # LANGKAH 5: Ekstrak hasil
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(prediction))
        confidence_percentage = round(confidence * 100, 2)

        print(f"📊 Index yang diprediksi: {predicted_index}")
        print(f"📊 Kelas yang diprediksi: {predicted_class}")
        print(f"📊 Confidence: {confidence_percentage}%")
        
        # Dapatkan top 3 prediksi untuk debugging
        top_indices = np.argsort(prediction[0])[::-1][:3]
        print("📊 Top 3 prediksi:")
        for i, idx in enumerate(top_indices):
            print(f"   {i+1}. {class_names[idx]}: {prediction[0][idx]*100:.2f}%")

        # Tentukan apakah tanaman sehat
        is_plant_healthy = is_healthy_plant(predicted_class)
        print(f"📊 Apakah sehat: {is_plant_healthy}")

        # Dapatkan informasi penyakit
        disease_info = get_disease_info(predicted_class)
        
        # Convert gambar ke base64 untuk response
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        print("✅ Prediksi berhasil")
        return jsonify({
            'success': True,
            'data': {
                'classification': {
                    'class': predicted_class,
                    'class_name': disease_info['name'],
                    'confidence': confidence,
                    'confidence_percentage': confidence_percentage,
                    'is_healthy': is_plant_healthy,
                    'predicted_index': int(predicted_index)
                },
                'disease_info': disease_info,
                'validation_info': {
                    'leaf_confidence': leaf_confidence,
                    'passed_pre_validation': True,
                    'passed_model_validation': True
                },
                'debug_info': {
                    'top_predictions': [
                        {
                            'class': class_names[idx],
                            'confidence': float(prediction[0][idx]),
                            'percentage': round(float(prediction[0][idx]) * 100, 2)
                        }
                        for idx in top_indices
                    ],
                    'model_input_shape': str(model.input_shape),
                    'preprocessing_applied': 'resnet50_preprocess'
                },
                'image_base64': image_base64
            }
        })

    except Exception as e:
        print(f"❌ Error prediksi: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Prediksi gagal: {str(e)}'}), 500

@app.route('/test-classes', methods=['GET'])
def test_classes():
    """Endpoint untuk testing urutan class names"""
    return jsonify({
        'success': True,
        'data': {
            'class_names': class_names,
            'num_classes': len(class_names),
            'model_output_shape': str(model.output_shape) if model else None
        }
    })

@app.route('/diseases', methods=['GET'])
def get_diseases_info():
    """Return daftar semua penyakit dan deskripsinya"""
    try:
        data = []
        for class_name in class_names:
            data.append({
                'class': class_name,
                'info': get_disease_info(class_name)
            })
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Memulai Enhanced Tomato Disease Classification API...")
    print(f"📦 Model dimuat: {'Ya' if model is not None else 'Tidak'}")
    if model:
        print(f"📊 Input shape model: {model.input_shape}")
        print(f"📊 Kelas output model: {len(class_names)}")
    
    # Dapatkan port dari environment variable (requirement Render)
    port = int(os.environ.get('PORT', 5000))
    
    print("🌐 Endpoints:")
    print("- GET  /health")
    print("- POST /predict (dengan validasi gambar)")
    print("- GET  /diseases")
    print("- GET  /test-classes")
    print("- GET  /debug")
    print("🔍 Fitur validasi gambar:")
    print("- Analisis warna (dominasi hijau)")
    print("- Deteksi struktur tepi") 
    print("- Validasi rasio aspek")
    print("- Cek kecerahan/kontras")
    print("- Validasi confidence model")
    print(f"🌐 Server dimulai di port {port}")
    
    app.run(host='0.0.0.0', port=port, debug=False)