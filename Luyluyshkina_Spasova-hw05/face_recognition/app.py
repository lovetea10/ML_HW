import os
from flask import Flask, request, jsonify
from face_cli import FaceRecognizer
from config import CELEBA_CONFIG

app = Flask(__name__)
model_path = os.path.join(CELEBA_CONFIG['models_dir'], 'facenet_model.pth')
recognizer = FaceRecognizer(model_path=model_path, threshold=0.7)

@app.route('/recognize', methods=['POST'])
def recognize_face():
    if 'target' not in request.files or 'test_dir' not in request.form:
        return jsonify({'error': 'Missing target image or test directory'}), 400

    target_file = request.files['target']
    test_dir = request.form['test_dir']

    target_path = os.path.join(CELEBA_CONFIG['output_dir'], 'temp_target.jpg')
    target_file.save(target_path)

    results = recognizer.compare_with_target(target_path, test_dir)
    os.remove(target_path)

    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)