"""
EduPredict Backend — Complete Flask API
Run: python app.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import joblib
import json
import numpy as np
import os
import datetime
import jwt
import bcrypt
from functools import wraps

app = Flask(__name__)
CORS(app, origins=['http://localhost:5173', 'http://localhost:3000'])

SECRET_KEY = 'edupredict-secret-key-2026'
DB_PATH = 'edupredict.db'

# ── Load ML model ─────────────────────────────────────────────────────────────
MODEL = joblib.load('model.pkl')
with open('model_metadata.json') as f:
    META = json.load(f)

# ── Database setup ────────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()

    c.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL DEFAULT 'Admin User',
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'Administrator',
            joined_on TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            email TEXT,
            grade TEXT DEFAULT '—',
            performance TEXT,
            confidence TEXT,
            risk_level TEXT,
            last_prediction TEXT DEFAULT '—',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT NOT NULL,
            student_id TEXT,
            grade_level TEXT,
            predicted_grade REAL,
            performance TEXT,
            risk_level TEXT,
            confidence TEXT,
            input_summary TEXT,
            predicted_on TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            type TEXT NOT NULL,
            date_range TEXT DEFAULT '—',
            format TEXT DEFAULT 'PDF',
            generated_by TEXT DEFAULT 'Admin',
            generated_on TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            algorithm TEXT NOT NULL,
            model_type TEXT DEFAULT 'Regression',
            r2_score TEXT DEFAULT '—',
            mae TEXT DEFAULT '—',
            rmse TEXT DEFAULT '—',
            trained_on TEXT DEFAULT 'Student Academic Performance',
            status TEXT DEFAULT 'Trained',
            last_trained TEXT DEFAULT '—',
            icon_color TEXT DEFAULT 'text-indigo-600',
            icon_bg TEXT DEFAULT 'bg-indigo-50'
        );

        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            source TEXT DEFAULT 'School System',
            records TEXT DEFAULT '—',
            features TEXT DEFAULT '—',
            last_updated TEXT DEFAULT '—',
            quality_score INTEGER DEFAULT 0,
            status TEXT DEFAULT 'Active'
        );

        CREATE TABLE IF NOT EXISTS settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            default_page TEXT DEFAULT 'dashboard',
            theme TEXT DEFAULT 'Light',
            language TEXT DEFAULT 'English',
            timezone TEXT DEFAULT '(UTC+01:00) West Africa Time',
            email_notifications INTEGER DEFAULT 1,
            performance_alerts INTEGER DEFAULT 1,
            model_updates INTEGER DEFAULT 1,
            system_announcements INTEGER DEFAULT 0,
            auto_backup INTEGER DEFAULT 1,
            data_retention TEXT DEFAULT '365 days',
            maintenance_mode INTEGER DEFAULT 0,
            two_factor INTEGER DEFAULT 0
        );
    ''')

    # Seed default admin user
    c.execute('SELECT COUNT(*) FROM users')
    if c.fetchone()[0] == 0:
        pw_hash = bcrypt.hashpw('admin123'.encode(), bcrypt.gensalt()).decode()
        c.execute(
            'INSERT INTO users (full_name, email, password_hash, role, joined_on) VALUES (?,?,?,?,?)',
            ('Admin User', 'admin@edupredict.app', pw_hash, 'Administrator',
             datetime.datetime.now().strftime('%d %b %Y'))
        )
        user_id = c.lastrowid
        c.execute('INSERT INTO settings (user_id) VALUES (?)', (user_id,))

    # Seed default models
    c.execute('SELECT COUNT(*) FROM models')
    if c.fetchone()[0] == 0:
        default_models = [
            ('Random Forest Regressor', 'Random Forest', 'Regression', '0.83', '0.91', '1.63', 'Student Academic Performance', 'Deployed', datetime.datetime.now().strftime('%d %b %Y, %H:%M'), 'text-green-600', 'bg-green-50'),
            ('Decision Tree Regressor', 'Decision Tree', 'Regression', '0.79', '0.99', '1.80', 'Student Academic Performance', 'Trained', datetime.datetime.now().strftime('%d %b %Y, %H:%M'), 'text-blue-600', 'bg-blue-50'),
            ('Support Vector Machine', 'SVM', 'Regression', '0.78', '1.07', '1.83', 'Student Academic Performance', 'Trained', datetime.datetime.now().strftime('%d %b %Y, %H:%M'), 'text-purple-600', 'bg-purple-50'),
            ('Linear Regression', 'Linear Regression', 'Regression', '0.80', '1.01', '1.74', 'Student Academic Performance', 'Trained', datetime.datetime.now().strftime('%d %b %Y, %H:%M'), 'text-orange-600', 'bg-orange-50'),
            ('XGBoost Regressor', 'XGBoost', 'Regression', '—', '—', '—', 'Student Academic Performance', 'Trained', '—', 'text-emerald-600', 'bg-emerald-50'),
            ('LightGBM Regressor', 'LightGBM', 'Regression', '—', '—', '—', 'Student Academic Performance', 'Trained', '—', 'text-cyan-600', 'bg-cyan-50'),
            ('Neural Network', 'MLP Regressor', 'Regression', '—', '—', '—', 'Student Academic Performance', 'Training', '—', 'text-pink-600', 'bg-pink-50'),
        ]
        c.executemany(
            'INSERT INTO models (name,algorithm,model_type,r2_score,mae,rmse,trained_on,status,last_trained,icon_color,icon_bg) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
            default_models
        )

    # Seed default datasets
    c.execute('SELECT COUNT(*) FROM datasets')
    if c.fetchone()[0] == 0:
        default_datasets = [
            ('Student Academic Performance', 'Academic, attendance and assessment data', 'School System', '1,044', '34', datetime.datetime.now().strftime('%d %b %Y'), 94, 'Active'),
            ('Student Demographics', 'Demographic and background information', 'School System', '1,044', '12', '—', 91, 'Active'),
            ('Assignment & Assessment Data', 'Assignments, quizzes and exam scores', 'LMS', '—', '15', '—', 89, 'Active'),
            ('Attendance Records', 'Daily attendance and absenteeism data', 'School System', '—', '8', '—', 93, 'Active'),
            ('Behavior & Engagement Data', 'Behavior, participation and engagement metrics', 'Third Party', '—', '14', '—', 85, 'Inactive'),
        ]
        c.executemany(
            'INSERT INTO datasets (name,description,source,records,features,last_updated,quality_score,status) VALUES (?,?,?,?,?,?,?,?)',
            default_datasets
        )

    conn.commit()
    conn.close()

# ── Auth helpers ──────────────────────────────────────────────────────────────

def make_token(user_id):
    payload = {
        'user_id': user_id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=7)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({'error': 'Token missing'}), 401
        try:
            data = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
            request.user_id = data['user_id']
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except Exception:
            return jsonify({'error': 'Invalid token'}), 401
        return f(*args, **kwargs)
    return decorated

def now_str():
    return datetime.datetime.now().strftime('%d %b %Y, %H:%M')

def row_to_dict(row):
    return dict(row) if row else None

# ── ML Prediction helper ───────────────────────────────────────────────────────

def run_prediction(data):
    categorical_cols = META['categorical_cols']
    numeric_cols = META['numeric_cols']
    cat_options = META['categorical_options']

    row = {}
    for col in categorical_cols:
        val = data.get(col, cat_options[col][0])
        if val not in cat_options[col]:
            val = cat_options[col][0]
        row[col] = val

    for col in numeric_cols:
        try:
            row[col] = float(data.get(col, 0))
        except (TypeError, ValueError):
            row[col] = 0.0

    import pandas as pd
    df_input = pd.DataFrame([row])
    predicted = float(MODEL.predict(df_input)[0])
    predicted = max(0.0, min(20.0, predicted))
    is_pass = predicted >= 10
    confidence = round((predicted / 20) * 100)
    performance = 'High Performer' if is_pass else 'At Risk'
    risk_level = 'Low Risk' if predicted >= 14 else ('Moderate Risk' if predicted >= 10 else 'High Risk')
    return {
        'predicted_grade': round(predicted, 1),
        'performance': performance,
        'risk_level': risk_level,
        'confidence': f'{confidence}%',
        'is_pass': is_pass
    }

# ── AUTH ROUTES ───────────────────────────────────────────────────────────────

@app.route('/api/auth/login', methods=['POST'])
@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    full_name = data.get('full_name', '').strip()
    email = data.get('email', '').strip()
    password = data.get('password', '')
    role = data.get('role', 'Teacher').strip()

    if not full_name or not email or not password:
        return jsonify({'error': 'Name, email and password are required'}), 400

    if len(password) < 8:
        return jsonify({'error': 'Password must be at least 8 characters'}), 400

    conn = get_db()

    # Check if email already exists
    existing = conn.execute('SELECT id FROM users WHERE email = ?', (email,)).fetchone()
    if existing:
        conn.close()
        return jsonify({'error': 'An account with this email already exists'}), 409

    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    conn.execute(
        'INSERT INTO users (full_name, email, password_hash, role, joined_on) VALUES (?,?,?,?,?)',
        (full_name, email, pw_hash, role, datetime.datetime.now().strftime('%d %b %Y'))
    )
    conn.commit()
    user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
    user_id = user['id']

    # Create default settings for new user
    conn.execute('INSERT INTO settings (user_id) VALUES (?)', (user_id,))
    conn.commit()
    conn.close()

    token = make_token(user_id)
    return jsonify({
        'token': token,
        'user': {
            'id': user_id,
            'full_name': full_name,
            'email': email,
            'role': role,
            'joined_on': datetime.datetime.now().strftime('%d %b %Y')
        }
    }), 201

@app.route('/api/auth/change-password', methods=['PUT'])
@token_required
def change_password():
    data = request.get_json()
    current = data.get('current_password', '')
    new_pw = data.get('new_password', '')

    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (request.user_id,)).fetchone()

    if not bcrypt.checkpw(current.encode(), user['password_hash'].encode()):
        conn.close()
        return jsonify({'error': 'Current password is incorrect'}), 400

    new_hash = bcrypt.hashpw(new_pw.encode(), bcrypt.gensalt()).decode()
    conn.execute('UPDATE users SET password_hash = ? WHERE id = ?', (new_hash, request.user_id))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Password updated successfully'})

# ── SETTINGS ROUTES ───────────────────────────────────────────────────────────

@app.route('/api/settings', methods=['GET'])
@token_required
def get_settings():
    conn = get_db()
    s = conn.execute('SELECT * FROM settings WHERE user_id = ?', (request.user_id,)).fetchone()
    conn.close()
    return jsonify(row_to_dict(s))

@app.route('/api/settings', methods=['PUT'])
@token_required
def update_settings():
    data = request.get_json()
    conn = get_db()
    conn.execute('''
        UPDATE settings SET
            default_page = ?,
            theme = ?,
            language = ?,
            timezone = ?,
            email_notifications = ?,
            performance_alerts = ?,
            model_updates = ?,
            system_announcements = ?,
            auto_backup = ?,
            data_retention = ?,
            maintenance_mode = ?,
            two_factor = ?
        WHERE user_id = ?
    ''', (
        data.get('default_page', 'dashboard'),
        data.get('theme', 'Light'),
        data.get('language', 'English'),
        data.get('timezone', '(UTC+01:00) West Africa Time'),
        int(data.get('email_notifications', 1)),
        int(data.get('performance_alerts', 1)),
        int(data.get('model_updates', 1)),
        int(data.get('system_announcements', 0)),
        int(data.get('auto_backup', 1)),
        data.get('data_retention', '365 days'),
        int(data.get('maintenance_mode', 0)),
        int(data.get('two_factor', 0)),
        request.user_id
    ))
    conn.commit()
    s = conn.execute('SELECT * FROM settings WHERE user_id = ?', (request.user_id,)).fetchone()
    conn.close()
    return jsonify(row_to_dict(s))

# ── STUDENT ROUTES ────────────────────────────────────────────────────────────

@app.route('/api/students', methods=['GET'])
@token_required
def get_students():
    search = request.args.get('search', '')
    grade = request.args.get('grade', '')
    performance = request.args.get('performance', '')
    risk = request.args.get('risk', '')

    query = 'SELECT * FROM students WHERE 1=1'
    params = []

    if search:
        query += ' AND (name LIKE ? OR student_id LIKE ?)'
        params += [f'%{search}%', f'%{search}%']
    if grade and grade != 'All Grades':
        query += ' AND grade = ?'
        params.append(grade)
    if performance and performance != 'All Performance':
        query += ' AND performance = ?'
        params.append(performance)
    if risk and risk != 'All Status':
        query += ' AND risk_level = ?'
        params.append(risk)

    query += ' ORDER BY id DESC'

    conn = get_db()
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/students', methods=['POST'])
@token_required
def add_student():
    data = request.get_json()
    name = data.get('name', '').strip()
    email = data.get('email', '').strip()
    grade = data.get('grade', '—')

    if not name:
        return jsonify({'error': 'Name is required'}), 400

    conn = get_db()
    count = conn.execute('SELECT COUNT(*) FROM students').fetchone()[0]
    student_id = f'STU{1000 + count}'

    existing = conn.execute('SELECT id FROM students WHERE LOWER(name) = LOWER(?)', (name,)).fetchone()
    if existing:
        conn.close()
        return jsonify({'error': 'Student already exists'}), 409

    email = email or f"{name.lower().replace(' ', '.')}@example.edu"

    conn.execute(
        'INSERT INTO students (student_id, name, email, grade, created_at) VALUES (?,?,?,?,?)',
        (student_id, name, email, grade, now_str())
    )
    conn.commit()
    student = conn.execute('SELECT * FROM students WHERE name = ?', (name,)).fetchone()
    conn.close()
    return jsonify(dict(student)), 201

@app.route('/api/students/<int:student_id>', methods=['PUT'])
@token_required
def update_student(student_id):
    data = request.get_json()
    conn = get_db()
    conn.execute('''
        UPDATE students SET
            name = COALESCE(?, name),
            email = COALESCE(?, email),
            grade = COALESCE(?, grade),
            performance = COALESCE(?, performance),
            confidence = COALESCE(?, confidence),
            risk_level = COALESCE(?, risk_level),
            last_prediction = COALESCE(?, last_prediction)
        WHERE id = ?
    ''', (
        data.get('name'), data.get('email'), data.get('grade'),
        data.get('performance'), data.get('confidence'),
        data.get('risk_level'), data.get('last_prediction'),
        student_id
    ))
    conn.commit()
    student = conn.execute('SELECT * FROM students WHERE id = ?', (student_id,)).fetchone()
    conn.close()
    return jsonify(dict(student))

@app.route('/api/students/<int:student_id>', methods=['DELETE'])
@token_required
def delete_student(student_id):
    conn = get_db()
    conn.execute('DELETE FROM students WHERE id = ?', (student_id,))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Student deleted'})

# ── PREDICTION ROUTES ─────────────────────────────────────────────────────────

@app.route('/api/predict', methods=['POST'])
@token_required
def predict():
    data = request.get_json()

    try:
        result = run_prediction(data)
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    student_name = data.get('selectedStudent', 'Unknown Student')
    grade_level = data.get('gradeLevel', '—')
    predicted_on = now_str()

    summary_parts = []
    if data.get('studyHours'): summary_parts.append(f"Study: {data['studyHours']}hrs")
    if data.get('gpa'): summary_parts.append(f"GPA: {data['gpa']}")
    if data.get('attendance'): summary_parts.append(f"Attendance: {data['attendance']}%")
    if data.get('assignments'): summary_parts.append(f"Assignments: {data['assignments']}%")
    input_summary = ', '.join(summary_parts) or 'No summary available'

    conn = get_db()
    conn.execute('''
        INSERT INTO predictions (student_name, student_id, grade_level, predicted_grade,
            performance, risk_level, confidence, input_summary, predicted_on)
        VALUES (?,?,?,?,?,?,?,?,?)
    ''', (
        student_name, data.get('studentId'), grade_level,
        result['predicted_grade'], result['performance'],
        result['risk_level'], result['confidence'],
        input_summary, predicted_on
    ))

    if student_name and student_name != 'Unknown Student':
        conn.execute('''
            UPDATE students SET
                performance = ?,
                risk_level = ?,
                confidence = ?,
                last_prediction = ?,
                grade = CASE WHEN grade = '—' THEN ? ELSE grade END
            WHERE LOWER(name) = LOWER(?)
        ''', (
            result['performance'], result['risk_level'],
            result['confidence'], predicted_on,
            grade_level, student_name
        ))

    conn.commit()
    conn.close()

    return jsonify({**result, 'predicted_on': predicted_on, 'input_summary': input_summary})

@app.route('/api/predictions', methods=['GET'])
@token_required
def get_predictions():
    conn = get_db()
    rows = conn.execute('SELECT * FROM predictions ORDER BY id DESC').fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

# ── REPORT ROUTES ─────────────────────────────────────────────────────────────

@app.route('/api/reports', methods=['GET'])
@token_required
def get_reports():
    search = request.args.get('search', '')
    report_type = request.args.get('type', '')

    query = 'SELECT * FROM reports WHERE 1=1'
    params = []

    if search:
        query += ' AND (name LIKE ? OR description LIKE ?)'
        params += [f'%{search}%', f'%{search}%']
    if report_type and report_type != 'All Report Types':
        query += ' AND type = ?'
        params.append(report_type)

    query += ' ORDER BY id DESC'

    conn = get_db()
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/reports', methods=['POST'])
@token_required
def create_report():
    data = request.get_json()
    name = data.get('name', '').strip()
    report_type = data.get('type', '').strip()

    if not name or not report_type:
        return jsonify({'error': 'Name and type are required'}), 400

    conn = get_db()
    conn.execute('''
        INSERT INTO reports (name, description, type, date_range, format, generated_by, generated_on)
        VALUES (?,?,?,?,?,?,?)
    ''', (
        name,
        data.get('description', ''),
        report_type,
        data.get('dateRange', '—'),
        data.get('format', 'PDF'),
        'Admin',
        now_str()
    ))
    conn.commit()
    report = conn.execute('SELECT * FROM reports ORDER BY id DESC LIMIT 1').fetchone()
    conn.close()
    return jsonify(dict(report)), 201

@app.route('/api/reports/<int:report_id>', methods=['PUT'])
@token_required
def update_report(report_id):
    data = request.get_json()
    conn = get_db()
    conn.execute('''
        UPDATE reports SET
            name = COALESCE(?, name),
            description = COALESCE(?, description),
            type = COALESCE(?, type),
            date_range = COALESCE(?, date_range),
            format = COALESCE(?, format)
        WHERE id = ?
    ''', (
        data.get('name'), data.get('description'),
        data.get('type'), data.get('dateRange'),
        data.get('format'), report_id
    ))
    conn.commit()
    report = conn.execute('SELECT * FROM reports WHERE id = ?', (report_id,)).fetchone()
    conn.close()
    return jsonify(dict(report))

@app.route('/api/reports/<int:report_id>', methods=['DELETE'])
@token_required
def delete_report(report_id):
    conn = get_db()
    conn.execute('DELETE FROM reports WHERE id = ?', (report_id,))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Report deleted'})

# ── MODEL ROUTES ──────────────────────────────────────────────────────────────

@app.route('/api/models', methods=['GET'])
@token_required
def get_models():
    conn = get_db()
    rows = conn.execute('SELECT * FROM models ORDER BY id ASC').fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/models', methods=['POST'])
@token_required
def train_model():
    data = request.get_json()
    name = data.get('name', '').strip()
    algorithm = data.get('algorithm', '').strip()

    if not name or not algorithm:
        return jsonify({'error': 'Name and algorithm are required'}), 400

    icon_map = {
        'Random Forest': ('text-green-600', 'bg-green-50'),
        'Decision Tree': ('text-blue-600', 'bg-blue-50'),
        'SVM': ('text-purple-600', 'bg-purple-50'),
        'Linear Regression': ('text-orange-600', 'bg-orange-50'),
        'XGBoost': ('text-emerald-600', 'bg-emerald-50'),
        'LightGBM': ('text-cyan-600', 'bg-cyan-50'),
        'MLP Regressor': ('text-pink-600', 'bg-pink-50'),
    }
    icon_color, icon_bg = icon_map.get(algorithm, ('text-indigo-600', 'bg-indigo-50'))

    conn = get_db()
    conn.execute('''
        INSERT INTO models (name, algorithm, model_type, r2_score, mae, rmse,
            trained_on, status, last_trained, icon_color, icon_bg)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    ''', (
        name, algorithm, 'Regression', '—', '—', '—',
        data.get('dataset', 'Student Academic Performance'),
        'Training', now_str(), icon_color, icon_bg
    ))
    conn.commit()
    model = conn.execute('SELECT * FROM models ORDER BY id DESC LIMIT 1').fetchone()
    conn.close()
    return jsonify(dict(model)), 201

@app.route('/api/models/<int:model_id>', methods=['PUT'])
@token_required
def update_model(model_id):
    data = request.get_json()
    conn = get_db()
    conn.execute('''
        UPDATE models SET
            name = COALESCE(?, name),
            status = COALESCE(?, status),
            r2_score = COALESCE(?, r2_score),
            mae = COALESCE(?, mae),
            rmse = COALESCE(?, rmse)
        WHERE id = ?
    ''', (data.get('name'), data.get('status'), data.get('r2_score'), data.get('mae'), data.get('rmse'), model_id))
    conn.commit()
    model = conn.execute('SELECT * FROM models WHERE id = ?', (model_id,)).fetchone()
    conn.close()
    return jsonify(dict(model))

@app.route('/api/models/<int:model_id>', methods=['DELETE'])
@token_required
def delete_model(model_id):
    conn = get_db()
    conn.execute('DELETE FROM models WHERE id = ?', (model_id,))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Model deleted'})

# ── DATASET ROUTES ────────────────────────────────────────────────────────────

@app.route('/api/datasets', methods=['GET'])
@token_required
def get_datasets():
    conn = get_db()
    rows = conn.execute('SELECT * FROM datasets ORDER BY id ASC').fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/datasets', methods=['POST'])
@token_required
def add_dataset():
    data = request.get_json()
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'error': 'Name is required'}), 400

    conn = get_db()
    conn.execute('''
        INSERT INTO datasets (name, description, source, records, features, last_updated, quality_score, status)
        VALUES (?,?,?,?,?,?,?,?)
    ''', (
        name,
        data.get('description', ''),
        data.get('source', 'School System'),
        data.get('records', '—'),
        data.get('features', '—'),
        now_str(),
        data.get('quality_score', 0),
        data.get('status', 'Active')
    ))
    conn.commit()
    dataset = conn.execute('SELECT * FROM datasets ORDER BY id DESC LIMIT 1').fetchone()
    conn.close()
    return jsonify(dict(dataset)), 201

@app.route('/api/datasets/<int:dataset_id>', methods=['PUT'])
@token_required
def update_dataset(dataset_id):
    data = request.get_json()
    conn = get_db()
    conn.execute('''
        UPDATE datasets SET
            name = COALESCE(?, name),
            description = COALESCE(?, description),
            source = COALESCE(?, source),
            status = COALESCE(?, status)
        WHERE id = ?
    ''', (data.get('name'), data.get('description'), data.get('source'), data.get('status'), dataset_id))
    conn.commit()
    dataset = conn.execute('SELECT * FROM datasets WHERE id = ?', (dataset_id,)).fetchone()
    conn.close()
    return jsonify(dict(dataset))

@app.route('/api/datasets/<int:dataset_id>', methods=['DELETE'])
@token_required
def delete_dataset(dataset_id):
    conn = get_db()
    conn.execute('DELETE FROM datasets WHERE id = ?', (dataset_id,))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Dataset deleted'})

# ── ANALYTICS ROUTE ───────────────────────────────────────────────────────────

@app.route('/api/analytics', methods=['GET'])
@token_required
def get_analytics():
    conn = get_db()

    total_students = conn.execute('SELECT COUNT(*) FROM students').fetchone()[0]
    high_performers = conn.execute("SELECT COUNT(*) FROM students WHERE performance = 'High Performer'").fetchone()[0]
    at_risk = conn.execute("SELECT COUNT(*) FROM students WHERE risk_level = 'High Risk'").fetchone()[0]
    moderate_risk = conn.execute("SELECT COUNT(*) FROM students WHERE risk_level = 'Moderate Risk'").fetchone()[0]
    low_risk = conn.execute("SELECT COUNT(*) FROM students WHERE risk_level = 'Low Risk'").fetchone()[0]

    predictions = conn.execute('SELECT confidence, predicted_grade FROM predictions').fetchall()
    avg_score = None
    avg_grade = None
    if predictions:
        confidences = []
        grades = []
        for p in predictions:
            try:
                confidences.append(float(p['confidence'].replace('%', '')))
                grades.append(float(p['predicted_grade']))
            except:
                pass
        if confidences:
            avg_score = round(sum(confidences) / len(confidences), 1)
        if grades:
            avg_grade = round(sum(grades) / len(grades), 1)

    total_predictions = conn.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
    total_reports = conn.execute('SELECT COUNT(*) FROM reports').fetchone()[0]
    total_models = conn.execute('SELECT COUNT(*) FROM models').fetchone()[0]

    conn.close()

    return jsonify({
        'total_students': total_students,
        'high_performers': high_performers,
        'at_risk': at_risk,
        'moderate_risk': moderate_risk,
        'low_risk': low_risk,
        'avg_confidence': avg_score,
        'avg_predicted_grade': avg_grade,
        'total_predictions': total_predictions,
        'total_reports': total_reports,
        'total_models': total_models,
        'overall_accuracy': avg_score,
    })

# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    init_db()
    print('EduPredict backend running on http://localhost:5000')
    print('Default login: admin@edupredict.app / admin123')
    app.run(debug=True, port=5000)
