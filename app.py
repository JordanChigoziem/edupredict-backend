"""
EduPredict Backend — Complete Flask API with per-user data isolation
Run: python app.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import joblib
import json
import datetime
import jwt
import os
import bcrypt
from functools import wraps

app = Flask(__name__)
CORS(app, origins=['http://localhost:5173', 'http://localhost:3000', 'https://edupredict-frontend-10.vercel.app'])

SECRET_KEY = 'edupredict-secret-key-2026'
DB_PATH = 'edupredict.db'

MODEL = joblib.load('model.pkl')
with open('model_metadata.json') as f:
    META = json.load(f)

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
            user_id INTEGER NOT NULL,
            student_id TEXT NOT NULL,
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
            user_id INTEGER NOT NULL,
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
            user_id INTEGER NOT NULL,
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
            user_id INTEGER,
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
            icon_bg TEXT DEFAULT 'bg-indigo-50',
            is_global INTEGER DEFAULT 0
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
            user_id INTEGER NOT NULL UNIQUE,
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

    # Seed admin user
    if c.execute('SELECT COUNT(*) FROM users').fetchone()[0] == 0:
        pw_hash = bcrypt.hashpw('admin123'.encode(), bcrypt.gensalt()).decode()
        c.execute('INSERT INTO users (full_name,email,password_hash,role,joined_on) VALUES (?,?,?,?,?)',
            ('Admin User','admin@edupredict.app',pw_hash,'Administrator',datetime.datetime.now().strftime('%d %b %Y')))
        uid = c.lastrowid
        c.execute('INSERT OR IGNORE INTO settings (user_id) VALUES (?)', (uid,))

    # Seed global datasets
    if c.execute('SELECT COUNT(*) FROM datasets').fetchone()[0] == 0:
        c.executemany('INSERT INTO datasets (name,description,source,records,features,last_updated,quality_score,status) VALUES (?,?,?,?,?,?,?,?)', [
            ('Student Academic Performance','Academic, attendance and assessment data','School System','1,044','34',datetime.datetime.now().strftime('%d %b %Y'),94,'Active'),
            ('Student Demographics','Demographic and background information','School System','1,044','12','—',91,'Active'),
            ('Assignment & Assessment Data','Assignments, quizzes and exam scores','LMS','—','15','—',89,'Active'),
            ('Attendance Records','Daily attendance and absenteeism data','School System','—','8','—',93,'Active'),
            ('Behavior & Engagement Data','Behavior, participation and engagement metrics','Third Party','—','14','—',85,'Inactive'),
        ])

    # Seed global ML models
    if c.execute('SELECT COUNT(*) FROM models WHERE is_global=1').fetchone()[0] == 0:
        c.executemany('INSERT INTO models (user_id,name,algorithm,model_type,r2_score,mae,rmse,trained_on,status,last_trained,icon_color,icon_bg,is_global) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)', [
            (None,'Random Forest Regressor','Random Forest','Regression','0.83','0.91','1.63','Student Academic Performance','Deployed',datetime.datetime.now().strftime('%d %b %Y, %H:%M'),'text-green-600','bg-green-50',1),
            (None,'Decision Tree Regressor','Decision Tree','Regression','0.79','0.99','1.80','Student Academic Performance','Trained',datetime.datetime.now().strftime('%d %b %Y, %H:%M'),'text-blue-600','bg-blue-50',1),
            (None,'Support Vector Machine','SVM','Regression','0.78','1.07','1.83','Student Academic Performance','Trained',datetime.datetime.now().strftime('%d %b %Y, %H:%M'),'text-purple-600','bg-purple-50',1),
            (None,'Linear Regression','Linear Regression','Regression','0.80','1.01','1.74','Student Academic Performance','Trained',datetime.datetime.now().strftime('%d %b %Y, %H:%M'),'text-orange-600','bg-orange-50',1),
            (None,'XGBoost Regressor','XGBoost','Regression','—','—','—','Student Academic Performance','Trained','—','text-emerald-600','bg-emerald-50',1),
            (None,'LightGBM Regressor','LightGBM','Regression','—','—','—','Student Academic Performance','Trained','—','text-cyan-600','bg-cyan-50',1),
            (None,'Neural Network','MLP Regressor','Regression','—','—','—','Student Academic Performance','Training','—','text-pink-600','bg-pink-50',1),
        ])

    conn.commit()
    conn.close()

def make_token(user_id):
    return jwt.encode({'user_id':user_id,'exp':datetime.datetime.utcnow()+datetime.timedelta(days=7)}, SECRET_KEY, algorithm='HS256')

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization','').replace('Bearer ','')
        if not token:
            return jsonify({'error':'Token missing'}),401
        try:
            data = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
            request.user_id = data['user_id']
        except jwt.ExpiredSignatureError:
            return jsonify({'error':'Token expired'}),401
        except:
            return jsonify({'error':'Invalid token'}),401
        return f(*args, **kwargs)
    return decorated

def now_str():
    return datetime.datetime.now().strftime('%d %b %Y, %H:%M')

def row_to_dict(row):
    return dict(row) if row else None

def run_prediction(data):
    import pandas as pd
    categorical_cols = META['categorical_cols']
    numeric_cols = META['numeric_cols']
    cat_options = META['categorical_options']
    row = {}
    for col in categorical_cols:
        val = data.get(col, cat_options[col][0])
        if val not in cat_options[col]: val = cat_options[col][0]
        row[col] = val
    for col in numeric_cols:
        try: row[col] = float(data.get(col,0))
        except: row[col] = 0.0
    predicted = float(MODEL.predict(pd.DataFrame([row]))[0])
    predicted = max(0.0, min(20.0, predicted))
    is_pass = predicted >= 10
    confidence = round((predicted/20)*100)
    return {
        'predicted_grade': round(predicted,1),
        'performance': 'High Performer' if is_pass else 'At Risk',
        'risk_level': 'Low Risk' if predicted>=14 else ('Moderate Risk' if predicted>=10 else 'High Risk'),
        'confidence': f'{confidence}%',
        'is_pass': is_pass
    }

# AUTH
@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email', '').strip()
    password = data.get('password', '')
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
    conn.close()
    if not user or not bcrypt.checkpw(password.encode(), user['password_hash'].encode()):
        return jsonify({'error': 'Invalid email or password'}), 401
    token = make_token(user['id'])
    return jsonify({'token': token, 'user': {'id': user['id'], 'full_name': user['full_name'], 'email': user['email'], 'role': user['role'], 'joined_on': user['joined_on']}})

@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    full_name = data.get('full_name','').strip()
    email = data.get('email','').strip()
    password = data.get('password','')
    role = data.get('role','Teacher').strip()
    if not full_name or not email or not password:
        return jsonify({'error':'Name, email and password are required'}),400
    if len(password) < 8:
        return jsonify({'error':'Password must be at least 8 characters'}),400
    conn = get_db()
    if conn.execute('SELECT id FROM users WHERE email=?',(email,)).fetchone():
        conn.close()
        return jsonify({'error':'An account with this email already exists'}),409
    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    conn.execute('INSERT INTO users (full_name,email,password_hash,role,joined_on) VALUES (?,?,?,?,?)',
        (full_name,email,pw_hash,role,datetime.datetime.now().strftime('%d %b %Y')))
    conn.commit()
    user = conn.execute('SELECT * FROM users WHERE email=?',(email,)).fetchone()
    conn.execute('INSERT OR IGNORE INTO settings (user_id) VALUES (?)',(user['id'],))
    conn.commit()
    conn.close()
    return jsonify({'token':make_token(user['id']),'user':{'id':user['id'],'full_name':full_name,'email':email,'role':role,'joined_on':datetime.datetime.now().strftime('%d %b %Y')}}),201

@app.route('/api/auth/me', methods=['GET'])
@token_required
def get_me():
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id=?',(request.user_id,)).fetchone()
    conn.close()
    if not user: return jsonify({'error':'User not found'}),404
    return jsonify({'id':user['id'],'full_name':user['full_name'],'email':user['email'],'role':user['role'],'joined_on':user['joined_on']})

@app.route('/api/auth/profile', methods=['PUT'])
@token_required
def update_profile():
    data = request.get_json()
    conn = get_db()
    conn.execute('UPDATE users SET full_name=?,email=? WHERE id=?',(data.get('full_name'),data.get('email'),request.user_id))
    conn.commit()
    user = conn.execute('SELECT * FROM users WHERE id=?',(request.user_id,)).fetchone()
    conn.close()
    return jsonify({'id':user['id'],'full_name':user['full_name'],'email':user['email'],'role':user['role']})

@app.route('/api/auth/change-password', methods=['PUT'])
@token_required
def change_password():
    data = request.get_json()
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id=?',(request.user_id,)).fetchone()
    if not bcrypt.checkpw(data.get('current_password','').encode(), user['password_hash'].encode()):
        conn.close()
        return jsonify({'error':'Current password is incorrect'}),400
    conn.execute('UPDATE users SET password_hash=? WHERE id=?',(bcrypt.hashpw(data.get('new_password','').encode(),bcrypt.gensalt()).decode(),request.user_id))
    conn.commit()
    conn.close()
    return jsonify({'message':'Password updated'})

# SETTINGS
@app.route('/api/settings', methods=['GET'])
@token_required
def get_settings():
    conn = get_db()
    s = conn.execute('SELECT * FROM settings WHERE user_id=?',(request.user_id,)).fetchone()
    if not s:
        conn.execute('INSERT OR IGNORE INTO settings (user_id) VALUES (?)',(request.user_id,))
        conn.commit()
        s = conn.execute('SELECT * FROM settings WHERE user_id=?',(request.user_id,)).fetchone()
    conn.close()
    return jsonify(row_to_dict(s))

@app.route('/api/settings', methods=['PUT'])
@token_required
def update_settings():
    data = request.get_json()
    conn = get_db()
    conn.execute('''INSERT INTO settings (user_id,default_page,theme,language,timezone,
        email_notifications,performance_alerts,model_updates,system_announcements,
        auto_backup,data_retention,maintenance_mode,two_factor)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(user_id) DO UPDATE SET
        default_page=excluded.default_page,theme=excluded.theme,language=excluded.language,
        timezone=excluded.timezone,email_notifications=excluded.email_notifications,
        performance_alerts=excluded.performance_alerts,model_updates=excluded.model_updates,
        system_announcements=excluded.system_announcements,auto_backup=excluded.auto_backup,
        data_retention=excluded.data_retention,maintenance_mode=excluded.maintenance_mode,
        two_factor=excluded.two_factor''',
        (request.user_id,data.get('default_page','dashboard'),data.get('theme','Light'),
         data.get('language','English'),data.get('timezone','(UTC+01:00) West Africa Time'),
         int(data.get('email_notifications',1)),int(data.get('performance_alerts',1)),
         int(data.get('model_updates',1)),int(data.get('system_announcements',0)),
         int(data.get('auto_backup',1)),data.get('data_retention','365 days'),
         int(data.get('maintenance_mode',0)),int(data.get('two_factor',0))))
    conn.commit()
    s = conn.execute('SELECT * FROM settings WHERE user_id=?',(request.user_id,)).fetchone()
    conn.close()
    return jsonify(row_to_dict(s))

# STUDENTS — per user
@app.route('/api/students', methods=['GET'])
@token_required
def get_students():
    search = request.args.get('search','')
    grade = request.args.get('grade','')
    performance = request.args.get('performance','')
    risk = request.args.get('risk','')
    query = 'SELECT * FROM students WHERE user_id=?'
    params = [request.user_id]
    if search:
        query += ' AND (name LIKE ? OR student_id LIKE ?)'
        params += [f'%{search}%',f'%{search}%']
    if grade and grade != 'All Grades':
        query += ' AND grade=?'; params.append(grade)
    if performance and performance != 'All Performance':
        query += ' AND performance=?'; params.append(performance)
    if risk and risk != 'All Status':
        query += ' AND risk_level=?'; params.append(risk)
    query += ' ORDER BY id DESC'
    conn = get_db()
    rows = conn.execute(query,params).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/students', methods=['POST'])
@token_required
def add_student():
    data = request.get_json()
    name = data.get('name','').strip()
    if not name: return jsonify({'error':'Name is required'}),400
    conn = get_db()
    if conn.execute('SELECT id FROM students WHERE user_id=? AND LOWER(name)=LOWER(?)',(request.user_id,name)).fetchone():
        conn.close()
        return jsonify({'error':'Student already exists'}),409
    count = conn.execute('SELECT COUNT(*) FROM students WHERE user_id=?',(request.user_id,)).fetchone()[0]
    student_id = f'STU{1000+count}'
    email = data.get('email','').strip() or f"{name.lower().replace(' ','.')}@example.edu"
    conn.execute('INSERT INTO students (user_id,student_id,name,email,grade,created_at) VALUES (?,?,?,?,?,?)',
        (request.user_id,student_id,name,email,data.get('grade','—'),now_str()))
    conn.commit()
    student = conn.execute('SELECT * FROM students WHERE user_id=? AND name=?',(request.user_id,name)).fetchone()
    conn.close()
    return jsonify(dict(student)),201

@app.route('/api/students/<int:sid>', methods=['PUT'])
@token_required
def update_student(sid):
    data = request.get_json()
    conn = get_db()
    conn.execute('''UPDATE students SET name=COALESCE(?,name),email=COALESCE(?,email),
        grade=COALESCE(?,grade),performance=COALESCE(?,performance),
        confidence=COALESCE(?,confidence),risk_level=COALESCE(?,risk_level),
        last_prediction=COALESCE(?,last_prediction) WHERE id=? AND user_id=?''',
        (data.get('name'),data.get('email'),data.get('grade'),data.get('performance'),
         data.get('confidence'),data.get('risk_level'),data.get('last_prediction'),sid,request.user_id))
    conn.commit()
    student = conn.execute('SELECT * FROM students WHERE id=? AND user_id=?',(sid,request.user_id)).fetchone()
    conn.close()
    return jsonify(dict(student))

@app.route('/api/students/<int:sid>', methods=['DELETE'])
@token_required
def delete_student(sid):
    conn = get_db()
    conn.execute('DELETE FROM students WHERE id=? AND user_id=?',(sid,request.user_id))
    conn.commit()
    conn.close()
    return jsonify({'message':'Student deleted'})

# PREDICTIONS — per user
@app.route('/api/predict', methods=['POST'])
@token_required
def predict():
    data = request.get_json()
    try:
        result = run_prediction(data)
    except Exception as e:
        return jsonify({'error':f'Prediction failed: {str(e)}'}),500
    student_name = data.get('selectedStudent','Unknown Student')
    grade_level = data.get('gradeLevel','—')
    predicted_on = now_str()
    parts = []
    if data.get('studyHours'): parts.append(f"Study: {data['studyHours']}hrs")
    if data.get('gpa'): parts.append(f"GPA: {data['gpa']}")
    if data.get('attendance'): parts.append(f"Attendance: {data['attendance']}%")
    if data.get('assignments'): parts.append(f"Assignments: {data['assignments']}%")
    input_summary = ', '.join(parts) or 'No summary available'
    conn = get_db()
    conn.execute('''INSERT INTO predictions (user_id,student_name,student_id,grade_level,
        predicted_grade,performance,risk_level,confidence,input_summary,predicted_on)
        VALUES (?,?,?,?,?,?,?,?,?,?)''',
        (request.user_id,student_name,data.get('studentId'),grade_level,
         result['predicted_grade'],result['performance'],result['risk_level'],
         result['confidence'],input_summary,predicted_on))
    if student_name and student_name != 'Unknown Student':
        conn.execute('''UPDATE students SET performance=?,risk_level=?,confidence=?,
            last_prediction=?,grade=CASE WHEN grade='—' THEN ? ELSE grade END
            WHERE LOWER(name)=LOWER(?) AND user_id=?''',
            (result['performance'],result['risk_level'],result['confidence'],
             predicted_on,grade_level,student_name,request.user_id))
    conn.commit()
    conn.close()
    return jsonify({**result,'predicted_on':predicted_on,'input_summary':input_summary})

@app.route('/api/predictions', methods=['GET'])
@token_required
def get_predictions():
    conn = get_db()
    rows = conn.execute('SELECT * FROM predictions WHERE user_id=? ORDER BY id DESC',(request.user_id,)).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

# REPORTS — per user
@app.route('/api/reports', methods=['GET'])
@token_required
def get_reports():
    search = request.args.get('search','')
    rtype = request.args.get('type','')
    query = 'SELECT * FROM reports WHERE user_id=?'
    params = [request.user_id]
    if search:
        query += ' AND (name LIKE ? OR description LIKE ?)'; params += [f'%{search}%',f'%{search}%']
    if rtype and rtype != 'All Report Types':
        query += ' AND type=?'; params.append(rtype)
    query += ' ORDER BY id DESC'
    conn = get_db()
    rows = conn.execute(query,params).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/reports', methods=['POST'])
@token_required
def create_report():
    data = request.get_json()
    name = data.get('name','').strip()
    rtype = data.get('type','').strip()
    if not name or not rtype: return jsonify({'error':'Name and type are required'}),400
    conn = get_db()
    conn.execute('INSERT INTO reports (user_id,name,description,type,date_range,format,generated_by,generated_on) VALUES (?,?,?,?,?,?,?,?)',
        (request.user_id,name,data.get('description',''),rtype,data.get('dateRange','—'),data.get('format','PDF'),'Admin',now_str()))
    conn.commit()
    report = conn.execute('SELECT * FROM reports WHERE user_id=? ORDER BY id DESC LIMIT 1',(request.user_id,)).fetchone()
    conn.close()
    return jsonify(dict(report)),201

@app.route('/api/reports/<int:rid>', methods=['PUT'])
@token_required
def update_report(rid):
    data = request.get_json()
    conn = get_db()
    conn.execute('''UPDATE reports SET name=COALESCE(?,name),description=COALESCE(?,description),
        type=COALESCE(?,type),date_range=COALESCE(?,date_range),format=COALESCE(?,format)
        WHERE id=? AND user_id=?''',
        (data.get('name'),data.get('description'),data.get('type'),data.get('dateRange'),data.get('format'),rid,request.user_id))
    conn.commit()
    report = conn.execute('SELECT * FROM reports WHERE id=? AND user_id=?',(rid,request.user_id)).fetchone()
    conn.close()
    return jsonify(dict(report))

@app.route('/api/reports/<int:rid>', methods=['DELETE'])
@token_required
def delete_report(rid):
    conn = get_db()
    conn.execute('DELETE FROM reports WHERE id=? AND user_id=?',(rid,request.user_id))
    conn.commit()
    conn.close()
    return jsonify({'message':'Report deleted'})

# MODELS — global + user's own
@app.route('/api/models', methods=['GET'])
@token_required
def get_models():
    conn = get_db()
    rows = conn.execute('SELECT * FROM models WHERE is_global=1 OR user_id=? ORDER BY id ASC',(request.user_id,)).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/models', methods=['POST'])
@token_required
def train_model():
    data = request.get_json()
    name = data.get('name','').strip()
    algorithm = data.get('algorithm','').strip()
    if not name or not algorithm: return jsonify({'error':'Name and algorithm required'}),400
    icon_map = {'Random Forest':('text-green-600','bg-green-50'),'Decision Tree':('text-blue-600','bg-blue-50'),
        'SVM':('text-purple-600','bg-purple-50'),'Linear Regression':('text-orange-600','bg-orange-50'),
        'XGBoost':('text-emerald-600','bg-emerald-50'),'LightGBM':('text-cyan-600','bg-cyan-50'),
        'MLP Regressor':('text-pink-600','bg-pink-50')}
    ic,ib = icon_map.get(algorithm,('text-indigo-600','bg-indigo-50'))
    conn = get_db()
    conn.execute('INSERT INTO models (user_id,name,algorithm,model_type,r2_score,mae,rmse,trained_on,status,last_trained,icon_color,icon_bg,is_global) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0)',
        (request.user_id,name,algorithm,'Regression','—','—','—',data.get('dataset','Student Academic Performance'),'Training',now_str(),ic,ib))
    conn.commit()
    model = conn.execute('SELECT * FROM models WHERE user_id=? ORDER BY id DESC LIMIT 1',(request.user_id,)).fetchone()
    conn.close()
    return jsonify(dict(model)),201

@app.route('/api/models/<int:mid>', methods=['PUT'])
@token_required
def update_model(mid):
    data = request.get_json()
    conn = get_db()
    conn.execute('''UPDATE models SET name=COALESCE(?,name),status=COALESCE(?,status),
        r2_score=COALESCE(?,r2_score),mae=COALESCE(?,mae),rmse=COALESCE(?,rmse)
        WHERE id=? AND (user_id=? OR is_global=1)''',
        (data.get('name'),data.get('status'),data.get('r2_score'),data.get('mae'),data.get('rmse'),mid,request.user_id))
    conn.commit()
    model = conn.execute('SELECT * FROM models WHERE id=?',(mid,)).fetchone()
    conn.close()
    return jsonify(dict(model))

@app.route('/api/models/<int:mid>', methods=['DELETE'])
@token_required
def delete_model(mid):
    conn = get_db()
    conn.execute('DELETE FROM models WHERE id=? AND user_id=? AND is_global=0',(mid,request.user_id))
    conn.commit()
    conn.close()
    return jsonify({'message':'Model deleted'})

# DATASETS — shared/global
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
    name = data.get('name','').strip()
    if not name: return jsonify({'error':'Name is required'}),400
    conn = get_db()
    conn.execute('INSERT INTO datasets (name,description,source,records,features,last_updated,quality_score,status) VALUES (?,?,?,?,?,?,?,?)',
        (name,data.get('description',''),data.get('source','School System'),data.get('records','—'),data.get('features','—'),now_str(),data.get('quality_score',0),data.get('status','Active')))
    conn.commit()
    dataset = conn.execute('SELECT * FROM datasets ORDER BY id DESC LIMIT 1').fetchone()
    conn.close()
    return jsonify(dict(dataset)),201

@app.route('/api/datasets/<int:did>', methods=['PUT'])
@token_required
def update_dataset(did):
    data = request.get_json()
    conn = get_db()
    conn.execute('UPDATE datasets SET name=COALESCE(?,name),description=COALESCE(?,description),source=COALESCE(?,source),status=COALESCE(?,status) WHERE id=?',
        (data.get('name'),data.get('description'),data.get('source'),data.get('status'),did))
    conn.commit()
    dataset = conn.execute('SELECT * FROM datasets WHERE id=?',(did,)).fetchone()
    conn.close()
    return jsonify(dict(dataset))

@app.route('/api/datasets/<int:did>', methods=['DELETE'])
@token_required
def delete_dataset(did):
    conn = get_db()
    conn.execute('DELETE FROM datasets WHERE id=?',(did,))
    conn.commit()
    conn.close()
    return jsonify({'message':'Dataset deleted'})

# ANALYTICS — per user
@app.route('/api/analytics', methods=['GET'])
@token_required
def get_analytics():
    conn = get_db()
    uid = request.user_id
    total_students = conn.execute('SELECT COUNT(*) FROM students WHERE user_id=?',(uid,)).fetchone()[0]
    high_performers = conn.execute("SELECT COUNT(*) FROM students WHERE user_id=? AND performance='High Performer'",(uid,)).fetchone()[0]
    at_risk = conn.execute("SELECT COUNT(*) FROM students WHERE user_id=? AND risk_level='High Risk'",(uid,)).fetchone()[0]
    moderate_risk = conn.execute("SELECT COUNT(*) FROM students WHERE user_id=? AND risk_level='Moderate Risk'",(uid,)).fetchone()[0]
    low_risk = conn.execute("SELECT COUNT(*) FROM students WHERE user_id=? AND risk_level='Low Risk'",(uid,)).fetchone()[0]
    predictions = conn.execute('SELECT confidence FROM predictions WHERE user_id=?',(uid,)).fetchall()
    avg_score = None
    if predictions:
        c = [float(p['confidence'].replace('%','')) for p in predictions if p['confidence']]
        if c: avg_score = round(sum(c)/len(c),1)
    total_predictions = conn.execute('SELECT COUNT(*) FROM predictions WHERE user_id=?',(uid,)).fetchone()[0]
    total_reports = conn.execute('SELECT COUNT(*) FROM reports WHERE user_id=?',(uid,)).fetchone()[0]
    conn.close()
    return jsonify({'total_students':total_students,'high_performers':high_performers,'at_risk':at_risk,
        'moderate_risk':moderate_risk,'low_risk':low_risk,'overall_accuracy':avg_score,
        'total_predictions':total_predictions,'total_reports':total_reports})

if __name__ == '__main__':
    init_db()
    print('EduPredict backend running on http://localhost:5000')
    print('Default login: admin@edupredict.app / admin123')
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
