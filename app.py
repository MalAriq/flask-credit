from flask import Flask, request, render_template, redirect, url_for, send_file, flash
from joblib import load
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
import os
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_principal import Principal, Permission, RoleNeed, Identity, identity_loaded, identity_changed, AnonymousIdentity


app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Initialize Flask-Login
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Konfigurasi untuk penyimpanan file
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Load the model, scaler, and features
model = load('model/ngboost_model.joblib')  # Ganti dengan nama file model Anda
scaler = load('model/scaler.joblib')  # Ganti dengan nama file scaler Anda
features = load('model/features.pkl')  # Ganti dengan nama file fitur Anda

# Initialize Flask-Principal
principals = Principal(app)

# Define roles
admin_permission = Permission(RoleNeed('admin'))
user_permission = Permission(RoleNeed('user'))

# In-memory user store (for demonstration purposes)
users = {
    'admin': {'password': 'adminpass', 'role': 'admin'},
    'user': {'password': 'userpass', 'role': 'user'}
}

class User(UserMixin):
    def __init__(self, username, role):
        self.id = username
        self.role = role

@login_manager.user_loader
def load_user(user_id):
    user = users.get(user_id)
    if user:
        return User(user_id, user['role'])
    return None

@identity_loaded.connect_via(app)
def on_identity_loaded(sender, identity):
    identity.user = current_user
    if hasattr(current_user, 'role'):
        identity.provides.add(RoleNeed(current_user.role))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users.get(username)
        if user and user['password'] == password:
            user_obj = User(username, user['role'])
            login_user(user_obj)
            identity_changed.send(app, identity=Identity(user_obj.id))
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    identity_changed.send(app, identity=AnonymousIdentity())
    return redirect(url_for('login'))

@app.route('/')
def landing():
    return render_template('landing_page.html')

@app.route('/home')
@login_required
def index():
    return render_template('index1.html')

@app.route('/about')
@login_required
def about():
    return render_template('about.html')

@app.route('/form')
@login_required
@user_permission.require(http_exception=403)
def form():
    return render_template('form1.html')

@app.route('/formcsv')
@login_required
@user_permission.require(http_exception=403)
def home():
    return render_template('index.html')

def preprocess_data(df):
    def text_to_months(text):
        if pd.isna(text):
            return 0
        words = text.split()
        years = int(words[0])
        months = int(words[3]) if len(words) > 3 else 0
        return years * 12 + months
    
    df['Credit_History_Age'] = df['Credit_History_Age'].apply(text_to_months)

    def replace_special_character(text):
        if "NM" in str(text):
            return "No"
        if "payments" in str(text) or "_" not in str(text):
            return text
        clean_text = str(text).replace("_", "")
        try:
            clean_text = pd.to_numeric(clean_text)
        except ValueError as e:
            clean_text = np.nan
        return np.nan if clean_text == "nan" or clean_text == "" else clean_text
    
    df['Age'] = pd.to_numeric(df['Age'].apply(replace_special_character))
    df['Annual_Income'] = pd.to_numeric(df['Annual_Income'].apply(replace_special_character))
    df['Changed_Credit_Limit'] = pd.to_numeric(df['Changed_Credit_Limit'].apply(replace_special_character))
    df['Outstanding_Debt'] = pd.to_numeric(df['Outstanding_Debt'].apply(replace_special_character))
    df['Num_of_Delayed_Payment'] = pd.to_numeric(df['Num_of_Delayed_Payment'].apply(replace_special_character))
    df['Amount_invested_monthly'] = pd.to_numeric(df['Amount_invested_monthly'].apply(replace_special_character))
    df['Monthly_Balance'] = pd.to_numeric(df['Monthly_Balance'].apply(replace_special_character))
    df['Num_of_Loan'] = pd.to_numeric(df['Num_of_Loan'].apply(replace_special_character))

    def remove_outliers(x,f):
        try:
            x = pd.to_numeric(x, errors='raise')
            if x < 0:
                return np.nan
            else:
                if f == 'age':
                    if x>=100:
                        return np.nan
                    else:
                        return x
                else:
                    return x
        except ValueError:
            return np.nan

    df['Age'] = df.apply(lambda x: remove_outliers(x['Age'], 'age'), axis=1)
    df['Num_of_Delayed_Payment'] = df.apply(lambda x: remove_outliers(x['Num_of_Delayed_Payment'], 'none'), axis=1)
    df['Num_of_Loan'] = df.apply(lambda x: remove_outliers(x['Num_of_Loan'], 'none'), axis=1)

    df['Annual_Income'].fillna(df['Annual_Income'].median())
    df['Changed_Credit_Limit'].fillna(df['Changed_Credit_Limit'].median())
    df['Outstanding_Debt'].fillna(df['Outstanding_Debt'].median())
    df['Monthly_Inhand_Salary'].fillna(df['Monthly_Inhand_Salary'].median())
    df['Num_of_Delayed_Payment'].fillna(df['Num_of_Delayed_Payment'].median())
    df['Num_Credit_Inquiries'].fillna(df['Num_Credit_Inquiries'].median())
    df['Amount_invested_monthly'].fillna(df['Amount_invested_monthly'].median())
    df['Monthly_Balance'].fillna(df['Monthly_Balance'].median())
    df = df.dropna(subset=['Num_of_Loan'])
    
    df = df.dropna(subset=['Type_of_Loan'])

    Type_of_Loan = df['Type_of_Loan'].str.split(',\s*and\s*|\s*,\s*')
    Type_of_Loan = Type_of_Loan.explode()
    loan_type_label = Type_of_Loan.unique()

    def replace_and(text):
        clean_text = str(text).replace(" and", "")
        return np.nan if clean_text == "nan" else clean_text

    df['Type_of_Loan'] = df['Type_of_Loan'].apply(replace_and)
    for loan_type in loan_type_label:
        df['Count_' + loan_type] = df['Type_of_Loan'].apply(lambda x: x.split(', ').count(loan_type))

    df.drop(['Num_of_Loan'], axis=1, inplace=True, errors="ignore")
    df.drop(["Type_of_Loan"], axis=1, inplace=True, errors="ignore")

    df.dropna(inplace=True)
    df["Age"] = pd.to_numeric(df["Age"])
    df["Annual_Income"] = pd.to_numeric(df["Annual_Income"])
    df["Num_of_Delayed_Payment"] = pd.to_numeric(df["Num_of_Delayed_Payment"])
    df["Changed_Credit_Limit"] = pd.to_numeric(df["Changed_Credit_Limit"])
    df["Outstanding_Debt"] = pd.to_numeric(df["Outstanding_Debt"])
    df["Amount_invested_monthly"] = pd.to_numeric(df["Amount_invested_monthly"])
    df["Monthly_Balance"] = pd.to_numeric(df["Monthly_Balance"])

    df.drop(df[df['Occupation'] == '_______'].index, inplace = True)
    df.drop(df[df['Payment_of_Min_Amount'] == 'NM'].index, inplace = True)

    df.drop(df[df['Credit_Mix'] == '_'].index, inplace = True)

    df = pd.get_dummies(df, columns=['Occupation', 'Credit_Mix','Payment_of_Min_Amount'], dtype=int)

    df['Spent Amount Payment_Behaviour'] = df['Payment_Behaviour'].str.extract(r'(\w+)_spent')
    df['Value Amount Payment_Behaviour'] = df['Payment_Behaviour'].str.extract(r'_spent_(\w+)_value')

    df.drop(["Payment_Behaviour"], axis=1, inplace=True, errors="ignore")

    spent_mapping = {'Low': 0, 'High': 1}
    value_mapping = {'Small': 0, 'Medium': 1, 'Large': 2}
    month_mapping = {'January': 1,'February': 2,'March': 3,'April': 4,'May': 5,'June': 6,'July': 7,'August': 8,'September': 9,'October': 10,'November': 11,'December': 12}
    df['Spent Amount Payment_Behaviour'] = df['Spent Amount Payment_Behaviour'].map(spent_mapping)
    df['Value Amount Payment_Behaviour'] = df['Value Amount Payment_Behaviour'].map(value_mapping)
    df['Month'] = df['Month'].map(month_mapping)

    df.fillna(0, inplace=True)

    return df

def categorize_credit_score(score):
    if score > 800:
        return 'Good'
    elif score >= 600:
        return 'Standard'
    else:
        return 'Poor'

@app.route('/predict', methods=['POST'])
@login_required
@user_permission.require(http_exception=403)
def predict():
    if 'file' in request.files and request.files['file'].filename != '':
        # Jika input adalah file CSV
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Baca CSV dan proses
            input_df = pd.read_csv(filepath)
            
            # Proses data
            input_df = preprocess_data(input_df)
            
            # Pastikan semua fitur ada dalam DataFrame
            for col in features:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            # Urutkan kolom sesuai fitur
            input_df = input_df[features]
            
            # Pastikan tidak ada NaN di DataFrame sebelum transformasi
            input_df.fillna(0, inplace=True)
            
            # Normalisasi fitur
            input_scaled = scaler.transform(input_df)
            
            # Prediksi probabilitas
            predicted_probs = model.predict_proba(input_scaled)
            
            # Hitung skor kredit
            scores = [int(600 + 80 * (np.log2(prob[1] / (1 - prob[1])))) for prob in predicted_probs]
            
            # Tambahkan skor dan kategori ke DataFrame
            input_df['Credit_Score_Prediction'] = scores
            input_df['Credit_Score_Category'] = input_df['Credit_Score_Prediction'].apply(categorize_credit_score)
            
            # Simpan hasil prediksi ke file baru
            output_filename = 'predictions.csv'
            output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            input_df.to_csv(output_filepath, index=False)
            
            return redirect(url_for('download_file', filename=output_filename))

@app.route('/uploads/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

@app.route('/result', methods=['POST'])
@login_required
@user_permission.require(http_exception=403)
def result():
    if request.method == 'POST':
        # Extract form data
        form_data = request.form
        
        # Prepare input data
        input_data = []
        for feature in features:
            value = form_data.get(feature)
            if value is None or value == '':
                value = 0  # or handle missing values appropriately
            input_data.append(float(value))
        
        # Scale input data
        scaled_data = scaler.transform([input_data])
        
        # Make prediction
        prediction_proba = model.predict_proba(scaled_data)[0, 1]
        prediction_label = "Good/Standard" if prediction_proba > 0.5 else "Poor"
        
        return render_template('result.html', prediction=prediction_label, probability=prediction_proba)



@app.errorhandler(403)
def forbidden(e):
    return render_template('403.html'), 403

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
