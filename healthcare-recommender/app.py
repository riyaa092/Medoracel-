from flask import Flask, render_template, request, redirect, url_for, session, flash 
import sqlite3
import pickle

# ✅ Import the helper
from extract_symptoms import get_symptom_details  # 🔥 Used in /predict

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure secret key in production

# ----------- Load ML model and encoders -----------
model = pickle.load(open("model.pkl", "rb"))
all_symptoms = pickle.load(open("symptom_encoder.pkl", "rb"))
disease_encoder = pickle.load(open("disease_encoder.pkl", "rb"))
symptom_index = {symptom: i for i, symptom in enumerate(all_symptoms)}

# ----------- Initialize SQLite database -----------
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ----------- ROUTES -----------

@app.route('/')
@app.route('/login', methods=['GET'])
def login_form():
    if 'username' in session:
        return redirect(url_for('home'))
    return render_template("login.html")

@app.route('/login', methods=['POST'])
def login_user():
    username = request.form['username']
    password = request.form['password']

    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = c.fetchone()
    conn.close()

    if user:
        session['username'] = username
        flash("Login successful!", "success")
        return redirect(url_for('home'))
    else:
        flash("Invalid username or password. Please try again.", "danger")
        return redirect(url_for('login_form'))

@app.route('/signup', methods=['GET', 'POST'])
def signup_form():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            flash("Signup successful! Please login.", "success")
            return redirect(url_for('login_form'))
        except sqlite3.IntegrityError:
            flash("Username already exists. Try a different one.", "warning")
        conn.close()
    return render_template("signup.html")

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("You have been logged out.", "info")
    return redirect(url_for('login_form'))

@app.route('/home')
def home():
    if 'username' not in session:
        flash("Please login to continue.", "warning")
        return redirect(url_for('login_form'))

    # Prepare dropdown options
    formatted_symptoms = [(s, s.replace('_', ' ').capitalize()) for s in all_symptoms]
    return render_template("index.html", symptoms=formatted_symptoms, username=session['username'])

@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        flash("Please login first.", "warning")
        return redirect(url_for('login_form'))

    # Get selected symptoms
    selected_symptoms = [request.form.get(f"Symptom_{i}") for i in range(1, 18)]
    selected_symptoms = [s for s in selected_symptoms if s and s.strip() != ""]

    if not selected_symptoms:
        flash("Please select at least one symptom.", "danger")
        return redirect(url_for('home'))

    # Create symptom vector
    input_vector = [0] * len(all_symptoms)
    for symptom in selected_symptoms:
        if symptom in symptom_index:
            input_vector[symptom_index[symptom]] = 1

    # Predict disease
    try:
        prediction_index = model.predict([input_vector])[0]
        prediction = disease_encoder.inverse_transform([prediction_index])[0]
    except Exception as e:
        print("Prediction error:", e)
        prediction = "Unable to predict. Please try again."

    # ✅ New: Extract explanation & severity info
    symptom_details = get_symptom_details(selected_symptoms)

    return render_template("result.html",
                           prediction=prediction,
                           symptoms=selected_symptoms,
                           symptom_details=symptom_details,
                           username=session['username'])
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)

