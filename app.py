# -*- coding: utf-8 -*-
from flask import Flask, request, render_template, send_file, session, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from pdfHandle import PDF_Class
import re, json, time
import os, requests
import tempfile
import threading
from post import post_processing
from flask_socketio import SocketIO, emit
from flask_session import Session

async_mode = None
app = Flask(__name__, template_folder='templates')
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
socketio = SocketIO(app, async_mode=async_mode)
app.config['UPLOAD_FOLDER'] = './PDF'
result_folder = "results"
ALLOWED_EXTENSIONS = {'pdf','zip'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def delete_file(file_path, delay):
    # Function to delete the file after a specified delay
    time.sleep(delay)
    if os.path.exists(file_path):
        os.remove(file_path)
@app.route('/set_sess', methods=['POST'])  # Specify the HTTP method as POST
def setSession():
    username = request.json.get('username')  # Access the username from the JSON request
    session['username'] = username  # Set the session variable
    return "okay"        
@app.route('/download/<path:filename>', methods=['GET'])
def download(filename):
    # file_path = os.path.join(result_folder, filename)
    if not os.path.exists(filename):
        return "Already downloaded or not existed"
    # Send the file as an attachment
    response = send_file(filename, as_attachment=True)
    # Delete the file after sending
    # Delete the file after a delay of 5 seconds (adjust as needed)
    deletion_thread = threading.Thread(target=delete_file, args=(filename, 5))
    deletion_thread.start()
    return response

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_pdf(file_path):
    PDF = PDF_Class(file_path)
    return PDF.parse_doc(socketio, session.get('username'))
def getTime():
    response = requests.get("http://worldtimeapi.org/api/timezone/Etc/UTC")
    data = response.json()
    year = int(data["datetime"][:4])
    month = int(data["datetime"][5:7])
    day = int(data["datetime"][8:10])
    current_time = data["datetime"][11:19]  # HH:MM:SS    
    return year, month, day, current_time
@app.route('/upload', methods=['POST'])
def upload():
    year, month, day, current_time = getTime()
    if year > 2025 or (year==2025 and month>3) or (year ==2025 and month==3 and day>5):
        return {"status":400, "result": "Error: Request to Developer"}
    if 'file' not in request.files:
        return {"status":400, "result": 'No file uploaded'}

    files = request.files.getlist('file')  # Get multiple files
    
    if not files or all(f.filename == '' for f in files):
        return {"status":400,"result": 'Empty file name'}
    if len(files)>200:
        return {"status":400, "result": "Reduce file count. Max size : 200 PDF"}
    ALL_RESULTS, xlsx_file_paths = [], []
    zipPath = os.path.join(result_folder, str(int(time.time() * 100000)) + "_results.zip")

    try:
        for file in files:
            if not allowed_file(file.filename):
                return {"status":400, "result": f'{file.filename}的文件扩展名无效'}        
            filename = secure_filename(file.filename)
            filename = re.sub("PDF", "pdf", filename)
            user_file = str(int(time.time() * 100000))
            file_path = os.path.join("upload", '_'.join([user_file, filename]))
            file.save(file_path)
            # Process the PDF file
            RESULT = process_pdf(file_path)
            ALL_RESULTS.append(RESULT)
            os.remove(file_path)  # Remove original file after processing
            # Generate Excel file
            xlsx_filename = os.path.splitext(filename)[0] + '.xlsx'
            xlsx_file_path = os.path.join(result_folder, xlsx_filename)
            xlsx_file_paths.append(xlsx_file_path)

        post_processing(ALL_RESULTS, xlsx_file_paths, zipPath)
        try:
            for xlsx_file_path in xlsx_file_paths:
                os.remove(xlsx_file_path)
        except:
            pass
    except Exception as e:
        socketio.emit('process', {'data': f"{file.filename} 处理过程中遇到错误: {str(e)}", 'username': session.get('username')})
        return {"status":400, "result":"None"}

    return {"status":200, "result":f"download/{zipPath}"}
@app.route('/success')
def success():
    download_url = request.args.get('download_url', '#')  # Default to '#' if not provided
    return render_template('success.html', download_url=download_url)
@app.route('/')
@app.route('/home')
def index():
    return render_template('index.html')
@app.route('/stop', methods=['GET'])
def stop():
    running = False
    return jsonify({'message': 'Stopping server...'})
@socketio.event
def my_event(message):
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': "已连接 !", 'count': session['receive_count']})
@socketio.event
def connect():
    emit('my_response', {'data': '正在连接...', 'count': 0})
@socketio.event
def disconnect():
    emit('my_response', {'data': '断开连接', 'count': 0})
if __name__ == '__main__':
    # from waitress import serve
    # serve(app, host="0.0.0.0", port=5000)
    PORT = int(os.getenv("PORT"))
    app.run("0.0.0.0", port=PORT)
