from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config["SECRET_KEY"] = "your-secret-key"
socketio = SocketIO(app)


@app.route("/")
def home():
    return render_template("index.html")


@socketio.on("send_message")
def handle_message(data):
    message = data["message"]

    # Initial processing
    processed_output = process_input(message)
    emit(
        "new_message",
        {"message": message, "processed_output": processed_output},
        broadcast=True,
    )

    # Delayed processing
    socketio.sleep(2)  # Wait for 2 seconds
    additional_output = process_input_additional(message)
    emit("additional_output", {"additional_output": additional_output}, broadcast=True)


def process_input(input_text):
    # Replace this with your actual processing logic
    return f"**Bold text**: {input_text}\n\n*Italic text*: {input_text}\n\n- List item 1\n- List item 2"


def process_input_additional(input_text):
    # Replace this with your additional processing logic
    return f"## Additional processing result:\n\n1. First item: {input_text.upper()}\n2. Second item: {input_text.lower()}"


if __name__ == "__main__":
    socketio.run(app, debug=True)
