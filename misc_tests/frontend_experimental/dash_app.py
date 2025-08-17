import dash
from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate

app = dash.Dash(__name__)

# Load the HTML file
with open("dash_index.html") as f:
    app.index_string = f.read()

app.layout = html.Div(id="page-content")


@app.callback(
    Output("chat-messages", "children"),
    Output("chat-input-field", "value"),
    Input("send-button", "n_clicks"),
    State("chat-input-field", "value"),
    State("chat-messages", "children"),
)
def update_chat(n_clicks, input_value, chat_messages):
    if n_clicks is None or not input_value:
        raise PreventUpdate

    user_message = html.Div(f"You: {input_value}", className="message user-message")
    bot_response = html.Div(
        f"Bot: You said: {input_value}", className="message bot-message"
    )

    if chat_messages is None:
        chat_messages = []
    chat_messages.extend([user_message, bot_response])

    return chat_messages, ""


if __name__ == "__main__":
    app.run_server(debug=True)
