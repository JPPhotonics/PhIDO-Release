import json
from io import BytesIO

from flask import Flask, jsonify, request, send_file

from PhotonicsAI.Photon import llm_api, semantic_similarity, utils

app = Flask(__name__)


@app.route("/get_spec", methods=["POST"])
def get_spec():
    user_input = request.json["prompt"]
    response = llm_api.parse_specs(user_input)
    # print('== user_input ==>', user_input)
    # print('== response ==>', response)
    with open("./flask_logs/prompt_logs.txt", "a") as file:
        file.write(user_input + "\n")
        file.write(json.dumps(response) + "\n")
        file.write("############\n")

    return jsonify({"response": response})


@app.route("/search", methods=["POST"])
def search():
    user_input = request.json["prompt"]

    db_docs = utils.search_directory_for_docstrings()
    list_of_desc = [i["Description"] for i in db_docs]

    i, s = semantic_similarity.dragon(user_input, list_of_desc)
    ii, ss = semantic_similarity.bm25(user_input, list_of_desc)
    final_i, final_s = semantic_similarity.combine_scores([s, ss])

    sorted_names = [db_docs[i]["Name"] for i in final_i]

    r = [
        f"{b}             [{round(a, 3)}]"
        for a, b in zip(sorted(final_s)[::-1], sorted_names)
    ]

    response = "NAME   [SCORE]\n\n" + "\n\n".join(r)
    return jsonify({"response": response, "main_r": sorted_names[0]})


@app.route("/get_gds", methods=["POST"])
def get_gds():
    class_name = request.json["main_r"]

    img = utils.get_gds_from_name(class_name)

    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    return send_file(img_byte_arr, mimetype="image/png")


@app.route("/get_splot", methods=["POST"])
def get_splot():
    class_name = request.json["main_r"]

    img = utils.get_splot_from_name(class_name)

    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    return send_file(img_byte_arr, mimetype="image/png")


@app.route("/status", methods=["GET"])
def status():
    # You can include any logic here
    return jsonify({"status": "online!"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8899, debug=True, use_reloader=False)
