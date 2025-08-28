import joblib
from flask import Flask, render_template, jsonify, request
from konlpy.tag import Okt

app=Flask(__name__)
            # 파일명

okt = Okt()

# 한국어 토크나이저가 없어서 java에서 가져와사용
def tw_tokenizer(text):
    tokens_ko = okt.morphs(text)
    return tokens_ko

try:
    model = joblib.load("model/lr_v1.pkl")
    vec = joblib.load("model/lr_cnt_v1.pkl")
except Exception as e:
    print("모델 로드 중 오류 발생: {str(e)}")
    raise

@app.route("/")
            # 루트로 접속하면 함수 실행
def hello_world():
    # templates 폴더를 만들고 , index.html 작성
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "텍스트가 흠.."})
    text = data["text"]
    if not text.strip():
        return jsonify({"error": "이건...텍스트가 흠.."})
    text_tfidf = vec.transform([text])
    predict = model.predict(text_tfidf)[0]
    return jsonify({"emotion" : str(predict)})

if __name__ == "__main__" :
    app.run(debug=True, host="0.0.0.0", port=8000)