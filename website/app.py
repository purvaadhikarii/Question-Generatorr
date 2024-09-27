import os
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
import pytesseract
from transformers import AutoModelWithLMHead, AutoTokenizer, BertTokenizer, BertForQuestionAnswering
from torch import tensor, argmax

app = Flask(__name__,static_url_path='/static')

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

tokenizer_q = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
model_q = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")

model_a = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer_a = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            text = extract_text_from_image(file_path)
            if text is None:
                return "Error extracting text from image", 500

            questions = generate_questions(text)

            return render_template('result.html', text=text, questions=questions)
    except Exception as e:
        app.logger.error(f"Error processing image: {e}")
        return "Error processing image", 500

def extract_text_from_image(image_path):
    try:
        with Image.open(image_path) as img:
            text = pytesseract.image_to_string(img)
        return text.strip() 
    except Exception as e:
        app.logger.error(f"Error extracting text from image: {e}")
        return None

def generate_questions(input_text, max_length=64):
    try:
        qns = []
        sentences = input_text.split('.')
        for sentence in sentences[:-1]:
            input_text = "answer: %s  context: %s </s>" % ('', sentence)
            features = tokenizer_q([input_text], return_tensors='pt')

            output = model_q.generate(input_ids=features['input_ids'], 
                       attention_mask=features['attention_mask'],
                       max_length=max_length)
            qns.append(tokenizer_q.decode(output[0]).replace('<pad> question: ','').replace('</s>',''))
        return qns
    except Exception as e:
        app.logger.error(f"Error generating questions: {e}")
        return []

@app.route('/answer', methods=['POST'])
def answer_question_route():
    try:
        question = request.form['question']
        context = request.form['context']
        answer = answer_question(question, context)
        return render_template('answer.html', question=question, context=context, answer=answer)
    except Exception as e:
        app.logger.error(f"Error answering question: {e}")
        return "Error answering question", 500

def answer_question(question, context):
    try:
        input_ids = tokenizer_a.encode(question, context)
        sep_index = input_ids.index(tokenizer_a.sep_token_id)
        num_seg_a = sep_index + 1
        num_seg_b = len(input_ids) - num_seg_a
        segment_ids = [0]*num_seg_a + [1]*num_seg_b
        assert len(segment_ids) == len(input_ids)
        outputs = model_a(tensor([input_ids]),
                        token_type_ids=tensor([segment_ids]),
                        return_dict=True) 
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        answer_start = argmax(start_scores)
        answer_end = argmax(end_scores)
        tokens = tokenizer_a.convert_ids_to_tokens(input_ids)
        answer = tokens[answer_start]
        for i in range(answer_start + 1, answer_end + 1):
            if tokens[i][0:2] == '##':
                answer += tokens[i][2:]
            else:
                answer += ' ' + tokens[i]
        return answer
    except Exception as e:
        app.logger.error(f"Error answering question: {e}")
        return "Sorry, I couldn't find an answer."

if __name__ == '__main__':
    app.run(debug=True)
