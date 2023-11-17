from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import cv2
import tensorflow as tf

app = Flask(__name__)

model = tf.keras.models.load_model('cyenet_model.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    description = ""
    solution = ""
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (224, 224))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            prediction = model.predict(img)
            label = np.argmax(prediction)
            print(prediction)

            if label == 0:
                description = "هذا النوع يظهر الجلد الداخلي للرحم بوضوح. يتميز بلونه الأحمر."
                solution = "الحل المقترح لهذا النوع هو مراجعة الطبيب المختص للحصول على المشورة الطبية المناسبة."
            elif label == 1:
                description = "هذا النوع هو مزيج بين النوعين 1 و 3. بينما يكون الجلد الأحمر مرئيًا، يكون بعض الجلد الوردي موجودًا أيضًا في الداخل."
                solution = "يُفضل الرجوع إلى الطبيب للحصول على تقييم دقيق وتوجيهات محددة حول الخطوات التالية."
            elif label == 2:
                description = "في هذا النوع، يقع معظم النسيج داخل الرحم. ونتيجة لذلك، يكون الجلد الوردي مرئيًا فقط من الخارج."
                solution = "يُنصح بإجراء فحوصات دورية والتشاور مع الطبيب المختص لضمان الصحة الجيدة."

            return render_template('index.html', label=label, prediction=prediction, description=description, solution=solution)
    return render_template('index.html', label=None)

if __name__ == '__main__':
    app.run(debug=True, port=8001)
