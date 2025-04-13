import tensorflow as tf
import numpy as np
from PIL import Image
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

class PredictView(APIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # تحميل النموذج المدرب
        self.model = tf.keras.models.load_model(r'trained_model.h5')

    def post(self, request):
        try:
            # الحصول على الصورة من الطلب
            file = request.FILES['file']
            image = Image.open(file)
            image = image.resize((128, 128))  # تغيير الحجم بما يتناسب مع نموذجك
            input_arr = np.array(image)
            input_arr = np.expand_dims(input_arr, axis=0)  # تحويل الصورة إلى batch

            # التنبؤ
            prediction = self.model.predict(input_arr)
            result_index = np.argmax(prediction)

            # فئات النموذج
            class_names = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 
                'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 
                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
                'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
                'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
                'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
            ]

            # إرجاع النتيجة كاستجابة JSON
            return Response({"prediction": class_names[result_index]})
        
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
