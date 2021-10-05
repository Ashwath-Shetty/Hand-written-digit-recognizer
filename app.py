import tensorflow as tf
import gradio as gr
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0, 
x_test = x_test / 255.0
print("---------------------------",tf.__version__,"-------------------------------------------")
mo=tf.keras.models.load_model('cnn_model.h5')
#print(mo.predict(x_train[0]).tolist()[0])
def classify(image):
    image=image.reshape(-1,28,28)
    image=image/255.0
    prediction = mo.predict(image).tolist()[0]
    print(prediction)
    return {str(i): prediction[i] for i in range(10)}
    
#sketchpad = gr.inputs.Sketchpad()
label = gr.outputs.Label(num_top_classes=3)
#interface = gr.Interface(classify, "sketchpad", "label", live=True, capture_session=True)
gr.Interface(fn=classify, inputs="sketchpad", outputs='label',live=True, capture_session=True).launch(share=True)
#interface.launch(share=True)
#gr.Interface(fn=recognize_digit, inputs="sketchpad", outputs="label").launch()
