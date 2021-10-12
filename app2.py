import tensorflow as tf
import gradio as gr
import numpy as np
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0, 
x_test = x_test / 255.0
# print("---------------------------",tf.__version__,"-------------------------------------------")
# print("--------------------------------------",gr.__version__)
#print("train",x_train[0].shape)
mo=tf.keras.models.load_model('mnist_1L_50e_32b.h5')
# print(mo.predict(x_train[0][0]))
# print(mo.predict(x_train[0][0]).shape)
#print(mo.predict(x_train[0][0]).tolist()[0])
# l=[0,1,2,3,4,5,6,7,8,9]

# for one in range(48,117):
#     l.append(chr(one))
l={'!': 0,
 '"': 1,
 '#': 2,
 '$': 3,
 '%': 4,
 '&': 5,
 "'": 6,
 '(': 7,
 ')': 8,
 '*': 9,
 '+': 10,
 ',': 11,
 '-': 12,
 '.': 13,
 '/': 14,
 '0': 15,
 '1': 16,
 '2': 17,
 '3': 18,
 '4': 19,
 '5': 20,
 '6': 21,
 '7': 22,
 '8': 23,
 '9': 24,
 ':': 25,
 ';': 26,
 '<': 27,
 '=': 28,
 '>': 29,
 '?': 30,
 '@': 31,
 'A': 32,
 'B': 33,
 'C': 34,
 'D': 35,
 'E': 36,
 'F': 37,
 'G': 38,
 'H': 39,
 'I': 40,
 'J': 41,
 'K': 42,
 'L': 43,
 'M': 44,
 'N': 45,
 'O': 46,
 'P': 47,
 'Q': 48,
 'R': 49,
 'S': 50,
 'T': 51,
 'U': 52,
 'V': 53,
 'W': 54,
 'X': 55,
 'Y': 56,
 'Z': 57,
 '[': 58,
 '\\': 59,
 ']': 60,
 '^': 61,
 '_': 62,
 '`': 63,
 'a': 64,
 'b': 65,
 'c': 66,
 'd': 67,
 'e': 68,
 'f': 69,
 'g': 70,
 'h': 71,
 'i': 72,
 'j': 73,
 'k': 74,
 'l': 75,
 'm': 76,
 'n': 77,
 'o': 78,
 'p': 79,
 'q': 80,
 'r': 81,
 's': 82,
 't': 83,
 'u': 84,
 'v': 85,
 'w': 86,
 'x': 87,
 'y': 88,
 'z': 89,
 '{': 90,
 '|': 91,
 '}': 92,
 '~': 93
 }
inv_map = {v: k for k, v in l.items()}
def classify(image):
    image=image.reshape(-1,28,28,1)
    #image = np.expand_dims(image, axis=0)
    image=image/255.0
    prediction = mo.predict(image).flatten()
    #key=np.argmax(prediction,axis = 1) 
    print("-----------------------e",mo.predict(image).shape)
    print("latest------------",prediction.shape)
    print("you-latest")
    #return {str(i): prediction[i] for i in range(26)}
    print({inv_map[i]: float(prediction[i]) for i in range(0,94)})
    return {inv_map[i]: float(prediction[i]) for i in range(0,94)}
    
# #sketchpad = gr.inputs.Sketchpad()
label = gr.outputs.Label(num_top_classes=3,label='Prediction')
# #interface = gr.Interface(classify, "sketchpad", "label", live=True, capture_session=True)
desc="Draw any single character on the canvas to predict. for e.x: a, 1, @ etc."
ar="Input can be - Any Alphabets(A-Z or a-z), Numbers(0-9), Any Symbols and Special Characters available on computer keyboard(for e.x: @, ], # etc.) "
gr.Interface(fn=classify, inputs="sketchpad", outputs=label,live=True, capture_session=True,title="Character Predictor",description=desc,article=ar).launch(share=True,debug=True)
#interface.launch(share=True)
#gr.Interface(fn=recognize_digit, inputs="sketchpad", outputs="label").launch()
