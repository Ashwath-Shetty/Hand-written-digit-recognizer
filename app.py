import tensorflow as tf
import gradio as gr
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0, 
x_test = x_test / 255.0
# print("---------------------------",tf.__version__,"-------------------------------------------")
# print("--------------------------------------",gr.__version__)
#print("train",x_train[0].shape)
mo=tf.keras.models.load_model('emnist_model.h5')
# print(mo.predict(x_train[0][0]))
# print(mo.predict(x_train[0][0]).shape)
# print(mo.predict(x_train[0][0]).tolist()[0])
# l=[0,1,2,3,4,5,6,7,8,9]
l=[]
k={0 :48,
1:49,
2:50,
3:51,
4:52,
5:53,
6 :54,
7 :55,
8 :56,
9 :57,
10 :65,
11 :66,
12 :67,
13 :68,
14 :69,
15 :70,
16 :71,
17 :72,
18 :73,
19 :74,
20 :75,
21 :76,
22 :77,
23 :78,
24 :79,
25 :80,
26 :81,
27 :82,
28 :83,
29 :84,
30 :85,
31 :86,
32 :87,
33 :88,
34 :89,
35 :90,
36 :97,
37 :98,
38 :100,
39 :101,
40 :102,
41 :103,
42: 104,
43 :110,
44 :113,
45 :114,
46 :116,
}
for one in range(48,117):
    l.append(chr(one))
#setuptools==41.0.0
def classify(image):
    image=image.reshape(-1,28,28,1)
    image=image/255.0
    prediction = mo.predict(image).tolist()[0]
    print("-----------------------e",mo.predict(image))
    print("------------",prediction)
    print("you")
    #return {str(i): prediction[i] for i in range(26)}
    return {chr(k[i]): float(prediction[i]) for i in range(0,47)}
    
# #sketchpad = gr.inputs.Sketchpad()
label = gr.outputs.Label(num_top_classes=26,label=l)
# #interface = gr.Interface(classify, "sketchpad", "label", live=True, capture_session=True)
gr.Interface(fn=classify, inputs="sketchpad", outputs=label,live=True, capture_session=True).launch(share=True,debug=True)
# #interface.launch(share=True)
#gr.Interface(fn=recognize_digit, inputs="sketchpad", outputs="label").launch()
