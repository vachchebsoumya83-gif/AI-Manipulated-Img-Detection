import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# ==========================
# Load Trained Model
# ==========================

try:
    model = load_model("model/best_model.keras")
    print("Best model loaded")
except:
    model = load_model("model/final_model.keras")
    print("Final model loaded")


selected_image_path = ""


# ==========================
# Browse Image
# ==========================

def browse_image():

    global selected_image_path

    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")
        ]
    )

    if file_path:

        selected_image_path = file_path

        img = Image.open(file_path)
        img = img.resize((300, 300))

        photo = ImageTk.PhotoImage(img)

        image_label.config(image=photo)
        image_label.image = photo

        result_label.config(text="")
        confidence_label.config(text="")


# ==========================
# Predict Image
# ==========================

def predict_image():

    if selected_image_path == "":
        messagebox.showwarning(
            "Warning",
            "Please select an image first"
        )
        return


    img = image.load_img(
        selected_image_path,
        target_size=(224,224)
    )


    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = img_array / 255.0


    prediction = model.predict(img_array)


    # ==========================
    # Handle Model Output
    # ==========================

    if prediction.shape[1] == 1:

        # Sigmoid model

        score = prediction[0][0]


        if score >= 0.5:

            result = "REAL IMAGE"
            confidence = score * 100
            color = "green"

        else:

            result = "AI GENERATED IMAGE"
            confidence = (1-score) * 100
            color = "red"


    else:

        # Softmax model

        fake_probability = prediction[0][0]
        real_probability = prediction[0][1]


        if real_probability > fake_probability:

            result = "REAL IMAGE"
            confidence = real_probability * 100
            color = "green"

        else:

            result = "AI GENERATED IMAGE"
            confidence = fake_probability * 100
            color = "red"



    result_label.config(
        text="Prediction : " + result,
        fg=color
    )


    confidence_label.config(
        text="Confidence : {:.2f}%".format(confidence)
    )



# ==========================
# Main Window
# ==========================

root = tk.Tk()

root.title(
    "AI Image Detection Using Deep Learning"
)

root.geometry(
    "700x700"
)

root.configure(
    bg="white"
)



# ==========================
# Heading
# ==========================

title = tk.Label(
    root,
    text="AI IMAGE DETECTION USING DEEP LEARNING",
    font=("Arial",18,"bold"),
    bg="white",
    fg="blue"
)

title.pack(
    pady=20
)



# ==========================
# Browse Button
# ==========================

browse_btn = tk.Button(
    root,
    text="Browse Image",
    command=browse_image,
    bg="green",
    fg="white",
    font=("Arial",14),
    width=20
)

browse_btn.pack(
    pady=10
)



# ==========================
# Image Display
# ==========================

image_label = tk.Label(
    root,
    bg="white"
)

image_label.pack(
    pady=20
)



# ==========================
# Predict Button
# ==========================

predict_btn = tk.Button(
    root,
    text="Predict",
    command=predict_image,
    bg="blue",
    fg="white",
    font=("Arial",14),
    width=20
)

predict_btn.pack(
    pady=10
)



# ==========================
# Result Display
# ==========================

result_label = tk.Label(
    root,
    text="",
    font=("Arial",16,"bold"),
    bg="white"
)

result_label.pack(
    pady=15
)



confidence_label = tk.Label(
    root,
    text="",
    font=("Arial",15),
    bg="white"
)

confidence_label.pack()



# ==========================
# Run Application
# ==========================

root.mainloop()