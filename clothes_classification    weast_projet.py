
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageOps, ImageTk
from tensorflow.keras.models import load_model


# =====================================================
# SETTINGS
# =====================================================

APP_TITLE = "ClothAI - Clothes Classification"

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "keras_model.h5"
LABELS_PATH = BASE_DIR / "model" / "labels.txt"

IMAGE_SIZE = (224, 224)
PREVIEW_SIZE = (400, 300)


# =====================================================
# COLORS
# =====================================================

BG_COLOR = "#F3F5F9"
SIDEBAR_COLOR = "#202B40"
SIDEBAR_HOVER = "#34445F"
ACCENT_COLOR = "#5271FF"
WHITE = "#FFFFFF"
TEXT_COLOR = "#202B40"
GRAY = "#8993A4"
GREEN = "#16A085"


# =====================================================
# LOAD LABELS
# =====================================================

def load_labels():
    labels = []

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        for line in file:
            label = line.strip()

            if not label:
                continue

            parts = label.split(" ", 1)

            if len(parts) == 2 and parts[0].isdigit():
                labels.append(parts[1])
            else:
                labels.append(label)

    return labels


# =====================================================
# LOAD MODEL
# =====================================================

try:
    model = load_model(MODEL_PATH, compile=False)
    class_names = load_labels()

except Exception as error:
    raise RuntimeError(
        f"Model could not be loaded:\n{error}"
    )


# =====================================================
# IMAGE PREPARATION
# =====================================================

def prepare_image(image_path):

    data = np.ndarray(
        shape=(1, 224, 224, 3),
        dtype=np.float32
    )

    image = Image.open(image_path).convert("RGB")

    image = ImageOps.fit(
        image,
        IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(image)

    normalized_image_array = (
        image_array.astype(np.float32) / 127.5
    ) - 1

    data[0] = normalized_image_array

    return data


# =====================================================
# PREDICTION
# =====================================================

def predict_clothes(image_path):

    data = prepare_image(image_path)

    prediction = model.predict(
        data,
        verbose=0
    )

    index = int(np.argmax(prediction[0]))

    if index >= len(class_names):
        raise ValueError("Model classes and labels do not match.")

    return {
        "clothes_class": class_names[index],
        "confidence": float(prediction[0][index])
    }


# =====================================================
# MAIN APPLICATION
# =====================================================

class ClothesApp:

    def __init__(self, root):

        self.root = root

        self.root.title(APP_TITLE)
        self.root.geometry("1050x680")
        self.root.minsize(850, 580)
        self.root.configure(bg=BG_COLOR)

        self.preview_image = None
        self.current_image = None

        self.create_sidebar()
        self.create_main_content()

    # =================================================
    # SIDEBAR / TASKBAR
    # =================================================

    def create_sidebar(self):

        self.sidebar = tk.Frame(
            self.root,
            bg=SIDEBAR_COLOR,
            width=220
        )

        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        logo = tk.Label(
            self.sidebar,
            text="✦ ClothAI",
            bg=SIDEBAR_COLOR,
            fg=WHITE,
            font=("Segoe UI", 22, "bold")
        )

        logo.pack(pady=(35, 45))

        self.create_menu_button("⌂   Dashboard")
        self.create_menu_button("▣   Classify", active=True)
        self.create_menu_button("◷   History")
        self.create_menu_button("⚙   Settings")

        tk.Frame(
            self.sidebar,
            bg=SIDEBAR_COLOR
        ).pack(expand=True)

        version = tk.Label(
            self.sidebar,
            text="ClothAI v1.0",
            bg=SIDEBAR_COLOR,
            fg="#9BA8BC",
            font=("Segoe UI", 9)
        )

        version.pack(pady=20)

    def create_menu_button(self, text, active=False):

        color = SIDEBAR_HOVER if active else SIDEBAR_COLOR

        button = tk.Button(
            self.sidebar,
            text=text,
            bg=color,
            fg=WHITE,
            activebackground=SIDEBAR_HOVER,
            activeforeground=WHITE,
            relief="flat",
            anchor="w",
            padx=25,
            font=("Segoe UI", 11),
            cursor="hand2",
            command=lambda: self.menu_action(text)
        )

        button.pack(fill="x", padx=12, pady=5, ipady=12)

    def menu_action(self, menu):

        if "Classify" in menu:
            return

        if "Dashboard" in menu:
            self.reset_result()
            return

        if "History" in menu:
            messagebox.showinfo(
                "History",
                "Prediction history is not implemented yet."
            )

        if "Settings" in menu:
            messagebox.showinfo(
                "Settings",
                "ClothAI Settings"
            )

    # =================================================
    # MAIN CONTENT
    # =================================================

    def create_main_content(self):

        self.main = tk.Frame(
            self.root,
            bg=BG_COLOR
        )

        self.main.pack(
            side="left",
            fill="both",
            expand=True,
            padx=35,
            pady=25
        )

        # Header

        header = tk.Frame(
            self.main,
            bg=BG_COLOR
        )

        header.pack(fill="x", pady=(0, 25))

        tk.Label(
            header,
            text="Clothes Classification",
            bg=BG_COLOR,
            fg=TEXT_COLOR,
            font=("Segoe UI", 23, "bold")
        ).pack(anchor="w")

        tk.Label(
            header,
            text="Upload an image and let AI recognize your clothes.",
            bg=BG_COLOR,
            fg=GRAY,
            font=("Segoe UI", 11)
        ).pack(anchor="w", pady=5)

        # Image card

        self.image_card = tk.Frame(
            self.main,
            bg=WHITE,
            highlightbackground="#E2E6EF",
            highlightthickness=1
        )

        self.image_card.pack(
            fill="both",
            expand=True
        )

        tk.Label(
            self.image_card,
            text="Upload Your Image",
            bg=WHITE,
            fg=TEXT_COLOR,
            font=("Segoe UI", 15, "bold")
        ).pack(anchor="w", padx=25, pady=(20, 10))

        self.image_label = tk.Label(
            self.image_card,
            text="📷\n\nChoose an image to start classification",
            bg="#F7F8FC",
            fg=GRAY,
            font=("Segoe UI", 12),
            relief="flat"
        )

        self.image_label.pack(
            fill="both",
            expand=True,
            padx=25,
            pady=10
        )

        # Upload button

        self.upload_button = tk.Button(
            self.image_card,
            text="＋  Choose Clothes Image",
            command=self.choose_image,
            bg=ACCENT_COLOR,
            fg=WHITE,
            activebackground="#3D5BE0",
            activeforeground=WHITE,
            font=("Segoe UI", 11, "bold"),
            relief="flat",
            cursor="hand2",
            padx=20,
            pady=12
        )

        self.upload_button.pack(pady=15)

        # Result card

        self.result_card = tk.Frame(
            self.main,
            bg=WHITE,
            highlightbackground="#E2E6EF",
            highlightthickness=1
        )

        self.result_card.pack(
            fill="x",
            pady=(20, 0)
        )

        tk.Label(
            self.result_card,
            text="AI Prediction Result",
            bg=WHITE,
            fg=TEXT_COLOR,
            font=("Segoe UI", 14, "bold")
        ).pack(anchor="w", padx=20, pady=(15, 8))

        self.result_label = tk.Label(
            self.result_card,
            text="No prediction yet",
            bg=WHITE,
            fg=GRAY,
            font=("Segoe UI", 16, "bold")
        )

        self.result_label.pack(anchor="w", padx=20)

        self.confidence_label = tk.Label(
            self.result_card,
            text="Confidence: -",
            bg=WHITE,
            fg=GRAY,
            font=("Segoe UI", 11)
        )

        self.confidence_label.pack(
            anchor="w",
            padx=20,
            pady=(5, 15)
        )

    # =================================================
    # CHOOSE IMAGE
    # =================================================

    def choose_image(self):

        image_path = filedialog.askopenfilename(
            title="Choose Clothes Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*")
            ]
        )

        if not image_path:
            return

        self.current_image = image_path

        self.show_preview(image_path)
        self.predict(image_path)

    # =================================================
    # SHOW IMAGE
    # =================================================

    def show_preview(self, image_path):

        image = Image.open(image_path).convert("RGB")

        image.thumbnail(PREVIEW_SIZE)

        self.preview_image = ImageTk.PhotoImage(image)

        self.image_label.configure(
            image=self.preview_image,
            text=""
        )

    # =================================================
    # PREDICT
    # =================================================

    def predict(self, image_path):

        self.upload_button.configure(
            state="disabled",
            text="Classifying..."
        )

        self.root.update_idletasks()

        try:

            result = predict_clothes(image_path)

            clothes = result["clothes_class"]
            confidence = result["confidence"]

            self.result_label.configure(
                text=f"✓  {clothes}",
                fg=GREEN
            )

            self.confidence_label.configure(
                text=f"Confidence: {confidence:.2%}",
                fg=TEXT_COLOR
            )

        except Exception as error:

            messagebox.showerror(
                "Prediction Error",
                str(error)
            )

        finally:

            self.upload_button.configure(
                state="normal",
                text="＋  Choose Clothes Image"
            )

    # =================================================
    # RESET
    # =================================================

    def reset_result(self):

        self.current_image = None
        self.preview_image = None

        self.image_label.configure(
            image="",
            text="📷\n\nChoose an image to start classification"
        )

        self.result_label.configure(
            text="No prediction yet",
            fg=GRAY
        )

        self.confidence_label.configure(
            text="Confidence: -"
        )


# =====================================================
# RUN APPLICATION
# =====================================================

def main():

    root = tk.Tk()

    app = ClothesApp(root)

    root.mainloop()


if __name__ == "__main__":
    main()
