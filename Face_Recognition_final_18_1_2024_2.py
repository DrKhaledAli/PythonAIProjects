import os
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from skimage import feature
import matplotlib.pyplot as plt

class FaceRecognitionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("3D Face Recognition App")

        # Set initial dimensions for half-maximized window
        window_width = self.master.winfo_screenwidth() // 2
        window_height = self.master.winfo_screenheight() // 2
        self.master.geometry(f"{window_width}x{window_height//1}+{window_width//4}+{window_height//4}")
     
        

        # Center the window on the screen
        self.master.eval('tk::PlaceWindow . center')

        # Create a menu bar
        menubar = tk.Menu(master)
        master.config(menu=menubar)

        # Create a Start Processing menu
        start_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Processing", menu=start_menu)
        start_menu.add_command(label="Select Database", command=self.select_dataset)
        start_menu.add_command(label="Extract Features", command=self.feature_extraction)
        start_menu.add_command(label="Train Classifier", command=self.train_svm)
        #start_menu.add_command(label="Test Single Image", command=self.test_single_image)
        #start_menu.add_command(label="Test All Images", command=self.test_all_images)

        # Create a Testing menu
        testing_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Testing", menu=testing_menu)
        testing_menu.add_command(label="Test unique Image", command=self.test_single_image)
        testing_menu.add_command(label="Test All Images", command=self.test_all_images)

        # Create an About menu
        about_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="About", menu=about_menu)
        about_menu.add_command(label="About", command=self.display_about_message)
        about_menu.add_command(label="Close", command=self.master.destroy)

    



        # Initialize SVM classifier
        self.clf = svm.SVC(kernel='linear')

        # Initialize feature extractor (LBP)
        self.lbp_radius = 1
        self.lbp_points = 8

        # Variables to store dataset and features
        self.dataset_path = None
        self.features = None

        # Create buttons
        button_width = 15
        
        self.select_dataset_button = tk.Button(master, text="Select Database", command=self.select_dataset, width=button_width)
        self.select_dataset_button.pack(side="left", padx=15, pady=10)

        self.feature_extraction_button = tk.Button(master, text="Feature Extraction", command=self.feature_extraction, width=button_width)
        self.feature_extraction_button.pack(side="left", padx=15, pady=10)

        self.train_svm_button = tk.Button(master, text="Train SVM", command=self.train_svm, width=button_width)
        self.train_svm_button.pack(side="left", padx=15, pady=10)

        self.test_single_image_button = tk.Button(master, text="Test unique Image", command=self.test_single_image, width=button_width)
        self.test_single_image_button.pack(side="left", padx=15, pady=10)

        self.test_all_images_button = tk.Button(master, text="Test All Images", command=self.test_all_images, width=button_width)
        self.test_all_images_button.pack(side="left", padx=15, pady=10)

    def select_dataset(self):
        self.dataset_path = filedialog.askdirectory(title="Select Database Folder")
        if self.dataset_path:
            messagebox.showinfo("Dataset Selected", f"Dataset selected: {self.dataset_path}")

    def feature_extraction(self):
        if not self.dataset_path:
            messagebox.showerror("Error", "Please select a dataset first.")
            return

        X, y = self.load_3d_images(self.dataset_path)
        X_lbp = self.extract_lbp_features(X)
        self.features = X_lbp
        messagebox.showinfo("Feature Extraction", "LBP features extracted successfully")

    def train_svm(self):
        if self.features is None:
            messagebox.showerror("Error", "Please perform feature extraction first.")
            return

        y = self.load_labels(self.dataset_path)
        X_train, X_test, y_train, y_test = train_test_split(self.features, y, test_size=0.2, random_state=42)

        # Train the SVM model
        self.clf.fit(X_train, y_train)

        messagebox.showinfo("SVM Training", "SVM trained successfully")

    def test_single_image(self):
        if not self.dataset_path or (self.features is not None and not self.features.any()):
         messagebox.showerror("Error", "Please select a dataset and perform feature extraction first.")
         return


        test_image_path = filedialog.askopenfilename(title="Select Test Image", filetypes=[("Image files", "*.jpg;*.png")])
        if test_image_path:
            image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)

            lbp_features = self.extract_single_lbp_features(image)
            predicted_label = self.clf.predict([lbp_features])

             # Evaluate the model for the single image
            true_label = int(test_image_path.split('_')[3])
            accuracy = 1 if predicted_label == true_label else 0  # 1 if correct, 0 if incorrect
            ######################################new code
            # Display the predicted label and accuracy in a messagebox
            result_message = f"Predicted Label: {predicted_label}\nTrue Label: {true_label}\nAccuracy: {accuracy}"
            messagebox.showinfo("Prediction", result_message)

            # Display the images with information using matplotlib
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            # Subplot 1: Original Image with File Name
            axs[0].imshow(image, cmap='gray')
            axs[0].set_title(f"Tested Image\nFile Name: {os.path.basename(test_image_path)}")

            # Subplot 2: LBP Features
            axs[1].bar(range(len(lbp_features)), lbp_features)
            axs[1].set_title("LBP Features Extracted")
            axs[1].set_xlabel("Feature Index")
            axs[1].set_ylabel("Feature Value")

            # Subplot 3: Final Image
            final_image = cv2.putText(image.copy(), f"Predicted: {predicted_label}", (10, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            axs[2].imshow(final_image, cmap='gray')
            axs[2].set_title("Final Image after Testing")

            plt.tight_layout()
            plt.show()
            ##############################################
            # Display the predicted label and accuracy in a messagebox
            #messagebox.showinfo("Prediction", f"Predicted Label: {predicted_label}\nTrue Label: {true_label}\nAccuracy: {accuracy}")

    def test_all_images(self):
        if not self.dataset_path or (self.features is not None and not self.features.any()):
            messagebox.showerror("Error", "Please select a dataset and perform feature extraction first.")
            return

        y = self.load_labels(self.dataset_path)
        predictions = self.clf.predict(self.features)

        # Evaluate the model for all images
        accuracy = accuracy_score(y, predictions)
        report = classification_report(y, predictions)

        messagebox.showinfo("Results", f"Testing all images completed\nAccuracy: {accuracy}\n\n{report}")
    
    def display_about_message(self):
        about_message = "3D Face Recognition Application\nVersion 1"
        messagebox.showinfo("About", about_message)

    def load_3d_images(self, data_dir):
      X = []  # List to store images
      y = []  # List to store labels

      for file_name in os.listdir(data_dir):
        if file_name.endswith((".jpg", ".png")):
            file_path = os.path.join(data_dir, file_name)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            # Assume the label is the integer part of the file name before an underscore
            label = int(file_name.split('_')[3])

            X.append(img)  # Append image to X
            y.append(label)  # Append label to y

      return np.array(X), np.array(y)

    def load_labels(self, data_dir):
        y = []

        for file_name in os.listdir(data_dir):
            if file_name.endswith((".jpg", ".png")):
                label = int(file_name.split('_')[3])
                y.append(label)

        return np.array(y)

    def extract_lbp_features(self, images):
        lbp_features = []

        for img in images:
            lbp = feature.local_binary_pattern(img, self.lbp_points, self.lbp_radius, method="uniform")
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, self.lbp_points + 3), range=(0, self.lbp_points + 2))
            hist = hist.astype("float") / hist.sum()

            lbp_features.append(hist)

        return np.array(lbp_features)

    def extract_single_lbp_features(self, image):
        lbp = feature.local_binary_pattern(image, self.lbp_points, self.lbp_radius, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, self.lbp_points + 3), range=(0, self.lbp_points + 2))
        hist = hist.astype("float") / hist.sum()

        return hist

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
