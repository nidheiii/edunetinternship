
Anomaly Detection in Surveillance Images Using CNN

This project is focused on using Convolutional Neural Networks (CNNs) along with Transfer Learning to classify surveillance images as either normal or anomaly. The objective is to enhance security systems by enabling automatic detection of unusual behavior or events in surveillance footage.

 Installation Steps

 1. Clone the Repository or Download the ZIP File
   - To start, download the project files from the GitHub repository. You can do this either by cloning the repository using Git or by downloading the ZIP file from the repository and extracting it to your computer.
   
   Clone using Git:
   Open the terminal (or command prompt) and run the following command:
   ```
   git clone https://github.com/yourusername/anomaly-detection.git
   ```
   This will create a folder named `anomaly-detection` with all the project files.

   Download ZIP:
   Alternatively, you can go to the GitHub page, click on the “Download ZIP” button, and extract it on your local machine.

 2. Navigate to the Project Folder
   Once you have the project files, go into the project folder using the terminal (or command prompt). If you cloned the repository, use:
   ```
   cd anomaly-detection
   ```

 3. Create a Virtual Environment (Optional but Recommended)
   It's recommended to create a virtual environment to isolate the project dependencies. Run the following commands based on your operating system:
   
   - For Windows:
     ```
     python -m venv env
     ```
   
   - For macOS/Linux:
     ```
     python3 -m venv env
     ```

4. Activate the Virtual Environment
   After creating the virtual environment, activate it:
   
   - For Windows:
     ```
     venv\Scripts activate
     ```

   - For macOS/Linux:
     ```
     source env/bin/activate
     ```
5. Install Dependencies
   With the virtual environment activated, install the necessary libraries by running:
   ```
   pip install tensorflow opencv-python numpy scikit-learn matplotlib pillow
   ```

   This will install the core libraries needed for the project, including TensorFlow for deep learning, OpenCV for image processing, and other essential libraries.

---

 Usage

 1. Data Preprocessing
   - Prepare your dataset by organizing it into two folders: `normal/` and `anomalous/`.
   - Apply data augmentation to increase the diversity of your dataset. You can perform transformations like rotation, flipping, and scaling to make the model more robust to variations in input images.

 2. Model Setup
   - Use a pre-trained CNN model, such as VGG16, and modify it by adding custom layers for binary classification (normal vs. anomalous).
   - For example, use the following code to load the pre-trained model and add custom layers:
     ```python
     model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
     ```

3. Train the Model
   - Once the model is set up, train it using the training dataset. You can define the number of epochs and provide validation data to monitor the model's performance.
   ```python
   model.fit(train_data, epochs=10, validation_data=val_data)
   ```

4. Classify New Images
   - After training, the model can classify new images as either normal or anomalous. Preprocess an image and use the model to predict the class:
   ```python
   img = image.load_img('new_image.jpg', target_size=(224, 224))
   prediction = model.predict(img_array)
   ```
 5. Save and Load the Model
   - Save the trained model to avoid retraining every time you use it:
   ```python
   model.save('anomaly_detection_model.h5')
   ```
   - You can load the model later for predictions or further training using:
   ```python
   model = load_model('anomaly_detection_model.h5')
   ```

---

For more details and to access the full code, visit the project repository on GitHub.
