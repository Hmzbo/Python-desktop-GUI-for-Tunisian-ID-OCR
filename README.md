# Python desktop application for Tunisian ID OCR
## Introduction
A python desktop application to automatically extract data from Tunisian ID cards.\
This project is built using the Yolov5 model developed for the Tunisian ID OCR web application project, [repo link](https://github.com/Hmzbo/Tunisian-ID-OCR). A Real-ESRGAN model is used to enhance low quality captured images, and and EasyOCR model is used to extract data from detected fields.

## Installation
To install this desktop application you can either clone this repository, or download it as a .zip file and extract it in the desired location. To clone this repo, you can use follow these steps:
1. open file explorer in the directory that you wish to add the cloned repository
2. enter cmd into directory bar
3. in cmd enter:\
  `git clone https://github.com/Hmzbo/Windows-desktop-GUI-for-Tunisian-ID-OCR.git`\
  `cd Windows-desktop-GUI-for-Tunisian-ID-OCR`

To launch the GUI use the following command:\
`python cin_detection_gui.py`
But make sure you have the following requirements met:
1. Python => 3.8
2. Install dependencies from `requirements.txt` file.

## GUI
![main](https://user-images.githubusercontent.com/62519374/168561648-415f7458-fd77-41b7-bf02-0446daefa5c4.png)

The interface is pretty simple and user-friendly, It is composed of 3 columns:
- Left column is used to display tables.
- Middle column contains the control buttons.
- Right column is used to display the camera capture.

We have only 4 buttons in the main window:
1. Start/stop detection: This is the main button that will trigger the CIN OCR process through your webcam.
2. Open file: This is used to open a .csv, or .xlsx file to add new entires.
3. Create file: This is used to create a blank .csv or .xlsx file with only column headers.
4. Auto-save changes: This checkbox allows you to auto-save changes made manually or automatically to the opened table.
***
![live_detection](https://user-images.githubusercontent.com/62519374/168579055-69ac780e-48dc-4839-b5ed-9ce2a29cd4ea.png)

Once, you click on "Start detection", the right columns gets wider to display the video capture with 3 textual indication about the detection status:
- Detecting..
- CIN Front Detcted.
- CIN Back Detected.

When the app detect one face of the CIN with a certain degree of confidence, the corresponnding text changes to Green color to indicate that you can switch to the other CIN face.
***
![results_table](https://user-images.githubusercontent.com/62519374/168579232-50621d52-407a-4690-867c-4085941de4b8.png)

Finally, when both CIN faces are detected the app starts the image processing and OCRing, this usually takes about 6-8 seconds, and then the results are shown on the table space in the left column. In case no table was opened before the detection process begun, a new table is automatically created and saved in a folder called `created_table_files`.
***
There are several settings that can be changed through the settings window (shortcut F1), as shown in the image below:
![settings](https://user-images.githubusercontent.com/62519374/168561683-a57d077f-bcf0-499e-b206-f704f0e14f19.png)

1. Camera resolution: This is used to capture frames from the camera, use max resolution available for your camera for better results.
2. Camera index: This is the index of the camera, don't change it if you have only one camera.
3. Detection confidence threshold: This is the threshold for the Yolo algorithm.
4. Sharpness threshold: This is used to avoid capturing blurry images. (The bigger the number the sharper the image is required to be.)
5. Save detection recording: This is used to indicate whether the detection process will be saved as a video file or not in the `recorded detections` folder.
