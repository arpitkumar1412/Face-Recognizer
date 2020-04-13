# Face-Recognizer

Running the tests

Run Tester.py script on commandline to train recognizer on training images and also predict test_img:

python tester.py
1.Place some test images in TestImages folder that you want to predict in tester.py file
2.Place Images for training the classifier in trainingImages folder.If you want to train clasifier to recognize multiple people then add each persons folder in separate label markes as 0,1,2,etc and then add their names along with labels in tester.py/videoTester.py file in 'name' variable.
3.To generate test images for training classifier use videoimg.py file.
4.To do test run via tester.py give the path of image in test_img variable
5.Use "videoTester.py" script for predicting faces realtime via your webcam.(But ensure that you run tester.py first since it generates training.yml file that is being used in "videoTester.py" script.
