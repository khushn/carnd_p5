# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, I've written a software pipeline to detect vehicles in a video (started with the test_video.mp4 and later implemented on  the full project_video.mp4). In the end, I also combined the processing of the carnd_p4 to generate a video having both car detections and lane marking. 


The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

The links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   We can also take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment the training data.  

The detailed project writeup is at:
---
[writeup](/writeup.pdf)

The IPython notebook having the entire code is at :
---
[vehicle_detection.ipynb](/vehicle_detection.ipynb)


The output (vehicle detected) of test images: 
[vehicle detected test images](/output_images)

Vehicle detected Video output:
---

[Vehicle video](/video_out.mp4)

Challenge part, with p4 and this processing combined:
---

[Challenge output video](/video_vehicles_and_lanes_marked.mp4)
