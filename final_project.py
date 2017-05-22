from __future__ import division
import numpy as np
import time
import sys
import random
import PIL.Image
import os
import scipy.misc
import skimage as sk
import skimage.io
import cv2
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import mode
from scipy import misc
from itertools import product

#np.set_printoptions(threshold=np.nan)

MASKING = False
NO_OF_BINS = 180
NO_OF_SEGMENTS = 5

# Status: program orders images based on histogram similarity with a given image,
# currently implementing clustering approach.

#--------------------------------------------------------------------------------------------------

# Returns histogram representing hue distribution given an RGB image
def get_hue_histogram(rgb_image, mask=None):

	if MASKING == True:
		return get_refined_hue_histogram(rgb_image)

	# Convert the rgb image to hsv
	hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

	# Generate histogram
	hue_hist = cv2.calcHist([hsv_image], [0], mask, [NO_OF_BINS], [0, NO_OF_BINS])
	hue_hist = cv2.normalize(hue_hist, hue_hist).flatten()

	# Show the histogram- uncomment to show histograms during execution
	# plt.plot(hue_hist,color = 'r')
	# plt.xlim([0,NO_OF_BINS])
	# plt.xlabel("Hue Bin Number")
	# plt.ylabel("Frequency")
	# plt.show()

	return hue_hist

#--------------------------------------------------------------------------------------------------

def show_hue_histogram(rgb_image, mask=None):
	hue_hist = None
	if MASKING == True:
		return get_refined_hue_histogram(rgb_image)

	# Convert the rgb image to hsv
	hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

	# Generate histogram
	hue_hist = cv2.calcHist([hsv_image], [0], mask, [NO_OF_BINS], [0, NO_OF_BINS])
	hue_hist = cv2.normalize(hue_hist, hue_hist).flatten()

	# Show the histogram- uncomment to show histograms during execution
	plt.plot(hue_hist,color = 'r')
	plt.xlim([0,NO_OF_BINS])
	plt.xlabel("Hue Bin Number")
	plt.ylabel("Frequency")
	plt.show()

	return hue_hist

#--------------------------------------------------------------------------------------------------

# Gets the hue histogram with the whites and blues removed
def get_refined_hue_histogram(rgb_image):
	hsv_image = np.array(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV), dtype=np.uint16)
	# HSV images in range 0 - 255 so values must be normalised- we want to remove
	# blue which is in the range 180 to 240 which is (0.5 to 0.66) * 255

	h,s,v = cv2.split(hsv_image)
	
	#hue_mask = np.uint8(np.logical_not(np.logical_and(h >= 90, h <= 120)))
	#saturation_mask = np.uint8(s > 0.1 * 255)
	value_mask = np.uint8(v > 0.05 * 255)

	mask = value_mask

	hue_hist = cv2.calcHist([hsv_image], [0], mask, [NO_OF_BINS], [0, NO_OF_BINS])
	hue_hist = cv2.normalize(hue_hist, hue_hist).flatten()

	return hue_hist

#--------------------------------------------------------------------------------------------------

def show_mask(rgb_image):
	hsv_image = np.array(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV), dtype=np.uint16)
	rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
	# HSV images in range 0 - 255 so values must be normalised- we want to remove
	# blue which is in the range 180 to 240 which is (0.5 to 0.66) * 255

	h,s,v = cv2.split(hsv_image)
	
	hue_mask = np.uint8(np.logical_not(np.logical_and(h >= 90, h <= 120)))
	saturation_mask = np.uint8(s > 0.1 * 255)
	value_mask = np.uint8(v > 0.05 * 255)
	
	mymask = value_mask

	# plt.imshow(cv2.bitwise_and(rgb, rgb, mask = mymask))
	# plt.show()

	rgb[np.where(mymask == False)] = np.array([255, 0, 0], dtype=np.uint8)
	return rgb

	#return cv2.bitwise_and(rgb, rgb, mask = mymask)

#--------------------------------------------------------------------------------------------------

def save_masks():
	for filename in os.listdir(os.getcwd()):
		if filename[-4:] != '.jpg' and filename[-4:] != '.JPG':
			print 'File ' + str(filename) + ' is not an image'
			continue
		img = show_mask(cv2.imread(filename))
		plt.imsave("./clusters/masks/"+str(filename), img)

# # Shows the hue histogram of an RGB image (assumes values in range 0-255)
# def show_hue_histogram(rgb_image):
# 	# Normalise RGB values to float values in range [0,1]
# 	rgb_image = (rgb_image.astype(float)) / 255
# 	hsv_image = matplotlib.colors.rgb_to_hsv(rgb_image) * NO_OF_BINS
# 	# Generate histogram and show it
# 	plt.hist(hsv_image[:,:, 0].ravel(), NO_OF_BINS, [0,NO_OF_BINS])
# 	plt.show()

#--------------------------------------------------------------------------------------------------

# Takes an image, looks in a folder, and orders the images in ascending order of similarity
def compare_image_to_others(rgb_image):
	# Results array- one column for filenumber, the other for the distance between
	# the histogram of the image and the given image
	results = np.zeros((0,2), dtype=float)
	filenames = np.zeros((0), dtype=str)

	hue_hist = get_hue_histogram(rgb_image)

	i = 0
	#os.chdir("datasets/all")
	for filename in os.listdir(os.getcwd()):
		if filename[-4:] != '.jpg' and filename[-4:] != '.JPG':
			print 'File ' + str(filename) + ' is not an image'
			continue
		print 'Processing image: ' + str(filename) + '...'
		# Calculate the current image's histogram
		current_image = cv2.imread(filename)
		current_histogram = get_hue_histogram(current_image)
		current_objective_function = cv2.compareHist(hue_hist, current_histogram, 
			                                         cv2.HISTCMP_CHISQR)
		results = np.concatenate((results, [[i, current_objective_function]]), axis=0)
		filenames = np.concatenate((filenames, [filename]), axis=1)
		i = i + 1

	results = results[results[:,1].argsort()]

	filenames = filenames[results.astype(np.uint16)[:, 0]]
	print results
	plt.scatter(np.arange(results.shape[0]), results[:,1])
	plt.xlabel("Data Point")
	plt.ylabel("Distance From Query Image")
	#plt.show()
	print '-------------------\n'
	print filenames
	return filenames

#--------------------------------------------------------------------------------------------------

def save_images(filenames):
	for i in xrange(0, filenames.size):
		plt.imsave("./CBIR/"+str(i)+".jpg", plt.imread(filenames[i]))
		print "saving " + str(i)

#--------------------------------------------------------------------------------------------------
# Scans current directory for images and views the names given in the filenames
# array
def show_images(filenames):
	for i in xrange(0, filenames.size):
		cv2.imshow('image', cv2.imread(filenames[i]))
		cv2.waitKey(0)

#--------------------------------------------------------------------------------------------------

# Takes a 3D array of hue histograms and the number of clusters in the dataset
def group_images_kmeans(hue_hists, n_clusters):
    # The k-means stuff
    kmeans = KMeans(n_clusters)
    kmeans.fit_predict(hue_hists)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    return labels

#--------------------------------------------------------------------------------------------------

def get_kmeans_accuracy(predicted_labels, true_labels):
	print predicted_labels
	# Array which stores how accurate each individual label prediction was
	label_accuracies = np.zeros((np.unique(predicted_labels).size), dtype=np.float16)
	overall_accuracy = 0

	# For each cluster in the true labels array...
	for current_label in np.unique(true_labels):
		current_indexes = np.where(true_labels == current_label)
		# Get the labels that were supposed to be labelled with current_label
		current_cluster_labels = predicted_labels[current_indexes]
		# Number of occurences of each label in the predicted labels
		label_counts = np.bincount(current_cluster_labels)
		current_accuracy = np.float16(np.max(label_counts) / np.sum(label_counts))
		label_accuracies[current_label] = current_accuracy
		overall_accuracy = overall_accuracy + (current_accuracy * (current_cluster_labels.size / true_labels.size))

	# print overall_accuracy
	# print label_accuracies

	return overall_accuracy

#--------------------------------------------------------------------------------------------------

def save_image_clusters(filenames, predicted_labels, true_labels, segmented=False):
	count = 0
	# For each cluster
	for cluster_count in xrange(0, np.unique(predicted_labels).size):
		current_cluster = np.unique(predicted_labels)[cluster_count]
		# For each image in that cluster
		for image_count in xrange(0, predicted_labels[np.where(predicted_labels == current_cluster)].size):
			current_image = cv2.imread(filenames[count])
			if segmented == True:
				path = "./segmented_clusters/" + str(cluster_count) + "-" + str(image_count) + ".jpg"
			else:
				path = "./clusters/" + str(cluster_count) + "-" + str(image_count) + ".jpg"
			cv2.imwrite(str(path), current_image)
			#sk.io.imsave(str(path), current_image)
			count = count + 1

#--------------------------------------------------------------------------------------------------

# Returns histograms of an image from different segments. Current:
# Source: http://www.pyimagesearch.com/2014/12/01/complete-guide-building-image-search-engine-python-opencv/
def get_segmented_histograms(rgb_image):
	# Our features array. It contains no_of_regions rows and NO_OF_BINS columns where n is the
	# number of regions. Each row in the array is a histogram for a region in the image.
	features = np.zeros((0, NO_OF_BINS), dtype=np.float32)

	# divide the image into four rectangles/segments (top-left,
	# top-right, bottom-right, bottom-left)
	w = rgb_image.shape[0]
	h = rgb_image.shape[1]
	cX =w / 2
	cY = h / 2

	segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
		(0, cX, cY, h)]

	# construct an elliptical mask representing the center of the
	# image
	(axesX, axesY) = (int(w * 0.75) / 2, int(h * 0.75) / 2)
	ellipMask = np.zeros(rgb_image.shape[:2], dtype=np.uint8)
	cv2.ellipse(ellipMask, (int(cX), int(cY)), (int(axesX), int(axesY)), 0, 0, 360, 255, -1)

	# loop over the segments
	for (startX, endX, startY, endY) in segments:
		# construct a mask for each corner of the image, subtracting
		# the elliptical center from it
		cornerMask = np.zeros(rgb_image.shape[:2], dtype=np.uint8)
		cv2.rectangle(cornerMask, (int(startX), int(startY)), (int(endX), int(endY)), 255, -1)
		cornerMask = cv2.subtract(cornerMask, ellipMask)
		#show_hue_histogram(rgb_image, cornerMask)
		# extract a color histogram from the image, then update the
		# feature vector
		features = np.insert(features, 0, get_hue_histogram(rgb_image, cornerMask), axis=0)

	# extract a color histogram from the elliptical region and
	# update the feature vector
	# plt.imshow(cv2.bitwise_and(rgb,rgb,mask = ellipMask))
	# plt.show()

	features = np.insert(features, 0, get_hue_histogram(rgb_image, ellipMask), axis=0)
	#show_hue_histogram(rgb_image, ellipMask)
	
	# Return the features
	return features

#--------------------------------------------------------------------------------------------------

def get_elliptical_histogram(rgb_image):
	# Our features array. It contains no_of_regions rows and NO_OF_BINS columns where n is the
	# number of regions. Each row in the array is a histogram for a region in the image.
	features = np.zeros((0, NO_OF_BINS), dtype=np.float32)

	# divide the image into four rectangles/segments (top-left,
	# top-right, bottom-right, bottom-left)
	w = rgb_image.shape[0]
	h = rgb_image.shape[1]
	cX =w / 2
	cY = h / 2

	# construct an elliptical mask representing the center of the
	# image
	(axesX, axesY) = (int(w * 0.75) / 2, int(h * 0.75) / 2)
	ellipMask = np.zeros(rgb_image.shape[:2], dtype=np.uint8)
	cv2.ellipse(ellipMask, (int(cX), int(cY)), (int(axesX), int(axesY)), 0, 0, 360, 255, -1)

	# extract a color histogram from the elliptical region and
	# update the feature vector
	return get_hue_histogram(rgb_image, ellipMask)

#--------------------------------------------------------------------------------------------------

def test_segmented_kmeans():
	os.chdir("datasets/INRIA")

	# Array of histograms- stores the features
	hists = np.zeros((0, NO_OF_BINS * NO_OF_SEGMENTS))
	filenames = []

	file_count = 0
	for filename in os.listdir(os.getcwd()):
		# If the current filename isn't an image skip this iteration
		if filename[-4:] != '.jpg' and filename[-4:] != '.JPG':
			print 'File ' + str(filename) + ' is not an image'
			continue
		print 'Processing image: ' + str(filename) + '...'
		file_count = file_count + 1
		filenames.append(filename)
		# Calculate the current image's histogram
		current_image = cv2.imread(filename)
		current_features = np.reshape(get_segmented_histograms(current_image), (1, NO_OF_BINS * NO_OF_SEGMENTS))
		hists = np.insert(hists, 0, current_features, axis=0)

		# Reverse the histogram so that the features are stored along one row
		# rather than NO_OF_BINS rows.
		# UNCOMMENT FOR NON-SEGMENETED IMAGES
		# current_histogram = current_histogram.reshape(-1)
		# hists = np.vstack((hists, current_histogram))

	predicted_labels = group_images_kmeans(hists, 12)
	true_labels = np.load('true_labels.npy')
	accuracy = get_kmeans_accuracy(predicted_labels, true_labels)
	save_image_clusters(filenames, predicted_labels, true_labels)
	os.chdir("../..")
	print "\nAccuracy: " + str(accuracy)
	
#--------------------------------------------------------------------------------------------------

def test_kmeans():
	os.chdir("datasets/italy")

	# Array of histograms- stores the features
	hists = np.zeros((0, NO_OF_BINS))
	filenames = []

	file_count = 0
	for filename in os.listdir(os.getcwd()):
		# If the current filename isn't an image skip this iteration
		if filename[-4:] != '.jpg' and filename[-4:] != '.JPG':
			print 'File ' + str(filename) + ' is not an image'
			continue
		print 'Processing image: ' + str(filename) + '...'
		file_count = file_count + 1
		filenames.append(filename)
		# Calculate the current image's histogram
		current_image = cv2.imread(filename)
		# current_features = np.reshape(get_segmented_histograms(current_image), (1, NO_OF_BINS * NO_OF_SEGMENTS))
		# hists = np.insert(hists, 0, current_features, axis=0)

		# Reverse the histogram so that the features are stored along one row
		# rather than NO_OF_BINS rows.
		# UNCOMMENT FOR NON-SEGMENETED IMAGES
		#current_histogram = get_elliptical_histogram(current_image)
		current_histogram = get_hue_histogram(current_image)
		current_histogram = current_histogram.reshape(-1)
		hists = np.vstack((hists, current_histogram))

	predicted_labels = group_images_kmeans(hists, 4)
	true_labels = np.load('true_labels.npy')
	accuracy = get_kmeans_accuracy(predicted_labels, true_labels)
	save_image_clusters(filenames, predicted_labels, true_labels)

	print "\nAccuracy: " + str(accuracy)
	os.chdir("../..")
	return accuracy

#--------------------------------------------------------------------------------------------------

def tune_k_means():
	os.chdir("datasets/italy")
	results = np.zeros((0), dtype=np.float16)

	text_file = open("results.txt", "w")
	permutations = product([0,1,2], repeat=NO_OF_SEGMENTS)
	best_perm = None
	best_acc = 0
	best_labels = None
	
	for p in permutations:
		current_permutation = np.fromiter(p, dtype=np.uint8)
		if np.array_equal(current_permutation, np.array([0,0,0,0,0])):
			print "nope"
			continue

		hists = np.zeros((0, NO_OF_BINS * NO_OF_SEGMENTS))
		filenames = []

		file_count = 0
		for filename in os.listdir(os.getcwd()):
			# If the current filename isn't an image skip this iteration
			if filename[-4:] != '.jpg' and filename[-4:] != '.JPG':
				print 'File ' + str(filename) + ' is not an image'
				continue
			print 'Processing image: ' + str(filename) + '...'
			file_count = file_count + 1
			filenames.append(filename)
			# Calculate the current image's histogram
			current_image = cv2.imread(filename)
			current_features = np.reshape(get_segmented_histograms(current_image), (1, NO_OF_BINS * NO_OF_SEGMENTS))
			hists = np.insert(hists, 0, current_features, axis=0)

		current_permutation = np.fromiter(p, dtype=np.uint8)
		hists = np.repeat(current_permutation, NO_OF_BINS) * hists

		predicted_labels = group_images_kmeans(hists, 4)
		print predicted_labels
		true_labels = np.load('true_labels.npy')
		accuracy = get_kmeans_accuracy(predicted_labels, true_labels)
		if accuracy > best_acc:
			best_acc = accuracy
			best_perm = current_permutation
			best_labels = predicted_labels
		results = np.append(results, accuracy)

		text_file.write("Weighting: " + str(current_permutation) +  " | Accuracy: "+ str(accuracy) + "\n")
	print "Max accuracy: " + str(np.max(results))
	print "Best permutation: " + str(best_perm)
	save_image_clusters(filenames, best_labels, true_labels)
	text_file.close()
	plt.plot(results)
	plt.xlabel("Weighting Combination")
	plt.ylabel("Accuracy")
	plt.savefig("results1.png")
	#plt.show()
	os.chdir("../..")

#--------------------------------------------------------------------------------------------------

def tune_number_of_bins():
	#os.chdir("datasets/INRIA")
	global NO_OF_BINS
	bins = np.array([3,4,5,6,8,9,10,12,15,18,20,24,30,36,40,45,60,72,90,120,180,360])
	best_acc = 0
	best_bins = 36
	results = np.zeros((bins.size))
	for i in xrange(0,bins.size):
		NO_OF_BINS = bins[i]
		results[i] = test_kmeans()
		if results[i] > best_acc:
			best_acc = results[i]
			best_bins = bins[i]
	print "Best accuracy: " + str(best_acc)
	print "Best bins: " + str(best_bins)
	plt.xlabel("Number of Bins")
	plt.ylabel("Accuracy")
	plt.plot(bins, results)
	plt.show()

#--------------------------------------------------------------------------------------------------

def resize_images():
	#os.chdir("datasets/italy")
	filenames = []

	file_count = 0
	for filename in os.listdir(os.getcwd()):
		# If the current filename isn't an image skip this iteration
		if filename[-4:] != '.jpg' and filename[-4:] != '.JPG':
			print 'File ' + str(filename) + ' is not an image'
			continue
		print 'Processing image: ' + str(filename) + '...'
		file_count = file_count + 1
		filenames.append(filename)
		# Calculate the current image's histogram
		current_image = plt.imread(filename)
		current_image = misc.imresize(current_image, 0.25)
		sk.io.imsave(filename, current_image)

#--------------------------------------------------------------------------------------------------

def basic_cbir():
	os.chdir("datasets/INRIA")
	img = cv2.imread("103900.jpg")
	filenames = compare_image_to_others(img)
	save_images(filenames)
	os.chdir("../..")

#--------------------------------------------------------------------------------------------------

def scramble_image(rgb_image):
	size = rgb_image[:,:,0].size
	random = np.random.choice(size, size, replace=False)
	r = np.reshape(np.reshape(rgb_image[:,:,0], -1)[random], (rgb_image.shape[0], rgb_image.shape[1]))
	g = np.reshape(np.reshape(rgb_image[:,:,1], -1)[random], (rgb_image.shape[0], rgb_image.shape[1]))
	b = np.reshape(np.reshape(rgb_image[:,:,2], -1)[random], (rgb_image.shape[0], rgb_image.shape[1]))

	new_image = np.zeros((rgb_image.shape), dtype=np.uint8)
	print new_image.shape
	new_image[:,:,0] = r
	new_image[:,:,1] = g
	new_image[:,:,2] = b

	plt.imshow(new_image)
	plt.show()

#--------------------------------------------------------------------------------------------------

def test():
	#test_kmeans()

	tune_k_means()

	# os.chdir("datasets")
	# img = cv2.imread('9.jpg')
	# print get_hue_histogram(img).shape
	# get_hue_histogram(img)
	# show_hue_histogram(img)
	#filenames = compare_image_to_others(img)
	#show_images(filenames)

#--------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    start_time = time.time()
    # Code
    tune_number_of_bins()
    print("\n--- Execution Time: %s seconds ---" % (time.time() - start_time))