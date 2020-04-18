#!/usr/bin/env python3
import pandas as pd


def gaussiam_hmm(epochs=1000, n_hidden_states=4):
    """
    +----------------+
    | Assignment 2.1 |
    +----------------+
    
    Fit an Hidden Markov Model with Gaussian emissions to the data in DSET1: 
    it is sufficient to focus on the “Appliances” and “Lights” columns of 
    the dataset which measure the energy consumption of appliances and lights, 
    respectively, across a period of 4.5 months. Consider the two columns in isolation, 
    i.e. train two separate HMM, one for appliances and one for light.
    Experiment with HMMs with a varying number of hidden states (e.g. at least 2, 3 and 4).
    Once trained the HMMs, perform Viterbi on a reasonably sized subsequence 
    (e.g. 1 month of data) and plot the timeseries data highlighting 
    (e.g. with different colours) the hidden state assigned to each timepoint by the Viterbi algorithm.
    Then, try sampling a sequence of at least 100 points from the trained HMMs.
    """
    from hmmlearn.hmm import GaussianHMM
    appliances_series = pd.read_csv('datasets/energy_data/energydata_complete.csv', 
                              header=0, usecols=[1])
    lights_series = pd.read_csv('datasets/energy_data/energydata_complete.csv',
                              header=0, usecols=[2])
    
    # Create an HMM and fit it to appliances data
    model_a = GaussianHMM(n_components=n_hidden_states,
                          covariance_type="diag",
                          n_iter=epochs,
                          algorithm='viterbi').fit(appliances_series)
    # Decode the optimal sequence of internal hidden state (Viterbi)
    hs_a = model_a.predict(appliances_series)
    # Generate new samples (visible , hidden)
    X_a, Z_a = model_a.sample(100)
    
    # Create an HMM and fit it to lights data
    model_l = GaussianHMM(n_components=n_hidden_states,
                          covariance_type="diag",
                          n_iter=epochs,
                          algorithm='viterbi').fit(lights_series)
    hs_l = model_l.predict(lights_series)
    X_l, Z_l = model_l.sample(100)
    return hs_a, hs_l 


def lda():
    """
    +----------------+
    | Assignment 2.2 |
    +----------------+
    
    Implement a simple image understanding application for DSET2 using the LDA 
    model and the bag of visual terms approach described in Lecture 9. 
    For details on how to implement the approach see the BOW demo and paper [10] 
    referenced on the Moodle site.  
    Keep one picture for each image subset (identified by the initial digit in the filename) 
    out of training for testing. In short:
    For each image (train and test) extract the SIFT descriptors for the interest 
    points identified by the SIFT detector (you are free to use a different detector 
    if you wish; even grid sampling).
    Learn a 500-dimensional codebook (i.e. run k-means with k=500 clusters) from  
    the SIFT descriptors of the training images (you can choose a subsample of them 
    if k-means takes too long).
    Generate the bag of visual terms for each image (train and test): 
        use the bag of terms for the training images to train an LDA model 
        (use one of the available implementations). 
    Choose the number of topics as you wish (but reasonably).
    Test the trained LDA model on test images: plot (a selection of) them with 
    overlaid visual patches coloured with different colours depending on the 
    most likely topic predicted by LDA.
    """
    from gensim.corpora.dictionary import Dictionary
    from gensim.models import LdaModel
    from sklearn.cluster import KMeans
    import numpy as np
    import cv2 as cv
    import pickle
    import os
    
    
    class BOVW_msrc(object):
    
        def __init__(self, n_cluster=500):
            self.n_clusters = n_cluster
            # RGB class mapping
            self.topic_rgb_map = {
                "void": [0, 0, 0],
                "building": [128, 0, 0],
                "grass": [0, 128, 0],
                "tree": [128, 128, 0],
                "cow": [0, 0, 128],
                "horse": [128, 0, 128],
                "sheep": [0, 128, 128],
                "sky": [128, 128, 128],
                "mountain": [64, 0, 0],
                "aeroplane": [192, 0, 0],
                "water": [64, 128, 0],
                "car": [64, 0, 128],
                "bicycle": [192, 0, 128] }
            self.dataset_dir = 'datasets/msrc_data'
            self.train_images = []
            self.test_images = []
            # all BOVW features of train/test
            self.train_sift_descriptors = []
            self.train_sift_descriptors_map = {}
            self.test_sift_descriptors = []
            self.test_sift_descriptors_map = {}
            # codebooks of train/test
            self.train_descriptors = []
            self.test_descriptors = []
            
            self.__compute_descriptors()
            self.__compute_clusters()
            
        def __compute_descriptors(self):
            print('Computing dataset descriptors...')
            idx = 1
            for filename in os.listdir(self.dataset_dir):
                if 'GT' not in filename and filename.endswith('.bmp'):
                    if idx <= 180:
                        self.train_images.append(filename)
                        path = os.path.join(self.dataset_dir, filename)
                        image = cv.imread(path)
                        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                        extractor = cv.xfeatures2d.SIFT_create()
                        _, descriptor = extractor.detectAndCompute(gray, None)
                        self.train_sift_descriptors.append(descriptor)
                        self.train_sift_descriptors_map[filename] = descriptor
                        self.train_descriptors.append(descriptor)
                    else:
                        self.test_images.append(filename)
                        path = os.path.join(self.dataset_dir, filename)
                        image = cv.imread(path)
                        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                        extractor = cv.xfeatures2d.SIFT_create()
                        _, descriptor = extractor.detectAndCompute(gray, None)
                        self.test_sift_descriptors.append(descriptor)
                        self.test_sift_descriptors_map[filename] = descriptor
                        self.test_descriptors.append(descriptor)
                    idx += 1
            self.train_descriptors = np.asarray(self.train_descriptors)
            self.train_descriptors = np.concatenate(self.train_descriptors, axis=0)
            self.test_descriptors = np.asarray(self.test_descriptors)
            self.test_descriptors = np.concatenate(self.test_descriptors, axis=0)
        
        def __compute_clusters(self):
            print('Computing clusters...')
            # SIFT descriptor clusters model
            self.kmeans_train_model = None
            self.kmeans_test_model = None
            model_train_filename = 'lda_kmeans_train.model'
            model_test_filename = 'lda_kmeans_test.model'
            if os.path.isfile(model_train_filename):
                print('Model train already exists')
                self.kmeans_train_model = pickle.load(open(model_train_filename, 'rb'))
            else:
                print('Model train not exists')
                self.kmeans_train_model = KMeans(n_clusters=self.n_clusters).fit(self.train_descriptors)
                pickle.dump(self.kmeans_train_model, open(model_train_filename, 'wb'))
            
            if os.path.isfile(model_test_filename):
                print('Model test already exists')
                self.kmeans_test_model = pickle.load(open(model_test_filename, 'rb'))
            else:
                print('Model test not exists')
                self.kmeans_test_model = KMeans(n_clusters=self.n_clusters).fit(self.test_descriptors)
                pickle.dump(self.kmeans_test_model, open(model_test_filename, 'wb'))
            self.visual_words_train = self.kmeans_train_model.cluster_centers_
            self.visual_words_test = self.kmeans_test_model.cluster_centers_
            
        def __compute_train_visterms(self):
            print('Computing training set visterms...')
            self.train_predictions_map = {}
            for filename, descriptor in self.train_sift_descriptors_map.items():
                prediction = self.kmeans_train_model.predict(descriptor)
                self.train_predictions_map[filename] = prediction
            # dictionary of visterms of each train image
            self.train_histograms_map = {}
            for filename, prediction in self.train_predictions_map.items():
                histogram, _ = np.histogram(prediction, bins=self.n_clusters)
                self.train_histograms_map[filename] = histogram
        
        def __compute_test_visterms(self):
            print('Computing test set visterms...')
            self.test_predictions_map = {}
            for filename, descriptor in self.test_sift_descriptors_map.items():
                prediction = self.kmeans_test_model.predict(descriptor)
                self.test_predictions_map[filename] = prediction
            # dictionary of visterms of each test image
            self.test_histograms_map = {}
            for filename, prediction in self.test_predictions_map.items():
                histogram, _ = np.histogram(prediction, bins=self.n_clusters)
                self.test_histograms_map[filename] = histogram
        
        def num_topics(self):
            return len(self.topic_rgb_map)
    
        def train_vocabulary(self):
            self.__compute_train_visterms()
            
            self.train_image_topics_map = {}
            for filename in self.train_images:
                topics = []
                prefix, suffix = filename.split('.')[0], filename.split('.')[1]
                topic_image_path = os.path.join(self.dataset_dir, ''.join([prefix, '_GT.', suffix]))
                image = cv.imread(topic_image_path)
                for key, value in self.topic_rgb_map.items():
                    mask = cv.inRange(image, np.array(value), np.array(value))
                    if cv.countNonZero(mask)>0:
                        topics.append(key)
                self.train_image_topics_map[filename] = topics
            
            self.train_topic_images_map = {}
            for image, topics in self.train_image_topics_map.items():
                for topic in self.topic_rgb_map.keys():
                    if topic in topics:
                        imgs = self.train_topic_images_map.get(topic, [])
                        imgs.append(image)
                        self.train_topic_images_map[topic] = imgs
            idx = 0
            self.train_vocabulary = []
            self.train_doc_id_map = {}
            for filename, codebook in self.train_histograms_map.items():
                self.train_vocabulary.append(list(codebook))
                self.train_doc_id_map[filename] = idx
                idx += 1
            return np.array(self.train_vocabulary), self.train_doc_id_map


    bovw = BOVW_msrc()
    vocabulary, doc_mapping = bovw.train_vocabulary()
    
    num_topics = bovw.num_topics()
    dictionary = Dictionary(vocabulary)
    corpus = [dictionary.doc2bow(doc) for doc in vocabulary]
    
    lda_model = LdaModel(corpus, num_topics=num_topics)
    
    
    
    
    
    
    
    
    
    