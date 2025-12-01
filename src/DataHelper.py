import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys

class DataHelper:
  def __init__(self, input_data, n_classes):
    self.input_data = input_data 
    self.n_classes = n_classes
    self.images_by_class = None
    self.avg_images = None
    self.IMG_SIZE = 28
    return

  """
  Goal: create a dictionary where each key is a unique label and the 
        value is a list of the images within that class.
  """
  def group_data_by_class(self):
    # if we already did the work, then return that dictionary
    if self.images_by_class != None:
      return self.images_by_class

    images_by_class = {}
    for i, data in enumerate(self.input_data):
      inputs, labels = data # each input is (1, 28, 28)
      for input_num, input_img in enumerate(inputs):
        curr_label = labels[input_num].item()
        if curr_label in images_by_class:
          images_by_class[curr_label].append(input_img)
        else:
          images_by_class[curr_label] = [input_img]

    self.images_by_class = images_by_class
    return images_by_class

  """
  Goal: generate a dictionary that maps the keys (labels/unique class) to the 
        average image. This will only work if self.images_by_class is already populated.
  """
  def mean_class_image(self):
    # sanity check
    if self.images_by_class == None:
      return None

    avg_images = {}
    for curr_class, images in self.images_by_class.items():
      curr_avg = np.zeros((self.IMG_SIZE, self.IMG_SIZE))
      for image in images:
        image = image.detach().numpy()
        curr_avg = np.add(curr_avg, image)
      curr_avg = np.divide(curr_avg, len(images))
      avg_images[curr_class] = curr_avg
    self.avg_images = avg_images
    return avg_images

  # the point here is to plot a histogram of the raw outputs (pre-softmax) that appear for each image, as well as a colored representation of whether or not
  # it was classified
  """
  Goal: Given a model, this function will return a dictionary that represents the probability of each class for an image given the 
        model. For example, hist_counts[0] will have two keys: 'raw model output values' which represents the probability for that
        class, and 'labels' which represents that images actual label. The list for each of these is ordered, and each one is the 
        same length as the total size of the training set given at initialization.
  Input: model, model_type ("fcnn" or "resnet")
  Output: dictionary of dictionary of str/lists for key/val pair
  """
  def get_class_count(self, model, model_type="fcnn"):
    # each key should actually contain another dict with all the classes and their values for that class
    hist_counts = {}

    for data_index, data in enumerate(self.input_data):
      inputs, labels = data # each input is (1, 28, 28)
      if model_type == "fcnn":
        for input_num, input_im in enumerate(inputs):
          logits = model(input_im) # should be the size = # of classes (in this case 10)
          true_label = labels[input_num].item()

          for i in range(self.n_classes): # should consistently loop from 0 to 9
            curr_prob = logits[:, i].item()
            if i in hist_counts:
              hist_counts[i]["raw model output values"].append(curr_prob)
              hist_counts[i]["labels"].append(true_label)
            else:
              hist_counts[i] = {"raw model output values": [curr_prob], "labels": [true_label]}

      elif model_type == "resnet":
        logits = model(inputs)
        for j, label in enumerate(labels):
          for i in range(self.n_classes): # should consistently loop from 0 to 9
            curr_prob = logits[j, i].item()
            if i in hist_counts:
              hist_counts[i]["raw model output values"].append(curr_prob)
              hist_counts[i]["labels"].append(label.item())
            else:
              hist_counts[i] = {"raw model output values": [curr_prob], "labels": [label.item()]}
    return hist_counts

  """
  Goal: This function will plot the probabilities for each class given by the model, and color code them by their actual
        label.
  Input: hist_counts (calculated from get_class_count function), data_path (where to save the plots)
  """
  def plot_raw_histograms(self, hist_counts, data_path):
    for curr_class in hist_counts.keys():
      curr_data = hist_counts[curr_class]
      df = pd.DataFrame(curr_data)

      fig, ax = plt.subplots(figsize=(self.n_classes, self.n_classes))
      other_clrs = ["red", "orange", "yellow", "purple", "blue", "brown", "pink", "#3A0CA3", "#4361EE", "#4CC9F0"]
      colors = {}

      for i in range(self.n_classes):
        if i == curr_class:
          colors[i] = 'green'
        else:
          colors[i] = other_clrs[i]

      sns.histplot(data=df, x='raw model output values', hue='labels', palette=colors, stat='count', edgecolor=None)
      ax.set_title('Raw Model Outputs for Class ' + str(curr_class))
      fig.savefig(data_path + str(curr_class))