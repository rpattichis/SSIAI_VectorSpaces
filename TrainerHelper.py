import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.autograd import Variable
import random

class TrainerHelper():
  def __init__(self, model, device, trainloader, valloader):
    self.model = model
    self.device = device
    self.trainloader = trainloader
    self.valloader = valloader
    return

  """
  Goal: This trains a single epoch and updates the progress.
  """
  def train_one_epoch(self, loss_fn, optimizer, epoch_index, tb_writer):
      running_loss = 0.
      last_loss = 0.

      for i, data in enumerate(self.trainloader):
          # Every data instance is an input + label pair
          inputs, labels = data

          # Zero your gradients for every batch!
          optimizer.zero_grad()

          # Make predictions for this batch
          outputs = self.model(inputs)

          # Compute the loss and its gradients
          loss = loss_fn(outputs, labels)
          loss.backward()

          # Adjust learning weights
          optimizer.step()

          # Gather data and report
          running_loss += loss.item()
          if i % 1000 == 999:
              last_loss = running_loss / 1000 # loss per batch
              print('  batch {} loss: {}'.format(i + 1, last_loss))
              tb_x = epoch_index * len(self.trainloader) + i + 1
              tb_writer.add_scalar('Loss/train', last_loss, tb_x)
              running_loss = 0.

      return last_loss
  
  """ 
  Goal: This trains a single epoch on the given input image (for the INN implementation). Meant to replace
        traing_one_epoch.
  """
  def train_input_epoch(self, loss_fn_mse, optimizer, input_image, target_out, epoch_index, tb_writer):
      running_loss = 0.
      last_loss = 0.

      optimizer.zero_grad()

      output = self.model(input_image)

      # Compute the loss and its gradients
      loss = loss_fn_mse(output, target_out)
      loss.backward()

      # Adjust learning weights
      optimizer.step()

      print("Current loss: " + str(loss))

      return last_loss

  """ 
  Goal: This trains a model given the loss fn, optimizer, and number of epochs. It supports both regular training as well as the INN implementation.

  """
  def custom_epoch_train(self, loss_fn, optimizer, n_epochs=10, train_input=False, input_image=None, target_out=None):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/mnist_trainer_{}'.format(timestamp))
    epoch_number = 0

    # validation loss
    best_vloss = 1_000_000.

    for epoch in range(n_epochs):
      print('EPOCH {}:'.format(epoch_number + 1))

      # Make sure gradient tracking is on, and do a pass over the data
      avg_loss = None
      # NOTE: Fix the parameters of the one epoch train
      if train_input:
        avg_loss = self.train_input_epoch(loss_fn, optimizer, input_image, target_out, epoch_number, writer)
      else:
        self.model.train(True)
        avg_loss = self.train_one_epoch(loss_fn, optimizer, epoch_number, writer)


      if train_input == False:
        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        self.model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(self.valloader):
                vinputs, vlabels = vdata
                voutputs = self.model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(self.model.state_dict(), model_path)

      epoch_number += 1
    return

  """ 
  Goal: Find the image closest to the target image in terms of MSE loss. If there is a threshold,
        then we will grab the closest images within that threshold and average those as our starting
        image rather than the closest image.
        Basically, it supports starting the trainable input by:
          - image within the class with the lowest MSE loss to the target image
          - 
  """
  def find_closest_image(self, class_images, target, model_type, threshold=None):
    image_scores = []
    loss_fn_mse = torch.nn.MSELoss()
    lowest_loss = float('inf')
    lowest_image = None
    for curr_image in class_images:
      if model_type == "resnet":
        curr_image = curr_image.unsqueeze(0).repeat(32, 1, 1, 1)
      
      scores = self.model(curr_image)
      curr_loss = loss_fn_mse(scores, target)
      if curr_loss < lowest_loss:
        lowest_loss = curr_loss
        lowest_image = curr_image
      image_scores.append((curr_loss, curr_image))
    
    # if there's a threshold, take average of the minimum distance images within that threshold
    # otherwise, it is just the minimum image used for training
    if threshold != None:
      image_scores = sorted(image_scores, key=lambda x: x[0])
      n_samples = int(len(image_scores) * threshold)
      image_samples = image_scores[:n_samples]
      lowest_image = torch.zeros(image_samples[0][1].shape)
      for image in image_samples:
        lowest_image = torch.add(lowest_image, image[1])
      lowest_image = torch.divide(lowest_image, n_samples)

    if model_type == "resnet":
      lowest_image = lowest_image[1, :, :, :]

    return lowest_image, lowest_loss

  """ 
  Goal: Constructs the target image, the starting image, and actually trains the input for (currently) 100 epochs 
        using MSE loss.
  """ 
  def find_invertable_input(self, hist_counts, input_image, n_classes, img_class_dict=None, model_type="fcnn", 
                            save_path="drive/MyDrive/LinAlgProject/graphs/resnet/invert_image_class", 
                            min_max=True, random=False ,threshold=None):

    # freeze the model's parameters
    for param in self.model.parameters():
      param.requires_grad = False

    distance_scores = []
    for class_name in hist_counts.keys():
      counts = hist_counts[class_name]['raw model output values']

      min_val, max_val = min(counts), max(counts)
      if min_max == False:
        min_val = np.percentile(counts, 25)
        max_val = np.percentile(counts, 75)

      # Step 1: build target probabilities tensor based on min and max values desired
      target = [min_val] * n_classes
      target[class_name] = max_val
      target = torch.tensor(target)

      if model_type == "resnet":
        target = target.unsqueeze(0).repeat(32, 1, 1, 1)
      target = Variable(torch.tensor(target), requires_grad=False)

      # Step 2: construct the starting input image that we will learn. Supports either finding the closest image in the training set,
      # as well as adding random noise to a given input image desired.
      curr_input = None
      # if there is an image, it will be the average image over all the images in that class for our implementation
      if input_image == None:
        curr_input, curr_loss = self.find_closest_image(img_class_dict[int(class_name)], target, model_type=model_type, threshold=threshold)
        print("Current loss before training: " + str(curr_loss))
      else:    
        curr_input = torch.tensor(input_image[int(class_name)]).float()
        if random:
          rand_input = torch.rand(1, 28, 28)
          curr_input = torch.add(rand_input, curr_input)

      if model_type == "resnet":
        curr_input = curr_input.unsqueeze(0).repeat(32, 1, 1, 1)
        
      # make the input trainable
      z = Variable(curr_input.to(self.device), requires_grad=True)

      optim = torch.optim.SGD([z], lr=0.0001)
      loss_fn_mse = torch.nn.MSELoss()

      self.custom_epoch_train(loss_fn_mse, optim, n_epochs=100, train_input=True, input_image=z, target_out=target)
      
      if model_type == "resnet":
        z = z[0, :, :, :]
      
      plt.imshow(np.reshape(z.detach().numpy(), (28, 28)), cmap="gray", interpolation="nearest")
      plt.axis('off')
      plt.savefig(save_path + str(class_name))

  def compute_distance(image_scores, target_scores):
    difference = torch.subtract(image_scores, target_scores)
    diff = torch.sum(torch.multiply(difference, difference))
    numerator = math.sqrt(diff)
    denom = torch.sum(torch.multiply(target_scores))
    return numerator / denom

  def get_accuracy(self, valloader):
    correct = 0.
    total = 0.
    softmax = nn.Softmax(dim=1)

    for data in valloader:
      inputs, labels = data
      total += len(inputs)
      outputs = self.model(inputs)
      probs = softmax(outputs)
      outputs = torch.argmax(probs, dim=-1)
      correct += (outputs == labels).sum().item()
    accuracy = correct / total
    return accuracy