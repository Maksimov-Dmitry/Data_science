import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

def plot_value_array(i, predictions_array, true_label, length):
  predictions_array, true_label = predictions_array[i], true_label[i]
  
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  thisplot = plt.bar(range(length), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                                100*np.max(predictions_array),
                                true_label),
                                color=color)
  plt.ylabel('probability')
  plt.xlabel('class')

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

def draw_row(i, count, size, images):
    for j in range (0, count):
        for k in range (0, count - 2):
            draw =  ImageDraw.Draw(images[i])
            draw. ellipse((k*size,j*size,k*size + size,j*size + size), fill="black")
            draw. ellipse(((k+1)*size,j*size,(k+1)*size + size,j*size + size), fill="black")
            draw. ellipse(((k+2)*size,j*size,(k+2)*size + size,j*size + size), fill="black")
            i += 1
    return i

def draw_col(i, count, size, images):
    for j in range (0, count):
        for k in range (0, count - 2):
            draw =  ImageDraw.Draw(images[i])
            draw. ellipse((j*size,k*size,j*size + size,k*size + size), fill="black")
            draw. ellipse((j*size,(k+1)*size,j*size + size,(k+1)*size + size), fill="black")
            draw. ellipse((j*size,(k+2)*size,j*size + size,(k+2)*size + size), fill="black")
            i += 1
    return i

def draw_up_right(i, count, size, images):
    for j in range (0, count):
        for k in range (0, count):
            if (j != 0 and k != (count - 1)):
                draw =  ImageDraw.Draw(images[i])
                draw. ellipse((k*size,j*size,k*size + size,j*size + size), fill="black")
                draw. ellipse(((k + 1)*size,j*size,(k + 1)*size + size,j*size + size), fill="black")
                draw. ellipse((k*size,(j - 1)*size,k*size + size,(j - 1)*size + size), fill="black")
                i += 1
    return i

def draw_up_left(i, count, size, images):
    for j in range (0, count):
        for k in range (0, count):
            if (j != 0 and k != 0):
                draw =  ImageDraw.Draw(images[i])
                draw. ellipse((k*size,j*size,k*size + size,j*size + size), fill="black")
                draw. ellipse(((k - 1)*size,j*size,(k - 1)*size + size,j*size + size), fill="black")
                draw. ellipse((k*size,(j - 1)*size,k*size + size,(j - 1)*size + size), fill="black")
                i += 1
    return i

def draw_down_right(i, count, size, images):
    for j in range (0, count):
        for k in range (0, count):
            if (j != (count - 1) and k != (count - 1)):
                draw =  ImageDraw.Draw(images[i])
                draw. ellipse((k*size,j*size,k*size + size,j*size + size), fill="black")
                draw. ellipse(((k + 1)*size,j*size,(k + 1)*size + size,j*size + size), fill="black")
                draw. ellipse((k*size,(j + 1)*size,k*size + size,(j + 1)*size + size), fill="black")
                i += 1
    return i

def draw_down_left(i, count, size, images):
    for j in range (0, count):
        for k in range (0, count):
            if (j != (count - 1) and k != 0):
                draw =  ImageDraw.Draw(images[i])
                draw. ellipse((k*size,j*size,k*size + size,j*size + size), fill="black")
                draw. ellipse(((k - 1)*size,j*size,(k - 1)*size + size,j*size + size), fill="black")
                draw. ellipse((k*size,(j + 1)*size,k*size + size,(j + 1)*size + size), fill="black")
                i += 1
    return i

def draw_picture(image):
        plt.figure()
        plt.imshow(image)
        plt.grid(False)
        plt.show()

def get_pictures(count, size):
    images = []
    for i in range (0, 94):
        images.append(Image.new("1", (size * count, size * count), "white"))

    for k in images:
        draw = ImageDraw.Draw(k)
        for i in range (0, count):
            for j in range (0, count):
                draw. ellipse((j*size,i*size,j*size + size,i*size + size), fill="white", outline="black")

    i = 0           
    i = draw_row(i, count, size, images)
    i = draw_col(i, count, size, images)
    i = draw_up_right(i, count, size, images)
    i = draw_up_left(i, count, size, images)
    i = draw_down_right(i, count, size, images)
    i = draw_down_left(i, count, size, images)

    return images