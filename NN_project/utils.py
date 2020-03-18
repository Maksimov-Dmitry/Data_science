import numpy as np

import cv2
 
def save_images(array, images):
    img_array = []
    for i in array:
        for j in i:
            number = np.argmax(j)
            image = images[number].convert('RGB') 
            im = np.array(image) 
            im = im[:, :, ::-1].copy() 
            height, width, layers = im.shape
            size = (width,height)
            img_array.append(im)
    out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 3, size)
 
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def get_images_from_array(array, images):
    temp_list = []
    for i in array:
        temptemp_list = []
        for j in i:
            temptemp_list.append(images[j])
        temp_list.append(temptemp_list)
    new_array = np.array(temp_list)
            
    return new_array

def get_new_video(array):
    new_array = np.array([array[i] for i in range(1, array.size)])
    return np.concatenate([new_array, [array[0]]])

def get_full_circle(array):
    new_array = np.vstack((array, get_new_video(array)))
    for i in range(1, array.size):
        next_array = get_new_video(new_array[i])
        new_array = np.vstack((new_array, next_array))
    return new_array
    
def get_data_set(arg, count_train, count_test, image_norm):
    full_arg = get_full_circle(arg)
#    print(full_arg)
    
    train_x = full_arg[:count_train]
    train_x = get_images_from_array(train_x, image_norm)
    train_y = full_arg[1:count_train + 1]
    
    test_x = full_arg[count_train:count_train + count_test]
    test_x = get_images_from_array(test_x, image_norm)
    test_y = full_arg[count_train + 1:count_train + 1 + count_test]
    
    return train_x, train_y, test_x, test_y

def get_dataset_from_video(count_train, count_test, images_norm, *args):
    full_train_x, full_train_y, full_test_x, full_test_y = get_data_set(args[0], count_train,
                                                                        count_test, images_norm)
    
    for i in range (1, len(args)):
        arg_train_x, arg_train_y, arg_test_x, arg_test_y = get_data_set(args[i], count_train,
                                                                        count_test, images_norm)
        full_train_x = np.vstack((full_train_x, arg_train_x))
        full_train_y = np.vstack((full_train_y, arg_train_y))
        full_test_x = np.vstack((full_test_x, arg_test_x))
        full_test_y = np.vstack((full_test_y, arg_test_y))

    return full_train_x, full_train_y, full_test_x, full_test_y
    
        
        