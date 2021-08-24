from django.shortcuts import render
from .models import Document
from .forms import DocumentForm
import os
import numpy as np
import forgery.E2E.parameters as parameters
from forgery.E2E.detection import detection,preload
from forgery.E2E.dataloaders.data_loader import loader
import torch
from sklearn import metrics
import cv2
import sys
sys.path.insert(0, 'C:/Users/urvas/Desktop/Avermass-Internship/django-app-3/forgery/E2E')

def upload_file(request):
    results = ""
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            newdoc = Document(docfile = request.FILES['docfile'])
            newdoc.save()
            filename = newdoc.docfile.name
            results = detect_forgery(filename = filename)
            # results['img_path_result'] = img_path_result
            # results['label'] = 'Forged' if is_pair_found else 'Not Forged'
    else:
        form = DocumentForm()
        return render(request, 'detectApp/upload_file.html', {'form': form})

    return render(request, 'detectApp/upload_file.html', {'results': results})

# Create your views here.
def process_image(img_path, parameters):
    # print('Processing image: ',img_path)
    score = np.nan
    try:
        X, RGB, NP, RGN,im_mode = loader(img_path, parameters.mode)
    except Exception as e:
        print("Error in opening image file: ",img_path)
        return score

    if np.min(X.shape[0:2])< (parameters.tile_size+parameters.tile_stride):
        print('Image is too small:'+ img_path)
    else:
        try:
            score = detection(X, RGB, NP, RGN, parameters.mode)
            print('> Score: {}'.format(score))
            # Update Output
        except Exception as e:
            print("Error in processing the image: ",img_path)
            raise e

    return score

def detect_forgery(filename):
    import argparse
    import glob
    parser = argparse.ArgumentParser()
    #parser.add_argument('-g', '--gpu'   , type=str, default=None)
    parser.add_argument('-m','--mode', type=str, default='FUSION') #RGB | N | RGN | FUSION
    parser.add_argument('-tile_size','--tile_size', type=int, default=256)
    parser.add_argument('-tile_stride','--tile_stride', type=int, default=192)
    parser.add_argument('-train_dataset','--train_dataset', type=str, default='E2E')


    config, _ = parser.parse_known_args()
    parameters.use_cuda = torch.cuda.is_available() # To run on CPU when GPUs are not available
    parameters.mode = config.mode
    parameters.tile_size = config.tile_size
    parameters.tile_stride = config.tile_stride
    parameters.ds = config.train_dataset
    preload(parameters.mode)


    images_path = glob.glob('media/' + filename)
    result = ""

    for img_p in images_path:
        s = process_image(img_path=img_p,parameters=parameters)
        print(f"Model prediction value: {s}")
        if s > 0.5:
            img = cv2.imread(img_p)
            result = "Forged" 
            cv2.putText(img, "Forged Image",(100, 100), cv2.FONT_HERSHEY_SIMPLEX, 10,(0, 0, 255), 2)
            img = cv2.resize(img, (640, 640))
            cv2.imshow("image", img)
            key = cv2.waitKey()
            if key == 27:
                break
        else:
            img = cv2.imread(img_p)
            result = "Not Forged"
            cv2.putText(img, "Not Forged",(100, 100), cv2.FONT_HERSHEY_SIMPLEX, 10,(0, 0, 255), 2)
            img = cv2.resize(img, (640, 640))
            cv2.imshow("image", img)
            key = cv2.waitKey()
            if key == 27:
                break

    return result