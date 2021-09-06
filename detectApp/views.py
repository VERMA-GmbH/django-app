from django.shortcuts import render
from rest_framework.views import APIView
from .models import Document
from .forms import DocumentForm
import numpy as np
import forgery.E2E.parameters as parameters
from forgery.E2E.detection import detection,preload
from forgery.E2E.dataloaders.data_loader import loader
import torch
from django.contrib.auth.models import User, Group
from rest_framework import viewsets
from rest_framework import permissions
from .serializers import UserSerializer, GroupSerializer
from rest_framework.views import APIView

import sys
sys.path.insert(0, 'C:/Users/urvas/Desktop/Avermass-Internship/django-app-3/forgery/E2E')

class UserViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = User.objects.all().order_by('-date_joined')
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]


class GroupViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows groups to be viewed or edited.
    """
    queryset = Group.objects.all()
    serializer_class = GroupSerializer
    permission_classes = [permissions.IsAuthenticated]

class UPLOADFILE(APIView):

    def upload_file(self, request):
        results = ""
        if request.method == 'POST':
            form = DocumentForm(request.POST, request.FILES)
            if form.is_valid():
                newdoc = Document(docfile = request.FILES['docfile'])
                newdoc.save()
                filename = newdoc.docfile.name
                results = self.detect_forgery(filename = filename)
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

    def detect_forgery(self, filename):
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
            s = self.process_image(img_path=img_p,parameters=parameters)
            print(f"Model prediction value: {s}")
            if s > 0.5:
                result = "Forged" 
            else:
                result = "Not Forged"

        return result