from django.shortcuts import render

# Create your views here.

from rest_framework.viewsets import ViewSet
from rest_framework.response import Response
from .serializers import UploadSerializer
from .profmatch import *
from django.core.files.storage import DefaultStorage, default_storage
from django.core.files.base import ContentFile
from CVinfoProfMatch import settings
import os


class UploadViewSet(ViewSet):
    serializer_class = UploadSerializer

    def list(self, request):
        return Response("GET API")

    def create(self, request):
        file = request.FILES['file']
        file_name = default_storage.save(file.name, file)
        file_path = os.path.join(settings.MEDIA_ROOT, file_name)
        response = cv_profiling(file_path)
        return Response(response)
