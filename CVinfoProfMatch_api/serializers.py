from rest_framework.serializers import Serializer, FileField

class UploadSerializer(Serializer):
    file = FileField()
    class Meta:
        fields = ['file']