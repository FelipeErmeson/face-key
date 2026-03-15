import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from insightface.utils import face_align

# app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# app.prepare(ctx_id=0, det_size=(640,640))

class RecFace():

    def __init__(self, app, img):
        self.app = app
        self.img = img
        self.quant_faces = None
        self.faces = None
        self.dados = []

    # def desenhar_bbox(self):
    #     rimg = app.draw_on(self.img, faces)
    #     cv2.imwrite('./t1_output.jpg', rimg)

    def create_embedding(self):
        self.faces = self.app.get(self.img)
        self.quant_faces = len(self.faces)

    def create_img_with_bbox(self):
        img_with_faces = self.app.draw_on(self.img, self.faces)
        # output_filename = "face_det_output.jpg"
        # cv2.imwrite(output_filename, img_with_faces)
        return img_with_faces

    def get_info(self):
        for face in self.faces:
            dado = {}
            dado['bbox'] = face.bbox
            dado['landmark'] = face.kps
            dado['confidence_det'] = face.det_score
            dado['normed_emb'] = face.normed_embedding
            dado['value_l2'] = face.embedding_norm
            dado['sex'] = face.sex
            dado['age'] = face.age
            aligned_face = face_align.norm_crop(self.img, landmark=face.kps)
            dado['aligned_face'] = aligned_face
            self.dados.append(dado)
            # cv2.imwrite("rosto_alinhado.jpg", aligned_face)

        return self.dados

        