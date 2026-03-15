import cv2

class FileReadException(Exception):
    def __init__(self, mensagem):
        super().__init__(mensagem)

class FileConvertColorException(Exception):
    def __init__(self, mensagem):
        super().__init__(mensagem)

def read_img(img_path):
    try:
        imagem = cv2.imread(str(img_path))
    except Exception as ex:
        raise FileReadException("Erro ao tentar ler imagem com OpenCV.")

    return imagem

def converter_canal(img_orig, transformacao=cv2.COLOR_BGR2RGB):
    try:
        new_img = cv2.cvtColor(img_orig, transformacao)
    except Exception as ex:
        raise FileConvertColorException("Erro ao tentar alterar canal da imagem.")
    return new_img