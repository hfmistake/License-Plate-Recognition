import time
import cv2
import numpy as np
import torch
import uuid as uid
from PIL import Image
import pytesseract
import easyocr


def pre_process_plate_image(plate_image):
    """
    Pré-processa a imagem da placa para melhorar a leitura do OCR

    :param plate_image: Imagem da placa
    :return: Retorna a imagem pre-processada
    """

    plate_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    plate_image = cv2.threshold(plate_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    plate_image = cv2.medianBlur(plate_image, 3)
    plate_image = cv2.resize(plate_image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return plate_image


def tesseract_read(plate_image):
    """
    Realiza a leitura da placa utilizando o Tesseract OCR

    :param plate_image: Imagem da placa
    :return: Retorna o texto extraído da placa
    """
    return pytesseract.image_to_string(plate_image,
                                       config="--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


def easy_ocr_read(plate_image):
    """
    Realiza a leitura da placa utilizando o EasyOCR

    :param plate_image: Imagem da placa
    :return: Retorna o texto extraído da placa
    """
    reader = easyocr.Reader(['en'], gpu=True)
    return reader.readtext(plate_image, detail=0, paragraph=True, batch_size=1, workers=0,
                           allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


def handle_ocr(plate_image, pre_processing=False):
    """
    Gerencia a leitura da placa utilizando o Tesseract OCR ou EasyOCR com opção de pré-processamento da imagem

    :param plate_image: Imagem da placa no formato PIL
    :param pre_processing: Opção de pré-processamento da imagem
    :return: Retorna o texto extraído da placa
    """
    plate_image = np.array(plate_image)
    if pre_processing:
        plate_image = pre_process_plate_image(plate_image)
    cv2.imshow("Plate", plate_image)
    result = tesseract_read(plate_image)
    if result:
        return result


class Prediction:
    """
    Classe responsável por realizar a detecção de veículos e captura de placas
    """

    def __init__(self, plate_model, pre_treined_model, data: dict):
        self.pre_trained_model = pre_treined_model
        self.data = data
        self.id_blacklist = set()
        self.last_capture = 0
        self.plate_model = plate_model

    @staticmethod
    def check_gpu():
        """
        Verifica se a GPU está disponível e fornece informações sobre a GPU

        :return: None
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
            print(f"Starting with GPU: {gpu_name}")
        else:
            print("GPU not available. Starting with CPU.")

    def validate_collision(self, center_x, center_y, track_id):
        """
        Valida se o objeto colidiu com a linha de captura e não está na lista negra

        :param center_x: Centro do objeto no eixo x
        :param center_y: Centro do objeto no eixo y
        :param track_id: Id de rastreamento do objeto
        :return: Retorna True se o objeto colidir com a linha de captura
        """
        entry = self.data["entry"]
        line_x1, line_y1, line_x2, line_y2 = self.data["line"]
        if center_y < line_y1 - 50 and entry:
            self.id_blacklist.add(track_id)
        if center_y > line_y1 + 50 and not entry:
            self.id_blacklist.add(track_id)
        is_inside_line = line_x1 < center_x < line_x2 and line_y1 - 15 < center_y < line_y1 + 15
        if is_inside_line and track_id not in self.id_blacklist:
            return True

        return False

    def handle_capture(self, original_frame, box):
        """
        Verifica se uma captura não foi realizada recentemente para evitar capturas duplicadas

        :param original_frame:  Frame original
        :param box: Caixa delimitadora do objeto
        :return: None
        """
        if time.time() - self.last_capture > 2:
            self.draw_capture(original_frame, box)
            self.last_capture = time.time()

    def draw_line(self, frame):
        """
        Desenha a linha de captura no frame
        :param frame: Frame original
        :return: None
        """
        cv2.line(frame, (self.data["line"][0], self.data["line"][1]),
                 (self.data["line"][2], self.data["line"][3]),
                 (88, 85, 232), 4)

    def is_plate_inside_vehicle(self, plate_box, vehicle_box):
        """
        Verifica se a placa está dentro da caixa delimitadora do veículo

        :param plate_box: Caixa delimitadora da placa
        :param vehicle_box: Caixa delimitadora do veículo
        :return: Retorna True se a placa estiver dentro da caixa delimitadora do veículo
        """
        plate_center_x, plate_center_y = self.get_center_point(plate_box)
        x1, y1, x2, y2 = self.get_coords(vehicle_box)
        return x1 < plate_center_x < x2 and y1 < plate_center_y < y2

    @staticmethod
    def draw_center_point(frame, center_x, center_y):
        """
        Desenha o ponto central do objeto no frame

        :param frame: Frame original
        :param center_x: Centro do objeto no eixo x
        :param center_y: Centro do objeto no eixo y
        :return: None
        """
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

    def draw_capture(self, original_frame, box):
        """
        Desenha os elementos visuais quando uma captura é realizada

        :param original_frame: Frame original
        :param box: Caixa delimitadora do objeto
        :return: None
        """
        vehicle_type = self.pre_trained_model.names[int(box.cls[0])]
        plate_results = self.plate_model(original_frame)
        for plate_result in plate_results:
            boxes = plate_result.boxes
            for plate_box in boxes:
                if self.is_plate_inside_vehicle(plate_box, box):
                    plate_x1, plate_y1, plate_x2, plate_y2 = self.get_coords(plate_box)
                    plate_image = Image.fromarray(plate_result.orig_img[plate_y1:plate_y2, plate_x1:plate_x2])
                    plate_text = handle_ocr(plate_image, pre_processing=True)
                    if not plate_text:
                        plate_text = "Placa não identificada"
                    cv2.putText(original_frame, f"Placa: {plate_text} Vehicle type: {vehicle_type}", (0, 200),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2)
        cv2.imshow("Capture", original_frame)

    @staticmethod
    def get_coords(box):
        """
        Retorna as coordenadas da caixa delimitadora

        :param box: Caixa delimitadora
        :return: Retorna as coordenadas da caixa delimitadora
        """
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        return x1, y1, x2, y2

    def get_center_point(self, box):
        """
        Retorna o ponto central do eixo x e y da caixa delimitadora

        :param box: Caixa delimitadora
        :return: Retorna o ponto central do eixo x e y da caixa delimitadora
        """
        x1, y1, x2, y2 = self.get_coords(box)
        width, height = x2 - x1, y2 - y1
        center_x, center_y = x1 + width // 2, y1 + height // 2
        return center_x, center_y

    def _process_frame(self, frame):
        """
        Processa o frame com utilizando o model para realizar a detecção de objetos

        :param frame: Frame original
        :return: Retorna o frame com os elementos visuais da detecção de objetos
        """
        results = self.pre_trained_model.track(frame, classes=self.data["object_indices"], conf=0.6, iou=0.5,
                                               persist=True)
        return self.draw_visualization_elements(results[0], results[0].plot())

    def check_collision(self, results, original_frame):
        """
        Verifica se houve colisão com a linha de captura e chama o gerenciador de captura

        :param results: Resultados da detecção de objetos
        :param original_frame: Frame original
        :return: None
        """
        print(self.id_blacklist)
        boxes = results.boxes
        for box in boxes:
            center_x, center_y = self.get_center_point(box)
            self.draw_center_point(original_frame, center_x, center_y)
            if box.id is not None and self.validate_collision(center_x, center_y, int(box.id.item())):
                self.handle_capture(original_frame, box)

    def draw_visualization_elements(self, results, original_frame):
        """
        Responsável por todos os elementos visuais da detecção de objetos e eventos de captura

        :param results: Resultados da detecção de objetos
        :param original_frame: Frame original
        :return:  Retorna o frame com os elementos visuais da detecção de objetos e eventos de captura
        """
        self.draw_line(original_frame)
        self.check_collision(results, original_frame)
        return original_frame

    def visualize(self):
        """
        Inicia a visualização do vídeo com a detecção de objetos

        :return: None
        """
        Prediction.check_gpu()
        capture = cv2.VideoCapture(self.data["video_source"])
        paused = False
        while capture.isOpened():
            if not paused:
                success, frame = capture.read()
                if not success:
                    break
                annotated_frame = self._process_frame(frame)
                cv2.imshow("YOLOv8 Inference", annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p"):
                paused = not paused
        capture.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def simple_track_and_capture(self):
        """
        Realiza a detecção de objetos e captura a imagem em um determinado diretório

        :return: None
        """
        Prediction.check_gpu()
        directory = self.data["directory"]
        object_indices = self.data["object_indices"]
        results = self.pre_trained_model.track(self.data["video_source"], stream=True, iou=0.5, classes=object_indices,
                                               conf=0.5)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width, height = x2 - x1, y2 - y1
                center_x, center_y = x1 + width // 2, y1 + height // 2
                if box.id is not None and self.validate_collision(center_x, center_y, int(box.id.item())):
                    im = Image.fromarray(result.orig_img)
                    plate_image = Image.fromarray(result.orig_img[y1:y2, x1:x2])
                    plate_text = handle_ocr(plate_image)
                    if plate_text:
                        print(f"Placa: {plate_text}")
                    im.save(f'{directory}/{uid.uuid4()}.jpg')
                    print("captura realizada")
