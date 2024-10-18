import numpy as np
import cv2 as cv
from tensorflow.keras.models import load_model
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# teachablemachine.withgoogle.com

model = load_model('keras_model.h5', compile=False)

moedas = [
    "1 real",
    "50 centavos",
    "25 antiga",
    "25 nova",
    "10 antiga",
    "10 nova",
    "5 antiga",
    "5 nova"
]

def main():
    video = cv.VideoCapture(1)

    while True:
        imagemOriginal = video.read()[1]
        imagemProcessada = processarImagem(imagemOriginal)
        contornos = pegarContornos(imagemProcessada)

        qtd1real = 0
        qtd50centavos = 0
        qtd25centavos = 0
        qtd10centavos = 0
        qtd5centavos = 0

        for contorno in contornos:
            x, y, w, h = cv.boundingRect(contorno)

            area = cv.contourArea(contorno)

            if h / w < 0.4 or h / w > 1.4:
                continue

            if area < 6000:
                continue

            recorte = imagemOriginal.copy()[y: y + h, x: x + w]

            moeda, confianca = detectarMoeda(recorte)

            if confianca < 0.5:
                continue

            cv.rectangle(imagemOriginal, (x, y), (x + w, y + h), (255, 255, 0), 2)

            cv.putText(imagemOriginal, str(moeda), (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv.putText(imagemOriginal, f'{moeda} ({confianca:.2f})', (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if moeda == moedas[0]:
                qtd1real += 1

            if moeda == moedas[1]:
                qtd50centavos += 1

            if moeda in [moedas[2], moedas[3]]:
                qtd25centavos += 1

            if moeda in [moedas[4], moedas[5]]:
                qtd10centavos += 1

            if moeda in [moedas[6], moedas[7]]:
                qtd5centavos += 1

        soma = qtd1real + qtd50centavos * 0.5 + qtd25centavos * 0.25 + qtd10centavos * 0.1 + qtd5centavos * 0.05
        somaTexto = "{:10.2f}".format(soma)

        cv.putText(imagemOriginal, "soma:" + somaTexto, (440, 47), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv.imshow('Principal', imagemOriginal)
        #cv.imshow('Principal', imagemProcessada)

        if cv.waitKey(1) == ord('q'):
            break


def processarImagem(imagem: cv.typing.MatLike) -> cv.typing.MatLike:
    kernel = np.ones((4, 4), np.uint8)
    imagemProcessada = imagem.copy()
    imagemProcessada = cv.cvtColor(imagemProcessada, cv.COLOR_BGR2GRAY)
    imagemProcessada = cv.GaussianBlur(imagemProcessada, (19, 19), 3)
    imagemProcessada = cv.dilate(imagemProcessada, kernel, iterations=2)
    imagemProcessada = cv.erode(imagemProcessada, kernel, iterations=1)
    imagemProcessada = cv.Canny(imagemProcessada, 90, 140)
    return imagemProcessada


def pegarContornos(imagem: cv.typing.MatLike):
    contornos = cv.findContours(imagem, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    return contornos


def detectarMoeda(imagem: cv.typing.MatLike):
    data = np.ndarray((1, 224, 224, 3), np.float32)
    imagemMoeda = cv.resize(imagem, (224, 224))
    arrayMoeda = np.asarray(imagemMoeda)
    imgMoedaNormalize = (arrayMoeda.astype(np.float32) / 127.0) - 1
    data[0] = imgMoedaNormalize
    predicao = model.predict(data)
    index = np.argmax(predicao)
    confianca = predicao[0][index]
    moeda = moedas[index]
    return moeda, confianca

if __name__ == "__main__":
    main()