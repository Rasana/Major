import numpy as np
import cv2
import copy
maiorArea = 0
cap = cv2.VideoCapture('Disparity.avi')

while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow("Video", frame)
    bkg= copy.copy(frame)
    fundo = cv2.GaussianBlur(bkg, (3,3), 0)
    print("OK")
    if cv2.waitKey(1) == 32:
        cv2.destroyWindow("Video")
        break
    while True:
        ret, imagem = cap.read()
        mascara = copy.copy(imagem)
        cinza = copy.copy(imagem)

        imagem = cv2.GaussianBlur(imagem, (3,3), 0)
        cv2.absdiff(imagem, fundo, mascara)
        gray = cv2.cvtColor(mascara, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(thresh1, kernel, iterations = 18)
        cinza = cv2.erode(dilated, kernel, iterations = 10)
        contorno, heir = cv2.findContours(cinza,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE,None,None,None)

        for cnt in contorno:
            vertices_do_retangulo = cv2.boundingRect(cnt)
            if (cv2.contourArea(cnt) >  maiorArea):
                maiorArea = cv2.contourArea(cnt)
                retangulo_de_interesse = vertices_do_retangulo

            ponto1 = (retangulo_de_interesse[0], retangulo_de_interesse[1])
            ponto2 = (retangulo_de_interesse[0] + retangulo_de_interesse[2], retangulo_de_interesse[1] + retangulo_de_interesse[3])
            cv2.rectangle(imagem, ponto1, ponto2,(0, 0, 0), 2)
            cv2.rectangle(cinza, ponto1, ponto2, (255, 255, 255), 1)
            largura = ponto2[0] - ponto1[0]
            altura = ponto2[1] - ponto1[1]
            cv2.line(cinza, (ponto1[0]+largura/2, ponto1[1]), (ponto1[0]+largura/2, ponto2[1]), (255, 255, 255), 1)
            cv2.line(cinza, (ponto1[0], ponto1[1]+altura/2), (ponto2[0], ponto1[1]+altura/2), (255, 255, 255), 1)

        #cv2.imshow("Mascara", mascara)
        cv2.imshow("Cinza", cinza)
        cv2.imshow("Video", imagem)
        #cv2.imshow("Dilated", thresh1)
        cv2.waitKey(30)



# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()
