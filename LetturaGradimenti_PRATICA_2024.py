# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 13:36:34 2022

@author: Pavan Davide
"""
#import sys
#sys.modules[__name__].__dict__.clear()
import sys
import os
import numpy as np
import cv2
from tkinter.filedialog import askopenfilename

# Salvo le immagini con le risposte riconosciute?
IMGsave_opt=True

#start_clock = datetime.datetime.now()

####################### FUNZIONI #########################

def PDFimg_read(IMGfile,CreateFolder):
    import fitz # PyMuPDF
    import numpy as np
    import cv2
    import sys
    import os
    
    
    IMGpath, IMGfilename = os.path.split(os.path.abspath(IMGfile))
    SaveDir=IMGpath+'\\'+IMGfilename[:-4]+'_Result'
    
    if CreateFolder:
        # if does not exist subfolder, then create it
        os.makedirs(SaveDir, exist_ok=True)
    
    
    
    # open the file
    pdf_file = fitz.open(IMGfile)
    
    # define erosion kernel
    kernel = np.ones((3, 3), np.uint8)
    
    # define images dimension
    IMGdim=(1230,1740)
    
    # define return varibles
    IMGgrayTOT =  np.zeros([IMGdim[1], IMGdim[0], 1], dtype=np.uint8)
    IMGbinTOT  =  np.zeros([IMGdim[1], IMGdim[0], 1], dtype=np.uint8)
    IMGerodeTOT = np.zeros([IMGdim[1], IMGdim[0], 1], dtype=np.uint8)
    
    # iterate over PDF pages
    
    for page_index in range(len(pdf_file)):
        # get the page itself
        page = pdf_file[page_index]
        # get image list
        image_list = page.get_images()
        # printing number of images found in this page
        if len(image_list)==1:
            print(f"[OK] Caricamento immagine {page_index+1}...")
        else:
            if len(image_list)>1:
                print(f"[WARNING!!!] Found a total of {len(image_list)} images in page {page_index+1}")
                sys.exit()
            else:
                print("[!] No images found on page", page_index)
                
        for image_index, img in enumerate(image_list, start=1):
            # get the XREF of the image
            xref = img[0]
            # extract the image bytes
            image_dict = pdf_file.extract_image(xref)
            
            # get the image extension
            image_ext = image_dict["ext"]
            
            IMGgray_TMP = cv2.imdecode(np.frombuffer(image_dict["image"],
                                   np.dtype(f'u{image_dict["bpc"] // 8}')
                                   ),
                     cv2.IMREAD_GRAYSCALE)
            
            #images will ALWAYS have same size
            IMGgray=cv2.resize(IMGgray_TMP,IMGdim, interpolation = cv2.INTER_CUBIC)
            
            del IMGgray_TMP
            
            IMGbin = cv2.adaptiveThreshold(IMGgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,9,60)
            
            # IMGdilat = cv2.dilate(IMGbin, kernel, iterations=2)
            IMGerode = cv2.erode(IMGbin, kernel, iterations=2)
            
            
            # fill return variables
            if np.mean(IMGgrayTOT) == 0:
                IMGgrayTOT[:,:,0]=IMGgray
            else:
               IMGgrayTOT = np.dstack((IMGgrayTOT,IMGgray))
               
            if np.mean(IMGbinTOT) == 0:
                IMGbinTOT[:,:,0]=IMGbin
            else:
               IMGbinTOT = np.dstack((IMGbinTOT,IMGbin))
               
            if np.mean(IMGerodeTOT) == 0:
                IMGerodeTOT[:,:,0]=IMGerode
            else:
               IMGerodeTOT = np.dstack((IMGerodeTOT,IMGerode))
               
            # cv2.imwrite(SaveDir+f"\\imageGRAY_{page_index+1}.{image_ext}", IMGgray)
            # cv2.imwrite(SaveDir+f"\\imageBIN_{page_index+1}.{image_ext}", IMGbin)
            # cv2.imwrite(SaveDir+f"\\imageERODE_{page_index+1}.{image_ext}", IMGerode)
            # cv2.imwrite(SaveDir+f"\\imageDILATE_{page_index+1}.{image_ext}", IMGdilat)
            
    return IMGgrayTOT, IMGbinTOT, IMGerodeTOT, SaveDir, IMGfilename#, CUTsize


def registration(img2register1, img2register2, img2register3, REFimg):
    import cv2
    import numpy as np
    
    height, width = REFimg.shape
    
    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(5000)
    
    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    # (which is not required in this case).
    kp1, d1 = orb_detector.detectAndCompute(img2register1, None)
    kp2, d2 = orb_detector.detectAndCompute(REFimg, None)
    
    # Match features between the two images.
    # We create a Brute Force matcher with
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    
    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)
    matches = list(matches)
    
    # Sort matches on the basis of their Hamming distance.
    matches.sort(key = lambda x: x.distance)
    matches = tuple(matches)
    
    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches)*0.9)]
    no_of_matches = len(matches)
    
    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))
    
    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt
    
    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
    
    # Use this matrix to transform the
    # colored image wrt the reference image.
    transformed_img1 = cv2.warpPerspective(img2register1,
    					homography, (width, height))
    transformed_img2 = cv2.warpPerspective(img2register2,
    					homography, (width, height))
    transformed_img3 = cv2.warpPerspective(img2register3,
    					homography, (width, height))
    
    return transformed_img1, transformed_img2, transformed_img3


def question_cut_6(img2cut,Q_num):
    import numpy as np
    
    Q_coord =  np.array([[437,480,430,930],#coordinate quadratini domande
                         [553,596,430,930],
                         [662,705,430,930],
                         [770,813,430,930],
                         [875,918,430,930],
                         [928,971,430,930],
                         [981,1023,430,930],
                         [1033,1076,430,930],
                         [1139,1182,430,930],
                         [1246,1289,430,930],
                         [1353,1396,430,930],
                         [1500,1526,183,420],   # ALIS
                         [1569,1595,183,420],   # ALIS2
                         [1500,1526,448,690],   # ERGO-POSTURAL
                         [1569,1595,448,690],   # ERGO-THERAPY
                         [1500,1526,724,940],   # NIGHT_LIFE
                         [1569,1595,724,940],   # LCC
                         [1500,1526,966,1150],  # SMART_BRAIN
                         [1569,1595,966,1150]]) # STRESSLESS
    
    Question = img2cut[Q_coord[Q_num,0]:Q_coord[Q_num,1],Q_coord[Q_num,2]:Q_coord[Q_num,3]]
    
    return Question

def blobDetect_6(Question):
    import numpy as np
    Question = Question!=0
    
    Where=np.argwhere(Question)
    Answer_coord=np.median(Where[:,1])+430
    
    if Answer_coord>=430 and Answer_coord<=589:
        Answer=1
    elif Answer_coord>=590 and Answer_coord<=626:
        Answer=2
    elif Answer_coord>=627 and Answer_coord<=664:
        Answer=3
    elif Answer_coord>=665 and Answer_coord<=701:
        Answer=4
    elif Answer_coord>=702 and Answer_coord<=738:
        Answer=5
    elif Answer_coord>=739 and Answer_coord<=930:
        Answer=6
    else:
        Answer=float("NaN")
    
    
    return Answer_coord, Answer

def blobDetect_1(Question):
    import numpy as np
    Question = Question!=0
    
    Where=np.argwhere(Question)
    Answer_coord=np.median(Where[:,1])
    
    if np.isnan(Answer_coord):
        Answer=float("NaN")
    else:
        Answer=1
    
    
    return Answer_coord, Answer
    
    
def sameShape(img1, img2):
    import numpy as np
    
    if np.shape(img1)[0]<np.shape(img2)[0]:
        img1=np.vstack((img1,np.ones([np.shape(img2)[0]-np.shape(img1)[0],np.shape(img1)[1],np.shape(img1)[2]], dtype=np.uint8)*255))
        
    elif np.shape(img1)[0]>np.shape(img2)[0]:
        img1=img1[:np.shape(img2)[0],:,:]

    if np.shape(img1)[1]<np.shape(img2)[1]:
        img1=np.hstack((img1,np.ones([np.shape(np.shape(img1)[0],img2)[1]-np.shape(img1)[1],np.shape(img1)[2]], dtype=np.uint8)*255))
        
    elif np.shape(img1)[1]>np.shape(img2)[1]:
        img1=img1[:,:np.shape(img2)[1],:]
    
    return img1, img2

def writeCircles(imgIN,Answers):
    import numpy as np
    import cv2
    
    imgIN_RGB = cv2.cvtColor(imgIN,cv2.COLOR_GRAY2RGB)
    
    Ans_coord_vert_6 = np.array([459,575,684,792,897,950,1002,1055,1161,1268,1375])#coordinate verticali domande a 6 risposte
    Ans_coord_horz_6 = np.array([571,608,646,683,720,758])#coordinate risposte 1-6
    
    Ans_coord_vert_1 = np.array([1513,  # ALIS     #coordinate VERTICALI domande a 1 risposta
                                 1582,  # ALIS2
                                 1513,  # ERGO-POSTURAL
                                 1582,  # ERGO-THERAPY
                                 1513,  # NIGHT_LIFE
                                 1582,  # LCC
                                 1513,  # SMART_BRAIN
                                 1582]) # STRESSLESS
    
    Ans_coord_horz_1 = np.array([196,   # ALIS     #coordinate ORIZZONTALI domande a 1 risposta
                                 196,   # ALIS2
                                 461,   # ERGO-POSTURAL
                                 461,   # ERGO-THERAPY
                                 737,   # NIGHT_LIFE
                                 737,   # LCC
                                 979,   # SMART_BRAIN
                                 979])  # STRESSLESS
    
    # Radius of circle
    radius = 20
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2
    
    for q in range(19):
        
        # Center coordinates
        if np.isnan(Answers[q]):
            print(f"Risposta {q+1} non riconosciuta, viene lasciata vuota...")
        else:
            if q < Ans_coord_vert_6.size:
                center_coordinates = (np.int64(Ans_coord_horz_6[np.int64(Answers[q])-1]), np.int64(Ans_coord_vert_6[q]))
            else:
                center_coordinates = (np.int64(Ans_coord_horz_1[q-Ans_coord_vert_6.size]), np.int64(Ans_coord_vert_1[q-Ans_coord_vert_6.size]))
            
            imgIN_RGB = cv2.circle(imgIN_RGB, center_coordinates, radius, color, thickness)
    return imgIN_RGB


def manualAnswer(Manual_coord):
    
    Quest_coord=Manual_coord[1]
    Answer_coord=Manual_coord[0]
        
    if Quest_coord>437 and Quest_coord<480:
        qN=0
        if Answer_coord>=430 and Answer_coord<=589:
            Answer=1
        elif Answer_coord>=590 and Answer_coord<=626:
            Answer=2
        elif Answer_coord>=627 and Answer_coord<=664:
            Answer=3
        elif Answer_coord>=665 and Answer_coord<=701:
            Answer=4
        elif Answer_coord>=702 and Answer_coord<=738:
            Answer=5
        elif Answer_coord>=739 and Answer_coord<=930:
            Answer=6
        else:
            Answer=float("NaN")
    elif Quest_coord>553 and Quest_coord<596:
        qN=1
        if Answer_coord>=430 and Answer_coord<=589:
            Answer=1
        elif Answer_coord>=590 and Answer_coord<=626:
            Answer=2
        elif Answer_coord>=627 and Answer_coord<=664:
            Answer=3
        elif Answer_coord>=665 and Answer_coord<=701:
            Answer=4
        elif Answer_coord>=702 and Answer_coord<=738:
            Answer=5
        elif Answer_coord>=739 and Answer_coord<=930:
            Answer=6
        else:
            Answer=float("NaN")
    elif Quest_coord>662 and Quest_coord<705:
        qN=2
        if Answer_coord>=430 and Answer_coord<=589:
            Answer=1
        elif Answer_coord>=590 and Answer_coord<=626:
            Answer=2
        elif Answer_coord>=627 and Answer_coord<=664:
            Answer=3
        elif Answer_coord>=665 and Answer_coord<=701:
            Answer=4
        elif Answer_coord>=702 and Answer_coord<=738:
            Answer=5
        elif Answer_coord>=739 and Answer_coord<=930:
            Answer=6
        else:
            Answer=float("NaN")
    elif Quest_coord>770 and Quest_coord<813:
        qN=3
        if Answer_coord>=430 and Answer_coord<=589:
            Answer=1
        elif Answer_coord>=590 and Answer_coord<=626:
            Answer=2
        elif Answer_coord>=627 and Answer_coord<=664:
            Answer=3
        elif Answer_coord>=665 and Answer_coord<=701:
            Answer=4
        elif Answer_coord>=702 and Answer_coord<=738:
            Answer=5
        elif Answer_coord>=739 and Answer_coord<=930:
            Answer=6
        else:
            Answer=float("NaN")
    elif Quest_coord>875 and Quest_coord<918:
        qN=4
        if Answer_coord>=430 and Answer_coord<=589:
            Answer=1
        elif Answer_coord>=590 and Answer_coord<=626:
            Answer=2
        elif Answer_coord>=627 and Answer_coord<=664:
            Answer=3
        elif Answer_coord>=665 and Answer_coord<=701:
            Answer=4
        elif Answer_coord>=702 and Answer_coord<=738:
            Answer=5
        elif Answer_coord>=739 and Answer_coord<=930:
            Answer=6
        else:
            Answer=float("NaN")
    elif Quest_coord>928 and Quest_coord<971:
        qN=5
        if Answer_coord>=430 and Answer_coord<=589:
            Answer=1
        elif Answer_coord>=590 and Answer_coord<=626:
            Answer=2
        elif Answer_coord>=627 and Answer_coord<=664:
            Answer=3
        elif Answer_coord>=665 and Answer_coord<=701:
            Answer=4
        elif Answer_coord>=702 and Answer_coord<=738:
            Answer=5
        elif Answer_coord>=739 and Answer_coord<=930:
            Answer=6
        else:
            Answer=float("NaN")
    elif Quest_coord>981 and Quest_coord<1023:
        qN=6
        if Answer_coord>=430 and Answer_coord<=589:
            Answer=1
        elif Answer_coord>=590 and Answer_coord<=626:
            Answer=2
        elif Answer_coord>=627 and Answer_coord<=664:
            Answer=3
        elif Answer_coord>=665 and Answer_coord<=701:
            Answer=4
        elif Answer_coord>=702 and Answer_coord<=738:
            Answer=5
        elif Answer_coord>=739 and Answer_coord<=930:
            Answer=6
        else:
            Answer=float("NaN")
    elif Quest_coord>1033 and Quest_coord<1076:
        qN=7
        if Answer_coord>=430 and Answer_coord<=589:
            Answer=1
        elif Answer_coord>=590 and Answer_coord<=626:
            Answer=2
        elif Answer_coord>=627 and Answer_coord<=664:
            Answer=3
        elif Answer_coord>=665 and Answer_coord<=701:
            Answer=4
        elif Answer_coord>=702 and Answer_coord<=738:
            Answer=5
        elif Answer_coord>=739 and Answer_coord<=930:
            Answer=6
        else:
            Answer=float("NaN")
    elif Quest_coord>1139 and Quest_coord<1182:
        qN=8
        if Answer_coord>=430 and Answer_coord<=589:
            Answer=1
        elif Answer_coord>=590 and Answer_coord<=626:
            Answer=2
        elif Answer_coord>=627 and Answer_coord<=664:
            Answer=3
        elif Answer_coord>=665 and Answer_coord<=701:
            Answer=4
        elif Answer_coord>=702 and Answer_coord<=738:
            Answer=5
        elif Answer_coord>=739 and Answer_coord<=930:
            Answer=6
        else:
            Answer=float("NaN")
    elif Quest_coord>1246 and Quest_coord<1289:
        qN=9
        if Answer_coord>=430 and Answer_coord<=589:
            Answer=1
        elif Answer_coord>=590 and Answer_coord<=626:
            Answer=2
        elif Answer_coord>=627 and Answer_coord<=664:
            Answer=3
        elif Answer_coord>=665 and Answer_coord<=701:
            Answer=4
        elif Answer_coord>=702 and Answer_coord<=738:
            Answer=5
        elif Answer_coord>=739 and Answer_coord<=930:
            Answer=6
        else:
            Answer=float("NaN")
    elif Quest_coord>1353 and Quest_coord<1396:
        qN=10
        if Answer_coord>=430 and Answer_coord<=589:
            Answer=1
        elif Answer_coord>=590 and Answer_coord<=626:
            Answer=2
        elif Answer_coord>=627 and Answer_coord<=664:
            Answer=3
        elif Answer_coord>=665 and Answer_coord<=701:
            Answer=4
        elif Answer_coord>=702 and Answer_coord<=738:
            Answer=5
        elif Answer_coord>=739 and Answer_coord<=930:
            Answer=6
        else:
            Answer=float("NaN")
            
            
    ######## risposte singole per i corsi ############
    elif Quest_coord>1500 and Quest_coord<1526:
        
        if Answer_coord>=183 and Answer_coord<=420:
            qN=11 #ALIS
            Answer=1
        elif Answer_coord>=448 and Answer_coord<=690:
            qN=13 #EP
            Answer=1
        elif Answer_coord>=724 and Answer_coord<=940:
            qN=15 #NL
            Answer=1
        elif Answer_coord>=966 and Answer_coord<=1150:
            qN=17 #SB
            Answer=1
        else:
            Answer=float("NaN")
        
    elif Quest_coord>1569 and Quest_coord<1595:
        
        if Answer_coord>=183 and Answer_coord<=420:
            qN=12 #ALIS2
            Answer=1
        elif Answer_coord>=448 and Answer_coord<=690:
            qN=14 #ET
            Answer=1
        elif Answer_coord>=724 and Answer_coord<=940:
            qN=16 #LCC
            Answer=1
        elif Answer_coord>=966 and Answer_coord<=1150:
            qN=18 #SL
            Answer=1
        else:
            Answer=float("NaN")
            
            
    
    return qN, Answer

def writeXLSXresult(Data_Array, SaveDir, Filename):
    import numpy as np
    import xlsxwriter
    
    # Replace NaN values with zero in the last 8 columns (ALIS, ALIS2, ...)
    Data_Array[:, -8:] = np.nan_to_num(Data_Array[:, -8:], nan=0)
    
    # Workbook() takes one, non-optional, argument
    # which is the filename that we want to create.
    workbook = xlsxwriter.Workbook(SaveDir+'\\'+Filename+'.xlsx')
    
    # The workbook object is then used to add new
    # worksheet via the add_worksheet() method.
    worksheet = workbook.add_worksheet()
    
    # Use the worksheet object to write
    # data via the write() method.
    
    # Start from the first cell.
    # Rows and columns are zero indexed.
    col = 0
    
    heading = ["Risp_1",
               "Risp_2",
               "Risp_3",
               "Risp_4",
               "Risp_5a",
               "Risp_5b",
               "Risp_5c",
               "Risp_5d",
               "Risp_6",
               "Risp_7",
               "Risp_8",
               "ALIS",
               "ALIS2",
               "EP",
               "ET",
               "NL",
               "LCC",
               "SB",
               "SL"]
    
    # iterating through content list
    for item in heading :
    
        # write operation perform
        worksheet.write(0, col, item)
    
        # incrementing the value of row by one
        # with each iterations.
        col += 1
    
    
    #Data_Array[np.isnan(Data_Array)] = None
    
    for row in range(np.size(Data_Array,0)):
        for col in range(np.size(Data_Array,1)):
            try:
                worksheet.write(row+1, col, Data_Array[row, col])
            except:
                pass

    # Finally, close the Excel file
    # via the close() method.
    workbook.close()
        


####################### SCRIPT ###########################


# select empty form
#VUOTOfile = askopenfilename(title= "Selezionare il PDF con il questionario VUOTO")
VUOTOfile = os.path.dirname(os.path.abspath(sys.argv[0]))+'\\Gradimento_VUOTO_pratica2024.pdf' #stessa cartella dello script in esecuzione

IMGgrayTOT, IMGbinTOT, IMGerodeTOT, SaveDir, IMGfilenameVUOTO = PDFimg_read(VUOTOfile,CreateFolder=False)

VUOTOerodeTOT=np.mean(IMGerodeTOT, dtype=np.uint8, axis=2)
VUOTOgrayTOT=np.mean(IMGgrayTOT, dtype=np.uint8, axis=2)
VUOTObinTOT=np.mean(IMGbinTOT, dtype=np.uint8, axis=2)
del IMGgrayTOT, IMGbinTOT, IMGerodeTOT, SaveDir, IMGfilenameVUOTO

IMGfile = askopenfilename(title= "Selezionare il PDF con i gradimenti COMPILATI") # show an "Open" dialog box and return the path to the selected file

IMGgrayTOT, IMGbinTOT, IMGerodeTOT, SaveDir, IMGfilename = PDFimg_read(IMGfile,CreateFolder=True)
#del IMGgrayTOT, IMGbinTOT

#tutte le immagini hanno la stessa dimensione

IMGgrayTOT,VUOTOerodeTOT = sameShape(IMGgrayTOT,VUOTOerodeTOT)
IMGbinTOT,VUOTOerodeTOT = sameShape(IMGbinTOT,VUOTOerodeTOT)
IMGerodeTOT,VUOTOerodeTOT = sameShape(IMGerodeTOT,VUOTOerodeTOT)

IMGelabTOT=IMGbinTOT

imgSUBCTRACT = np.zeros([np.shape(IMGelabTOT)[0],np.shape(IMGelabTOT)[1],np.shape(IMGelabTOT)[2]], dtype=np.uint8)
IMGcirclesTOT = np.zeros([np.shape(IMGelabTOT)[0],np.shape(IMGelabTOT)[1],3,np.shape(IMGelabTOT)[2]], dtype=np.uint8)

K=np.ones((2, 2), np.uint8)

# registro tutti i questionari compilati sul questionario vuoto
for i in range(np.shape(IMGelabTOT)[2]):
    print(f'Registrazione pagina {i+1}...')
    IMGerodeTOT[:,:,i], IMGelabTOT[:,:,i], IMGgrayTOT[:,:,i] = registration(img2register1=IMGerodeTOT[:,:,i], img2register2=IMGelabTOT[:,:,i], img2register3=IMGgrayTOT[:,:,i], REFimg=VUOTOerodeTOT.astype(np.uint8))
    # sottrazione immagine
    imgSUBCTRACT[:,:,i]=cv2.subtract(VUOTOerodeTOT, IMGelabTOT[:,:,i])
    #imgSUBCTRACT[:,:,i] = cv2.adaptiveThreshold(imgSUBCTRACT[:,:,i],255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,9,60)
    imgSUBCTRACT[:,:,i]=cv2.dilate(cv2.erode(imgSUBCTRACT[:,:,i],K, iterations=1),K, iterations=4)
    #imgSUBCTRACT[:,:,i]=cv2.dilate(cv2.erode(imgSUBCTRACT[:,:,i],K, iterations=1),K, iterations=1)
    #imgSUBCTRACT[:,:,i]=cv2.dilate(cv2.erode(imgSUBCTRACT[:,:,i],K, iterations=1),K, iterations=1)
    
    # cv2.imwrite(SaveDir+f"\\imageSUBTRACT_{i+1}.jpeg", imgSUBCTRACT[:,:,i])

Answer_coord=np.zeros((np.shape(imgSUBCTRACT)[2], 19), np.float64)
Answer=np.zeros((np.shape(imgSUBCTRACT)[2], 19), np.float64)

for n in range(np.shape(imgSUBCTRACT)[2]):
    print(f"Riconoscimento pagina {n+1}...")
    for q in range(19):
        #estraggo i qudratini delle risposte
        Question=question_cut_6(img2cut=imgSUBCTRACT[:,:,n],Q_num=q)
        
        
        # # if does not exist subfolder, then create it
        # os.makedirs(SaveDir+'\\Questions', exist_ok=True)
        # cv2.imwrite(SaveDir+f"\\Questions\\Question_{n+1}_n{q+1}.jpeg", Question)
        
        #riconosco la risposta con bulb detection
        if q<=10:
            Answer_coord[n,q], Answer[n,q]=blobDetect_6(Question)
        else:
            Answer_coord[n,q], Answer[n,q]=blobDetect_1(Question)
            
    if any(np.isnan(Answer[n,:10])) or all(np.isnan(Answer[n,11:])):
        
        print('Risposte non completamente individuate')
        IMGcirclesTMP=writeCircles(imgIN=IMGgrayTOT[:,:,n],Answers=Answer[n,:])
        print('-------------------------------------------')
        print('')
        print('BARRA SPAZIATRICE  ---> RISPOSTE ACCETTATE')
        print('')
        print('TASTO "ESC"        ---> INSERIMENTO MANUALE')
        print('')
        print('-------------------------------------------')
        width = int(IMGcirclesTMP.shape[1] * 1/2)
        height = int(IMGcirclesTMP.shape[0] * 1/2)
        dim = (width, height)
        
        # resize image
        IMGcirclesTMP = cv2.resize(IMGcirclesTMP, dim, interpolation = cv2.INTER_AREA)

        # Showing original image
        cv2.imshow("Gradimento non riconosciuto", IMGcirclesTMP)
        kk=cv2.waitKey(0)
        
        if kk%256 == 32:
            # SPACEBAR pressed
            #K in multiples of 32 np.int64s the spacebar key was pressed
            
            # close all the opened windows.
            cv2.destroyAllWindows()
            #continue
                
        
        elif kk%256 == 27:
            # ESC pressed
            #K in multiples of 27 means the esc key was pressed
            
            # close all the opened windows.
            cv2.destroyAllWindows()
            print()
            print()
            print()
            print('-------------------------------------------')
            print()
            print('Ogni click del mouse inserisce una risposta')
            print()
            print('NON è possibile correggere le risposte date')
            print()
            print('          -------------------')
            print()
            print('BARRA SPAZIATRICE  ---> CONFERMA RISPOSTE')
            print()
            print('In caso di errore eseguire nuovamente il programma')
            print()
            print('-------------------------------------------')
            # Create point matrix get coordinates of mouse click on image
            posList = []
            counter = 0
            def mousePoints(event,x,y,flags,params):
                global counter
                global posList
                
                # Left button mouse click event opencv
                if event == cv2.EVENT_LBUTTONDOWN and counter<19:
                    posList.append([x, y])
                    counter = counter + 1
                    
            
            
            # Read image
            img=cv2.cvtColor(IMGgrayTOT[:,:,n],cv2.COLOR_GRAY2RGB)
            
            width = int(img.shape[1] * 1/2)
            height = int(img.shape[0] * 1/2)
            dim = (width, height)
            
            # resize image
            img = cv2.resize(cv2.cvtColor(IMGgrayTOT[:,:,n],cv2.COLOR_GRAY2RGB), dim, interpolation = cv2.INTER_AREA)
            
            while True:
                
    
                    
                # Showing original image
                cv2.imshow("Gradimento non riconosciuto", img)
                
                for i in range(len(posList)):
                    cv2.circle(img,(posList[-1:][0]),10,(0,255,0),2)
                
                # Mouse click event on original image
                cv2.setMouseCallback("Gradimento non riconosciuto", mousePoints)
                
                    
                # Refreshing window all time
                k=cv2.waitKey(1)
                
                if k%256 == 32:
                    # SPACEBAR pressed
                    #K in multiples of 32 means the spacebar key was pressed
                    break
                
                # if k%256 == 27:
                #     # ESC pressed
                #     #K in multiples of 27 means the esc key was pressed
                #     if len(posList)!=0:
                #         posList=posList[0:len(posList)-1] #REMOVE LAST ENTRY
                #         continue
                
            posNp = np.array(posList)*2 
            
            # close all the opened windows.
            cv2.destroyAllWindows()
            
            #modifica la tabella risposte
            Answer_coord[n,:]=float("NaN")
            Answer[n,:]=float("NaN")
            
            for qq in range(len(posNp)):
                Manual_coord=posNp[qq,:]
                
                qN, Asw=manualAnswer(Manual_coord)
                
                Answer_coord[n,qN]=Manual_coord[0]
                Answer[n,qN]=Asw
            
    IMGcirclesTOT[:,:,:,n]=writeCircles(imgIN=IMGgrayTOT[:,:,n],Answers=Answer[n,:])
    if IMGsave_opt:
        cv2.imwrite(SaveDir+f"\\Pagina_{n+1}.jpeg", IMGcirclesTOT[:,:,:,n])


#salvataggio in excel

writeXLSXresult(Answer, SaveDir, IMGfilename[:-4])
    
#adesso_clock = datetime.datetime.now()

#elapsed_time = adesso_clock-start_clock
print('-------------------------------------------')
print ()
print ('Elaborazione COMPLETATA')
print ()
print ('Risultati salvati in: '+IMGfilename[:-4]+'.xlsx')
print ()
print('-------------------------------------------')
#print('File creato il %s/%s/%s alle ore %s:%s' % (adesso_clock.day, adesso_clock.month, adesso_clock.year,adesso_clock.hour, adesso_clock.minute))
#print()
#print(f'Tempo impiegato: {elapsed_time.total_seconds()} secondi')

sys.modules[__name__].__dict__.clear()