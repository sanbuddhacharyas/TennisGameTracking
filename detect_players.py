from ultralytics import YOLO
import cv2
import numpy as np

def area_of_box(box):
    height = box[3] - box[1]
    width = box[2] - box[0]
    return height * width

def remove_court_backgroud(img, court_pos, margin=10):
    """
        Removes the background except the A4 paper section

        Args:
            img:     input image
            corners: marker detected corners

        Return:
            output: background removed image
    """
    top = [[court_pos[0] - margin, court_pos[1] - margin], [court_pos[2] + margin, court_pos[1] - margin],\
        [court_pos[2] + margin, court_pos[3] + margin], [court_pos[0] - margin, court_pos[3] + margin]]

    original = img.copy()
    top      = np.array(top).reshape((-1, 1, 2))


    mask_top    = np.zeros(img.shape[:2], np.uint8)

    cv2.fillPoly(mask_top,[top],(255,255, 255))
   


    top_img    = cv2.bitwise_and(original, original, mask = mask_top)

    return top_img

def detect_court_net(model, img):

    player_1     = [None, None, None, None]
    player_2     = [None, None, None, None]
    court_pos    = [None, None, None, None]
    net_pos      = [None, None, None, None]

    court_score  = 0.0
    net_score    = 0.0
    player_score = []
    player_pos   = []
  
    result = model(img,verbose=False)[0]

    boxes = result.boxes  # Boxes object for bbox outputs
    boxes = boxes.to('cpu')
    boxes = boxes.numpy()

    for box in boxes:

        if (box.cls[0]==0) and (box.conf[0] > 0.80):

            if box.conf[0]>court_score:
                court_pos = [int(i) for i in box.xyxy[0]]
                court_score = box.conf[0]

        elif (box.cls[0]==1) and (box.conf[0] > 0.70) :
            if box.conf[0]>net_score:
                net_pos =  [int(i) for i in box.xyxy[0]]
                net_score = box.conf[0]

        elif (box.cls[0] == 2) and (box.conf[0] > 0.40):
            player_score.append(box.conf[0])
            player  =  [int(i) for i in box.xyxy[0]]
            player_pos.append(player)

    if len(player_pos)>=2:
        area_score = [area_of_box(i) for i in player_pos]
        player_1 = player_pos[np.argmax(area_score)]
        player_2 = player_pos[np.argmin(area_score)]
                

    return player_1, player_2, court_pos, net_pos

def detect_player_ball(model, img):

    player_1 = [None, None, None, None]
    player_2 = [None, None, None, None]
    ball_pos = [None, None]

    player_1_score = 0.0
    player_2_score = 0.0
    ball_score     = 0.0

    result = model(img,verbose=False)[0]

    boxes = result.boxes  # Boxes object for bbox outputs
    boxes = boxes.to('cpu')
    boxes = boxes.numpy()

    for box in boxes:

        if (box.cls[0]==1) and (box.conf[0] > 0.80):

            if box.conf[0]>player_1_score:
                player_1 = [int(i) for i in box.xyxy[0]]
                player_1_score = box.conf[0]

        elif (box.cls[0]==0) and (box.conf[0] > 0.70) :
            if box.conf[0]>player_2_score:
                player_2 =  [int(i) for i in box.xyxy[0]]
                player_2_score = box.conf[0]

        elif (box.cls[0] == 2) and (box.conf[0] > 0.40):
            if box.conf[0]>ball_score:
                ball_pos  =  [int(i) for i in box.xyxy[0]]
                ball_score = box.conf[0]

    return player_1, player_2, ball_pos


if __name__=='__main__':
    model = YOLO('./weights/player_court_net_best.pt')
    img   = cv2.imread('frames/100.jpg')

    player_1, player_2, court_pos, net_pos = detect_court_net(model, img)

    img = cv2.rectangle(img, (court_pos[0], court_pos[1]), (court_pos[2], court_pos[3]), (255, 0, 0), 3)
    img = cv2.rectangle(img, (net_pos[0], net_pos[1]), (net_pos[2], net_pos[3]), (0, 255, 0), 3)
    img = cv2.rectangle(img, (player_1[0], player_1[1]), (player_1[2], player_1[3]), (0, 0, 255), 3)
    img = cv2.rectangle(img, (player_2[0], player_2[1]), (player_2[2], player_2[3]), (255, 0, 255), 3)

    
    backgroud_removed = remove_court_backgroud(img, court_pos)

    cv2.imshow('img', backgroud_removed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()