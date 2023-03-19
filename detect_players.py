from ultralytics import YOLO
import cv2

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
    model = YOLO('./weights/best.pt')
    img   = cv2.imread('frames/440.jpg')

    player_1, player_2, ball_pos = detect_player_ball(model, img)

    img = cv2.rectangle(img, (player_1[0], player_1[1]), (player_1[2], player_1[3]), (255, 0, 0), 3)
    img = cv2.rectangle(img, (player_2[0], player_2[1]), (player_2[2], player_2[3]), (0, 255, 0), 3)
    # img = cv2.rectangle(img, (ball_pos[0], ball_pos[1]), (ball_pos[2], ball_pos[3]), (0, 0, 255), 3)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()