import cv2
from cvzone.HandTrackingModule import HandDetector
import pyautogui
import threading
import numpy as np
import time

# Get screen size
screen_w, screen_h = pyautogui.size()

# Setup camera
cap = cv2.VideoCapture(0)
cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Dynamic camera width
cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Dynamic camera height
frameR = 100  # Frame reduction for more reliable tracking at edges
scale_factor = 0.5  # Reduce image size by half for faster processing

detector = HandDetector(detectionCon=0.9, maxHands=1)

# Delay flags
l_delay = 0
r_delay = 0
double_delay = 0

# Delay functions
def l_clk_delay():
    global l_delay
    global l_clk_thread
    time.sleep(1)
    l_delay = 0
    l_clk_thread = threading.Thread(target=l_clk_delay)

def r_clk_delay():
    global r_delay
    global r_clk_thread
    time.sleep(1)
    r_delay = 0
    r_clk_thread = threading.Thread(target=r_clk_delay)

def double_clk_delay():
    global double_delay
    global double_clk_thread
    time.sleep(2)
    double_delay = 0
    double_clk_thread = threading.Thread(target=double_clk_delay)

# Initial thread creation
l_clk_thread = threading.Thread(target=l_clk_delay)
r_clk_thread = threading.Thread(target=r_clk_delay)
double_clk_thread = threading.Thread(target=double_clk_delay)

while True:
    success, img = cap.read()
    if success:
        img = cv2.flip(img, 1)
        hands, img = detector.findHands(img, flipType=False)
        cv2.rectangle(img, (frameR, frameR), (cam_w - frameR, cam_h - frameR), (255, 0, 255), 2)

        action_text = ""  # Text to display
        if hands:
            lmlist = hands[0]['lmList']
            ind_x, ind_y = lmlist[8][0], lmlist[8][1]
            mid_x, mid_y = lmlist[12][0], lmlist[12][1]
            cv2.circle(img, (ind_x, ind_y), 5, (0, 255, 255), 2)
            cv2.circle(img, (mid_x, mid_y), 5, (0, 255, 255), 2)
            fingers = detector.fingersUp(hands[0])

            # Mouse movement
            if fingers[1] == 1 and fingers[2] == 0 and fingers[0] == 1:
                conv_x = np.interp(ind_x, (frameR, cam_w - frameR), (0, screen_w))
                conv_y = np.interp(ind_y, (frameR, cam_h - frameR), (0, screen_h))
                pyautogui.moveTo(conv_x, conv_y)
                action_text = "Moving Mouse "

            # Mouse Button Clicks
            if fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 1 and abs(ind_x - mid_x) < 25:
                if fingers[4] == 0 and l_delay == 0:
                    pyautogui.click(button='left')
                    l_delay = 1
                    action_text = "Left Click "
                    l_clk_thread.start()
                elif fingers[4] == 1 and r_delay == 0:
                    pyautogui.click(button='right')
                    r_delay = 1
                    action_text = "Right Click"
                    r_clk_thread.start()

            # Mouse Scrolling
            if fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 0:
                scroll_direction = 1 if fingers[4] == 1 else -1
                pyautogui.scroll(scroll_direction)
                action_text = "Scrolling "

            # Double Mouse Click
            if fingers[1] == 1 and fingers[2] == 0 and fingers[0] == 0 and fingers[4] == 0 and double_delay == 0:
                x, y = np.interp(ind_x, (frameR, cam_w - frameR), (0, screen_w)), np.interp(ind_y, (frameR, cam_h - frameR), (0, screen_h))
                double_delay = 1
                pyautogui.doubleClick(x, y, button='left')
                action_text = "Double Click "
                double_clk_thread.start()

        # Display action text
        cv2.putText(img, action_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Camera Feed", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
