import cv2
import numpy as np

# 차량 윤곽의 최소 가로와 세로 길이 비율
min_aspect_ratio = 0.5
max_aspect_ratio = 2.0

# 차량 윤곽의 최소 면적
min_area = 1000

# 동영상 캡처 객체 생성
video_capture = cv2.VideoCapture('영상3.mp4')

# 동영상 캡처 객체가 열려있는지 확인
if video_capture.isOpened():
    ret, frame1 = video_capture.read()
else:
    ret = False

# 초기 프레임 읽기
ret, frame1 = video_capture.read()
ret, frame2 = video_capture.read()

while ret:
    # 현재 프레임과 이전 프레임의 차이 계산
    diff = cv2.absdiff(frame1, frame2)
    
    # 차이 영상을 그레이스케일로 변환
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # 그레이스케일 영상 이진화
    _, threshold = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    
    # 모폴로지 연산 적용
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    
    # 윤곽선 검출
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # 윤곽선을 둘러싸는 사각형 그리기
        x, y, w, h = cv2.boundingRect(contour)
        
        # 차량 윤곽의 가로와 세로 길이 비율 계산
        aspect_ratio = float(w) / h
        
        # 윤곽선이 차량으로 판단되는지 확인
        if aspect_ratio >= min_aspect_ratio and aspect_ratio <= max_aspect_ratio:
            # 차량 윤곽의 면적 계산
            area = cv2.contourArea(contour)
            
            # 윤곽선이 충분히 큰 면적을 가지는지 확인
            if area >= min_area:
                cv2.rectangle(frame1, (x-10, y-10), (x+w+10, y+h+10), (0, 0, 255), 1)
    
    # 결과 프레임 출력
    cv2.imshow("Vehicle Detection", frame1)
   
    # 'Esc' 키를 누르면 종료
    if cv2.waitKey(30) == 27:
        break
    
    # 프레임 업데이트
    frame1 = frame2
    ret, frame2 = video_capture.read()

# 동영상 캡처 객체 해제
video_capture.release()

# 창 닫기
cv2.destroyAllWindows()
