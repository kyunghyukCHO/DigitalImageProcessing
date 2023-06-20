import cv2
import numpy as np

class ObjectDetection():
    def convert_to_black(self, rgb_frames):
        # 프레임에서 초록색 채널 추출
        for i, frame in enumerate(rgb_frames):
            green_channel = frame[:, :, 1] 
            # 초록색 채널 값이 80보다 작거나 200보다 큰 픽셀을 검정색으로 변환
            frame[(green_channel <= 80) | (green_channel > 200)] = [0, 0, 0]
            rgb_frames[i] = frame
        return rgb_frames
    
    def apply_morphology(self, frames, kernel_size):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # 모폴로지 연산을 적용하여 노이즈 제거
        return np.array([cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel) for frame in frames])
    
    def apply_dilation(self, frames, kernel_size):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # 팽창 연산을 적용하여 객체 윤곽 강화
        return np.array([cv2.dilate(frame, kernel, iterations=1) for frame in frames])
    
    def frames_rgb_to_hsv(self, frames):
        hsv_frames = []
        # RGB 프레임을 HSV 형식으로 변환
        for frame in frames:
            hsv_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
        return np.array(hsv_frames, dtype='uint8')
    
    def frames_hsv_to_rgb(self, frames):
        rgb_frames = []
        # HSV 프레임을 RGB 형식으로 변환
        for frame in frames:
            rgb_frames.append(cv2.cvtColor(frame, cv2.COLOR_HSV2BGR))
        return np.array(rgb_frames, dtype='uint8')
    
    def delete_background(self, frames):
        avg = np.zeros_like(frames[0], dtype='uint64')
        num_frames = len(frames)
        # 프레임의 평균값 계산
        for frame in frames:
            avg += frame
        avg = np.array(avg / num_frames, dtype='uint8') 
        # 프레임에서 평균값을 빼서 배경 제거
        for i, frame in enumerate(frames):
            frames[i] = frame - avg
        return frames
    
    def thresholding(self, frames, thres_min, thres_max):
        for i, frame in enumerate(frames):
            _, frames[i] = cv2.threshold(frame, thres_min, 255, cv2.THRESH_TOZERO)
            _, frames[i] = cv2.threshold(frame, thres_max, 0, cv2.THRESH_TOZERO_INV)
        return frames
    
    def thresholding_and_set_range(self, frames, ranges):
        for i, frame in enumerate(frames):
            result = np.zeros_like(frame)
            for (thres_min, thres_max) in ranges:
                # 임계값 범위에 해당하는 픽셀을 255로 설정
                within_threshold_indices = (frame >= thres_min) & (frame <= thres_max)
                result[within_threshold_indices] = 255
            frames[i] = result
        return frames
    
    def connect_components(self, frames, original_frames, area_min):
        for k, frame in enumerate(frames):
            # 연결된 컴포넌트 탐색
            cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(frame[:, :, 2])
            for i in range(1, cnt):
                (x, y, w, h, area) = stats[i]
                if area < area_min:
                    continue
                # 객체 주변에 사각형 그리기
                cv2.rectangle(original_frames[k], (x, y), (x+w, y+h), (0, 0, 255), 2)
        return original_frames
    
    def hand_detection(self, frames):
        # RGB 프레임을 HSV로 변환
        hsv_frames = self.frames_rgb_to_hsv(frames)
        # 배경 제거
        back_deleted_frames = self.delete_background(hsv_frames[:, :, :, 2].copy())
        # 임계값 처리
        hsv_frames[:, :, :, 2] = self.thresholding_and_set_range(back_deleted_frames, [(0, 250)])
        # HSV 프레임을 RGB로 변환
        rgb_frames = self.frames_hsv_to_rgb(hsv_frames)
        # 초록색을 검정색으로 변환
        rgb_frames = self.convert_to_black(rgb_frames)
        # 모폴로지 연산 적용
        rgb_frames = self.apply_morphology(rgb_frames, 20)
        # 팽창 연산 적용
        rgb_frames = self.apply_dilation(rgb_frames, 40)

        # 객체 연결 컴포넌트 탐색 후 사각형 그리기
        connected_frames = self.connect_components(rgb_frames, frames, 50000)

        return rgb_frames


class GetVideo():
    def get_video_frames(self, filename):
        video_address = filename
        capture = cv2.VideoCapture(video_address)
        frames = []
        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break
            img = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
            frames.append(img)
        capture.release()
        return np.array(frames, dtype='uint8')


# 비디오 프레임 가져오기
frames = GetVideo().get_video_frames("영상1.mp4")

# 객체 검출 클래스 인스턴스 생성
object_detection = ObjectDetection()

# 손 검출 수행
result_frames = object_detection.hand_detection(frames)

# 결과 프레임 출력
for frame in frames:
    cv2.imshow("Video1", frame)
    if cv2.waitKey(30) == 27:
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
