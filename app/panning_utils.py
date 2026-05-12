import cv2
import numpy as np

class PanningCorrector:
    """
    Normalizes coordinates from a panning camera into a global static coordinate system
    using background feature tracking and homography.
    """
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher()
        self.homographies = [] # Stores H matrix for each frame relative to frame 0

    def estimate_camera_movement(self, frames):
        """
        Processes video frames to find the transformation matrix for every frame
        relative to the first frame.
        """
        if not frames: return
        
        # Reference frame (usually the first frame)
        prev_img = frames[0]
        kp1, des1 = self.sift.detectAndCompute(prev_img, None)
        
        # Cumulative Homography (Identity to start)
        h_cumulative = np.eye(3)
        self.homographies = [h_cumulative]

        for i in range(1, len(frames)):
            curr_img = frames[i]
            kp2, des2 = self.sift.detectAndCompute(curr_img, None)
            
            # Match background features (ignoring the moving runner)
            matches = self.bf.knnMatch(des1, des2, k=2)
            
            # Ratio test for high-quality matches
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)

            if len(good) > 10:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                
                # Find the transformation from frame i-1 to i
                h_matrix, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                
                # Chain the homography to relate back to the very first frame
                h_cumulative = h_cumulative @ h_matrix
                self.homographies.append(h_cumulative)
                
                # Update references for next iteration
                kp1, des1 = kp2, des2
            else:
                # If tracking fails, assume the camera stayed still (fallback)
                self.homographies.append(h_cumulative)

    def transform_landmark(self, landmark_xy, frame_idx):
        """
        Converts a [x, y] pixel from a specific frame into the 'Global Canvas' space.
        """
        if frame_idx >= len(self.homographies):
            return landmark_xy
            
        h = self.homographies[frame_idx]
        # Reshape for perspectiveTransform
        point = np.array([[landmark_xy]], dtype='float32')
        transformed_point = cv2.perspectiveTransform(point, h)
        
        return transformed_point[0][0]

def normalize_sprint_sequence(landmarks_sequence, video_frames):
    """
    Wrapper to correct a whole sequence of MediaPipe/mmpose landmarks.
    """
    corrector = PanningCorrector()
    print("Analyzing camera pan...")
    corrector.estimate_camera_movement(video_frames)
    
    corrected_sequence = []
    for f_idx, frame_landmarks in enumerate(landmarks_sequence):
        # frame_landmarks is [33, 3] (MediaPipe)
        corrected_frame = []
        for joint in frame_landmarks:
            # joint is [x, y, z]
            # We only transform X and Y; Z (depth) is treated separately
            global_xy = corrector.transform_landmark([joint[0], joint[1]], f_idx)
            corrected_frame.append([global_xy[0], global_xy[1], joint[2]])
            
        corrected_sequence.append(corrected_frame)
        
    return corrected_sequence
