import cv2
import mediapipe as mp
import numpy as np

def build_3d_face(image_path, output_obj_path):
    mp_mesh = mp.solutions.face_mesh
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    with mp_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,  # Uses the exact 468 canonical vertices
        min_detection_confidence=0.5
    ) as detector:
        results = detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            print("No face detected! Use a clear front-facing portrait.")
            return

        landmarks = results.multi_face_landmarks[0].landmark

        # 1. Collect 468 3D points
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

        # 2. Re-center around origin (0, 0, 0)
        center = np.mean(pts, axis=0)
        pts -= center

        # Invert Y so up is positive and Z so depth matches the viewport
        pts[:, 1] = -pts[:, 1]
        pts[:, 2] = -pts[:, 2]

        # 3. Normalize scale to match the HUD coordinate space
        span = np.max(np.abs(pts))
        if span > 0:
            pts = (pts / span) * 1.5

        # 4. Write out Wavefront OBJ using MediaPipe's canonical face triangulation
        with open(output_obj_path, 'w') as f:
            f.write("# Jai-Vardhan-Reddy Canonical Face Mesh\n")
            for x, y, z in pts:
                f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")

            # Write standard triangular faces
            for tri in mp_mesh.FACEMESH_TESSELATION:
                f.write(f"f {tri[0] + 1} {tri[1] + 1}\n")

        print(f"New 3D face mesh written to {output_obj_path} ({len(pts)} vertices).")

if __name__ == "__main__":
    build_3d_face("core/my_face.jpg", "core/face_model.obj")