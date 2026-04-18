import os
import shutil
import subprocess

PROJECT_ROOT = ".."
FRAMES_DIR = f"{PROJECT_ROOT}/frames"
COLMAP_IMAGES = f"{PROJECT_ROOT}/colmap_project/images"
COLMAP_DB = f"{PROJECT_ROOT}/colmap_project/database/database.db"
COLMAP_SPARSE = f"{PROJECT_ROOT}/colmap_project/sparse"
COLMAP_DENSE = f"{PROJECT_ROOT}/colmap_project/dense"

def copy_frames():
    os.makedirs(COLMAP_IMAGES, exist_ok=True)
    for f in os.listdir(FRAMES_DIR):
        if f.endswith(".jpg"):
            shutil.copy(f"{FRAMES_DIR}/{f}", f"{COLMAP_IMAGES}/{f}")

def run(cmd):
    subprocess.run(cmd, shell=True, check=True)

def main():
    print("Copying frames...")
    copy_frames()

    print("Feature extraction...")
    run(f"colmap feature_extractor --database_path {COLMAP_DB} --image_path {COLMAP_IMAGES}")

    print("Matching...")
    run(f"colmap exhaustive_matcher --database_path {COLMAP_DB}")

    print("Sparse reconstruction...")
    run(f"colmap mapper --database_path {COLMAP_DB} --image_path {COLMAP_IMAGES} --output_path {COLMAP_SPARSE}")

    print("Undistorting...")
    run(f"colmap image_undistorter --image_path {COLMAP_IMAGES} --input_path {COLMAP_SPARSE}/0 --output_path {COLMAP_DENSE} --output_type COLMAP")

    print("Dense reconstruction...")
    run(f"colmap patch_match_stereo --workspace_path {COLMAP_DENSE} --workspace_format COLMAP")

    print("Fusion...")
    run(f"colmap stereo_fusion --workspace_path {COLMAP_DENSE} --workspace_format COLMAP --output_path {COLMAP_DENSE}/fused.ply")

    print("DONE: Model created at dense/fused.ply")

if __name__ == "__main__":
    main()