import open3d as o3d
import os

INPUT = "../colmap_project/dense/fused.ply"
OUTPUT_DIR = "../outputs/model"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("Loading raw COLMAP model...")

    pcd = o3d.io.read_point_cloud(INPUT)

    print("Downsampling for clean model...")
    pcd = pcd.voxel_down_sample(voxel_size=0.01)

    print("Removing noise...")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0)

    output_path = f"{OUTPUT_DIR}/coral_model.ply"

    print("Saving final model...")
    o3d.io.write_point_cloud(output_path, pcd)

    print("DONE: Final 3D coral model saved at:", output_path)

if __name__ == "__main__":
    main()