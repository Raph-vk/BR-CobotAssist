import numpy as np
import os
import time
import yaml
import math
import meshcat
import meshcat.geometry as g
from urdfpy import URDF
import roboticstoolbox as rtb
import imageio
from PIL import Image
import io

class RobotVisualizer:
    def __init__(self, yaml_path = 'utils/visualization/config_visualization.yaml'):

        # Load inputs from yaml file
        with open(yaml_path, 'r') as file:
            settings = yaml.safe_load(file)

        self.DT = settings['dt_robot']
        self.urdf_path = settings['urdf_path']
        print(f"Loaded robot visualization settings from: {os.path.abspath(yaml_path)}")


        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")

        self.vis = meshcat.Visualizer()
        self.vis.open()
        time.sleep(2)

        self.init_objects()

        time.sleep(1)

    def init_objects(self):
        # Define colors
        FANUC_YELLOW = 0xFFD700
        # GREY = 0x888888
        GREY = 0x696969
        BLACK = 0x000000

        # get the urdf file
        assert os.path.exists(self.urdf_path), f"URDF not found: {self.urdf_path}"
        self.robot = URDF.load(self.urdf_path)
        base_dir = os.path.dirname(self.urdf_path)

        # print([l.name for l in robot.links])
        for link in self.robot.links:
            for visual in link.visuals:
                mesh_spec = visual.geometry.mesh
                if mesh_spec is None:
                    continue

                # full path to the STL
                mesh_file = os.path.join(base_dir, mesh_spec.filename)
                if not os.path.exists(mesh_file):
                    print(f"⚠️  Missing mesh file: {mesh_file}")
                    continue

                # load MeshCat geometry
                geom = g.StlMeshGeometry.from_file(mesh_file)

                # color geometry
                link_color = GREY if link.name in ['base_link', 'link_gripper'] else BLACK if link.name in ['link_6'] else FANUC_YELLOW
                mat  = g.MeshLambertMaterial(color=link_color, side="double")
                
                node = f"/{link.name}"
                self.vis[node].set_object(geom, mat)
                fk = self.robot.link_fk({link:0})  
                self.vis[node].set_transform(fk[link])

    def move_joints(self, joint_positions, videopath = None, recording=False, extract_frame=False):
        if joint_positions.shape[1] == 7:
            # remove the last joint (vacuum on/off)
            joint_positions = joint_positions[:, :-1]
            # print(f"Joint positions new shape: {joint_positions.shape}")

        if recording:
            print(f"Speed of movements only accurate in .mp4 file, not in MeshCat viewer.")
            frames = []
        
        for q in joint_positions:
            # build a dict of joint_name→value
            cfg = {name: math.radians(float(val)) for name, val in zip(self.robot.joint_map.keys(), q)}
            # compute link poses in the world frame
            fk = self.robot.link_fk(cfg)  
            # apply them to MeshCat
            for link, X in fk.items():
                # print(link.name, X)
                self.vis[f"/{link.name}"].set_transform(X)
            # save image if recording
            if recording:
                image = self.vis.get_image()
                frame = np.ascontiguousarray(image, dtype=np.uint8)
                frames.append(frame)
            else:
                time.sleep(self.DT)
        
        if recording:
            print(f'frames shape: {len(frames)}, frames[0].shape: {frames[0].shape}')
            imageio.mimwrite(videopath, frames, fps=1/self.DT, macro_block_size = None, quality = 8, ffmpeg_params=['-loglevel', 'error'])
            print(f'Saved robot movement video to: {videopath}')
            print(f'frames[0][0][0][0]: {frames[0][0][0][0]}')  # Print the first pixel value to check if the video is saved correctly

        if extract_frame:
            # Capture the normal/front view from current camera position
            frontview_image = self.vis.get_image()
            frontview = np.ascontiguousarray(frontview_image, dtype=np.uint8)
                       
            # Create side view transform matrix (90 degrees around Y-axis, positioned to the side)
            side_transform = meshcat.transformations.compose_matrix(
                angles=[0, 0, -np.pi/2],  # Rotate 90 degrees around Y-axis for side view
                translate=[0, 0, 0]      # Move viewpoint to the side and slightly up
            )
            
            # Temporarily set camera to side view position
            self.vis["/Cameras/default"].set_transform(side_transform)
            
            # Capture image from side viewpoint
            sideview_image = self.vis.get_image()
            sideview = np.ascontiguousarray(sideview_image, dtype=np.uint8)

            return frontview, sideview

    def move_joints_to_angle(self, joint_angles, q0 = np.zeros(6), timesteps = 250):
        # print(self.robot.joint_map.keys())
        if not len(joint_angles) == 6:
            raise ValueError("Joint angles must match the number of joints in the robot, so a list of length 6.")
        traj = rtb.jtraj(q0, joint_angles, timesteps)
        for q in traj.q:
            # build a dict of joint_name→value
            cfg = {name: float(val) for name, val in zip(self.robot.joint_map.keys(), q)}
            # print("cfg: ",cfg)
            # compute link poses in the world frame
            fk = self.robot.link_fk(cfg)  
            # apply them to MeshCat
            for link, X in fk.items():
                # print(link.name, X)
                self.vis[f"/{link.name}"].set_transform(X)
            time.sleep(self.DT)

    # def add_ball(self, name, position, radius=0.1, color=0xCD1C18):
    #     ball = g.Sphere(radius)
    #     self.vis[name].set_object(ball, g.MeshLambertMaterial(color=color))
    #     self.vis[name].set_transform(tf.translation_matrix(position))

    # def remove_ball(self, name):

    #     if name in self.vis:
    #         self.vis[name].delete()

if __name__ == "__main__":
    # Example usage
    visualizer = RobotVisualizer()
    # Example joint positions
    visualizer.move_joints_to_angle([np.pi/2,np.pi/4,0,0,0,0])