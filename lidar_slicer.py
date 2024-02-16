"""
This program can be launched directly.
To move the drone, you have to click on the map, then use the arrows on the keyboard
"""

import os
import sys
from typing import Type
import math
import numpy as np
from tabulate import tabulate

from spg.playground import Playground
from spg.utils.definitions import CollisionTypes

# This line add, to sys.path, the path to parent path of this file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from maps.walls_medium_02 import add_walls, add_boxes
from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.entities.rescue_center import RescueCenter, wounded_rescue_center_collision
from spg_overlay.gui_map.closed_playground import ClosedPlayground
from spg_overlay.gui_map.gui_sr import GuiSR
from spg_overlay.gui_map.map_abstract import MapAbstract
from spg_overlay.utils.misc_data import MiscData
from typing import Optional
import random

np.set_printoptions(suppress=True,linewidth=np.nan)

class MyDroneLidar(DroneAbstract):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


        self.is_parallel_to_obstacle = False
    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass

    def process_lidar_values(self):
        lidar_values_array = self.lidar().get_sensor_values()

        back = np.concatenate((lidar_values_array[:21], lidar_values_array[160:181]))
        right = lidar_values_array[23:68]
        front = lidar_values_array[69:114]
        left = lidar_values_array[114:159]

        right_right = right[:14]
        right_front = right[15:30]
        right_left = right[31:44]

        front_right = front[0:14]
        front_front = front[15:30]
        front_left = front[31:44] 

        left_right = left[:14]
        left_front = left[15:30]
        left_left = left[31:44]

        # Checking for collisions in each direction
        back_collided = min(back) < 50
        right_left_collided = min(right_left) < 50
        right_front_collided = min(right_front) < 50
        right_right_collided = min(right_right) < 50
        front_left_collided = min(front_left) < 50
        front_front_collided = min(front_front) < 50
        front_right_collided = min(front_right) < 50
        left_left_collided = min(left_left) < 50
        left_front_collided = min(left_front) < 50
        left_right_collided = min(left_right) < 50

        back_collided = min(back) < 50
        front_collided = min(front) < 50
        right_collided = min(right) < 50
        left_collided = min(left) < 50


        collision_dict = {
        "back": back_collided,
        "right": {
            "right": right_collided,
            "right_left": right_left_collided,
            "right_front": right_front_collided,
            "right_right": right_right_collided,
        },
        "front": {
            "front": front_collided,
            "front_left": front_left_collided,
            "front_front": front_front_collided,
            "front_right": front_right_collided,
        },
        "left": {
            "left": left_collided,
            "left_left": left_left_collided,
            "left_front": left_front_collided,
            "left_right": left_right_collided,
        },
                        }

        return collision_dict                

    def handle_front_collision(self, front_front_collided, front_right_collided, front_left_collided, right_left_collided):
        commands = {
            "forward": {"forward": 0.5, "lateral": 0.0, "rotation": 0.0, "grasper": 0},
            "slight_turn_right": {"forward": 0.0, "lateral": 0.0, "rotation": 0.5, "grasper": 0},
            "slight_turn_left": {"forward": 0.0, "lateral": 0.0, "rotation": -0.5, "grasper": 0},
            "turn_right": {"forward": 0.0, "lateral": 0.0, "rotation": 1, "grasper": 0},
        }

        if not front_front_collided:
            return commands["forward"]
        elif not front_right_collided:
            return commands["slight_turn_right"]
        elif not front_left_collided:
            return commands["slight_turn_left"]
        elif not right_left_collided:
            return commands["turn_right"]
        else:
            # Handle other cases or add additional logic here
            return commands["forward"]

    def control(self):
        """
        We only send a command to do nothing
        """
        commands = {
        "idle": {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0},
        "forward": {"forward": 0.5, "lateral": 0.0, "rotation": 0.0, "grasper": 0},
        "turn_left": {"forward": 0.0, "lateral": 0.0, "rotation": 1, "grasper": 0},
        "turn_right": {"forward": 0.0, "lateral": 0.0, "rotation": -1, "grasper": 0},
        "turn_back": {"forward": -1, "lateral": 1, "rotation": 1, "grasper": 0},
        "slight_turn_right": {"forward": 0.0, "lateral": 0.0, "rotation": 1, "grasper": 0},
        "slight_turn_left": {"forward": 0.0, "lateral": 0.0, "rotation": -1, "grasper": 0}
                    }

        lidar_values_array = self.lidar().get_sensor_values()
        back = np.concatenate((lidar_values_array[:21], lidar_values_array[160:181]))
        right = lidar_values_array[23:68]
        front = lidar_values_array[69:114]
        left = lidar_values_array[114:159]

        right_right = right[:14]
        right_front = right[15:30]
        right_left = right[31:44]

        front_right = front[0:14]
        front_front = front[15:30]
        front_left = front[31:44] 

        left_right = left[:14]
        left_front = left[15:30]
        left_left = left[31:44]

        collided_dict = self.process_lidar_values()

        back_collided = collided_dict["back"]

        right_left_collided = collided_dict["right"]["right_left"]
        right_front_collided = collided_dict["right"]["right_front"]
        right_right_collided = collided_dict["right"]["right_right"]

        front_left_collided = collided_dict["front"]["front_left"]
        front_front_collided = collided_dict["front"]["front_front"]
        front_right_collided = collided_dict["front"]["front_right"]

        left_left_collided = collided_dict["left"]["left_left"]
        left_front_collided = collided_dict["left"]["left_front"]
        left_right_collided = collided_dict["left"]["left_right"]


        front_collided = collided_dict["front"]["front"]
        left_collided = collided_dict["left"]["left"]
        right_collided = collided_dict["right"]["right"]

        if front_collided:
            return self.handle_front_collision(front_front_collided, front_right_collided, front_left_collided, right_left_collided)
        else:
            return commands["forward"]
        


class MyMapLidar(MapAbstract):

    def __init__(self):
        super().__init__()

        # PARAMETERS MAP
        self._size_area = (1113, 750)

        self._rescue_center = RescueCenter(size=(210, 90))
        self._rescue_center_pos = ((420, 315), 0)

        self._number_drones = 3
        #self._drones_pos = [((-50, 0), 0)]
        self._drones_pos = [((-50, 0), 0), ((-50, 30), 0), ((-50, 60), 0)]
        self._drones = []

    def construct_playground(self, drone_type: Type[DroneAbstract]) -> Playground:
        playground = ClosedPlayground(size=self._size_area)

        # RESCUE CENTER
        playground.add_interaction(CollisionTypes.GEM,
                                   CollisionTypes.ACTIVABLE_BY_GEM,
                                   wounded_rescue_center_collision)

        playground.add(self._rescue_center, self._rescue_center_pos)

        add_walls(playground)
        add_boxes(playground)

        # POSITIONS OF THE DRONES
        misc_data = MiscData(size_area=self._size_area,
                             number_drones=self._number_drones)
        for i in range(self._number_drones):
            drone = drone_type(identifier=i, misc_data=misc_data,
                               display_lidar_graph=False)
            self._drones.append(drone)
            playground.add(drone, self._drones_pos[i])

        return playground


def main():
    my_map = MyMapLidar()
    playground = my_map.construct_playground(drone_type=MyDroneLidar)

    # draw_lidar_rays : enable the visualization of the lidar rays
    # enable_visu_noises : to enable the visualization. It will show also a demonstration of the integration
    # of odometer values, by drawing the estimated path in red. The green circle shows the position of drone according
    # to the gps sensor and the compass
    gui = GuiSR(playground=playground,
                the_map=my_map,
                draw_lidar_rays=True,
                use_keyboard=False,
                enable_visu_noises=False,
                draw_gps=True,
                )
    gui.run()
        


if __name__ == '__main__':
    main()
