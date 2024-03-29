import sys
import threading
import time
import random
import numpy as np
import cv2
import numpy
from math import pi, sin, cos
from coppeliasimapi2 import CoppeliaSimAPI, Human, YouBot, RelationObject
import sys
import random
from genericworker import *
import math
import signal

from shapely.geometry import Point, Polygon

human_radius = 0.4
plant_radius = 0.4
laptop_radius = 0.4


def contained_with_radius(polygon, point, radius):
    point2 = Polygon(
        [
            [point.x - radius, point.y - radius],
            [point.x - radius, point.y + radius],
            [point.x + radius, point.y + radius],
            [point.x + radius, point.y - radius],
        ]
    )
    return point2.within(polygon)


class HumanMovementRandomiser(object):
    def __init__(self, human, period):
        super(HumanMovementRandomiser, self).__init__()
        self.period = period
        self.human = human
        self.last_update = time.time() - self.period
        self.moving_alone = True
        self.friend = None
        self.new_position = False

    def moving_with_friend(self, friend=None):
        self.moving_alone = False
        self.friend = friend

    def update(
        self,
        position_list,
        boundary,
        relation_to_human_map,
        relations_moving_humans,
        robot_position,
    ):
        if not self.moving_alone and self.friend is None:
            return

        pos1 = self.human.get_position()
        distance1_to_robot = math.sqrt(
            (pos1[0] - robot_position[0]) ** 2 + (pos1[1] - robot_position[1]) ** 2
        )
        distance2_to_robot = distance1_to_robot
        if not self.moving_alone:
            pos2 = self.friend.get_position()
            center = (pos1 + pos2) / 2.0
            length = math.sqrt(sum([(a - b) ** 2 for a, b in zip(pos1, pos2)]))
            orientation = (
                math.atan2(pos2[1] - pos1[1], pos2[0] - pos1[0]) + math.pi / 2.0
            )
            relations_moving_humans[relation_to_human_map].move(
                center[0],
                center[1],
                0.0,
                math.pi / 2.0,
                orientation,
                math.pi / 2.0,
                length,
            )
            distance2_to_robot = math.sqrt(
                (pos2[0] - robot_position[0]) ** 2 + (pos2[1] - robot_position[1]) ** 2
            )

        if self.human.distance_to_goal() < 0.001:
            self.new_position = False

        t = time.time()
        if t - self.last_update > self.period:
            self.last_update = t
            if distance1_to_robot >= 1.2 and distance2_to_robot >= 1.2:
                self.move_human(position_list, boundary)
                self.new_position = True
        else:
            if distance1_to_robot < 1.2 or distance2_to_robot < 1.2:
                if self.new_position:
                    self.human.stop()
                    if distance1_to_robot < 0.8:
                        self.human.cannot_move()
                    if not self.moving_alone:
                        self.friend.stop()
                        if distance2_to_robot < 0.8:
                            self.friend.cannot_move()
                    self.new_position = False
            else:
                self.human.can_move()
                if not self.moving_alone:
                    self.friend.can_move()

    def move_human(self, position_list, boundary):
        px = 6.0 * (random.random() - 0.5)
        py = 6.0 * (random.random() - 0.5)
        pose = [px, py]
        collision = True
        tries = 0
        while collision and tries < 5:
            px = 6.0 * (random.random() - 0.5)
            py = 6.0 * (random.random() - 0.5)
            pose = [px, py]
            collision = self.check_collision_position(pose, position_list)

            point = Point(px, py)
            if not contained_with_radius(boundary, point, human_radius):
                collision = True
            tries += 1

        if not collision:
            self.human.move([px, py, 0]) #-1.0])
            if not self.moving_alone:
                self.friend.move([0, 0.5, 0], relative_to=self.human.dummy_handle)

    def check_collision_position(self, pose, position_list):
        for p in position_list:
            if p[0] == pose[0] and p[1] == pose[1]:
                return True
        return False


class SODA:
    def __init__(
        self, proxy_map, data, scene_file="dataset_new.ttt", scene_path="../scenes/", frame_of_reference = 'R'
    ):
        super(SODA, self).__init__()
        self.coppelia = CoppeliaSimAPI(
            [os.path.join(os.path.dirname(__file__), scene_path)]
        )
        self.frame_of_reference = frame_of_reference
        self.coppelia.load_scene(scene_file, headless=True)
        self.goal_found = 0
        self.ini_data = data
        self.data = data
        self.object_list = []
        self.humans = []
        self.tables = []
        self.plants = []
        self.laptops = []
        self.walls = []
        self.relations = []
        self.relations_moving_humans = []
        self.relation_to_human_map = {}
        self.goal = None

        self.humans_IND = []
        self.tables_IND = []
        self.plants_IND = []
        self.laptops_IND = []
        self.walls_IND = []

        self.interacting_humans = []

        self.min_humans = None
        self.min_wandHumans = None
        self.min_plants = None
        self.min_tables = None
        self.min_laptops = None
        self.min_relations = None

        self.max_humans = None
        self.max_wandHumans = None
        self.max_plants = None
        self.max_tables = None
        self.max_laptops = None
        self.max_relations = None

        self.coppelia.start()

    def __del__(self):
        print("SONATA destructor")
        self.coppelia.shutdown()

    def room_setup(
        self,
        min_humans,
        min_wandHumans,
        min_plants,
        min_tables,
        min_relations,
        max_humans,
        max_wandHumans,
        max_plants,
        max_tables,
        max_relations,
        robot_random_pose = True,
        show_goal = True,
        show_relations = True
    ):
        self.coppelia.remove_objects(
            self.humans,
            self.tables,
            self.laptops,
            self.plants,
            self.goal,
            self.walls,
            self.relations,
            self.relations_moving_humans,
        )

        self.interacting_humans = []
        self.min_humans = min_humans
        self.min_wandHumans = min_wandHumans
        self.min_plants = min_plants
        self.min_tables = min_tables
        self.min_laptops = min_tables
        self.min_relations = min_relations

        self.max_humans = max_humans
        self.max_wandHumans = max_wandHumans
        self.max_plants = max_plants
        self.max_tables = max_tables
        self.max_laptops = max_tables
        self.max_relations = max_relations

        self.data = self.ini_data
        self.object_list = []
        self.position_list = (
            []
        )  # checking collision for wandering humans (has plants tables and static humans)

        self.humans = []
        self.wandering_humans = []
        self.tables = []
        self.plants = []
        self.laptops = []
        self.relations = []
        self.relations_moving_humans = []
        self.relation_to_human_map = {}
        self.wall_type = random.randint(
            0, 2
        )  # 0 for square, 1 for rectangle, 2 for L-shaped
        self.boundary = None
        self.human_indices = []
        self.table_indices = []
        self.plant_indices = []
        self.wandering_humans_indices = []
        self.length = None
        self.breadth = None
        self.break_point = 5  # when to stop creating an entity
        self.count = 0  # storing count till break point
        self.n_interactions = None

        self.humans_IND = []
        self.tables_IND = []
        self.plants_IND = []
        self.laptops_IND = []
        self.walls_IND = []

        IND = 1

        # Add four walls
        # SQUARE
        if self.wall_type == 0:
            length = random.randint(6, 10)
            self.walls_data = [
                ([length / 2, length / 2, 0.4], [length / 2, -length / 2, 0.4]),
                ([length / 2, -length / 2, 0.4], [-length / 2, -length / 2, 0.4]),
                ([-length / 2, -length / 2, 0.4], [-length / 2, length / 2, 0.4]),
                ([-length / 2, length / 2, 0.4], [length / 2, length / 2, 0.4]),
            ]
        # RECTANGLE
        elif self.wall_type == 1:
            breadth = random.randint(5, 10)
            length = random.randint(breadth, 10)
            self.walls_data = [
                ([length / 2, breadth / 2, 0.4], [length / 2, -breadth / 2, 0.4]),
                ([length / 2, -breadth / 2, 0.4], [-length / 2, -breadth / 2, 0.4]),
                ([-length / 2, -breadth / 2, 0.4], [-length / 2, breadth / 2, 0.4]),
                ([-length / 2, breadth / 2, 0.4], [length / 2, breadth / 2, 0.4]),
            ]
        # L Shaped
        elif self.wall_type == 2:
            length = random.randint(3, 4)
            self.walls_data = [
                (
                    [length / 2, -length / 2, 0.4],
                    [length / 2, -3 * length / 2, 0.4],
                ),  # bottom right connecting the upper right most
                (
                    [length / 2, -3 * length / 2, 0.4],
                    [3 * length / 2, -3 * length / 2, 0.4],
                ),  # upper right most
                (
                    [3 * length / 2, -3 * length / 2, 0.4],
                    [3 * length / 2, -length / 2, 0.4],
                ),  # up right connecting the upper right most
                ([3 * length / 2, -length / 2, 0.4], [3 * length / 2, 3 * length / 2, 0.4]),  # top
                ([3 * length / 2, 3 * length / 2, 0.4], [length / 2, 3* length / 2, 0.4]),  # up left
                ([length / 2 , 3 * length / 2, 0.4], [-3 * length / 2, 3 * length / 2, 0.4]),  # left bottom
                ([-3 * length / 2, 3 * length / 2, 0.4], [-3 * length / 2, -length / 2, 0.4]),  # bottom
                ([-3 * length / 2, -length / 2, 0.4], [length / 2, -length / 2, 0.4]),  # right bottom
            ]

        # elif self.wall_type == 2:
        #     length = random.randint(3, 4)
        #     self.walls_data = [
        #         (
        #             [length, -length, 0.4],
        #             [length, -2 * length, 0.4],
        #         ),  # bottom right connecting the upper right most
        #         (
        #             [length, -2 * length, 0.4],
        #             [2 * length, -2 * length, 0.4],
        #         ),  # upper right most
        #         (
        #             [2 * length, -2 * length, 0.4],
        #             [2 * length, -length, 0.4],
        #         ),  # up right connecting the upper right most
        #         ([2 * length, -length, 0.4], [2 * length, length, 0.4]),  # top
        #         ([2 * length, length, 0.4], [length, length, 0.4]),  # up left
        #         ([length, length, 0.4], [-length, length, 0.4]),  # left bottom
        #         ([-length, length, 0.4], [-length, -length, 0.4]),  # bottom
        #         ([-length, -length, 0.4], [length, -length, 0.4]),  # right bottom
        #     ]

        poly = []
        for i in range(len(self.walls_data)):
            if self.wall_type == 2 and i == len(self.walls_data) - 1:
                break
            poly.append((self.walls_data[i][0][0], self.walls_data[i][0][1]))

        self.boundary = Polygon(poly)

        self.walls = [self.coppelia.create_wall(w[0], w[1]) for w in self.walls_data]
        print(self.walls_data)
        self.data["walls"] = []

        for w in self.walls:
            self.walls_IND.append(IND)
            self.data["walls"].append(w)
            self.object_list.append(w)
            IND = IND + 1

        # Adding threshold
        breadth = 0
        if self.wall_type == 0:
            self.length = length / 2 - 0.6
            length = length / 2 - 0.6
        elif self.wall_type == 1:
            self.length = length / 2 - 0.6
            length = length / 2 - 0.6
            self.breadth = breadth / 2 - 0.6
            breadth = breadth / 2 - 0.6
        elif self.wall_type:
            self.length = length - 0.6
            length = length - 0.6

        # Add Goal
        x, y = self.set_entity_position()
        self.goal_data = [x, y]
        self.goal = self.coppelia.create_goal(self.goal_data[0], self.goal_data[1])
        self.data["goal"] = [self.goal]
        self.object_list.append(self.goal)
        self.goal.set_renderable(show_goal)

        self.robot = YouBot()
        if robot_random_pose:
            x, y = self.set_entity_position()
            while True:
                x, y = (random.random() - 0.5) * 20.0, (random.random() - 0.5) * 20.0
                self.robot.set_position([x, y, 0.09533682])
                if contained_with_radius(self.boundary, Point(x, y), human_radius):
                    break

            self.robot.set_orientation(
                [8.81747201e-06, 5.42909198e-04, random.random() * 2.0 * math.pi]
            )
        else:
            self.robot.set_position([0, 0, 0.09533682])
            self.robot.set_orientation([8.81747201e-06, 5.42909198e-04, random.randint(0, 3) * math.pi / 2.])

        self.object_list.append(self.robot)

        for child in self.coppelia.get_objects_in_tree():
            name = child.get_name()
            if name == "camera_fr":
                camera_fr = child
                camera_fr.set_parent(self.robot, keep_in_place=False)
                camera_fr.set_model(self.robot)
                camera_fr.set_position([0.0, 0.0, 0.0], relative_to=self.robot)
                camera_fr.set_orientation([0.0, 0.0, pi])
                camera_fr.set_model_respondable(False)
                camera_fr.set_model_dynamic(False)
                camera_fr.set_collidable(False)
                camera_fr.set_measurable(False)
                camera_fr.set_model_collidable(False)
                camera_fr.set_model_measurable(False)
            elif name == "Vision_sensor":
                vision_sensor = child
                entity_to_render = vision_sensor.get_entity_to_render()
                far_clipping_plane = vision_sensor.get_far_clipping_plane()
                near_clipping_plane = vision_sensor.get_near_clipping_plane()
                perspective_angle = vision_sensor.get_perspective_angle()
                perspective_mode = vision_sensor.get_perspective_mode()
                render_mode = vision_sensor.get_render_mode()
                resolution = vision_sensor.get_resolution()
                window_size = vision_sensor.get_windowed_size()

        n_humans = random.randint(self.min_humans, self.max_humans)

        for i in range(n_humans):
            a = self.coppelia.create_human()

            x, y = self.set_entity_position()
            ang = self.set_entity_orientation()

            p = [x, y, 0]
            a.set_position(p)
            a.move(p)
            a.set_orientation([0, 0, ang])

            self.count = 0
            while self.check_collision(a, self.object_list):
                x, y = self.set_entity_position()

                p = [x, y, 0]
                a.set_position(p)
                a.move(p)
                self.count += 1
                if self.count == self.break_point:
                    self.coppelia.remove_object(a)
                    n_humans -= 1
                    break

            if self.count != self.break_point:
                self.object_list.append(a)
                self.humans_IND.append(IND)
                IND = IND + 1
                self.humans.append(a)

        n_wandering_humans = random.randint(
            self.min_wandHumans, self.max_wandHumans
        )  # input to be taken from UI
        for i in range(n_wandering_humans):
            a = self.coppelia.create_human()

            x, y = self.set_entity_position()

            p = [x, y, 0]
            a.set_position(p)
            a.move(p)

            self.count = 0
            while self.check_collision(a, self.object_list):
                x, y = self.set_entity_position()

                p = [x, y, 0]
                a.set_position(p)
                a.move(p)
                self.count += 1
                if self.count == self.break_point:
                    self.coppelia.remove_object(a)
                    n_wandering_humans -= 1
                    break

            if self.count != self.break_point:
                self.object_list.append(a)
                self.humans_IND.append(IND)
                IND = IND + 1
                self.humans.append(a)

        self.wandering_humans = []
        for h in self.humans[n_humans:]:
            self.wandering_humans.append(HumanMovementRandomiser(h, 12))

        self.data["humans"] = tuple(self.humans)

        # Add tables and laptops
        n_tables = random.randint(self.min_tables, self.max_tables)  # input from UI
        for i in range(n_tables):

            a = self.coppelia.load_model("small_table.ttm")
            b = self.coppelia.load_model("models/office items/laptop.ttm")
            # human_present = random.randint(0,1)

            x, y = self.set_entity_position()

            p_a = [x, y, 0.85]
            p_b = [x, y, 0.975]
            a.set_position(p_a)
            b.set_position(p_b)

            self.count = 0
            while self.check_collision(a, self.object_list):
                x, y = self.set_entity_position()

                p_a = [x, y, 0.85]
                p_b = [x, y, 0.975]
                a.set_position(p_a)
                b.set_position(p_b)

                self.count += 1
                if self.count == self.break_point:
                    self.coppelia.remove_object(a)
                    self.coppelia.remove_object(b)
                    n_tables -= 1
                    break

            if self.count != self.break_point:
                self.object_list.append(a)
                self.object_list.append(b)
                self.tables_IND.append(IND)
                IND = IND + 1
                self.tables.append(a)
                self.laptops.append(b)

        ##just for adding index for laptop
        for iii in self.laptops:
            self.laptops_IND.append(IND)
            IND = IND + 1
        ##

        self.data["table"] = tuple(self.tables)
        self.data["laptop"] = tuple(self.laptops)

        # Add plants
        n_plants = random.randint(self.min_plants, self.max_plants)
        for i in range(n_plants):
            a = self.coppelia.load_model("models/furniture/plants/indoorPlant.ttm")

            x, y = self.set_entity_position()

            p_a = [x, y, 0.165]
            a.set_position(p_a)

            self.count = 0
            while self.check_collision(a, self.object_list):
                x, y = self.set_entity_position()
                p_a = [x, y, 0.165]
                a.set_position(p_a)

                self.count += 1
                if self.count == self.break_point:
                    self.coppelia.remove_object(a)
                    n_plants -= 1
                    break

            if self.count != self.break_point:
                self.object_list.append(a)
                self.plants_IND.append(IND)
                IND = IND + 1
                self.plants.append(a)

        self.data["plant"] = tuple(self.plants)

        num_relations = random.randint(self.min_relations, self.max_relations)
        for i in range(n_tables):
            self.table_indices.append(0)
        for i in range(n_plants):
            self.plant_indices.append(0)
        for i in range(n_humans):
            self.human_indices.append(0)
        for i in range(n_wandering_humans):
            self.wandering_humans_indices.append(0)

        for i in range(num_relations):

            relation_type = random.randint(
                0, 3
            )  # 0 for human-human (static); 1 for human-table, 2 for human-plant, 3 for human-human (wandering)
            relation_priority_list = [(i + relation_type) % 4 for i in range(4)]
            for rel in relation_priority_list:
                if self.create_interaction(rel, n_humans, length, breadth, show_relations):
                    break

        for t in self.tables:
            self.position_list.append(t.get_position())

        for h in self.humans:
            self.position_list.append(h.get_position())

        for p in self.plants:
            self.position_list.append(p.get_position())

        return self.data, self.wandering_humans

    def get_simulation_timestep(self):
        return self.coppelia.get_simulation_timestep()

    def soda_compute(self, people, objects, walls, goals):
        self.coppelia.step()
        if self.frame_of_reference == 'R':
            frame_of_reference = self.robot
        else:
            frame_of_reference = None
        # Get robot's position
        robot_position = self.robot.get_position()
        robot_orientation = self.robot.get_orientation()

        # Get the position of humans A/B and add it to the world
        for id_available, human in zip(self.humans_IND, self.data["humans"]):
            person = Person()
            person.id = id_available - len(self.walls_IND)
            person.x, person.y, _ = human.get_position(relative_to=frame_of_reference)
            person.angle = -human.get_orientation(relative_to=frame_of_reference)[2]
            people.append(person)

        # Update humans' trajectories
        for i, h in enumerate(self.wandering_humans):
            if i in self.relation_to_human_map.keys():
                h.update(
                    self.position_list,
                    self.boundary,
                    self.relation_to_human_map[i],
                    self.relations_moving_humans,
                    robot_position,
                )
            else:
                h.update(self.position_list, self.boundary, None, None, robot_position)

        # Add objects
        for id_available, obj in zip(self.tables_IND, self.data["table"]):
            obj2 = ObjectT()
            obj2.id = id_available - len(self.walls_IND)
            obj2.x, obj2.y, _ = obj.get_position(relative_to=frame_of_reference)
            obj2.angle = -obj.get_orientation(relative_to=frame_of_reference)[2]
            obj_shape = obj.get_model_bounding_box()
            obj2.bbx1 = -obj_shape[1]
            obj2.bbx2 = obj_shape[1]
            obj2.bby1 = -obj_shape[3]
            obj2.bby2 = obj_shape[3]
            objects.append(obj2)
            # id_available += 1

        # Add objects
        for id_available, obj in zip(self.laptops_IND, self.data["laptop"]):
            obj2 = ObjectT()
            obj2.id = id_available - len(self.walls_IND)
            obj2.x, obj2.y, _ = obj.get_position(relative_to=frame_of_reference)
            obj2.angle = -obj.get_orientation(relative_to=frame_of_reference)[2]
            obj_shape = obj.get_model_bounding_box()
            obj2.bbx1 = obj_shape[0]
            obj2.bbx2 = obj_shape[1]
            obj2.bby1 = obj_shape[2]
            obj2.bby2 = obj_shape[3]
            objects.append(obj2)
            # id_available += 1

        # Add objects
        for id_available, obj in zip(self.plants_IND, self.data["plant"]):
            obj2 = ObjectT()
            obj2.id = id_available - len(self.walls_IND)
            obj2.x, obj2.y, _ = obj.get_position(relative_to=frame_of_reference)
            obj2.angle = -obj.get_orientation(relative_to=frame_of_reference)[2]
            obj_shape = obj.get_model_bounding_box()

            obj2.bbx1 = obj_shape[0]
            obj2.bbx2 = obj_shape[1]
            obj2.bby1 = obj_shape[2]
            obj2.bby2 = obj_shape[3]
            objects.append(obj2)
            # id_available += 1

        # add robots as an object
        obj3 = ObjectT()
        obj3.id = -2
        obj3.x, obj3.y, _ = self.robot.get_position()
        obj3.angle = -self.robot.get_orientation()[2]
        obj_shape = [1, 1, 1, 1]
        obj3.bbx1 = obj_shape[0]
        obj3.bbx2 = obj_shape[1]
        obj3.bby1 = obj_shape[2]
        obj3.bby2 = obj_shape[3]
        objects.append(obj3)

        for id_available, wall in zip(self.walls_IND, self.data["walls"]):
            w = WallT()
            p = wall.get_position(relative_to=frame_of_reference)[0:2]
            ang = wall.get_orientation(relative_to=frame_of_reference)[2]
            l = wall.get_length()
            w.x1 = p[0] - cos(ang) * l / 2.0
            w.y1 = p[1] - sin(ang) * l / 2.0
            w.x2 = p[0] + cos(ang) * l / 2.0
            w.y2 = p[1] + sin(ang) * l / 2.0
            walls.append(w)

        interactions = []
        for inter in self.interacting_humans:
            i = InteractionT()
            i.idSrc = inter["src"] - len(self.walls_IND)
            i.idDst = inter["dst"] - len(self.walls_IND)
            i.type = inter["relationship"]
            interactions.append(i)

        for gg in self.data["goal"]:
            goall = GoalT()
            goall.x, goall.y, _ = gg.get_position(relative_to=frame_of_reference)
            goals.append(goall)

        return self.data, people, objects, interactions, walls, goals[0]

    def create_interaction(self, relation_type, n_humans, length, breadth, show_relations = True):
        flag = False
        if relation_type == 0:

            if (
                self.human_indices.count(0) >= 2
            ):  # human-human(static) (CHECKS: human list not empty)
                for i in range(len(self.human_indices)):
                    if self.human_indices[i] == 0:
                        ind_1 = i
                        break

                for i in range(len(self.human_indices)):
                    if self.human_indices[i] == 0 and ind_1 != i:
                        ind_2 = i
                        break

                if self.wall_type == 0 or self.wall_type == 2:
                    ind1, ind2, flag = self.two_static_person_talking(
                        length, length, ind_1, ind_2
                    )
                    if flag:
                        self.human_indices[ind1] = 1
                        self.human_indices[ind2] = 1
                        self.interacting_humans.append(
                            {
                                "src": self.humans_IND[ind_1],
                                "dst": self.humans_IND[ind_2],
                                "relationship": "two_static_person_talking",
                            }
                        )
                        # Add relation cylinder
                        cylinder = self.create_interaction_cylinder(
                            self.humans[ind_1].get_position(),
                            self.humans[ind_2].get_position(),
                            show_relations
                        )
                        self.relations.append(cylinder)

                else:
                    ind1, ind2, flag = self.two_static_person_talking(
                        length, breadth, ind_1, ind_2
                    )
                    if flag:
                        self.human_indices[ind1] = 1
                        self.human_indices[ind2] = 1
                        self.interacting_humans.append(
                            {
                                "src": self.humans_IND[ind_1],
                                "dst": self.humans_IND[ind_2],
                                "relationship": "two_static_person_talking",
                            }
                        )
                        # Add relation cylinder
                        cylinder = self.create_interaction_cylinder(
                            self.humans[ind_1].get_position(),
                            self.humans[ind_2].get_position(),
                            show_relations
                        )
                        self.relations.append(cylinder)

        elif relation_type == 1:

            if (
                self.human_indices.count(0) >= 1 and self.table_indices.count(0) >= 1
            ):  # human-table (CHECKS: table or human list not empty)
                for i in range(len(self.human_indices)):
                    if self.human_indices[i] == 0:
                        ind_h = i
                        break

                for i in range(len(self.table_indices)):
                    if self.table_indices[i] == 0:
                        ind_t = i
                        break

                if self.wall_type == 0 or self.wall_type == 2:
                    ind_h, ind_t, flag = self.human_laptop_interaction(
                        length, length, ind_h, ind_t
                    )
                    if flag:
                        self.human_indices[ind_h] = 1
                        self.table_indices[ind_t] = 1
                        self.interacting_humans.append(
                            {
                                "src": self.humans_IND[ind_h],
                                "dst": self.tables_IND[ind_t],
                                "relationship": "human_laptop_interaction",
                            }
                        )
                        # Add relation cylinder
                        cylinder = self.create_interaction_cylinder(
                            self.humans[ind_h].get_position(),
                            self.tables[ind_t].get_position(),
                            show_relations
                        )
                        self.relations.append(cylinder)
                else:
                    ind_h, ind_t, flag = self.human_laptop_interaction(
                        length, breadth, ind_h, ind_t
                    )
                    if flag:
                        self.human_indices[ind_h] = 1
                        self.table_indices[ind_t] = 1
                        self.interacting_humans.append(
                            {
                                "src": self.humans_IND[ind_h],
                                "dst": self.tables_IND[ind_t],
                                "relationship": "human_laptop_interaction",
                            }
                        )
                        # Add relation cylinder
                        cylinder = self.create_interaction_cylinder(
                            self.humans[ind_h].get_position(),
                            self.tables[ind_t].get_position(),
                            show_relations
                        )
                        self.relations.append(cylinder)

        elif relation_type == 2:
            if (
                self.human_indices.count(0) >= 1 and self.plant_indices.count(0) >= 1
            ):  # human-plant (CHECKS: plant or human list not empty)
                for i in range(len(self.human_indices)):
                    if self.human_indices[i] == 0:
                        ind_h = i
                        break

                for i in range(len(self.plant_indices)):
                    if self.plant_indices[i] == 0:
                        ind_p = i
                        break

                if self.wall_type == 0 or self.wall_type == 2:
                    ind_h, ind_p, flag = self.human_plant_interaction(
                        length, length, ind_p, ind_h
                    )
                    if flag:
                        self.human_indices[ind_h] = 1
                        self.plant_indices[ind_p] = 1
                        self.interacting_humans.append(
                            {
                                "src": self.humans_IND[ind_h],
                                "dst": self.plants_IND[ind_p],
                                "relationship": "human_plant_interaction",
                            }
                        )
                        # Add relation cylinder
                        cylinder = self.create_interaction_cylinder(
                            self.humans[ind_h].get_position(),
                            self.plants[ind_p].get_position(),
                            show_relations
                        )
                        self.relations.append(cylinder)
                else:
                    ind_h, ind_p, flag = self.human_plant_interaction(
                        length, breadth, ind_p, ind_h
                    )
                    if flag:
                        self.human_indices[ind_h] = 1
                        self.plant_indices[ind_p] = 1
                        self.interacting_humans.append(
                            {
                                "src": self.humans_IND[ind_h],
                                "dst": self.plants_IND[ind_p],
                                "relationship": "human_plant_interaction",
                            }
                        )
                        # Add relation cylinder
                        cylinder = self.create_interaction_cylinder(
                            self.humans[ind_h].get_position(),
                            self.plants[ind_p].get_position(),
                            show_relations
                        )
                        self.relations.append(cylinder)

        elif relation_type == 3:
            # print("trying to create wandering humans interacting")
            if (
                self.wandering_humans_indices.count(0) >= 2
            ):  # wandering human-human (CHECKS: wandering human list not empty)
                for i in range(len(self.wandering_humans_indices)):
                    if self.wandering_humans_indices[i] == 0:
                        ind_1 = i
                        break

                for i in range(len(self.wandering_humans_indices)):
                    if self.wandering_humans_indices[i] == 0 and ind_1 != i:
                        ind_2 = i
                        break

                if self.wall_type == 0 or self.wall_type == 2:
                    ind_1, ind_2, flag = self.wandering_human_interacting(
                        length, length, ind_1, ind_2
                    )
                    if flag:
                        self.wandering_humans_indices[ind_1] = 1
                        self.wandering_humans_indices[ind_2] = 1
                        self.interacting_humans.append(
                            {
                                "src": self.humans_IND[n_humans::][ind_1],
                                "dst": self.humans_IND[n_humans::][ind_2],
                                "relationship": "wandering_human_interacting",
                            }
                        )
                        # Add relation cylinder
                        cylinder = self.create_interaction_cylinder(
                            self.humans[n_humans::][ind_1].get_position(),
                            self.humans[n_humans::][ind_2].get_position(),
                            show_relations
                        )
                        self.relations_moving_humans.append(cylinder)
                        self.relation_to_human_map[ind_1] = (
                            len(self.relations_moving_humans) - 1
                        )
                        self.relation_to_human_map[ind_2] = (
                            len(self.relations_moving_humans) - 1
                        )
                else:
                    ind_1, ind_2, flag = self.wandering_human_interacting(
                        length, breadth, ind_1, ind_2
                    )
                    if flag:
                        self.wandering_humans_indices[ind_1] = 1
                        self.wandering_humans_indices[ind_2] = 1
                        self.interacting_humans.append(
                            {
                                "src": self.humans_IND[n_humans::][ind_1],
                                "dst": self.humans_IND[n_humans::][ind_2],
                                "relationship": "wandering_human_interacting",
                            }
                        )
                        # Add relation cylinder
                        cylinder = self.create_interaction_cylinder(
                            self.humans[n_humans::][ind_1].get_position(),
                            self.humans[n_humans::][ind_2].get_position(),
                            show_relations
                        )
                        self.relations_moving_humans.append(cylinder)
                        self.relation_to_human_map[ind_1] = (
                            len(self.relations_moving_humans) - 1
                        )
                        self.relation_to_human_map[ind_2] = (
                            len(self.relations_moving_humans) - 1
                        )
        return flag

    def create_interaction_cylinder(self, pos1, pos2, show_relations = True):
        center = (pos1 + pos2) / 2.0
        length = math.sqrt(sum([(a - b) ** 2 for a, b in zip(pos1, pos2)]))
        orientation = math.atan2(pos2[1] - pos1[1], pos2[0] - pos1[0]) + math.pi / 2.0
        cylinder = self.coppelia.create_relation(
            center[0], center[1], 0.0, math.pi / 2.0, orientation, math.pi / 2.0, length
        )
        cylinder.set_renderable(show_relations)
        return cylinder

    def check_collision(self, obj, obj_list):
        for obj2 in obj_list:
            if obj is not obj2:
                if self.coppelia.check_collision(obj, obj2):
                    return True
        return False

    def two_static_person_talking(self, length, breadth, ind_1, ind_2):
        collision1 = True
        collision2 = True

        self.count = 0
        flag = True

        prev_pose_1 = self.data["humans"][ind_1].get_position()
        prev_orientation_1 = self.data["humans"][ind_1].get_orientation()
        prev_pose_2 = self.data["humans"][ind_2].get_position()
        prev_orientation_2 = self.data["humans"][ind_2].get_orientation()

        ang = random.uniform(-20, 20)
        ang = float((ang * math.pi) / 180.0)

        while collision1 or collision2:
            x, y = self.set_entity_position()

            dist = random.uniform(1.5, 3.0)
            self.data["humans"][ind_1].set_position([x, y, 0])
            self.data["humans"][ind_2].set_position(
                [dist, 0, 0], relative_to=self.data["humans"][ind_1]
            )
            self.data["humans"][ind_2].set_orientation(
                [0, 0, math.pi + ang], relative_to=self.data["humans"][ind_1]
            )
            collision1 = self.check_collision(
                self.data["humans"][ind_1], self.object_list
            )
            collision2 = self.check_collision(
                self.data["humans"][ind_2], self.object_list
            )

            pos2 = self.data["humans"][ind_2].get_position()
            point = Point(pos2[0], pos2[1])

            if not contained_with_radius(self.boundary, point, human_radius):
                collision2 = True

            point = Point(x, y)

            if not contained_with_radius(self.boundary, point, human_radius):
                collision1 = True

            self.count += 1

            if self.count == self.break_point:
                self.data["humans"][ind_1].set_position(prev_pose_1)
                self.data["humans"][ind_1].set_orientation(prev_orientation_1)
                self.data["humans"][ind_2].set_position(prev_pose_2)
                self.data["humans"][ind_2].set_orientation(prev_orientation_2)
                flag = False
                break

        return ind_1, ind_2, flag

    def human_laptop_interaction(self, length, breadth, ind_h, ind_t):

        collision = True

        pose = self.tables[ind_t].get_position()

        self.count = 0
        flag = True

        prev_pose = self.data["humans"][ind_h].get_position()
        prev_orientation = self.data["humans"][ind_h].get_orientation()

        while collision:
            dist = random.uniform(1.0, 4.0)
            self.data["humans"][ind_h].set_position([pose[0], pose[1] - dist, 0])
            self.data["humans"][ind_h].set_orientation([0, 0, 1.57])

            point = Point(pose[0], pose[1] - dist)
            collision = self.check_collision(
                self.data["humans"][ind_h], self.object_list
            )

            if not contained_with_radius(self.boundary, point, laptop_radius):
                collision = True

            self.count += 1

            if self.count == self.break_point:
                self.data["humans"][ind_h].set_position(prev_pose)
                self.data["humans"][ind_h].set_orientation(prev_orientation)
                flag = False
                break

        return ind_h, ind_t, flag

    def human_plant_interaction(self, length, breadth, ind_p, ind_h):

        collision = True

        pose = self.plants[ind_p].get_position()

        self.count = 0
        flag = True

        prev_pose = self.data["humans"][ind_h].get_position()
        prev_orientation = self.data["humans"][ind_h].get_orientation()

        while collision:
            dist = random.uniform(1.0, 4.0)
            self.data["humans"][ind_h].set_position([pose[0], pose[1] - dist, 0])
            self.data["humans"][ind_h].set_orientation([0, 0, 1.57])

            point = Point(pose[0], pose[1] - dist)
            collision = self.check_collision(
                self.data["humans"][ind_h], self.object_list
            )

            if not contained_with_radius(self.boundary, point, plant_radius):
                collision = True

            self.count += 1

            if self.count == self.break_point:
                self.data["humans"][ind_h].set_position(prev_pose)
                self.data["humans"][ind_h].set_orientation(prev_orientation)
                flag = False
                break

        return ind_h, ind_p, flag

    def wandering_human_interacting(self, length, breadth, ind_1, ind_2):

        collision1 = True
        collision2 = True

        self.count = 0
        flag = True

        prev_pose_1 = self.data["humans"][ind_1].get_position()
        prev_orientation_1 = self.data["humans"][ind_1].get_orientation()
        prev_pose_2 = self.data["humans"][ind_2].get_position()
        prev_orientation_2 = self.data["humans"][ind_2].get_orientation()

        while collision1 or collision2:

            x, y = self.set_entity_position()

            self.wandering_humans[ind_1].human.set_position([x, y, 0])
            self.wandering_humans[ind_1].human.set_orientation([0, 0, 0])
            self.wandering_humans[ind_2].human.set_position([x, y + 0.5, 0])
            self.wandering_humans[ind_2].human.set_orientation([0, 0, 0])
            collision1 = self.check_collision(
                self.wandering_humans[ind_1].human, self.object_list
            )
            collision2 = self.check_collision(
                self.wandering_humans[ind_2].human, self.object_list
            )

            self.count += 1
            if self.count == self.break_point:
                self.data["humans"][ind_1].set_position(prev_pose_1)
                self.data["humans"][ind_1].set_orientation(prev_orientation_1)
                self.data["humans"][ind_2].set_position(prev_pose_2)
                self.data["humans"][ind_2].set_orientation(prev_orientation_2)
                flag = False
                break

        if self.count != self.break_point:
            self.wandering_humans[ind_2].human.set_position([x, y + 0.5, 0])
            self.wandering_humans[ind_2].human.set_orientation([0, 0, 0])
            self.wandering_humans[ind_1].moving_with_friend(
                self.wandering_humans[ind_2].human
            )
            self.wandering_humans[ind_2].moving_with_friend()

        return ind_1, ind_2, flag

    def set_entity_position(self):
        if self.wall_type == 0:
            x = random.uniform(-self.length, self.length)
            y = random.uniform(-self.length, self.length)
        elif self.wall_type == 1:
            y = random.uniform(-self.breadth, self.breadth)
            x = random.uniform(-self.length, self.length)
        elif self.wall_type == 2:
            x = random.uniform(-3* self.length / 2, 3 * self.length / 2)
            if x > self.length / 2 + 0.5:
                y = random.uniform(-3 * self.length / 2, 3* self.length / 2)
            else:
                y = random.uniform(-self.length / 2, 3 * self.length / 2)

        # elif self.wall_type == 2:
        #     x = random.uniform(-self.length, 2 * self.length)
        #     if x > self.length + 0.5:
        #         y = random.uniform(-2 * self.length, self.length)
        #     else:
        #         y = random.uniform(-self.length, self.length)

        return x, y

    def set_entity_orientation(self):
        ang = random.uniform(-180, 180)
        return float(ang * math.pi / 180.0)

    def room_setup_sc1(self):
        self.coppelia.remove_objects(
            self.humans,
            self.tables,
            self.laptops,
            self.plants,
            self.goal,
            self.walls,
            self.relations,
            self.relations_moving_humans,
        )

        self.interacting_humans = []

        self.data = self.ini_data
        self.object_list = []
        self.position_list = (
            []
        )  # checking collision for wandering humans (has plants tables and static humans)

        self.humans = []
        self.wandering_humans = []
        self.tables = []
        self.plants = []
        self.laptops = []
        self.relations = []
        self.relations_moving_humans = []
        self.relation_to_human_map = {}
        self.wall_type = random.randint(
            0, 2
        )  # 0 for square, 1 for rectangle, 2 for L-shaped
        self.boundary = None
        self.human_indices = []
        self.table_indices = []
        self.plant_indices = []
        self.wandering_humans_indices = []
        self.length = None
        self.breadth = None
        self.break_point = 5  # when to stop creating an entity
        self.count = 0  # storing count till break point
        self.n_interactions = None

        self.humans_IND = []
        self.tables_IND = []
        self.plants_IND = []
        self.laptops_IND = []
        self.walls_IND = []

        IND = 1

        # Add four walls
        self.wall_type = 1
        breadth = 8.0
        length = 8.0
        self.walls_data = [
            ([length / 2, breadth / 2, 0.4], [length / 2, -breadth / 2, 0.4]),
            ([length / 2, -breadth / 2, 0.4], [-length / 2, -breadth / 2, 0.4]),
            ([-length / 2, -breadth / 2, 0.4], [-length / 2, breadth / 2, 0.4]),
            ([-length / 2, breadth / 2, 0.4], [length / 2, breadth / 2, 0.4]),
        ]

        poly = []
        for i in range(len(self.walls_data)):
            if self.wall_type == 2 and i == len(self.walls_data) - 1:
                break
            poly.append((self.walls_data[i][0][0], self.walls_data[i][0][1]))

        self.boundary = Polygon(poly)

        self.walls = [self.coppelia.create_wall(w[0], w[1]) for w in self.walls_data]
        self.data["walls"] = []

        for w in self.walls:
            self.walls_IND.append(IND)
            self.data["walls"].append(w)
            self.object_list.append(w)
            IND = IND + 1

        # Adding threshold
        breadth = 0
        if self.wall_type == 0:
            self.length = length / 2 - 0.6
            length = length / 2 - 0.6
        elif self.wall_type == 1:
            self.length = length / 2 - 0.6
            length = length / 2 - 0.6
            self.breadth = breadth / 2 - 0.6
            breadth = breadth / 2 - 0.6
        elif self.wall_type:
            self.length = length - 0.6
            length = length - 0.6

        # Add Goal
        x, y = 3.3, 3.3
        self.goal_data = [x, y]
        self.goal = self.coppelia.create_goal(self.goal_data[0], self.goal_data[1])
        self.data["goal"] = [self.goal]
        self.object_list.append(self.goal)

        self.robot = YouBot()
        x, y = 0, -4
        self.robot.set_orientation([0, 0, 0])
        self.object_list.append(self.robot)

        for child in self.coppelia.get_objects_in_tree():
            name = child.get_name()
            if name == "camera_fr":
                camera_fr = child
                camera_fr.set_parent(self.robot, keep_in_place=False)
                camera_fr.set_model(self.robot)
                camera_fr.set_position([0.0, 0.0, 0.0], relative_to=self.robot)
                camera_fr.set_orientation([0.0, 0.0, pi])
                camera_fr.set_model_respondable(False)
                camera_fr.set_model_dynamic(False)
                camera_fr.set_collidable(False)
                camera_fr.set_measurable(False)
                camera_fr.set_model_collidable(False)
                camera_fr.set_model_measurable(False)
            elif name == "Vision_sensor":
                vision_sensor = child
                entity_to_render = vision_sensor.get_entity_to_render()
                far_clipping_plane = vision_sensor.get_far_clipping_plane()
                near_clipping_plane = vision_sensor.get_near_clipping_plane()
                perspective_angle = vision_sensor.get_perspective_angle()
                perspective_mode = vision_sensor.get_perspective_mode()
                render_mode = vision_sensor.get_render_mode()
                resolution = vision_sensor.get_resolution()
                window_size = vision_sensor.get_windowed_size()

        self.wandering_humans = []
        for x in [-2.6, -2.0, -1.4, -0.8, 3]:
            a = self.coppelia.create_human()
            y = 3.6
            p = [x, y, 0]
            a.set_position(p)
            self.object_list.append(a)
            self.humans_IND.append(IND)
            IND = IND + 1
            self.humans.append(a)
            a.move([x, -3.6, 0])
        for y in [-2.6, -2.0, -1.4, -0.8, 3]:
            a = self.coppelia.create_human()
            x = -3.6
            p = [x, y, 0]
            a.set_position(p)
            self.object_list.append(a)
            self.humans_IND.append(IND)
            IND = IND + 1
            self.humans.append(a)
            a.move([3.6, y, 0])

        self.data["humans"] = tuple(self.humans)

        # #Add tables and laptops
        # a = self.coppelia.load_model('small_table.ttm')
        # b = self.coppelia.load_model('models/office items/laptop.ttm')
        # x,y = self.set_entity_position()
        # p_a = [x,y,0.85]
        # p_b = [x,y,0.975]
        # a.set_position(p_a)
        # b.set_position(p_b)
        # self.object_list.append(a)
        # self.object_list.append(b)
        # self.tables_IND.append(IND)
        # IND = IND + 1
        # self.tables.append(a)
        # self.laptops.append(b)

        ##just for adding index for laptop
        for iii in self.laptops:
            self.laptops_IND.append(IND)
            IND = IND + 1
        ##

        self.data["table"] = tuple(self.tables)
        self.data["laptop"] = tuple(self.laptops)

        # Add plants
        a = self.coppelia.load_model("models/furniture/plants/indoorPlant.ttm")
        x, y = self.set_entity_position()
        p_a = [0, 0, 0.165]
        a.set_position(p_a)
        self.object_list.append(a)
        self.plants_IND.append(IND)
        IND = IND + 1
        self.plants.append(a)

        self.data["plant"] = tuple(self.plants)

        for i in range(len(self.tables)):
            self.table_indices.append(0)
        for i in range(len(self.plants)):
            self.plant_indices.append(0)
        for i in range(len(self.humans)):
            self.human_indices.append(0)
        for i in range(len(self.wandering_humans)):
            self.wandering_humans_indices.append(0)

        # num_relations = random.randint(self.min_relations,self.max_relations)
        # for i in range(num_relations):
        #     relation_type = random.randint(0,3) # 0 for human-human (static); 1 for human-table, 2 for human-plant, 3 for human-human (wandering)
        #     relation_priority_list = [ (i+relation_type)%4 for i in range(4)]
        #     for rel in relation_priority_list:
        #         if self.create_interaction(rel, n_humans, length, breadth):
        #             break

        for t in self.tables:
            self.position_list.append(t.get_position())

        for h in self.humans:
            self.position_list.append(h.get_position())

        for p in self.plants:
            self.position_list.append(p.get_position())

        return self.data, self.wandering_humans

    def room_setup_two_groups(self):
        self.coppelia.remove_objects(
            self.humans,
            self.tables,
            self.laptops,
            self.plants,
            self.goal,
            self.walls,
            self.relations,
            self.relations_moving_humans,
        )

        self.interacting_humans = []

        self.data = self.ini_data
        self.object_list = []
        self.position_list = (
            []
        )  # checking collision for wandering humans (has plants tables and static humans)

        self.humans = []
        self.wandering_humans = []
        self.tables = []
        self.plants = []
        self.laptops = []
        self.relations = []
        self.relations_moving_humans = []
        self.relation_to_human_map = {}
        self.wall_type = random.randint(
            0, 2
        )  # 0 for square, 1 for rectangle, 2 for L-shaped
        self.boundary = None
        self.human_indices = []
        self.table_indices = []
        self.plant_indices = []
        self.wandering_humans_indices = []
        self.length = None
        self.breadth = None
        self.break_point = 5  # when to stop creating an entity
        self.count = 0  # storing count till break point
        self.n_interactions = None

        self.humans_IND = []
        self.tables_IND = []
        self.plants_IND = []
        self.laptops_IND = []
        self.walls_IND = []

        IND = 1

        # Add four walls
        self.wall_type = 1
        breadth = 9.0
        length = 9.0
        self.walls_data = [
            ([length / 2, breadth / 2, 0.4], [length / 2, -breadth / 2, 0.4]),
            ([length / 2, -breadth / 2, 0.4], [-length / 2, -breadth / 2, 0.4]),
            ([-length / 2, -breadth / 2, 0.4], [-length / 2, breadth / 2, 0.4]),
            ([-length / 2, breadth / 2, 0.4], [length / 2, breadth / 2, 0.4]),
        ]

        poly = []
        for i in range(len(self.walls_data)):
            if self.wall_type == 2 and i == len(self.walls_data) - 1:
                break
            poly.append((self.walls_data[i][0][0], self.walls_data[i][0][1]))

        self.boundary = Polygon(poly)

        self.walls = [self.coppelia.create_wall(w[0], w[1]) for w in self.walls_data]
        self.data["walls"] = []

        for w in self.walls:
            self.walls_IND.append(IND)
            self.data["walls"].append(w)
            self.object_list.append(w)
            IND = IND + 1

        # Adding threshold
        breadth = 0
        if self.wall_type == 0:
            self.length = length / 2 - 0.6
            length = length / 2 - 0.6
        elif self.wall_type == 1:
            self.length = length / 2 - 0.6
            length = length / 2 - 0.6
            self.breadth = breadth / 2 - 0.6
            breadth = breadth / 2 - 0.6
        elif self.wall_type:
            self.length = length - 0.6
            length = length - 0.6

        # Add Goal
        x, y = 2.3, 1.3
        self.goal_data = [x, y]
        self.goal = self.coppelia.create_goal(self.goal_data[0], self.goal_data[1])
        self.data["goal"] = [self.goal]
        self.object_list.append(self.goal)

        self.robot = YouBot()
        x, y = 0., 0
        p = self.robot.get_position()
        self.robot.set_position([x, p[1], p[2]])
        self.robot.set_orientation([0, 0, 0])
        self.object_list.append(self.robot)

        for child in self.coppelia.get_objects_in_tree():
            name = child.get_name()
            if name == "camera_fr":
                camera_fr = child
                camera_fr.set_parent(self.robot, keep_in_place=False)
                camera_fr.set_model(self.robot)
                camera_fr.set_position([0.0, 0.0, 0.0], relative_to=self.robot)
                camera_fr.set_orientation([0.0, 0.0, pi])
                camera_fr.set_model_respondable(False)
                camera_fr.set_model_dynamic(False)
                camera_fr.set_collidable(False)
                camera_fr.set_measurable(False)
                camera_fr.set_model_collidable(False)
                camera_fr.set_model_measurable(False)
            elif name == "Vision_sensor":
                vision_sensor = child
                entity_to_render = vision_sensor.get_entity_to_render()
                far_clipping_plane = vision_sensor.get_far_clipping_plane()
                near_clipping_plane = vision_sensor.get_near_clipping_plane()
                perspective_angle = vision_sensor.get_perspective_angle()
                perspective_mode = vision_sensor.get_perspective_mode()
                render_mode = vision_sensor.get_render_mode()
                resolution = vision_sensor.get_resolution()
                window_size = vision_sensor.get_windowed_size()

        self.wandering_humans = []
        for x in [-2.2, -1.6, -1., -0.4]:
            a = self.coppelia.create_human()
            y = 3.6
            p = [x, y, 0]
            a.set_position(p)
            self.object_list.append(a)
            self.humans_IND.append(IND)
            IND = IND + 1
            self.humans.append(a)
            a.move([x, -3.6, 0])
        for y in [-2.6, -2.0, -1.4, -0.8]:
            a = self.coppelia.create_human()
            x = -3.6
            p = [x, y, 0]
            a.set_position(p)
            self.object_list.append(a)
            self.humans_IND.append(IND)
            IND = IND + 1
            self.humans.append(a)
            a.move([3.6, y, 0])

        for x in [3.0]:
            a = self.coppelia.create_human()
            y = 3.6
            p = [x, y, 0]
            a.set_position(p)
            self.object_list.append(a)
            self.humans_IND.append(IND)
            IND = IND + 1
            self.humans.append(a)

        for y in [3.0]:
            a = self.coppelia.create_human()
            x = -3.6
            p = [x, y, 0]
            a.set_position(p)
            self.object_list.append(a)
            self.humans_IND.append(IND)
            IND = IND + 1
            self.humans.append(a)


        self.data["humans"] = tuple(self.humans)
        self.data["table"] = tuple(self.tables)
        self.data["laptop"] = tuple(self.laptops)

        # Add plants
        a = self.coppelia.load_model("models/furniture/plants/indoorPlant.ttm")
        x, y = self.set_entity_position()
        p_a = [1., 0, 0.165]
        a.set_position(p_a)
        self.object_list.append(a)
        self.plants_IND.append(IND)
        IND = IND + 1
        self.plants.append(a)

        self.data["plant"] = tuple(self.plants)

        for i in range(len(self.tables)):
            self.table_indices.append(0)
        for i in range(len(self.plants)):
            self.plant_indices.append(0)
        for i in range(len(self.humans)):
            self.human_indices.append(0)
        for i in range(len(self.wandering_humans)):
            self.wandering_humans_indices.append(0)
        for t in self.tables:
            self.position_list.append(t.get_position())
        for h in self.humans:
            self.position_list.append(h.get_position())
        for p in self.plants:
            self.position_list.append(p.get_position())

        return self.data, self.wandering_humans

    def room_setup_one_group(self):
        self.coppelia.remove_objects(
            self.humans,
            self.tables,
            self.laptops,
            self.plants,
            self.goal,
            self.walls,
            self.relations,
            self.relations_moving_humans,
        )

        self.interacting_humans = []
        self.data = self.ini_data
        self.object_list = []
        self.position_list = (
            []
        )  # checking collision for wandering humans (has plants tables and static humans)

        self.humans = []
        self.wandering_humans = []
        self.tables = []
        self.plants = []
        self.laptops = []
        self.relations = []
        self.relations_moving_humans = []
        self.relation_to_human_map = {}
        self.wall_type = random.randint(
            0, 2
        )  # 0 for square, 1 for rectangle, 2 for L-shaped
        self.boundary = None
        self.human_indices = []
        self.table_indices = []
        self.plant_indices = []
        self.wandering_humans_indices = []
        self.length = None
        self.breadth = None
        self.break_point = 5  # when to stop creating an entity
        self.count = 0  # storing count till break point
        self.n_interactions = None

        self.humans_IND = []
        self.tables_IND = []
        self.plants_IND = []
        self.laptops_IND = []
        self.walls_IND = []

        IND = 1

        # Add four walls
        self.wall_type = 1
        breadth = 8.0
        length = 8.0
        self.walls_data = [
            ([length / 2, breadth / 2, 0.4], [length / 2, -breadth / 2, 0.4]),
            ([length / 2, -breadth / 2, 0.4], [-length / 2, -breadth / 2, 0.4]),
            ([-length / 2, -breadth / 2, 0.4], [-length / 2, breadth / 2, 0.4]),
            ([-length / 2, breadth / 2, 0.4], [length / 2, breadth / 2, 0.4]),
        ]

        poly = []
        for i in range(len(self.walls_data)):
            if self.wall_type == 2 and i == len(self.walls_data) - 1:
                break
            poly.append((self.walls_data[i][0][0], self.walls_data[i][0][1]))

        self.boundary = Polygon(poly)

        self.walls = [self.coppelia.create_wall(w[0], w[1]) for w in self.walls_data]
        self.data["walls"] = []

        for w in self.walls:
            self.walls_IND.append(IND)
            self.data["walls"].append(w)
            self.object_list.append(w)
            IND = IND + 1

        # Adding threshold
        breadth = 0
        if self.wall_type == 0:
            self.length = length / 2 - 0.6
            length = length / 2 - 0.6
        elif self.wall_type == 1:
            self.length = length / 2 - 0.6
            length = length / 2 - 0.6
            self.breadth = breadth / 2 - 0.6
            breadth = breadth / 2 - 0.6
        elif self.wall_type:
            self.length = length - 0.6
            length = length - 0.6

        # Add Goal
        x, y = 3.3, 3.3
        self.goal_data = [x, y]
        self.goal = self.coppelia.create_goal(self.goal_data[0], self.goal_data[1])
        self.data["goal"] = [self.goal]
        self.object_list.append(self.goal)

        self.robot = YouBot()
        x, y = 0, -1
        self.robot.set_orientation([0, 0, 0])
        self.object_list.append(self.robot)

        for child in self.coppelia.get_objects_in_tree():
            name = child.get_name()
            if name == "camera_fr":
                camera_fr = child
                camera_fr.set_parent(self.robot, keep_in_place=False)
                camera_fr.set_model(self.robot)
                camera_fr.set_position([0.0, 0.0, 0.0], relative_to=self.robot)
                camera_fr.set_orientation([0.0, 0.0, pi])
                camera_fr.set_model_respondable(False)
                camera_fr.set_model_dynamic(False)
                camera_fr.set_collidable(False)
                camera_fr.set_measurable(False)
                camera_fr.set_model_collidable(False)
                camera_fr.set_model_measurable(False)
            elif name == "Vision_sensor":
                vision_sensor = child
                entity_to_render = vision_sensor.get_entity_to_render()
                far_clipping_plane = vision_sensor.get_far_clipping_plane()
                near_clipping_plane = vision_sensor.get_near_clipping_plane()
                perspective_angle = vision_sensor.get_perspective_angle()
                perspective_mode = vision_sensor.get_perspective_mode()
                render_mode = vision_sensor.get_render_mode()
                resolution = vision_sensor.get_resolution()
                window_size = vision_sensor.get_windowed_size()

        self.wandering_humans = []
        for x in [-2.9, -2.4, -1.9, 0.5, 1.0, 2.5]:
            a = self.coppelia.create_human()
            y = 3.5
            p = [x, y, 0]
            a.set_position(p)
            self.object_list.append(a)
            self.humans_IND.append(IND)
            IND = IND + 1
            self.humans.append(a)
            a.move([x, -3.5, 0])
        for y in [-1.0]:
            a = self.coppelia.create_human()
            x = -3.2
            p = [x, y, 0]
            a.set_position(p)
            self.object_list.append(a)
            self.humans_IND.append(IND)
            IND = IND + 1
            self.humans.append(a)
            # a.move([-x, -y, 0])
        for x in [-1.0]:
            a = self.coppelia.create_human()
            y = 2.9
            p = [x, y, 0]
            a.set_position(p)
            a.set_orientation([0, 0, -3.141592 / 2.0])
            self.object_list.append(a)
            self.humans_IND.append(IND)
            IND = IND + 1
            self.humans.append(a)
            a = self.coppelia.create_human()
            y = -2.9
            p = [x, y, 0]
            a.set_position(p)
            a.set_orientation([0, 0, 3.141592 / 2.0])
            self.object_list.append(a)
            self.humans_IND.append(IND)
            IND = IND + 1
            self.humans.append(a)

        a = self.coppelia.create_human()
        x = 0
        y = 1.6
        p = [x, y, 0]
        a.set_position(p)
        a.set_orientation([0, 0, -3.141592 / 2.0])
        self.object_list.append(a)
        self.humans_IND.append(IND)
        IND = IND + 1
        self.humans.append(a)

        self.data["humans"] = tuple(self.humans)

        self.data["table"] = tuple(self.tables)
        self.data["laptop"] = tuple(self.laptops)

        # Add plants
        a = self.coppelia.load_model("models/furniture/plants/indoorPlant.ttm")
        x, y = self.set_entity_position()
        p_a = [0, 0, 0.165]
        a.set_position(p_a)
        self.object_list.append(a)
        self.plants_IND.append(IND)
        IND = IND + 1
        self.plants.append(a)

        self.data["plant"] = tuple(self.plants)

        for i in range(len(self.tables)):
            self.table_indices.append(0)
        for i in range(len(self.plants)):
            self.plant_indices.append(0)
        for i in range(len(self.humans)):
            self.human_indices.append(0)
        for i in range(len(self.wandering_humans)):
            self.wandering_humans_indices.append(0)
        for t in self.tables:
            self.position_list.append(t.get_position())
        for h in self.humans:
            self.position_list.append(h.get_position())
        for p in self.plants:
            self.position_list.append(p.get_position())

        return self.data, self.wandering_humans

    def room_setup_1(self, num_of_humans):
        self.coppelia.remove_objects(
            self.humans,
            self.tables,
            self.laptops,
            self.plants,
            self.goal,
            self.walls,
            self.relations,
            self.relations_moving_humans
        )

        self.interacting_humans = []

        self.data = self.ini_data
        self.object_list = []
        self.position_list = (
            []
        )  # checking collision for wandering humans (has plants tables and static humans)

        self.humans = []
        self.wandering_humans = []
        self.tables = []
        self.plants = []
        self.laptops = []
        self.relations = []
        self.relations_moving_humans = []
        self.relation_to_human_map = {}
        self.wall_type = random.randint(
            0, 2
        )  # 0 for square, 1 for rectangle, 2 for L-shaped
        self.boundary = None
        self.human_indices = []
        self.table_indices = []
        self.plant_indices = []
        self.wandering_humans_indices = []
        self.length = None
        self.breadth = None
        self.break_point = 5  # when to stop creating an entity
        self.count = 0  # storing count till break point
        self.n_interactions = None

        self.humans_IND = []
        self.tables_IND = []
        self.plants_IND = []
        self.laptops_IND = []
        self.walls_IND = []

        IND = 1

        # Add four walls
        self.wall_type = 2
        breadth = 4.0
        length = 4.0
        self.walls_data = [
            (
                [length, -length, 0.4],
                [length, -2 * length, 0.4],
            ),  # bottom right connecting the upper right most
            (
                [length, -2 * length, 0.4],
                [2 * length, -2 * length, 0.4],
            ),  # upper right most
            (
                [2 * length, -2 * length, 0.4],
                [2 * length, -length, 0.4],
            ),  # up right connecting the upper right most
            ([2 * length, -length, 0.4], [2 * length, length, 0.4]),  # top
            ([2 * length, length, 0.4], [length, length, 0.4]),  # up left
            ([length, length, 0.4], [-length, length, 0.4]),  # left bottom
            ([-length, length, 0.4], [-length, -length, 0.4]),  # bottom
            ([-length, -length, 0.4], [length, -length, 0.4]),  # right bottom
        ]

        poly = []
        for i in range(len(self.walls_data)):
            if self.wall_type == 2 and i == len(self.walls_data) - 1:
                break
            poly.append((self.walls_data[i][0][0], self.walls_data[i][0][1]))

        self.boundary = Polygon(poly)

        self.walls = [self.coppelia.create_wall(w[0], w[1]) for w in self.walls_data]
        self.data["walls"] = []

        for w in self.walls:
            self.walls_IND.append(IND)
            self.data["walls"].append(w)
            self.object_list.append(w)
            IND = IND + 1

        # Adding threshold
        breadth = 0
        if self.wall_type == 0:
            self.length = length / 2 - 0.6
            length = length / 2 - 0.6
        elif self.wall_type == 1:
            self.length = length / 2 - 0.6
            length = length / 2 - 0.6
            self.breadth = breadth / 2 - 0.6
            breadth = breadth / 2 - 0.6
        elif self.wall_type:
            self.length = length - 0.6
            length = length - 0.6

        # Add Goal
        x, y = 3.3, 3.3
        self.goal_data = [x, y]
        self.goal = self.coppelia.create_goal(self.goal_data[0], self.goal_data[1])
        self.data["goal"] = [self.goal]
        self.object_list.append(self.goal)

        self.robot = YouBot()
        x, y = 0, -4
        self.robot.set_orientation([0, 0, 0])
        self.object_list.append(self.robot)

        for child in self.coppelia.get_objects_in_tree():
            name = child.get_name()
            if name == "camera_fr":
                camera_fr = child
                camera_fr.set_parent(self.robot, keep_in_place=False)
                camera_fr.set_model(self.robot)
                camera_fr.set_position([0.0, 0.0, 0.0], relative_to=self.robot)
                camera_fr.set_orientation([0.0, 0.0, pi])
                camera_fr.set_model_respondable(False)
                camera_fr.set_model_dynamic(False)
                camera_fr.set_collidable(False)
                camera_fr.set_measurable(False)
                camera_fr.set_model_collidable(False)
                camera_fr.set_model_measurable(False)
            elif name == "Vision_sensor":
                vision_sensor = child
                entity_to_render = vision_sensor.get_entity_to_render()
                far_clipping_plane = vision_sensor.get_far_clipping_plane()
                near_clipping_plane = vision_sensor.get_near_clipping_plane()
                perspective_angle = vision_sensor.get_perspective_angle()
                perspective_mode = vision_sensor.get_perspective_mode()
                render_mode = vision_sensor.get_render_mode()
                resolution = vision_sensor.get_resolution()
                window_size = vision_sensor.get_windowed_size()

        self.wandering_humans = []
        for index, y in enumerate([-1.0, -2.0, 0.5, 0.2]):
            if num_of_humans == 0:
                break
            num_of_humans -= 1
            a = self.coppelia.create_human()
            x = -3.2 + index / 5
            p = [x, y, 0]
            a.set_position(p)
            self.object_list.append(a)
            self.humans_IND.append(IND)
            IND = IND + 1
            self.humans.append(a)
            # a.move([-x, -y, 0])
        for x in [-1.0, 2.0, 3.0]:
            if num_of_humans == 0:
                break
            num_of_humans -= 1
            a = self.coppelia.create_human()
            y = -2.0
            p = [x, y, 0]
            a.set_position(p)
            a.set_orientation([0, 0, -3.141592 / 2.0])
            self.object_list.append(a)
            self.humans_IND.append(IND)
            IND = IND + 1
            self.humans.append(a)

        for index, x in enumerate([0.4, -1.2, 2.6, -3.4, 2.3]):
            if num_of_humans == 0:
                break
            num_of_humans -= 1
            a = self.coppelia.create_human()
            y = 1.9 + index * 0.3
            p = [x, y, 0]
            a.set_position(p)
            a.set_orientation([0, 0, 3.141592 / 2.0])
            self.object_list.append(a)
            self.humans_IND.append(IND)
            IND = IND + 1
            self.humans.append(a)

        for y in [-2.4, 1.2]:
            if num_of_humans == 0:
                break
            num_of_humans -= 1
            a = self.coppelia.create_human()
            x = 2.7
            p = [x, y, 0]
            a.set_position(p)
            a.set_orientation([0, 0, -3.141592 / 2.0])
            self.object_list.append(a)
            self.humans_IND.append(IND)
            IND = IND + 1
            self.humans.append(a)

        if num_of_humans != 0:
            num_of_humans -= 1
            a = self.coppelia.create_human()
            x = 4.5
            y = -5.0
            p = [x, y, 0]
            a.set_position(p)
            a.set_orientation([0, 0, -3.141592 / 2.0])
            self.object_list.append(a)
            self.humans_IND.append(IND)
            IND = IND + 1
            self.humans.append(a)
        if num_of_humans != 0:
            num_of_humans -= 1
            a = self.coppelia.create_human()
            x = 3.8
            y = -4.5
            p = [x, y, 0]
            a.set_position(p)
            a.set_orientation([0, 0, -3.141592 / 2.0])
            self.object_list.append(a)
            self.humans_IND.append(IND)
            IND = IND + 1
            self.humans.append(a)
        if num_of_humans != 0:
            num_of_humans -= 1
            a = self.coppelia.create_human()
            x = 3.9
            y = -4.0
            p = [x, y, 0]
            a.set_position(p)
            a.set_orientation([0, 0, -3.141592 / 2.0])
            self.object_list.append(a)
            self.humans_IND.append(IND)
            IND = IND + 1
            self.humans.append(a)

        self.data["humans"] = tuple(self.humans)

        ##just for adding index for laptop
        for iii in self.laptops:
            self.laptops_IND.append(IND)
            IND = IND + 1
        ##

        self.data["table"] = tuple(self.tables)
        self.data["laptop"] = tuple(self.laptops)

        # Add plants
        # a = self.coppelia.load_model("models/furniture/plants/indoorPlant.ttm")
        # x, y = self.set_entity_position()
        # p_a = [0, 0, 0.165]
        # a.set_position(p_a)
        # self.object_list.append(a)
        # self.plants_IND.append(IND)
        # IND = IND + 1
        # self.plants.append(a)

        self.data["plant"] = tuple(self.plants)

        for i in range(len(self.tables)):
            self.table_indices.append(0)
        for i in range(len(self.plants)):
            self.plant_indices.append(0)
        for i in range(len(self.humans)):
            self.human_indices.append(0)
        for i in range(len(self.wandering_humans)):
            self.wandering_humans_indices.append(0)

        # num_relations = random.randint(self.min_relations,self.max_relations)
        # for i in range(num_relations):
        #     relation_type = random.randint(0,3) # 0 for human-human (static); 1 for human-table, 2 for human-plant, 3 for human-human (wandering)
        #     relation_priority_list = [ (i+relation_type)%4 for i in range(4)]
        #     for rel in relation_priority_list:
        #         if self.create_interaction(rel, n_humans, length, breadth):
        #             break

        for t in self.tables:
            self.position_list.append(t.get_position())

        for h in self.humans:
            self.position_list.append(h.get_position())

        for p in self.plants:
            self.position_list.append(p.get_position())

        return self.data, self.wandering_humans

    def room_setup_2(self):
        IND = self.initialize_room()
        # Add Goal
        x, y = 3.3, 3.3
        self.goal_data = [x, y]
        self.goal = self.coppelia.create_goal(self.goal_data[0], self.goal_data[1])
        self.data["goal"] = [self.goal]
        self.object_list.append(self.goal)

        self.robot = YouBot()
        x, y = 0, -4
        self.robot.set_orientation([0, 0, 0])
        self.object_list.append(self.robot)

        self.camera_setup()

        self.wandering_humans = []
        for index in range(3):
            a = self.coppelia.create_human()
            x = -2.0
            y = 4.0 - (4.0 / 5) * index
            p = [x, y, 0]
            a.set_position(p)
            a.set_orientation([0, 0, -3.141592 / 2.0])
            self.object_list.append(a)
            self.humans_IND.append(IND)
            IND = IND + 1
            a.move([x, -4.0, 0])
            self.humans.append(a)

        for index in range(4):
            a = self.coppelia.create_human()
            x = -3.0
            y = 0.0 - (4.0 / 5) * index
            p = [x, y, 0]
            a.set_position(p)
            a.set_orientation([0, 0, -3.141592 / 2.0])
            self.object_list.append(a)
            self.humans_IND.append(IND)
            IND = IND + 1
            a.move([x, -4.0, 0])
            self.humans.append(a)

        for index in range(3):
            a = self.coppelia.create_human()
            x = 2.0
            y = 4.0 - (4.0 / 5) * index
            p = [x, y, 0]
            a.set_position(p)
            a.set_orientation([0, 0, -3.141592 / 2.0])
            self.object_list.append(a)
            self.humans_IND.append(IND)
            IND = IND + 1
            a.move([x, -4.0, 0])
            self.humans.append(a)
        for index in range(4):
            a = self.coppelia.create_human()
            x = 2.0
            y = 0.0 - (4.0 / 3) * index
            p = [x, y, 0]
            a.set_position(p)
            a.set_orientation([0, 0, -3.141592 / 2.0])
            self.object_list.append(a)
            self.humans_IND.append(IND)
            IND = IND + 1
            a.move([x, -4.0, 0])
            self.humans.append(a)

        self.data["humans"] = tuple(self.humans)

        ##just for adding index for laptop
        for iii in self.laptops:
            self.laptops_IND.append(IND)
            IND = IND + 1
        ##

        self.data["table"] = tuple(self.tables)
        self.data["laptop"] = tuple(self.laptops)
        self.data["plant"] = tuple(self.plants)

        self.after_room_setup()

        return self.data, self.wandering_humans

    def room_setup_4(self):
        IND = self.initialize_room()
        # Add Goal
        x, y = 0, -3.1
        self.goal_data = [x, y]
        self.goal = self.coppelia.create_goal(self.goal_data[0], self.goal_data[1])
        self.data["goal"] = [self.goal]
        self.object_list.append(self.goal)

        self.robot = YouBot()
        x, y = 0, -4
        self.robot.set_orientation([0, 0, 0])
        self.object_list.append(self.robot)

        self.camera_setup()

        self.wandering_humans = []
        for index in range(7):
            a = self.coppelia.create_human()
            import math

            orientation = math.pi / 6.0 * index
            x = 3.3 * math.cos(orientation)
            y = 3.3 * math.sin(orientation) - 0.8
            p = [x, y, 0]
            a.set_position(p)
            a.set_orientation([0, 0, -orientation])
            self.object_list.append(a)
            self.humans_IND.append(IND)
            IND = IND + 1
            a.move([0, -3.1, 0])
            self.humans.append(a)
        self.data["humans"] = tuple(self.humans)

        ##just for adding index for laptop
        for iii in self.laptops:
            self.laptops_IND.append(IND)
            IND = IND + 1
        ##

        self.data["table"] = tuple(self.tables)
        self.data["laptop"] = tuple(self.laptops)
        self.data["plant"] = tuple(self.plants)

        self.after_room_setup()

        return self.data, self.wandering_humans

    def initialize_room(self):
        self.coppelia.remove_objects(
            self.humans,
            self.tables,
            self.laptops,
            self.plants,
            self.goal,
            self.walls,
            self.relations,
            self.relations_moving_humans,
        )

        self.interacting_humans = []

        self.data = self.ini_data
        self.object_list = []
        self.position_list = (
            []
        )  # checking collision for wandering humans (has plants tables and static humans)

        self.humans = []
        self.wandering_humans = []
        self.tables = []
        self.plants = []
        self.laptops = []
        self.relations = []
        self.relations_moving_humans = []
        self.relation_to_human_map = {}
        self.wall_type = random.randint(
            0, 2
        )  # 0 for square, 1 for rectangle, 2 for L-shaped
        self.boundary = None
        self.human_indices = []
        self.table_indices = []
        self.plant_indices = []
        self.wandering_humans_indices = []
        self.length = None
        self.breadth = None
        self.break_point = 5  # when to stop creating an entity
        self.count = 0  # storing count till break point
        self.n_interactions = None

        self.humans_IND = []
        self.tables_IND = []
        self.plants_IND = []
        self.laptops_IND = []
        self.walls_IND = []

        IND = 1

        # Add four walls
        self.wall_type = 1
        breadth = 9.0
        length = 9.0
        self.walls_data = [
            ([length / 2, breadth / 2, 0.4], [length / 2, -breadth / 2, 0.4]),
            ([length / 2, -breadth / 2, 0.4], [-length / 2, -breadth / 2, 0.4]),
            ([-length / 2, -breadth / 2, 0.4], [-length / 2, breadth / 2, 0.4]),
            ([-length / 2, breadth / 2, 0.4], [length / 2, breadth / 2, 0.4]),
        ]

        poly = []
        for i in range(len(self.walls_data)):
            if self.wall_type == 2 and i == len(self.walls_data) - 1:
                break
            poly.append((self.walls_data[i][0][0], self.walls_data[i][0][1]))

        self.boundary = Polygon(poly)

        self.walls = [self.coppelia.create_wall(w[0], w[1]) for w in self.walls_data]
        self.data["walls"] = []

        for w in self.walls:
            self.walls_IND.append(IND)
            self.data["walls"].append(w)
            self.object_list.append(w)
            IND = IND + 1

        # Adding threshold
        breadth = 0
        if self.wall_type == 0:
            self.length = length / 2 - 0.6
            length = length / 2 - 0.6
        elif self.wall_type == 1:
            self.length = length / 2 - 0.6
            length = length / 2 - 0.6
            self.breadth = breadth / 2 - 0.6
            breadth = breadth / 2 - 0.6
        elif self.wall_type:
            self.length = length - 0.6
            length = length - 0.6
        return IND

    def after_room_setup(self):
        for i in range(len(self.tables)):
            self.table_indices.append(0)
        for i in range(len(self.plants)):
            self.plant_indices.append(0)
        for i in range(len(self.humans)):
            self.human_indices.append(0)
        for i in range(len(self.wandering_humans)):
            self.wandering_humans_indices.append(0)

        for t in self.tables:
            self.position_list.append(t.get_position())

        for h in self.humans:
            self.position_list.append(h.get_position())

        for p in self.plants:
            self.position_list.append(p.get_position())

    def camera_setup(self):
        for child in self.coppelia.get_objects_in_tree():
            name = child.get_name()
            if name == "camera_fr":
                camera_fr = child
                camera_fr.set_parent(self.robot, keep_in_place=False)
                camera_fr.set_model(self.robot)
                camera_fr.set_position([0.0, 0.0, 0.0], relative_to=self.robot)
                camera_fr.set_orientation([0.0, 0.0, pi])
                camera_fr.set_model_respondable(False)
                camera_fr.set_model_dynamic(False)
                camera_fr.set_collidable(False)
                camera_fr.set_measurable(False)
                camera_fr.set_model_collidable(False)
                camera_fr.set_model_measurable(False)
            elif name == "Vision_sensor":
                vision_sensor = child
                entity_to_render = vision_sensor.get_entity_to_render()
                far_clipping_plane = vision_sensor.get_far_clipping_plane()
                near_clipping_plane = vision_sensor.get_near_clipping_plane()
                perspective_angle = vision_sensor.get_perspective_angle()
                perspective_mode = vision_sensor.get_perspective_mode()
                render_mode = vision_sensor.get_render_mode()
                resolution = vision_sensor.get_resolution()
                window_size = vision_sensor.get_windowed_size()

    def room_setup_5(self):
        IND = self.initialize_room()
        # Add Goal
        x, y = 0, 3.1
        self.goal_data = [x, y]
        self.goal = self.coppelia.create_goal(self.goal_data[0], self.goal_data[1])
        self.data["goal"] = [self.goal]
        self.object_list.append(self.goal)

        self.robot = YouBot()
        x, y = 0, -4
        self.robot.set_orientation([0, 0, 0])
        self.object_list.append(self.robot)

        self.camera_setup()

        self.wandering_humans = []
        for index in range(7):
            a = self.coppelia.create_human()
            import math

            orientation = -3.141592 / 6.0 * index
            x = 3.3 * math.cos(orientation)
            y = 3.3 * math.sin(orientation) - 0.8
            p = [x, y, 0]
            a.set_position(p)
            a.set_orientation([0, 0, -orientation])
            self.object_list.append(a)
            self.humans_IND.append(IND)
            IND = IND + 1
            a.move([0, 3.1, 0])
            self.humans.append(a)
        self.data["humans"] = tuple(self.humans)

        ##just for adding index for laptop
        for iii in self.laptops:
            self.laptops_IND.append(IND)
            IND = IND + 1
        ##
        self.data["table"] = tuple(self.tables)
        self.data["laptop"] = tuple(self.laptops)
        self.data["plant"] = tuple(self.plants)

        return self.data, self.wandering_humans

    def room_setup_3(self):
        IND = self.initialize_room()
        # Add Goal
        x, y = 0, 3.1
        self.goal_data = [x, y]
        self.goal = self.coppelia.create_goal(self.goal_data[0], self.goal_data[1])
        self.data["goal"] = [self.goal]
        self.object_list.append(self.goal)

        self.robot = YouBot()
        x, y = 0, -4
        self.robot.set_orientation([0, 0, 0])
        self.object_list.append(self.robot)

        self.camera_setup()

        self.wandering_humans = []
        for index in range(1):
            a = self.coppelia.create_human()
            import math

            x = -3.0
            y = 2.5
            p = [x, y, 0]
            a.set_position(p)
            a.set_orientation([0, 0, -math.pi / 2])
            self.object_list.append(a)
            self.humans_IND.append(IND)
            IND = IND + 1
            self.humans.append(a)
        for index in range(1):
            a = self.coppelia.create_human()
            import math

            x = -3.0
            y = -2.5
            p = [x, y, 0]
            a.set_position(p)
            a.set_orientation([0, 0, math.pi / 2])
            self.object_list.append(a)
            self.humans_IND.append(IND)
            IND = IND + 1
            self.humans.append(a)

        for index in range(1):
            a = self.coppelia.create_human()
            import math

            x = 2.7
            y = -2.0
            p = [x, y, 0]
            a.set_position(p)
            a.set_orientation([0, 0, math.pi / 2])
            self.object_list.append(a)
            self.humans_IND.append(IND)
            IND = IND + 1
            self.humans.append(a)

        # two horizontal humans
        for index in range(1):
            a = self.coppelia.create_human()
            import math

            x = 3.0
            y = 2.5
            p = [x, y, 0]
            a.set_position(p)
            a.set_orientation([0, 0, math.pi])
            self.object_list.append(a)
            self.humans_IND.append(IND)
            IND = IND + 1
            self.humans.append(a)
        for index in range(1):
            a = self.coppelia.create_human()
            import math

            x = 1.0
            y = 2.5
            p = [x, y, 0]
            a.set_position(p)
            a.set_orientation([0, 0, 0])
            self.object_list.append(a)
            self.humans_IND.append(IND)
            IND = IND + 1
            self.humans.append(a)

        for index in range(1):
            a = self.coppelia.create_human()
            import math

            x = 0
            y = -2.6
            p = [x, y, 0]
            a.set_position(p)
            a.set_orientation([0, 0, 0])
            self.object_list.append(a)
            self.humans_IND.append(IND)
            IND = IND + 1
            a.move([3.5, -2.6, 0])
            self.humans.append(a)
        for index in range(2):
            a = self.coppelia.create_human()
            import math

            x = -1.9 + 0.8 * index
            y = 2
            p = [x, y, 0]
            a.set_position(p)
            a.set_orientation([0, 0, 0])
            self.object_list.append(a)
            self.humans_IND.append(IND)
            IND = IND + 1
            a.move([x, -4.0, 0])
            self.humans.append(a)

        ##just for adding index for laptop
        for iii in self.laptops:
            self.laptops_IND.append(IND)
            IND = IND + 1
        ##
        self.after_room_setup()
        ind1 = 0
        ind2 = 1
        print(self.human_indices, ind1, ind2)
        self.human_indices[ind1] = 1
        self.human_indices[ind2] = 1
        self.interacting_humans.append(
            {
                "src": self.humans_IND[ind1],
                "dst": self.humans_IND[ind2],
                "relationship": "two_static_person_talking",
            }
        )
        ind1 = 3
        ind2 = 4
        print(self.human_indices, ind1, ind2)
        self.human_indices[ind1] = 1
        self.human_indices[ind2] = 1
        self.interacting_humans.append(
            {
                "src": self.humans_IND[ind1],
                "dst": self.humans_IND[ind2],
                "relationship": "two_static_person_talking",
            }
        )

        ind1 = 6
        ind2 = 7
        print("--", self.human_indices, ind1, ind2)
        self.interacting_humans.append(
            {
                "src": self.humans_IND[ind1],
                "dst": self.humans_IND[ind2],
                "relationship": "wandering_human_interacting",
            }
        )

        self.data["humans"] = tuple(self.humans)

        self.data["table"] = tuple(self.tables)
        self.data["laptop"] = tuple(self.laptops)
        self.data["plant"] = tuple(self.plants)
        return self.data, self.wandering_humans

    def room_setup_6(self):
        IND = self.initialize_room()
        # Add Goal
        x, y = 2.3, 0.5
        self.goal_data = [x, y]
        self.goal = self.coppelia.create_goal(self.goal_data[0], self.goal_data[1])
        self.data["goal"] = [self.goal]
        self.object_list.append(self.goal)
        # Place the robot
        self.robot = YouBot()
        x, y = 0, -4
        p = self.robot.get_position()
        self.robot.set_position([x, p[1], p[2]])
        self.robot.set_orientation([0, 0, 0])
        self.object_list.append(self.robot)

        self.camera_setup()

        self.wandering_humans = []
        a = self.coppelia.load_model("small_table.ttm")
        b = self.coppelia.load_model("models/office items/laptop.ttm")
        x, y = -2.5, 2.5
        p_a = [x, y, 0.85]
        p_b = [x, y, 0.975]
        a.set_position(p_a)
        b.set_position(p_b)
        self.object_list.append(a)
        self.object_list.append(b)
        self.tables_IND.append(IND)
        IND = IND + 1
        self.tables.append(a)
        self.laptops.append(b)

        for index in range(1):
            a = self.coppelia.create_human()
            import math

            x = -2.5
            y = -2.
            p = [x, y, 0]
            a.set_position(p)
            a.set_orientation([0, 0, math.pi / 2])
            self.object_list.append(a)
            self.humans_IND.append(IND)
            IND = IND + 1
            self.humans.append(a)

        # two horizontal humans
        for index in range(1):
            a = self.coppelia.create_human()
            import math

            x = 2.5
            y = 2.5
            p = [x, y, 0]
            a.set_position(p)
            a.set_orientation([0, 0, math.pi])
            self.object_list.append(a)
            self.humans_IND.append(IND)
            IND = IND + 1
            self.humans.append(a)
        for index in range(1):
            a = self.coppelia.create_human()
            import math

            x = -1.
            y = 2.5
            p = [x, y, 0]
            a.set_position(p)
            a.set_orientation([0, 0, 0])
            self.object_list.append(a)
            self.humans_IND.append(IND)
            IND = IND + 1
            self.humans.append(a)



        for index in range(1):
            a = self.coppelia.create_human()
            import math

            x = 2.3
            y = -2.6
            p = [x, y, 0]
            a.set_position(p)
            a.set_orientation([0, 0, math.pi])
            self.object_list.append(a)
            self.humans_IND.append(IND)
            IND = IND + 1
            a.move([-3.5, -2.6, 0])
            self.humans.append(a)

        ##just for adding index for laptop
        for iii in self.laptops:
            self.laptops_IND.append(IND)
            IND = IND + 1
        ##
        self.after_room_setup()

        self.interacting_humans.append(
            {
                "src": self.humans_IND[1],
                "dst": self.humans_IND[2],
                "relationship": "two_static_person_talking",
            }
        )

        cylinder = self.create_interaction_cylinder(
            self.humans[1].get_position(),
            self.humans[2].get_position(),
        )
        self.relations.append(cylinder)

        self.interacting_humans.append(
            {
                "src": self.humans_IND[0],
                "dst": self.tables_IND[0],
                "relationship": "human_laptop_interaction",
            }
        )
        # Add relation cylinder
        cylinder = self.create_interaction_cylinder(
            self.humans[0].get_position(),
            self.tables[0].get_position(),
        )
        self.relations.append(cylinder)


        self.data["humans"] = tuple(self.humans)
        self.data["table"] = tuple(self.tables)
        self.data["laptop"] = tuple(self.laptops)
        self.data["plant"] = tuple(self.plants)
        return self.data, self.wandering_humans
