import carla
import random
import numpy as np
import math

from distutils.util import strtobool
from srunner.scenariomanager.actorcontrols.basic_control import BasicControl
from srunner.scenariomanager.actorcontrols.visualizer import Visualizer
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

class WaypointVehicleControl(BasicControl):

    def __init__(self, actor, args=None):
        super(WaypointVehicleControl, self).__init__(actor)
        self._generated_waypoint_list = []
        self._waypoints_speed_np_array = None
        self._last_update = None
        self._consider_traffic_lights = False
        self._consider_obstacles = False
        self._proximity_threshold = float('inf')
        self._max_deceleration = None
        self._max_acceleration = None

        self._obstacle_sensor = None
        self._obstacle_distance = float('inf')
        self._obstacle_actor = None
        self._obstacle_distance_threshold = 0.5

        self._minimum_look_ahead_distance = 0.1

        self._visualizer = None

        self._brake_lights_active = False

        """hitory variable for every step"""
        self._last_waypoint_index = 0
        self._last_control_applied_time = 0
        self._last_speed = 0
        self._last_integral = 0
        self._last_derivative = 0

        self.speed_profile = []
        #self.last_position = None
        self.last_yaw = None
        self.final_index = -1
        bp = CarlaDataProvider.get_world().get_blueprint_library().find('sensor.other.obstacle')
        bp.set_attribute('distance', '10')
        bp.set_attribute('hit_radius', '1')
        bp.set_attribute('only_dynamics', 'False')
        self._obstacle_sensor = CarlaDataProvider.get_world().spawn_actor(
            bp, carla.Transform(carla.Location(x=self._actor.bounding_box.extent.x, z=1.0)), attach_to=self._actor)
        self._obstacle_sensor.listen(lambda event: self._on_obstacle(event))  # pylint: disable=unnecessary-lambda

        self._get_vehicle_info()
        self._initialized = False


        # add speed_profile
        if args and 'speed_profile' in args:
            self.speed_profile = args['speed_profile']



    def _get_vehicle_info(self):
        physics_control = self._actor.get_physics_control()
        rear_axle_center = (physics_control.wheels[2].position + physics_control.wheels[3].position)/200
        offset = rear_axle_center - self._actor.get_location()

        self._max_steer = physics_control.wheels[0].max_steer_angle
        self._wheelbase = np.linalg.norm([offset.x, offset.y, offset.z])
        self._throttle = 0
        self._brake = 0
        # self._actor.set_simulate_physics(True)


    def _on_obstacle(self, event):
        """
        Callback for the obstacle sensor

        Sets _obstacle_distance and _obstacle_actor according to the closest obstacle
        found by the sensor.
        """
        if not event:
            return
        self._obstacle_distance = event.distance
        self._obstacle_actor = event.other_actor

    def reset(self):
        """
        Reset the controller
        """

        if self._visualizer:
            self._visualizer.reset()
        if self._obstacle_sensor:
            self._obstacle_sensor.destroy()
            self._obstacle_sensor = None
        if self._actor and self._actor.is_alive:
            self._actor = None

    def run_step(self):
        """
        Execute on tick of the controller's control loop
        
        Follow the provided waypoints use pure pursuit algorithm and PID.
        """

        if self._reached_goal :
            # Reached the goal, so stop
            velocity = carla.Vector3D(0, 0, 0)
            control_cmd = carla.VehicleControl()
            control_cmd.brake = 10
            self._actor.apply_control(control_cmd)
            return
        """
        Goal is not reached, so continue to next waypoint
        """
        # if self._stop_in_front_of_obstacle():
        #     return

        if CarlaDataProvider.get_location(self._actor).z < -0.2:
            self._reached_goal = True

        if self._waypoints:
            if self._waypoints_speed_np_array is None:
                """intialize a np array for lateral control"""
                vector3D_to_list = lambda v: (v.x, v.y)
                self._waypoints_speed_np_array = np.array([(self._waypoints[i].location.x,
                                                      self._waypoints[i].location.y,
                                                      self.speed_profile[i]) for i in range(len(self._waypoints))])

            vehicle_speed = self._actor.get_velocity().length()
            self._next_waypoint_index = self._get_next_waypoint_index(self._last_waypoint_index,
                                                                    look_ahead_distance=max(vehicle_speed*0.5, self._minimum_look_ahead_distance))
            # self._next_waypoint_index = self._get_closest_waypoint_index(CarlaDataProvider.get_location(self._actor))+2
            if self._next_waypoint_index >= 0 and self._next_waypoint_index < len(self._waypoints):
                self._apply_control(self._next_waypoint_index)
                self._last_waypoint_index = self._next_waypoint_index
                self._last_control_applied_time = GameTime.get_time()
            else:
                self._reached_goal = True

    def _relative_location(self, waypoint):
        frame = CarlaDataProvider.get_transform(self._actor)
        location = waypoint.location
        origin = frame.location
        forward = frame.get_forward_vector()
        right = frame.get_right_vector()
        up = frame.get_up_vector()

        disp = location - origin
        x = np.dot([disp.x, disp.y, disp.z], [forward.x, forward.y, forward.z])
        y = np.dot([disp.x, disp.y, disp.z], [right.x, right.y, right.z])
        z = np.dot([disp.x, disp.y, disp.z], [up.x, up.y, up.z])
        return carla.Vector3D(x, y, z)

    def _apply_control(self, waypoint_index):
        if self._initialized == False:
            velocity = self._speed_to_velocity(self.speed_profile[waypoint_index] ,self._waypoints[waypoint_index].rotation)
            self._actor.set_target_velocity(velocity)
            self._initialized = True
            return

        control_cmd = carla.VehicleControl()

        steer = self._lateral_control(waypoint_index)
        throttle = self._longitudinal_control(waypoint_index)

        if throttle < 0:
            control_cmd.brake = -throttle
        else:
            control_cmd.throttle = throttle
        control_cmd.steer = steer

        self._actor.apply_control(control_cmd)

    def _longitudinal_control(self, waypoint_index):
        """
        A longitudinal PID controller
        """

        kp = 0.14
        ki = 0.3
        kd = 0.4

        vehicle_speed = self._actor.get_velocity().length()
        target_speed = self._get_target_speed(waypoint_index)
        delta_t = GameTime.get_time() - self._last_control_applied_time
        delta_v = target_speed - vehicle_speed
        integral = self._last_integral + delta_v * delta_t
        derivative = delta_v - self._last_derivative
        acc = kp * delta_v + ki * integral + kd * derivative

        throttle = np.tanh(acc)
        brake = 0

        throttle = np.clip(throttle, -1, 1)
        self._last_integral = integral
        self._last_derivative = derivative

        return throttle

    def _lateral_control(self, waypoint_index):
        vehicle_speed = self._actor.get_velocity().length()
        waypoint = self._waypoints[waypoint_index]
        wp_loc_rel = self._relative_location(waypoint) + carla.Vector3D(self._wheelbase, 0, 0)
        wp_ar = [wp_loc_rel.x, wp_loc_rel.y]
        d2 = wp_ar[0]**2 + wp_ar[1]**2
        steer_rad = math.atan(2 * self._wheelbase * wp_loc_rel.y / (d2*0.25))
        steer_deg = math.degrees(steer_rad)
        steer_deg = np.clip(steer_deg, -self._max_steer, self._max_steer)
        steer = steer_deg / self._max_steer
        steer = steer

        return steer

    def _stop_in_front_of_obstacle(self):
        vehicle_speed = self._actor.get_velocity().length()
        if self._obstacle_distance < max(vehicle_speed*0.5, self._obstacle_distance_threshold):
            control_cmd = carla.VehicleControl()
            control_cmd.brake = 2
            self._actor.apply_control(control_cmd)
            return True
        else:
            return False

    def _speed_to_velocity(self, speed, rotation):
        return rotation.get_forward_vector() * speed

    def _get_closest_waypoint_index(self, loc):
        current_xy = np.array([loc.x, loc.y])
        return np.argmin(np.sum((current_xy - self._waypoints_speed_np_array[:,:2])**2, axis=1))

    def _get_next_waypoint_index(self, last_index, min_distance = 1000, look_ahead_distance = 5):
        vehicle_location = self._actor.get_location()
        min_distance = 1000

        if last_index == len(self._waypoints) - 1:
            return -1
        next_index = last_index
        for i in range(last_index, len(self._waypoints)):
            waypoint = self._waypoints[i]
            waypoint_location = waypoint.location
            # print(vehicle_location.distance(waypoint_location))
            #Find the waypoint closest to the vehicle, but once vehicle is close to upcoming waypoint, search for next one
            if vehicle_location.distance(waypoint_location) < min_distance and vehicle_location.distance(waypoint_location) > look_ahead_distance:
                min_distance = vehicle_location.distance(waypoint_location)
                next_index = i
            if vehicle_location.distance(waypoint_location) > 2*look_ahead_distance:
                break
        return next_index

    def _get_target_speed(self, index):
        speed = 0
        if index >= len(self.speed_profile) :
            speed = self.speed_profile[-1]
        else:
            speed = self.speed_profile[index]

        self.update_target_speed(speed)
        return speed

