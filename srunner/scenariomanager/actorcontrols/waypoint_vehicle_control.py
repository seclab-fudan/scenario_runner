#!/usr/bin/env python

# write for adding waypoint speed

"""
This module extends the simple_vehicle_control to support waypoint speed in carla

"""
from distutils.util import strtobool
import math

import carla

from srunner.scenariomanager.actorcontrols.basic_control import BasicControl
from srunner.scenariomanager.actorcontrols.visualizer import Visualizer
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

class WaypointVehicleControl(BasicControl):

    def __init__(self, actor, args=None):
        super(WaypointVehicleControl, self).__init__(actor)
        self._generated_waypoint_list = []
        self._last_update = None
        self._consider_traffic_lights = False
        self._consider_obstacles = False
        self._proximity_threshold = float('inf')
        self._max_deceleration = None
        self._max_acceleration = None

        self._obstacle_sensor = None
        self._obstacle_distance = float('inf')
        self._obstacle_actor = None

        self._visualizer = None

        self._brake_lights_active = False

        self.speed_profile = []
        self.last_location = None
        self.index = 0

        if args and 'consider_obstacles' in args and strtobool(args['consider_obstacles']):
            self._consider_obstacles = strtobool(args['consider_obstacles'])
            bp = CarlaDataProvider.get_world().get_blueprint_library().find('sensor.other.obstacle')
            bp.set_attribute('distance', '250')
            if args and 'proximity_threshold' in args:
                self._proximity_threshold = float(args['proximity_threshold'])
                bp.set_attribute('distance', str(max(float(args['proximity_threshold']), 250)))
            bp.set_attribute('hit_radius', '1')
            bp.set_attribute('only_dynamics', 'True')
            self._obstacle_sensor = CarlaDataProvider.get_world().spawn_actor(
                bp, carla.Transform(carla.Location(x=self._actor.bounding_box.extent.x, z=1.0)), attach_to=self._actor)
            self._obstacle_sensor.listen(lambda event: self._on_obstacle(event))  # pylint: disable=unnecessary-lambda

        if args and 'consider_trafficlights' in args and strtobool(args['consider_trafficlights']):
            self._consider_traffic_lights = strtobool(args['consider_trafficlights'])

        if args and 'max_deceleration' in args:
            self._max_deceleration = float(args['max_deceleration'])

        if args and 'max_acceleration' in args:
            self._max_acceleration = float(args['max_acceleration'])

        if args and 'attach_camera' in args and strtobool(args['attach_camera']):
            self._visualizer = Visualizer(self._actor)
        
        # add speed_profile
        if args and 'speed_profile' in args:
            self.speed_profile = args['speed_profile']

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

        If _waypoints are provided, the vehicle moves towards the next waypoint
        with the given _target_speed, until reaching the final waypoint. Upon reaching
        the final waypoint, _reached_goal is set to True.

        If _waypoints is empty, the vehicle moves in its current direction with
        the given _target_speed.

        For further details see :func:`_set_new_velocity`
        """

        if self._reached_goal:
            # Reached the goal, so stop
            velocity = carla.Vector3D(0, 0, 0)
            self._actor.set_target_velocity(velocity)
            return

        if self._visualizer:
            self._visualizer.render()

        self._reached_goal = False

        if not self._waypoints:
            # No waypoints are provided, so we have to create a list of waypoints internally
            # get next waypoints from map, to avoid leaving the road
            self._reached_goal = False

            map_wp = None
            if not self._generated_waypoint_list:
                map_wp = CarlaDataProvider.get_map().get_waypoint(CarlaDataProvider.get_location(self._actor))
            else:
                map_wp = CarlaDataProvider.get_map().get_waypoint(self._generated_waypoint_list[-1].location)
            while len(self._generated_waypoint_list) < 50:
                map_wps = map_wp.next(2.0)
                if map_wps:
                    self._generated_waypoint_list.append(map_wps[0].transform)
                    map_wp = map_wps[0]
                else:
                    break

            # Remove all waypoints that are too close to the vehicle
            while (self._generated_waypoint_list and
                   self._generated_waypoint_list[0].location.distance(self._actor.get_location()) < 0.5):
                self._generated_waypoint_list = self._generated_waypoint_list[1:]

            direction_norm = self._set_new_velocity(self._offset_waypoint(self._generated_waypoint_list[0]))
            if direction_norm < 2.0:
                self._generated_waypoint_list = self._generated_waypoint_list[1:]
        else:
            # When changing from "free" driving without pre-defined waypoints to a defined route with waypoints
            # it may happen that the first few waypoints are too close to the ego vehicle for obtaining a
            # reasonable control command. Therefore, we drop these waypoints first.
            while self._waypoints and self._waypoints[0].location.distance(self._actor.get_location()) < 0.5:
                self._waypoints = self._waypoints[1:]

            self._reached_goal = False
            if not self._waypoints:
                self._reached_goal = True
            else:
                direction_norm = self._set_new_velocity_gbf(self._offset_waypoint(self._waypoints[0]))
                if direction_norm < 4.0 and self._reached_goal == False:
                    self._waypoints = self._waypoints[1:]
                    if not self._waypoints:
                        self._reached_goal = True

    def _offset_waypoint(self, transform):
        """
        Given a transform (which should be the position of a waypoint), displaces it to the side,
        according to a given offset

        Args:
            transform (carla.Transform): Transform to be moved

        returns:
            offset_location (carla.Transform): Moved transform
        """
        if self._offset == 0:
            offset_location = transform.location
        else:
            right_vector = transform.get_right_vector()
            offset_location = transform.location + carla.Location(x=self._offset*right_vector.x,
                                                                  y=self._offset*right_vector.y)

        return offset_location

    def _set_new_velocity(self, next_location):
        """
        Calculate and set the new actor veloctiy given the current actor
        location and the _next_location_

        If _consider_obstacles is true, the speed is adapted according to the closest
        obstacle in front of the actor, if it is within the _proximity_threshold distance.
        If _consider_trafficlights is true, the vehicle will enforce a stop at a red
        traffic light.
        If _max_deceleration is set, the vehicle will reduce its speed according to the
        given deceleration value.
        If the vehicle reduces its speed, braking lights will be activated.

        Args:
            next_location (carla.Location): Next target location of the actor

        returns:
            direction (carla.Vector3D): Length of direction vector of the actor
        """

        current_time = GameTime.get_time()
        target_speed = self._target_speed

        if not self._last_update:
            self._last_update = current_time

        current_speed = math.sqrt(self._actor.get_velocity().x**2 + self._actor.get_velocity().y**2)

        if self._consider_obstacles:
            # If distance is less than the proximity threshold, adapt velocity
            if self._obstacle_distance < self._proximity_threshold:
                distance = max(self._obstacle_distance, 0)
                if distance > 0:
                    current_speed_other = math.sqrt(
                        self._obstacle_actor.get_velocity().x**2 + self._obstacle_actor.get_velocity().y**2)
                    if current_speed_other < current_speed:
                        acceleration = -0.5 * (current_speed - current_speed_other)**2 / distance
                        target_speed = max(acceleration * (current_time - self._last_update) + current_speed, 0)
                else:
                    target_speed = 0

        if self._consider_traffic_lights:
            if (self._actor.is_at_traffic_light() and
                    self._actor.get_traffic_light_state() == carla.TrafficLightState.Red):
                target_speed = 0

        if target_speed < current_speed:
            if not self._brake_lights_active:
                self._brake_lights_active = True
                light_state = self._actor.get_light_state()
                light_state |= carla.VehicleLightState.Brake
                self._actor.set_light_state(carla.VehicleLightState(light_state))
            if self._max_deceleration is not None:
                target_speed = max(target_speed, current_speed - (current_time -
                                                                  self._last_update) * self._max_deceleration)
        else:
            if self._brake_lights_active:
                self._brake_lights_active = False
                light_state = self._actor.get_light_state()
                light_state &= ~carla.VehicleLightState.Brake
                self._actor.set_light_state(carla.VehicleLightState(light_state))
            if self._max_acceleration is not None:
                tmp_speed = min(target_speed, current_speed + (current_time -
                                                               self._last_update) * self._max_acceleration)
                # If the tmp_speed is < 0.5 the vehicle may not properly accelerate.
                # Therefore, we bump the speed to 0.5 m/s if target_speed allows.
                target_speed = max(tmp_speed, min(0.5, target_speed))

        # set new linear velocity
        velocity = carla.Vector3D(0, 0, 0)
        direction = next_location - CarlaDataProvider.get_location(self._actor)
        direction_norm = math.sqrt(direction.x**2 + direction.y**2)
        velocity.x = direction.x / direction_norm * target_speed
        velocity.y = direction.y / direction_norm * target_speed

        self._actor.set_target_velocity(velocity)

        # set new angular velocity
        current_yaw = CarlaDataProvider.get_transform(self._actor).rotation.yaw
        # When we have a waypoint list, use the direction between the waypoints to calculate the heading (change)
        # otherwise use the waypoint heading directly
        if self._waypoints:
            delta_yaw = math.degrees(math.atan2(direction.y, direction.x)) - current_yaw
        else:
            new_yaw = CarlaDataProvider.get_map().get_waypoint(next_location).transform.rotation.yaw
            delta_yaw = new_yaw - current_yaw

        if math.fabs(delta_yaw) > 360:
            delta_yaw = delta_yaw % 360

        if delta_yaw > 180:
            delta_yaw = delta_yaw - 360
        elif delta_yaw < -180:
            delta_yaw = delta_yaw + 360

        angular_velocity = carla.Vector3D(0, 0, 0)
        if target_speed == 0:
            angular_velocity.z = 0
        else:
            angular_velocity.z = delta_yaw / (direction_norm / target_speed)
        self._actor.set_target_angular_velocity(angular_velocity)

        self._last_update = current_time

        return direction_norm


    def _set_new_velocity_gbf(self, next_location):
        """
        Calculate and set the new actor veloctiy given the current actor
        location and the _next_location_

        If _consider_obstacles is true, the speed is adapted according to the closest
        obstacle in front of the actor, if it is within the _proximity_threshold distance.
        If _consider_trafficlights is true, the vehicle will enforce a stop at a red
        traffic light.
        If _max_deceleration is set, the vehicle will reduce its speed according to the
        given deceleration value.
        If the vehicle reduces its speed, braking lights will be activated.

        Args:
            next_location (carla.Location): Next target location of the actor

        returns:
            direction (carla.Vector3D): Length of direction vector of the actor
        """

        current_time = GameTime.get_time()
        # target_speed = self._target_speed
        target_speed = self._get_target_speed()

        if not self._last_update:
            self._last_update = current_time

        current_speed = math.sqrt(self._actor.get_velocity().x**2 + self._actor.get_velocity().y**2)

        if self._consider_obstacles:
            # If distance is less than the proximity threshold, adapt velocity
            if self._obstacle_distance < self._proximity_threshold:
                distance = max(self._obstacle_distance, 0)
                if distance > 0:
                    current_speed_other = math.sqrt(
                        self._obstacle_actor.get_velocity().x**2 + self._obstacle_actor.get_velocity().y**2)
                    if current_speed_other < current_speed:
                        acceleration = -0.5 * (current_speed - current_speed_other)**2 / distance
                        target_speed = max(acceleration * (current_time - self._last_update) + current_speed, 0)
                else:
                    target_speed = 0

        if self._consider_traffic_lights:
            if (self._actor.is_at_traffic_light() and
                    self._actor.get_traffic_light_state() == carla.TrafficLightState.Red):
                target_speed = 0

        if target_speed < current_speed:
            if not self._brake_lights_active:
                self._brake_lights_active = True
                light_state = self._actor.get_light_state()
                light_state |= carla.VehicleLightState.Brake
                self._actor.set_light_state(carla.VehicleLightState(light_state))
            if self._max_deceleration is not None:
                target_speed = max(target_speed, current_speed - (current_time -
                                                                  self._last_update) * self._max_deceleration)
        else:
            if self._brake_lights_active:
                self._brake_lights_active = False
                light_state = self._actor.get_light_state()
                light_state &= ~carla.VehicleLightState.Brake
                self._actor.set_light_state(carla.VehicleLightState(light_state))
            if self._max_acceleration is not None:
                tmp_speed = min(target_speed, current_speed + (current_time -
                                                               self._last_update) * self._max_acceleration)
                # If the tmp_speed is < 0.5 the vehicle may not properly accelerate.
                # Therefore, we bump the speed to 0.5 m/s if target_speed allows.
                target_speed = max(tmp_speed, min(0.5, target_speed))
    

        # set new linear velocity
        flag=True
        velocity = carla.Vector3D(0, 0, 0)
        # line a
        direction = next_location - CarlaDataProvider.get_location(self._actor)
        direction_norm = math.sqrt(direction.x**2 + direction.y**2)
        '''
        # check this point
        if self.last_location is not None:
            # line b
            direction_waypoint = self.last_location - next_location
            direction_waypoint_norm = math.sqrt(direction_waypoint.x**2 + direction_waypoint.y**2)
            # line c
            c = self.last_location - CarlaDataProvider.get_location(self._actor)
            c_len = math.sqrt(c.x**2 + c.y**2)
            if direction_norm == 0 or direction_waypoint_norm == 0:
                flag = False
            else:
                cos = (direction_norm * direction_norm + direction_waypoint_norm * direction_waypoint_norm - c_len * c_len)/(2 * direction_norm * direction_waypoint_norm)
                angle = math.degrees(math.acos(cos))
                if angle > 60 :
                    flag = False

        if flag == False:
            self._waypoints = self._waypoints[1:]
            if not self._waypoints:
                self._reached_goal = True
                return direction_norm
            else:
                next_location = self._offset_waypoint(self._waypoints[0])
                direction = next_location - CarlaDataProvider.get_location(self._actor)
                direction_norm = math.sqrt(direction.x**2 + direction.y**2)
        '''
        self.last_location = next_location

        velocity.x = direction.x / direction_norm * target_speed
        velocity.y = direction.y / direction_norm * target_speed

        self._actor.set_target_velocity(velocity)

        # set new angular velocity
        current_yaw = CarlaDataProvider.get_transform(self._actor).rotation.yaw
        # When we have a waypoint list, use the direction between the waypoints to calculate the heading (change)
        # otherwise use the waypoint heading directly
        if self._waypoints:
            delta_yaw = math.degrees(math.atan2(direction.y, direction.x)) - current_yaw
        else:
            new_yaw = CarlaDataProvider.get_map().get_waypoint(next_location).transform.rotation.yaw
            delta_yaw = new_yaw - current_yaw

        if math.fabs(delta_yaw) > 360:
            delta_yaw = delta_yaw % 360

        if delta_yaw > 180:
            delta_yaw = delta_yaw - 360
        elif delta_yaw < -180:
            delta_yaw = delta_yaw + 360

        angular_velocity = carla.Vector3D(0, 0, 0)
        if target_speed == 0:
            angular_velocity.z = 0
        else:
            angular_velocity.z = delta_yaw / (direction_norm / target_speed)
        self._actor.set_target_angular_velocity(angular_velocity)

        self._last_update = current_time

        return direction_norm
    
    def _get_target_speed(self):
        #print(current_time)
        speed = 0
        if self.index >= len(self.speed_profile) :
            speed = self.speed_profile[-1]
        else:
            speed = self.speed_profile[self.index]
            self.index = self.index + 1

        self.update_target_speed(speed)

        return speed


