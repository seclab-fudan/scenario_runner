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
        self.last_index = -1
        #self.last_position = None
        self.last_yaw = None
        self.final_index = -1

       
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

        if self._waypoints:
            # print(len(self._waypoints), self._actor.id)
            # When changing from "free" driving without pre-defined waypoints to a defined route with waypoints
            # it may happen that the first few waypoints are too close to the ego vehicle for obtaining a
            # reasonable control command. Therefore, we drop these waypoints first.
            self._reached_goal = False
            #if self.last_position is None:
                #self.last_position = CarlaDataProvider.get_location(self._actor)
            if self.last_yaw is None:
                self.last_yaw = CarlaDataProvider.get_transform(self._actor).rotation.yaw
            if self.final_index == -1:
                self.final_index = len(self._waypoints)
            # print(self.last_yaw)
            # get the time
            current_time = GameTime.get_time()
            # print(current_time)
            time_index = int(current_time * 10) 
            # print(self._waypoints[time_index].rotation.yaw)
            # print(time_index - current_time * 10)
            if time_index >= self.final_index:
                #direction_norm = self._set_new_velocity_gbf(time_index, self._offset_waypoint(self._waypoints[self.final_index]), self._get_target_speed(self.final_index), current_time)
                #if direction_norm < 4.0 and self._reached_goal == False:
                self._reached_goal = True
            elif time_index != self.last_index:
                self._set_new_velocity_gbf(time_index, self._offset_waypoint(self._waypoints[time_index]), self._get_target_speed(time_index), current_time)
                self.last_index = time_index

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
            print("never triggered")
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


    def _set_new_velocity_gbf(self, time_index, next_location, target_speed, current_time):
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

        # update the time
        if not self._last_update:
            self._last_update = current_time

        next_location = self._offset_waypoint(self._waypoints[time_index])
        last_location = self._offset_waypoint(self._waypoints[time_index - 1])

        #current_speed = math.sqrt(self._actor.get_velocity().x**2 + self._actor.get_velocity().y**2)

        # set new linear velocity
        velocity = carla.Vector3D(0, 0, 0)
        #direction = next_location - last_location
        direction = next_location - CarlaDataProvider.get_location(self._actor)
        direction_norm = math.sqrt(direction.x**2 + direction.y**2)
        if direction_norm != 0:
            velocity.x = direction.x / direction_norm * target_speed#(time_index - current_time * 10) #/ (current_time * 10 - time_index + 1)
            velocity.y = direction.y / direction_norm * target_speed#(time_index - current_time * 10) #/ (current_time * 10 - time_index + 1)
        else:
            return
        
        self._actor.set_target_velocity(velocity)

        # print(velocity)
        # set new angular velocity
        current_yaw = math.degrees(math.atan2(direction.y, direction.x))
        delta_yaw = current_yaw - self.last_yaw

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
            #angular_velocity.z = delta_yaw / (time_index - current_time * 10) #(current_time * 10 - time_index + 1)
            #print(current_time * 10 - time_index + 1)
        self._actor.set_target_angular_velocity(angular_velocity)

        self._last_update = current_time
        self.last_yaw = current_yaw
        #self.last_position = next_location

        return direction_norm
    
    def _get_target_speed(self, index):
        speed = 0
        if index >= len(self.speed_profile) :
            speed = self.speed_profile[-1]
        else:
            speed = self.speed_profile[index]

        self.update_target_speed(speed)

        return speed


