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
        self._last_update = None
        self._consider_traffic_lights = False
        self._consider_obstacles = False
        self._proximity_threshold = float('inf')
        self._max_deceleration = None
        self._max_acceleration = None

        self._obstacle_sensor = None
        self._obstacle_distance = float('inf')
        self._obstacle_actor = None
        self._obstacle_distance_threshold = 0.1

        self._visualizer = None

        self._brake_lights_active = False

        self._last_waypoint_index = 0
        self.speed_profile = []
        #self.last_position = None
        self.last_yaw = None
        self.final_index = -1
        bp = CarlaDataProvider.get_world().get_blueprint_library().find('sensor.other.obstacle')
        bp.set_attribute('distance', '10')
        bp.set_attribute('hit_radius', '1')
        bp.set_attribute('only_dynamics', 'True')
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
        self._throttle = 0.3
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

        If _waypoints are provided, the vehicle moves towards the next waypoint
        with the given _target_speed, until reaching the final waypoint. Upon reaching
        the final waypoint, _reached_goal is set to True.

        If _waypoints is empty, the vehicle moves in its current direction with
        the given _target_speed.

        For further details see :func:`_set_new_velocity`
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
        if self._waypoints:
            self._next_waypoint_index = self._get_next_waypoint_index(self._last_waypoint_index,
                                                                    look_ahead_distance=self._throttle * 5)
            if self._next_waypoint_index >= 0 and self._next_waypoint_index < len(self._waypoints):
                self._apply_control(self._next_waypoint_index)
                self._last_waypoint_index = self._next_waypoint_index
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
        time_delta = waypoint_index*1.0 - GameTime.get_time()
        time_delta = 1
        vehicle_speed = np.linalg.norm(CarlaDataProvider.get_velocity(self._actor))
        target_speed = self._get_target_speed(waypoint_index)
        kp = 0.1
        return (target_speed - vehicle_speed) / time_delta * kp

    def _lateral_control(self, waypoint_index):
        vehicle_speed = np.linalg.norm(CarlaDataProvider.get_velocity(self._actor))

        waypoint = self._waypoints[waypoint_index] 
        wp_loc_rel = self._relative_location(waypoint) + carla.Vector3D(self._wheelbase, 0, 0)
        wp_ar = [wp_loc_rel.x, wp_loc_rel.y]
        d2 = wp_ar[0]**2 + wp_ar[1]**2
        steer_rad = math.atan(2 * self._wheelbase * wp_loc_rel.y / d2)
        steer_deg = math.degrees(steer_rad)
        steer_deg = np.clip(steer_deg, -self._max_steer, self._max_steer)
        steer = steer_deg / self._max_steer
        steer = steer * vehicle_speed / math.sqrt(d2)  

        return steer

    def _stop_in_front_of_obstacle(self):
        vehicle_speed = np.linalg.norm(CarlaDataProvider.get_velocity(self._actor))
        if self._obstacle_distance < max(vehicle_speed*0.5, self._obstacle_distance_threshold):
            control_cmd = carla.VehicleControl()
            control_cmd.brake = 10
            self._actor.apply_control(control_cmd)
            return True
        else:
            return False

        
    def _speed_to_velocity(self, speed, rotation):
        return rotation.get_forward_vector() * speed

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


    def _set_new_velocity_gbf(self, time_index, next_location, target_speed, current_time):
        """
        """

        # update the time
        if not self._last_update:
            self._last_update = current_time


        #current_speed = math.sqrt(self._actor.get_velocity().x**2 + self._actor.get_velocity().y**2)

        # set new linear velocity
        velocity = carla.Vector3D(0, 0, 0)
        direction = next_location - CarlaDataProvider.get_location(self._actor)
        direction_norm = math.sqrt(direction.x**2 + direction.y**2)
        if direction_norm != 0:
            velocity.x = direction.x / direction_norm * target_speed#(time_index - current_time * 10) #/ (current_time * 10 - time_index + 1)
            velocity.y = direction.y / direction_norm * target_speed#(time_index - current_time * 10) #/ (current_time * 10 - time_index + 1)
        else:
            return
        
        self._actor.set_target_velocity(velocity)

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



def relative_location(frame, location):
    origin = frame.location
    forward = frame.get_forward_vector()
    right = frame.get_right_vector()
    up = frame.get_up_vector()
    disp = location - origin
    x = np.dot([disp.x, disp.y, disp.z], [forward.x, forward.y, forward.z])
    y = np.dot([disp.x, disp.y, disp.z], [right.x, right.y, right.z])
    z = np.dot([disp.x, disp.y, disp.z], [up.x, up.y, up.z])
    return carla.Vector3D(x, y, z)

def control_pure_pursuit(vehicle_tr, waypoint_tr, max_steer, wheelbase):
    # TODO: convert vehicle transform to rear axle transform
    wp_loc_rel = relative_location(vehicle_tr, waypoint_tr.location) + carla.Vector3D(wheelbase, 0, 0)
    wp_ar = [wp_loc_rel.x, wp_loc_rel.y]
    d2 = wp_ar[0]**2 + wp_ar[1]**2
    steer_rad = math.atan(2 * wheelbase * wp_loc_rel.y / d2)
    steer_deg = math.degrees(steer_rad)
    steer_deg = np.clip(steer_deg, -max_steer, max_steer)
    return steer_deg / max_steer

def get_next_waypoint(world, vehicle, waypoints, next_index,look_ahead_distance):
    vehicle_location = vehicle.get_location()
    min_distance = 1000

    for i in range(len(waypoints)):
        if i < next_index:
            continue
        waypoint = waypoints[i]
        waypoint_location = waypoint.location
        # print(vehicle_location.distance(waypoint_location))
        #Find the waypoint closest to the vehicle, but once vehicle is close to upcoming waypoint, search for next one
        if vehicle_location.distance(waypoint_location) < min_distance and vehicle_location.distance(waypoint_location) > look_ahead_distance:
            min_distance = vehicle_location.distance(waypoint_location)
            next_index = i
    return next_index

def spawn_waypoint_vehicle(world,waypoints,vehicle_bp,throttle=0.3):
    waypoints = [carla.Transform(carla.Location(x=p[0], y=p[1], z=0.5), carla.Rotation(yaw=0.0)) for p in waypoints]
    # vehicle_bp = world.get_blueprint_library().find('vehicle.tesla.model3') #random.choice(vehicles_bp)
    vehicle = world.spawn_actor(vehicle_bp, waypoints[0])
    world.wait_for_tick()
    print("spawn_waypoint_vehicle vehicle:{}".format(vehicle.id))
    physics_control = vehicle.get_physics_control()
    max_steer = physics_control.wheels[0].max_steer_angle
    rear_axle_center = (physics_control.wheels[2].position + physics_control.wheels[3].position)/200
    offset = rear_axle_center - vehicle.get_location()
    wheelbase = np.linalg.norm([offset.x, offset.y, offset.z])
    vehicle.set_simulate_physics(True)
    next_index = 1
    look_ahead_distance = throttle*5
    while next_index < len(waypoints)-1:
        #Get next waypoint
        next_index = get_next_waypoint(world, vehicle, waypoints,next_index,look_ahead_distance)
        waypoint = waypoints[next_index]
        #Control vehicle's throttle and steering
        vehicle_transform = vehicle.get_transform()
        steer = control_pure_pursuit(vehicle_transform, waypoint, max_steer, wheelbase)
        print("steer:",steer)
        control = carla.VehicleControl(throttle, steer)
        vehicle.apply_control(control)
        world.wait_for_tick()
    print("Finish waypoint!!!")

def spawn_autopilot_vehicle(world,vehicle_bp,position,tm_port,autopilot=False):
    spawn_point = carla.Transform(carla.Location(x=position[0], y=position[1], z=0.5), carla.Rotation(yaw=0.0)) 
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    world.wait_for_tick()
    vehicle.set_autopilot(autopilot,tm_port)
    return vehicle
