import gymnasium as gym
from gymnax.environments import environment
from gymnax.environments import spaces
import pygame
from math import sin, cos, radians
from pygame.math import Vector2
import os, random
from PIL import Image
from flax import struct
import jax
import jax.numpy as jnp
import jax.random as jrandom
import chex
from typing import Dict, Tuple, Any, Union, Optional

from max_speed import max_speed

#from player_movement import moves

from functions import get_angle, linesCollided, getCollisionPoint, dist, \
    get_direction, angle_to_wind, vel1, vel2, vel3, vel4

from display_size import displayHeight, displayWidth

import numpy as np


class Wall:

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = displayHeight - y1
        self.x2 = x2
        self.y2 = displayHeight - y2


    def draw(self, surface):
        pygame.draw.line(surface, (50,120,100), (self.x1, self.y1), (self.x2, self.y2), width=10)

    """
    returns true if the car object has hit this wall
    """

    def hitCar(self, car):
        cw = car.width
        # since the car sprite isn't perfectly square the hitbox is a little smaller than the width of the car
        ch = car.height - 4
        rightVector = Vector2(car.direction)
        upVector = Vector2(car.direction).rotate(-90)
        carCorners = []
        cornerMultipliers = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
        carPos = car.position
        for i in range(4):
            carCorners.append(carPos + (rightVector * cw / 2 * cornerMultipliers[i][0]) +
                              (upVector * ch / 2 * cornerMultipliers[i][1]))

        for i in range(4):
            j = i + 1
            j = j % 4
            if linesCollided(self.x1, self.y1, self.x2, self.y2, carCorners[i].x, carCorners[i].y, carCorners[j].x,
                              carCorners[j].y):
                return True
        return False


class RewardGate:

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.active = True

        self.center = Vector2((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

 
    def draw(self, surface):
        if self.active:
            pygame.draw.line(surface, (255,0,255), (self.x1, self.y1), (self.x2, self.y2), width=2)

    """
    returns true if the car object has hit this wall
    """

    def hitCar(self, car):
        if not self.active:
            return False

        cw = car.width
        # since the car sprite isn't perfectly square the hitbox is a little smaller than the width of the car
        ch = car.height - 4
        rightVector = Vector2(car.direction)
        upVector = Vector2(car.direction).rotate(-90)
        carCorners = []
        cornerMultipliers = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
        carPos = car.position
        for i in range(4):
            carCorners.append(carPos + (rightVector * cw / 2 * cornerMultipliers[i][0]) +
                              (upVector * ch / 2 * cornerMultipliers[i][1]))

        for i in range(4):
            j = i + 1
            j = j % 4
            if linesCollided(self.x1, self.y1, self.x2, self.y2, carCorners[i].x, carCorners[i].y, carCorners[j].x,
                              carCorners[j].y):
                return True
        return False


class Boat:
    def __init__(self, x, y, walls, rewardGates, width=10, height=15):
        self.x = x
        self.y = y
        self.position = Vector2(x, y)
        self.speed = 0
        self.angle = -30
        
        self.direction = get_direction(self.angle)
        
        self.dead = False
        self.width = width
        self.height = height
        self.lineCollisionPoints = []
        self.collisionLineDistances = []
        self.vectorLength = displayWidth #length of vision vectors
        
        self.turningLeft = False
        self.turningRight = False
        self.walls = walls
        self.rewardGates = rewardGates
        self.rewardNo = 0
        
        self.directionToRewardGate = self.rewardGates[self.rewardNo].center - self.position

        self.reward = 0

        self.score = 0
        self.lifespan = 0
        
        self.steps_between_gate = 0
        self.dist_between_gate = self.directionToRewardGate.length()
        
        self.image = pygame.image.load("./boaty_boat.png").convert_alpha()
        self.image = pygame.transform.scale(self.image, (width, height))
        self.image_orig = self.image.copy()
        
        self.max_speed = max_speed
        
        self.max_length_on_screen = (displayHeight ** 2 + displayWidth ** 2) ** 0.5 #distance from corner to corner
        
        self.last_distance_reward_gate = self.max_length_on_screen
        
        
    
    def reset(self):
        self.position = Vector2(self.x, self.y)
        self.speed = 0
        self.angle = -30
        
        self.direction = get_direction(self.angle)
        
        self.dead = False
        self.lineCollisionPoints = []
        self.collisionLineDistances = []

        self.turningLeft = False
        self.turningRight = False
    
        self.rewardNo = 0
        self.reward = 0
        
        self.directionToRewardGate = self.rewardGates[self.rewardNo].center - self.position
        
        self.steps_between_gate = 0
        self.dist_between_gate = self.directionToRewardGate.length()

        self.lifespan = 0
        self.score = 0
        for g in self.rewardGates:
            g.active = True
            
        self.last_distance_reward_gate = self.max_length_on_screen
        
    
    def checkRewardGates(self):
        self.reward = 0            
        if self.rewardGates[self.rewardNo].hitCar(self):
            self.rewardGates[self.rewardNo].active = False
            self.rewardNo += 1
            self.score += 1
            self.reward = 10000 * ((10000-self.steps_between_gate) / 10000) #10000 is the max steps per episode, this creates a time pressure
            self.steps_between_gate = 0
            if self.rewardNo == len(self.rewardGates):
                self.dist_between_gate = 100
            else:
                self.directionToRewardGate = self.rewardGates[self.rewardNo].center - self.position
                self.dist_between_gate = self.directionToRewardGate.length()
            #state = self.getState()
            #print(state[-3])
            if self.rewardNo == len(self.rewardGates):
                self.rewardNo = 0
                for g in self.rewardGates:
                    g.active = True      
         
        self.directionToRewardGate = self.rewardGates[self.rewardNo].center - self.position
        
        if self.last_distance_reward_gate > self.directionToRewardGate.length():
            self.reward += ((self.dist_between_gate - self.directionToRewardGate.length()) / self.dist_between_gate)
        elif self.last_distance_reward_gate < self.directionToRewardGate.length():
            self.reward -= 1 - ((self.dist_between_gate - self.directionToRewardGate.length()) / self.dist_between_gate)
            
        #print('{} last'.format(self.last_distance_reward_gate))
        #print('{} now'.format(self.directionToRewardGate.length()))
        
        self.last_distance_reward_gate = self.directionToRewardGate.length()

    """
    checks every wall and if the car has hit a wall returns true    
    """

    def hitAWall(self):
        for wall in self.walls:
            if wall.hitCar(self):
                return True

        return False


    def getState(self):
        self.setVisionVectors()
        normalizedVisionVectors = [1 - (max(1.0, line) / self.vectorLength) for line in self.collisionLineDistances]
        normalizedVisionVectors = [value + 1 if value == 0 else value for value in normalizedVisionVectors]
        normalizedVisionVectors = [values if values >= 0.97 else 0 for values in normalizedVisionVectors]
        
        normalizedAngleOfNextGate = (get_angle(self.direction) - get_angle(self.directionToRewardGate)) % 360
        if normalizedAngleOfNextGate > 180:
            normalizedAngleOfNextGate = -1 * (360 - normalizedAngleOfNextGate)

        normalizedAngleOfNextGate /= 180
        
        normalizedDirecToRewardGate_x = self.directionToRewardGate[0] / displayWidth
        normalizedDirecToRewardGate_y = self.directionToRewardGate[1] / displayHeight
        
        normalizedDistToRewardGate = self.directionToRewardGate.length() / self.max_length_on_screen
        
        normalizedAngleToWind = angle_to_wind(self.angle) / np.pi
        
        normalizedSpeed = self.speed / self.max_speed

        normalizedState = [*normalizedVisionVectors, #0 to 1
                           normalizedAngleOfNextGate,
                           #normalizedDirecToRewardGate_x, #-1 to 1
                           #normalizedDirecToRewardGate_y, #-1 to 1
                           normalizedDistToRewardGate, #0 to 1
                           normalizedAngleToWind, #0 to 1
                           normalizedSpeed] #0 to 1
        #print(normalizedState)
        return np.array(normalizedState)
    
    
    """
    by creating lines in many directions from the car and getting the closest collision point of that line
    we create  "vision vectors" which will allow the car to 'see' 
    kinda like a sonar system
    """
    def setVisionVectors(self):
        h = self.height# - 4
        w = self.width
        self.collisionLineDistances = []
        self.lineCollisionPoints = []
        self.setVisionVector(w / 2, 0, 0)
        
        self.setVisionVector(w / 2, -h / 2, -180 / 16)
        self.setVisionVector(w / 2, -h / 2, -180 / 4)
        #self.setVisionVector(w / 2, -h / 2, -4 * 180 / 8)

        self.setVisionVector(w / 2, h / 2, 180 / 16)
        self.setVisionVector(w / 2, h / 2, 180 / 4)
        #self.setVisionVector(w / 2, h / 2, 4 * 180 / 8)

        #self.setVisionVector(-w / 2, -h / 2, -6 * 180 / 8)
        #self.setVisionVector(-w / 2, h / 2, 6 * 180 / 8)
        #self.setVisionVector(-w / 2, 0, 180)
        
        
    
    def getPositionOnCarRelativeToCenter(self, right, up):
        rightVector = Vector2(self.direction)
        rightVector.normalize()
        upVector = self.direction.rotate(90)
        upVector.normalize()

        return self.position + ((rightVector * right) + (upVector * up))
        
    """
    returns the point of collision of a line (x1,y1,x2,y2) with the walls, 
    if multiple walls are hit it returns the closest collision point
    """

    def getCollisionPointOfClosestWall(self, x1, y1, x2, y2):
        minDist = 2 * displayWidth
        closestCollisionPoint = Vector2(0, 0)
        for wall in self.walls:
            collisionPoint = getCollisionPoint(x1, y1, x2, y2, wall.x1, wall.y1, wall.x2, wall.y2)
            if collisionPoint is None:
                continue
            if dist(x1, y1, collisionPoint.x, collisionPoint.y) < minDist:
                minDist = dist(x1, y1, collisionPoint.x, collisionPoint.y)
                closestCollisionPoint = Vector2(collisionPoint)
        return closestCollisionPoint

    """
    calculates and stores the distance to the nearest wall given a vector 
    """

    def setVisionVector(self, startX, startY, gamma):
        collisionVectorDirection = self.direction.rotate(gamma)
        collisionVectorDirection = collisionVectorDirection.normalize() * self.vectorLength
        startingPoint = self.getPositionOnCarRelativeToCenter(startX, startY)
        collisionPoint = self.getCollisionPointOfClosestWall(startingPoint.x, startingPoint.y,
                                                              startingPoint.x + collisionVectorDirection.x,
                                                              startingPoint.y + collisionVectorDirection.y)
        if collisionPoint.x == 0 and collisionPoint.y == 0:
            self.collisionLineDistances.append(self.vectorLength)
        else:
            self.collisionLineDistances.append(
                dist(startingPoint.x, startingPoint.y, collisionPoint.x, collisionPoint.y))
        self.lineCollisionPoints.append(collisionPoint)
    """
    shows dots where the collision vectors detect a wall 
    """

    def showCollisionVectors(self, surface):
        for point in self.lineCollisionPoints:
            if point != [0,0]:
                pygame.draw.line(surface, (131, 139, 139), (self.position.x, self.position.y), (point.x, point.y,), 1)
                pygame.draw.circle(surface, (0, 0, 0), (point.x, point.y), 5)
            
            
    def updateWithAction(self, actionNo):
        self.turningLeft = False
        self.turningRight = False

        if actionNo == 2:
            self.turningLeft = True
        elif actionNo == 1:
            self.turningRight = True
        elif actionNo == 0:
            pass
        
        totalReward = 0

        for i in range(1):
            if not self.dead:
                self.lifespan+=1
                self.steps_between_gate+=1
                self.move()

                if self.hitAWall():
                    self.dead = True
                    totalReward -= 100000
                    # return
                
                #totalReward += self.reward
                
                if self.score == 32: #finishes game after 4 laps
                    self.dead = True
                    #totalReward += 100000
                  
                self.checkRewardGates()
                totalReward += self.reward

        self.setVisionVectors()
        
        self.reward = totalReward
        
        return self.getState(), self.reward, self.dead, {}

        
            
    def move(self):
        
        self.speed = self.rew(angle_to_wind(self.angle)) * 4
        
        self.position.x = self.position.x - (self.speed * sin(radians(self.angle)))
        self.position.y = self.position.y - (self.speed * cos(radians(self.angle)))
        
        self.direction = get_direction(self.angle)
        
        if self.turningRight:
            self.angle -= 4
        elif self.turningLeft:
            self.angle += 4
    
    
    def rew(self, theta, theta_0=0, theta_dead=np.pi / 12):
        if angle_to_wind(self.angle) <= 7*np.pi/36:
            return vel1(theta, theta_0, theta_dead) * np.cos(theta)
        elif angle_to_wind(self.angle) > 7*np.pi/36 and angle_to_wind(self.angle) <= 5*np.pi/8:
            return vel2(theta)
        elif angle_to_wind(self.angle) > 5*np.pi/8 and angle_to_wind(self.angle) <= 3*np.pi/4:
            return vel3(theta)
        elif angle_to_wind(self.angle) > 3*np.pi/4 and angle_to_wind(self.angle) <= np.pi:
            return vel4(theta)



@struct.dataclass
class EnvState(environment.EnvState):
    boat_pos: jnp.ndarray
    marks: jnp.ndarray
    reward_gate: jnp.ndarray
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    max_steps_in_episode: int = 500  # v0 had only 200 steps!


class Game(environment.Environment[EnvState, EnvParams]):
    def __init__(self, player):
        super().__init__()

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters for CartPole-v1
        return EnvParams()

    def step_env(self, key: chex.PRNGKey, state: EnvState,  action: Union[int, float, chex.Array], params: EnvParams
                 ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        s, r, d, _ = self.boat.updateWithAction(action)
        self.render()
        if self.take_pics:
            self.save_pics(value)

        # Update state dict and evaluate termination conditions
        state = EnvState(
            x=x,
            x_dot=x_dot,
            theta=theta,
            theta_dot=theta_dot,
            time=state.time + 1,
        )
        done = self.is_terminal(state, params)

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            jnp.array(reward),
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
            self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        init_state = jax.random.uniform(key, minval=-0.05, maxval=0.05, shape=(4,))
        state = EnvState(
            x=init_state[0],
            x_dot=init_state[1],
            theta=init_state[2],
            theta_dot=init_state[3],
            time=0,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        """Applies observation function to state."""
        return jnp.array([state.x, state.x_dot, state.theta, state.theta_dot])

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        # Check termination criteria
        done1 = jnp.logical_or(
            state.x < -params.x_threshold,
            state.x > params.x_threshold,
        )
        done2 = jnp.logical_or(
            state.theta < -params.theta_threshold_radians,
            state.theta > params.theta_threshold_radians,
        )

        # Check number of steps in episode termination condition
        done_steps = state.time >= params.max_steps_in_episode
        done = jnp.logical_or(jnp.logical_or(done1, done2), done_steps)
        return done

    def render(self):
        state = self.get_state()
        state = state[-4]
        state *= 180
        self.screen.fill((127, 225, 212))

        for i in self.walls:
            i.draw(self.screen)
        for j in self.gates:
            j.draw(self.screen)

        self.boat.showCollisionVectors(self.screen)
        rotated = pygame.transform.rotate(self.boat.image_orig, self.boat.angle)
        rect = rotated.get_rect()
        self.screen.blit(rotated, self.boat.position - (rect.width / 2, rect.height / 2))

        pygame.draw.circle(self.screen, (0, 0, 0), (800, 100), 4)
        pygame.draw.circle(self.screen, (0, 0, 0), (350, 150), 4)
        pygame.draw.circle(self.screen, (0, 0, 0), (350, 600), 4)
        pygame.draw.circle(self.screen, (0, 0, 0), (700, 650), 4)
        pygame.draw.circle(self.screen, (0, 0, 0), (700, 550), 4)

        speedtext = self.myfont.render("Speed = " + str(int(self.boat.speed * 5)), 1, (0, 0, 0))
        self.screen.blit(speedtext, (7, 10))
        scoretext = self.myfont.render("Score = " + str(int(self.boat.score)), 1, (0, 0, 0))
        self.screen.blit(scoretext, (122, 10))
        rewardtext = self.myfont.render("Direc to Reward = " + str(self.boat.directionToRewardGate), 1, (0, 0, 0))
        self.screen.blit(rewardtext, (235, 10))
        gatetext = self.myfont.render("Angle To RwdGate = " + str(state), 1, (0, 0, 0))
        self.screen.blit(gatetext, (640, 10))

        pygame.display.update()

        if self.human:
            self.clock.tick(self.ticks)
        else:
            pass

    def save_pics(self, value):
        self.display_surface = pygame.display.get_surface()

        self.image3d = np.ndarray(
            (displayWidth, displayHeight, 3), np.uint8)

        pygame.pixelcopy.surface_to_array(
            self.image3d, self.display_surface)
        self.image3dT = np.transpose(self.image3d, axes=[1, 0, 2])
        im = Image.fromarray(self.image3dT)  # monochromatic image
        imrgb = im.convert('RGB')  # color image

        filename = ''.join(['Episode_',
                            str(value),
                            '-frame-',
                            str(self.boat.lifespan).zfill(5),
                            '.jpg'])
        foldername = ''.join(['./exported_frames/Episode_', str(value)])
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        filenamepath = os.path.join(foldername, filename)
        imrgb.save(filenamepath)

    @property
    def name(self) -> str:
        """Environment name."""
        return "SailingEnv-v1"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(2)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        high = jnp.array(
            [
                params.x_threshold * 2,
                jnp.finfo(jnp.float32).max,
                params.theta_threshold_radians * 2,
                jnp.finfo(jnp.float32).max,
            ]
        )
        return spaces.Box(-high, high, (4,), dtype=jnp.float32)


if __name__ == '__main__':
    env = Game('Human')
    time_steps = 800
    for _ in range(time_steps):
        s, r, d, _ = env.step(random.randint(0, 2), 'e')
        #print(f'state: {s}')
        #print(f'reward: {r}')
        env.render()
    #env.close()
    
    # env = Game('ai')
    # time_steps = 5000#800
    # for _ in range(time_steps):
    #     s, r, d, _ = env.step(moves[_], 'e')
    #     #print(f'state: {s}')
    #     #print(f'reward: {r}')
    #     env.render()
    # #env.close()