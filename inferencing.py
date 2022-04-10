import logging
import os
import PAIA
import cv2
from dqn_model import DQN
import torch


class MLPlay:
    def __init__(self):
        self.step_number = 0
        self.episode_number = 1
        self.action = 1
        self.state = [] 
        self.state_n = []
        self.device = 'cpu'
        self.net = DQN((4,28,63), 5).to(self.device)
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.dat")
        self.net.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        self.cnt = 0



    def decision(self, state: PAIA.State) -> PAIA.Action:

        if state.observation.images.front.data:
            img_array = PAIA.image_to_array(state.observation.images.front.data)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            img_array = cv2.resize(img_array, (63, 28))
            #cv2.imshow("img",img_array)
            #cv2.waitKey(20)
        else:
            img_array = None


        self.step_number += 1
        self.state_n.append(img_array)
        
        if self.step_number % 4 == 0:
            self.state =  self.state_n.copy()
            self.state_n = []

        if self.step_number % 4 == 0:
            state_v = torch.tensor([self.state]).to(self.device)
            q_vals_v = self.net(state_v.float())
            _, act_v = torch.max(q_vals_v, dim=1)
            self.action = int(act_v)

        action = PAIA.create_action_object(acceleration=True, brake=False, steering=0.0)

        if state.event == PAIA.Event.EVENT_NONE:
            action_table = [
                (False, False, -1.0), # 0 -> turn left
                (True, False, 0.0), # 1 -> forward
                (False, False, 1.0), # 2 -> turn right
                (True, False, -1.0), # 3 -> forward + turn left
                (True, False, 1.0) # 4 -> forward + turn right
            ]
            action = PAIA.create_action_object(*action_table[self.action])
        elif state.event != PAIA.Event.EVENT_NONE:
            action = PAIA.create_action_object(command=PAIA.Command.COMMAND_FINISH)
            logging.info('Progress: %.3f' %state.observation.progress )
        return action