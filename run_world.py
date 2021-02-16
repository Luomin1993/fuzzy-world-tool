#!/usr/bin/env python
# -*- coding: utf-8 -*-

__ENV__  =  'python3';
__author__ =  'hanss401';

import numpy as np;
import re;
import pyglet;
from pyglet.window import key;
import ratcave as rc;
from PIL import Image;
import cv2;
from scipy.ndimage import filters;
import matplotlib.pyplot as plt
from matplotlib.patches import Circle;
import detection as dete;
# /usr/local/lib/python3.6/dist-packages/ratcave/texture.py

# ============================== Define objects and states  ==================================
global_objs   = ['horse','bed','jeep','sw_pipe','music','cow','table','chair','truck'];
global_states = ['TEMP_NORMAL','TEMP_HIGH','MOVE','STOP','NAVIGATION'];
POS_LEFT_WALL  = -2;
POS_RIGHT_WALL = 2.5;
POS_DOWN_WALL = -2.5;
POS_UP_WALL = 2.2;
SPEED_AGENT = 0.7;
DIRE_NAVI = 1;
OBSTACLES_WALLS = [(POS_LEFT_WALL,'x'),(POS_RIGHT_WALL,'x'),(POS_DOWN_WALL,'z'),(POS_UP_WALL,'z')];
OBSTACLES_OBJS = [];


window = pyglet.window.Window(width=555, height=555,caption='Fullscreen');
window.set_location(555, 333);
#window.set_fullscreen(fullscreen=True, width=800, height=800);
keys = key.KeyStateHandler();
window.push_handlers(keys);


class Obj_Attr(object):
    """The Attributes of the object in world"""
    def __init__(self,mesh,position,move,temp,material_id,is_switch=False,turn_on='TURN_OFF'):
        super(Obj_Attr, self).__init__()
        self.mesh         =          mesh;
        self.position     =      position;
        self.material_id  =   material_id;
        self.TAB_STATES = {'TEMP_NORMAL':0,'TEMP_HIGH':0,'MOVE':0,'STOP':0,'NAVIGATION':0,'TURN_ON':0,'TURN_OFF':0};
        self.TAB_STATES[temp]=1;
        self.TAB_STATES[turn_on]=1;
        self.TAB_STATES[move]=1;
        self.SPEED = [0,0];

class World(object):
    """Wrapper of the scene"""
    def __init__(self):
        super(World, self).__init__();
        self.scene = rc.Scene(meshes=[],camera=rc.Camera(orientation0=(0, 0, -1),rotation=(0, 0, 0)))
        self.scene.bgColor = 133, 133, 33
        self.name  = 'World';
        self.objs  = [];
        self.objs_num = len(self.objs);
        self.rules = [];
        self.rules_num = len(self.rules);
        self.size  = 32;

    def make_world_from_fw(self,fw_path):
        f = open(fw_path);
        self.name     = re.match(r'WORLD_NAME:(.*);',f.readline()).groups()[0];
        self.size     = int(re.match(r'WORLD_SIZE:(.*);',f.readline()).groups()[0]);
        self.objs_num = int(re.match(r'OBJ_NUM:(.*);',f.readline()).groups()[0]);
        self.objs.append(Obj_Attr( rc.WavefrontReader('models/box.obj').get_mesh("box",position=(0, -.1, -1.5), scale=.03*self.size/20, rotation=(0, -90, 0)),
                         (0,0),0,0,0)); # add floor;
        self.objs[0].mesh.textures.append(rc.Texture().from_image('models/wall.jpg'));
        for i in range(self.objs_num):
            self.objs.append(self.resolve_obj( f.readline() ));
            if i>0:OBSTACLES_OBJS.append(self.objs[-1].position);
        self.rules_num = int(re.match(r'RULE_NUM:(.*);',f.readline()).groups()[0]);
        for i in range(self.rules_num):
            self.rules.append(self.resolve_rule( f.readline() ));    
        self.objs[1].mesh.textures.append(rc.Texture().from_image('models/roof_1.jpg'));    
            
    def resolve_rule(self,line):
        (state_1,state_2) = re.match(r'RULE:(.*)=>(.*);',line).groups();
        state_1 = re.match(r'"(.*)":"(.*)"',state_1).groups();
        state_2 = re.match(r'"(.*)":"(.*)"',state_2).groups();
        return ( (state_1[1],int(state_1[0]) ) ,(state_2[1],int(state_2[0]) ) );

    def resolve_obj(self,line):
        (ID,model,mesh,pos,temp,move) = re.match(r'OBJ:ID:"(.*)";MODEL:"(.*)";MESH:"(.*)";POS:(.*);TEMP:"(.*)";MOVE:"(.*)";',line).groups();
        #print (ID,model,mesh,pos);
        pos = pos.split(',');pos=[float(i) for i in pos];
        entity = rc.WavefrontReader(model).get_mesh(mesh,position=(pos[0], -.1,pos[1]), scale=.1, rotation=(0, 0, 0));
        #entity.uniforms['diffuse'] = 1, 1, 0 #give color;
        return Obj_Attr(entity,pos,move,temp,0);

    def make_world_to_fw(self,fw_path):
        pass;
    
    def add_obj(self):
        pass;  

    @window.event
    def show(self):
        with rc.default_shader:
            self.scene.meshes = [i.mesh for i in self.objs];
            print('ok');
            self.scene.draw()      

class Agent(object):
    """The agent in the fworld"""
    def __init__(self):
        super(Agent, self).__init__()
        self.actions  = None;
        self.reward   = 0;
        self.SG       = None; #Semantic Graph;
        self.see      = None;

    def do_action(self,ACT_ID,OBJ_ID):
        #return self.actions[np.random.randint(low=0, high=3)];
        global world;
        ACTION[ACT_ID](world.objs[OBJ_ID]);

class Teacher(object):
    """The teacher in the fworld"""
    def __init__(self):
        super(Teacher, self).__init__()
        self.name = None;

# ============================== basic functions  ==============================
'''make fuzzy-world map based on finite objs and rules'''
def make_world(path,index):
    global_objs   = ['horse','bed','jeep','sw_pipe','music','cow','table','chair','truck'];
    global_states = ['TEMP_NORMAL','TEMP_HIGH','MOVE','STOP'];
    #------ pick objs in the world -----------
    objs = [];
    while(True):
        obj = global_objs[np.random.randint(low=0, high=len(global_objs))];
        if obj not in objs:objs.append(obj);
        if len(objs)==5:break;
    #------ make pos in the world ------------
    # x = [-1,-0.5,0,0.5,1];
    # y = [-1,-0.5,0];
    x = [-1.5,-0.75,0,0.75,1.5];
    y = [-1.5,-0.75,0,0.75];
    poses = [];
    poses_in = [];
    for i in range(len(objs)):
        while 1:
            (x_this,y_this) = (np.random.randint(low=0, high=len(x)), np.random.randint(low=0, high=len(y))); 
            if (x_this,y_this) not in poses:poses_in.append((x_this,y_this));poses.append((x[x_this],y[y_this]));break;
            else:continue;
    #------ make rules ---------
    rules = [];
    for i in range(4):
        while 1:
            (id_1,state_1,id_2,state_2) = (np.random.randint(low=0,high=len(objs)),
                                         np.random.randint(low=0,high=len(global_states)),
                                         np.random.randint(low=0,high=len(objs)),
                                         np.random.randint(low=0,high=len(global_states)))
            if ( (id_1,state_1,id_2,state_2) not in rules ) and ( not (id_1==id_2 and state_1==state_2) ):
                rules.append((id_1,state_1,id_2,state_2));break;#print rules[-1];
            else:continue;
    #------ write into file ------
    f = open(path+'/sample_'+str(index)+'.fw', 'w');
    f.write( 'WORLD_NAME:sam'+str(index)+';\n'   );
    f.write( 'WORLD_SIZE:44;\n' );
    f.write( 'OBJ_NUM:'+str(len(objs))+';\n');
    for i in range(len(objs)):
        f.write( 'OBJ:ID:"'+str(i)+'";MODEL:"./models/'+objs[i]+'.obj";MESH:"'+objs[i]+'";POS:'+str(poses[i][0])+','+str(poses[i][1])+';TEMP:"TEMP_NORMAL";MOVE:"STOP";\n' );
    f.write( 'RULE_NUM:'+str(len(rules))+';\n' )
    for i in range(len(rules)):
        f.write( 'RULE:"'+str(rules[i][0])+'":"'+str(global_states[rules[i][1]])+'"=>"'+str(rules[i][2])+'":"'+str(global_states[rules[i][3]])+'";\n' );    
    f.close();

def take_screenshot(dt):
    """ takes a screenshot of the client size and saves the image """
    global times;times+=1;
    if times%1000==0:pyglet.image.get_buffer_manager().get_color_buffer().save('img/screenshot'+str(times)+'.png');

def take_filter(dt):
    """ give filter show of saved images """
    global times;
    if times == 0 or times ==10:return;
    if times%10==0: 
        im   = np.array(Image.open('img/screenshot'+str(times-10)+'.png').convert('L'));
        imx  = np.zeros(im.shape);filters.sobel(im,1,imx);
        imy  = np.zeros(im.shape);filters.sobel(im,0,imy);
        imxy = np.sqrt(imx**2+imy**2);
        Image.fromarray( np.append(np.append(imx,imy,axis=1),imxy,axis=1)  ).convert('RGB').save('img_filter/f'+str(times)+'.jpg');
        cv2.imshow('time',cv.imread('img_filter/f'+str(times)+'.jpg'));cv.waitKey(52);    

def give_dete(dt):
    """ give dete show of saved images """
    global times;
    if times == 0 or times ==10:return;
    if times%10==0: 
        img   = np.array(Image.open('img/screenshot'+str(times-10)+'.png').convert('L'));
        #---- draw bboxes ---------
        bboxes = dete.give_dete_bbox();
        for bbox in bboxes:
            cv2.rectangle(img,bbox[0], bbox[1],(255,0,0),2);
        #---- draw texts ---------
        texts = dete.give_class();
        for i in range(len(texts)):
            cv2.putText(img,texts[i],bboxes[i][0],cv2.FONT_HERSHEY_PLAIN,1.4,(111,111,255),1);
        #---- draw lines ---------
        states = dete.give_states();
        # print bboxes
        for i in range(len(states)):
            # w1     = bboxes[obj1][2];
            # h1     = bboxes[states[i][1]][3];
            # w2     = bboxes[obj2][2];
            # h2     = bboxes[states[i][2]][3];
            obj1 = states[i][1];
            obj2 = states[i][2];
            point1 = ( int((bboxes[obj1][0][0]+bboxes[obj1][1][0])/2) ,int((bboxes[obj1][0][1]+bboxes[obj1][1][1])/2));
            point2 = ( int((bboxes[obj2][0][0]+bboxes[obj2][1][0])/2) ,int((bboxes[obj2][0][1]+bboxes[obj2][1][1])/2));
            cv2.line(img, point1, point2, (111, 255,111),1);
            cv2.putText(img,states[i][0],(  int((point1[0]+point2[0])/2),  int((point1[1]+point2[1])/2)),cv2.FONT_HERSHEY_PLAIN,1.4,(222,222,11),1);
        #---- save and show -------
        Image.fromarray( img  ).convert('RGB').save('img_filter/f'+str(times)+'.jpg');
        #cv2.imshow('time',cv2.imread('img_filter/f'+str(times)+'.jpg'));cv2.waitKey(52);  

def give_color_by_temp(obj):
    if obj.TAB_STATES['TEMP_HIGH']==1:return 0,0,1;
    if obj.TAB_STATES['TEMP_NORMAL']==1:return 0,1,0;
    return 1,0,0;

def is_state(obj,STATE_NAME,STATE_VALUE):
	return obj.TAB_STATES[STATE_NAME]==STATE_VALUE;

def set_state(obj,STATE,STATE_VALUE):
	obj.TAB_STATES[STATE]=STATE_VALUE;
	return obj;

def move_camera(dt):
    global world;
    camera_speed = 3
    if keys[key.LEFT]:
        world.scene.camera.position.x -= camera_speed * dt
    if keys[key.RIGHT]:
        world.scene.camera.position.x += camera_speed * dt
    if keys[key.UP]:
        world.scene.camera.position.z += camera_speed * dt
    if keys[key.DOWN]:
        world.scene.camera.position.z -= camera_speed * dt
    if keys[key.K]:
        world.scene.camera.position.y += camera_speed * dt
    if keys[key.L]:
        world.scene.camera.position.y -= camera_speed * dt    
    if keys[key.H]:
        world.scene.camera.rotation.x += 13 * dt
    if keys[key.J]:
        world.scene.camera.rotation.x -= 13 * dt        
    if keys[key.A]:
        world.scene.camera.rotation.y += 13 * dt;#world.scene.camera.rotation.z += 13 * dt;
    if keys[key.D]:
        world.scene.camera.rotation.y -= 13 * dt;#world.scene.camera.rotation.z -= 13 * dt; 

#----- draw the objects animations and states in the scene ------------
def __draw__(dt):
    # =========== Follow Rule ===========
    for rule in world.rules:
        #[(('STOP', 1), ('MOVE', 2))]
        #if STATE_JUDGE[rule[0][0]]( world.objs[ int(rule[0][1]) ] ): 
        if is_state(world.objs[ int(rule[0][1]) ],rule[0][0],1):
            #world.objs[ int(rule[1][1]) ] = STATE_SET[rule[1][0]](world.objs[ int(rule[1][1]) ]);
            world.objs[ int(rule[1][1]) ] = set_state(world.objs[ int(rule[1][1]) ],rule[1][0],1);
    # =========== Follow State ==========
    for obj in world.objs[1:]:
        if obj.TAB_STATES["MOVE"]==1:STATE_IM["MOVE"](obj,dt);
        if obj.TAB_STATES["TEMP_HIGH"]==1:STATE_IM["TEMP_HIGH"](obj,dt);
        if obj.TAB_STATES["TEMP_NORMAL"]==1:STATE_IM["TEMP_NORMAL"](obj,dt);
    state_navigation(world.objs[1],dt);

#------------------------- STATE implementation ------------------
def state_stop(obj,dt):
    #if world.objs[id].move_speed==0 and world.objs[id].rot_speed==0 : return True;
    #return False;
    obj.mesh.rotation.y += 0*dt;

def state_move(obj,dt):
    #if world.objs[id].move_speed==0 and world.objs[id].rot_speed==0 : return False;
    #return True; 
    obj.mesh.rotation.y += 12*dt;

def compute_speed(SPEED,THETA):
    if THETA<=90:
        return SPEED*np.cos(THETA),-SPEED*np.sin(THETA);
    if THETA<=180:
        return -SPEED*np.cos(THETA),-SPEED*np.sin(THETA);
    if THETA<=270:
        return SPEED*np.cos(THETA),-SPEED*np.sin(THETA);
    if THETA<=360:
        return -SPEED*np.cos(THETA),-SPEED*np.sin(THETA);            

def close_to_obs(obj):
    for COORDINATE in OBSTACLES_WALLS:
        if COORDINATE[1]=='x':
            DISTANCE = abs(obj.mesh.position.x - COORDINATE[0]);
        if COORDINATE[1]=='z':
            DISTANCE = abs(obj.mesh.position.z - COORDINATE[0]);
        if DISTANCE <0.1:return True;
    for COORDINATE in OBSTACLES_OBJS:
        DISTANCE = np.sqrt((obj.mesh.position.x - COORDINATE[0])**2+(obj.mesh.position.z - COORDINATE[1])**2);
        if DISTANCE <0.3:return True;    
    return False;            

def state_navigation(obj,dt):
    DIRE_NAVI = 1;
    if close_to_obs(obj):
        obj.mesh.rotation.y = (obj.mesh.rotation.y+5)%360;
        obj.SPEED[0],obj.SPEED[1] = compute_speed(SPEED_AGENT,obj.mesh.rotation.y);
    print(obj.SPEED[0],obj.SPEED[1]);    
    obj.mesh.position.x += obj.SPEED[0]*dt;
    obj.mesh.position.z += obj.SPEED[1]*dt;
    print(obj.mesh.position.x,obj.mesh.position.z);

def state_temphigh(obj,dt):
    obj.mesh.uniforms['diffuse'] = give_color_by_temp(obj);

def action_donothing(scene):
    pass;

def action_move_forward(scene):
    scene.camera.position.z += .2;

def action_move_back(scene):
    scene.camera.position.z -= .2;


def agent_move(dt):
    agent = Agent();
    agent.actions = ['DO_NOTHING','MOVE_FORWARD','MOVE_BACK'];
    ACTION[agent.do_action()](world.scene);        

def agent_act(dt):
    global time_passed,agent;
    loss_1=np.sin(time_passed);loss_2=np.cos(time_passed);loss_3=np.tan(time_passed);
    #print(' \033[0;31m STEP: \033[0m  '+str(time_passed)+' \033[0;31m Loss: \033[0m'+str(loss_1)+' \033[0;31m Loss: \033[0m'+str(loss_2)+' \033[0;31m Loss: \033[0m'+str(loss_3)) 
    #ax.scatter(time_passed,loss_1,c='b',marker='.')  #散点图
    #plt.pause(0.001)
    time_passed+=dt;#print time_passed;
    #if time_passed>4:agent.do_action("MOVE",2);agent.do_action("TEMP_HIGH",2);

# =============================== STATES  ===============================
# STATE_JUDGE = {"STOP":is_state_stop,"MOVE":is_state_move,
#                "TURN_ON":is_state_turnon,"TURN_OFF":is_state_turnoff,
#                "TEMP_HIGH":is_state_temphigh,"TEMP_NORMAL":is_state_tempnormal};

# STATE_SET = {"STOP":set_state_stop,"MOVE":set_state_move,
#              "TURN_ON":set_state_turnon,"TURN_OFF":set_state_turnoff,
#              "TEMP_HIGH":set_state_temphigh,"TEMP_NORMAL":set_state_tempnormal};

STATE_IM = {"STOP":state_stop,"MOVE":state_move,"TEMP_HIGH":state_temphigh,"TEMP_NORMAL":state_temphigh,"NAVIGATION":state_navigation};



# ============================== STEP 0:4进1出表 ===================================    
times = 0;
#pyglet.clock.schedule(take_screenshot);
#pyglet.clock.schedule(take_filter);
#pyglet.clock.schedule(give_dete);
pyglet.clock.schedule(move_camera);
# pyglet.clock.schedule(agent_move)


world = World();
world.make_world_from_fw('world_defined/sample_7.fw');
world.scene.camera.position.y += 1.8; 
world.scene.camera.rotation.x -= 50;
world.scene.meshes = [i.mesh for i in world.objs];
pyglet.clock.schedule(__draw__);
world.objs[1].SPEED[0] = -SPEED_AGENT;
time_passed = 0;
agent = Agent();

pyglet.clock.schedule(agent_act);

@window.event
def on_draw():
    global world;
    with rc.default_shader:
        world.scene.draw()

pyglet.app.run();        