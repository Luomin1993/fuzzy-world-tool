# fuzzy-world-tool

Fuzzy World:A Tool Training Agent from Concept Cognitive to Logic Inference

Fuzzy world is a lightweight 3D virtual environment tool for multi-level intelligent test/development, and the internal logic and entity attributes of the environment are easy to redefine.

![image](https://github.com/Luomin1993/fuzzy-world/blob/master/img_train/fig_tool.png)

### Quick Start
The ratcave has a bug in resolving 3d-model,you can move my fixed into the ratcave's folder.

```
$ pip install ratcave
$ git clone https://github.com/Luomin1993/fuzzy-world-tool.git
$ cd fuzzy-world-tool
$ mv texture.py /usr/local/lib/python3.6/dist-packages/ratcave/
$ python3.6 run_world.py
```

which can start a virtual environment with logic tasks by a navigational agent using PID algorithm to avoid obstacles.

Even if the agent has enough information,logical reasoning tasks can not be completed efficiently through end-to-end models.Some advancing approaches that address the challenge of logic reasoning have been developed ,however, these models often rely on hard-ruled inference or pure fuzzy computation, which imposes limits on reasoning abilities of agents. Agent with logical reasoning abilities is not an inseparable end-to-end system,on the contrary,logical reasoning requires three levels of ability from the bottom up(the three-level learning paradigm):

- Basic recognition ability;
- Concept cognitive ability;
- Logic reasoning ability;

### Basic Recognition Ability

```
# world_object_dete.py
@window.event
def take_screenshot(dt):
    """ takes a screenshot of the client size and saves the image """
    global times,img_id;times+=1;
    if times%20==0:pyglet.image.get_buffer_manager().get_color_buffer().save('vae_data/'+obj_name+'_'+obj_color+'_'+str(img_id)+'.png');img_id+=1;
    if img_id == 1200:sys.exit(1);
pyglet.clock.schedule(take_screenshot);
```

![image](https://github.com/Luomin1993/fuzzy-world/blob/master/img_train/fig_recog.png)

The data generated can be also used for testing generative models(like VAE):

![image](https://github.com/Luomin1993/fuzzy-world/blob/master/img_train/fig_vae.png)

### Concept Cognitive Ability

```
# make_dataset.py
def make_scene(obj_list,scene):
    # Image size:400*400;Grid size 4*4;
    unit_size = 100;
    occupied  = [];
    bboxes    = [];
    sizes     = [];
    obj_names = [];
    # Traverse the list of objects added to the scene;
    for obj in obj_list:
        # Assign the spatial location of the entity;
        pos = (np.random.randint(4),np.random.randint(4));
        while pos in occupied:pos = (np.random.randint(4),np.random.randint(4));
        occupied.append(pos);
        # Image synthesis;
        scene = paste_alpha(obj.img_arr,scene,(unit_size*pos[0],0,unit_size*pos[1],0  ));
        # Record the bbox position coordinates of the entity in the figure;
        bboxes.append([unit_size*pos[1],unit_size*pos[0],obj.bbox[1]-obj.bbox[0],obj.bbox[3]-obj.bbox[2]]);
        # Record the pixel size of the entity;
        sizes.append([obj.bbox[1]-obj.bbox[0],obj.bbox[3]-obj.bbox[2]]);
    return (scene,occupied,bboxes,sizes);
```

![image](https://github.com/Luomin1993/fuzzy-world/raw/master/img_train/fig_concept.png)

### Logic Reasoning Ability

Baseline model:Guided feature transformation(gft): A neural language grounding module for embodied agents

```
# baseline model: model_baidu.py
print model.layers[17].name;
for i in range(len(model.layers)):
    if model.layers[i].name=='emb_1':print i;
history = LossHistory();
model.summary();
model.fit([data_l,data_o,data_a_last], data_a,epochs=30, batch_size=5);
model.save('gft.h5')
from keras.utils import plot_model
plot_model(model, to_file='model.png')
```

![image](url)

### Future Work
The task integrating the content learned by agent in virtual environment into real environment based on open source hardware.

![image](https://github.com/Luomin1993/fuzzy-world/blob/master/img_train/fig_expr.png)
