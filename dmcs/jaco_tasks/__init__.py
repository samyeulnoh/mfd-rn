from dmcs.jaco_tasks.custom import (
    cjaco_reach,              # easy 
    cjaco_push,               # medium
    cjaco_lift,               # medium
    cjaco_stack_box_obstacle, # hard
    cjaco_push_box_obstacle,  # hard
    cjaco_lift_box_obstacle,  # hard
)

def make_jaco(domain, task, seed):
    assert domain == "jaco"
    count = len(task.split("_"))
    if count == 2: # easy or medium
        task_name, prop_name = task.split("_", 1)
        if task_name == "reach":
            return cjaco_reach.make(prop_name, seed)
        elif task_name == "push":
            return cjaco_push.make(prop_name, seed)
        elif task_name == "lift":
            return cjaco_lift.make(prop_name, seed)
        else:
            raise ValueError(f"{task_name} is not found.")
    elif count == 3: # hard
        task_name, prop_name, constraint_name = task.split("_", 2)
        assert constraint_name == "obstacle"
        if task_name == "stack":
            return cjaco_stack_box_obstacle.make(prop_name, seed)
        elif task_name == "push":
            return cjaco_push_box_obstacle.make(prop_name, seed)
        elif task_name == "lift":
            return cjaco_lift_box_obstacle.make(prop_name, seed)
        else:
            raise ValueError(f"{task_name} is not found.")
    else:
        raise ValueError(f"{task} is not found.")
