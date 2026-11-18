import os, sys, json, runpy
from pathlib import Path
import warp as wp
wp.config.enable_backward = False
import contextlib
import isaaclab.app
original_launch=isaaclab.app.launch_simulation

def install_measurements():
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_tasks.contrib.lift.mdp.rewards import object_goal_distance
    from isaaclab.utils.math import combine_frame_transforms
    from rsl_rl.runners import OnPolicyRunner

    output=Path(os.environ['CUBE_DIAG_OUTPUT'])
    limit=int(os.environ.get('CUBE_DIAG_EPISODES','20'))
    episodes=[]
    current=None
    original_reward=object_goal_distance.__call__
    def measured_reward(self, env, *args, **kwargs):
        nonlocal current
        result=original_reward(self,env,*args,**kwargs)
        if not self._track_success:
            return result
        robot=env.scene['robot']; obj=env.scene['object']
        command=env.command_manager.get_command('object_pose')
        target,_=combine_frame_transforms(robot.data.root_pos_w.torch,robot.data.root_quat_w.torch,command[:,:3])
        pos=obj.data.root_pos_w.torch[0].detach().cpu().tolist()
        tgt=target[0].detach().cpu().tolist()
        dist=float(torch.linalg.vector_norm(target[0]-obj.data.root_pos_w.torch[0]))
        if current is None:
            current={'episode':len(episodes)+1,'steps':0,'min_distance_m':float('inf'),'max_height_m':-float('inf'),'success_5cm':False,'success_10cm':False,'steps_within_10cm':0,'start_cube':pos,'target':tgt,'target_quaternion':command[0,3:].detach().cpu().tolist(),'trace':[]}
            if not episodes:
                print('DIAG_CONFIG',type(env.cfg.sim.physics).__name__, 'dt',env.step_dt,'max_steps',env.max_episode_length,'threshold',kwargs.get('success_threshold'),flush=True)
        current['steps']+=1
        if dist<current['min_distance_m']:
            current['min_distance_m']=dist; current['closest_cube']=pos
        current['max_height_m']=max(current['max_height_m'],pos[2])
        current['success_5cm'] |= pos[2]>0.04 and dist<0.05
        current['success_10cm'] |= pos[2]>0.04 and dist<0.1
        current['steps_within_10cm']+=int(pos[2]>0.04 and dist<0.1)
        current['end_cube']=pos; current['end_distance_m']=dist
        if current['steps']%5==0:
            current['trace'].append({'step':current['steps'],'cube':pos,'distance_m':dist,'ee':env.scene['ee_frame'].data.target_pos_w.torch[0,0].detach().cpu().tolist()})
        return result
    import functools
    functools.update_wrapper(measured_reward, original_reward)
    object_goal_distance.__call__=measured_reward
    original_step=ManagerBasedRLEnv.step
    def measured_step(self,action):
        nonlocal current
        result=original_step(self,action)
        if bool(result[2][0]) or bool(result[3][0]):
            current['terminated']=bool(result[2][0]);current['timeout']=bool(result[3][0])
            current['logged_success']=float(result[4].get('log',{}).get('Metrics/success_rate',-1))
            episodes.append(current)
            print('DIAG_EPISODE',json.dumps({k:v for k,v in current.items() if k!='trace'}),flush=True)
            current=None
            output.write_text(json.dumps(episodes,indent=2))
            if len(episodes)>=limit:
                print('DIAG_DONE',len(episodes),flush=True)
                self.close()
                raise KeyboardInterrupt
        return result
    ManagerBasedRLEnv.step=measured_step
    # Avoid rewriting existing exported policies during a measurement run.
    OnPolicyRunner.export_policy_to_jit=lambda *args,**kwargs:None
    OnPolicyRunner.export_policy_to_onnx=lambda *args,**kwargs:None

@contextlib.contextmanager
def measured_launch(*args, **kwargs):
    with original_launch(*args, **kwargs) as app:
        install_measurements()
        yield app
isaaclab.app.launch_simulation=measured_launch
runpy.run_path('/home/hami/workspaces/deformable_isaac_ws/IsaacLab/scripts/reinforcement_learning/play.py',run_name='__main__')
