"""Play only the actor of an rsl_rl checkpoint, deterministically, on any sheet twist task: no runner, no critic.

``play.py`` builds the task's runner and loads actor and critic, so it only loads a checkpoint whose networks match
that task's agent config. The UR PPO checkpoints (a FastSAC actor on ``action_history``, a critic on ``ur_critic``)
do not load on the Franka task, which has no ``ur_critic`` group. This builds the actor alone
(``FastSACActorModel`` with ``UR10eFineTuneRunnerCfg``'s actor settings), whose 49 inputs every twist student task
has, so the same weights play on the Franka, UR10e and UR16e tasks. Every finished episode prints its ending and
its length: steps, simulated seconds (what the real arm would take) and wall-clock seconds (slower, with the viewer).
The task's training configuration is used (``play.py --train_env_cfg``); Hydra overrides follow the arguments.
``--camera <scene entity>`` shows that camera's colour stream in its own window while it plays (the UR16e camera
task, ``Template-Sheet-Rl-Platform-TwistStudent-UR16e-Camera-v0``, has ``d435i``); press q in it to stop.

Example (from the sheet_rl project, after active_isaac):
    PYOPENGL_PLATFORM=glx python scripts/play_actor.py --task Template-Sheet-Rl-Platform-TwistStudent-v0 \\
        --checkpoint logs/rsl_rl/sheet_platform_twist_ur10e_ppo/<run>/model_1924.pt \\
        --viz newton_gl --num_envs 1 --real-time env.actions.arm_action.scale=0.03
"""

import argparse
import sys
import time

import sheet_rl.tasks  # noqa: F401

import warp as wp

wp.config.enable_backward = False

import torch  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402
from isaaclab_rl.entrypoints.common import add_frontend_args, create_isaaclab_env  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import setup_preset_cli  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Template-Sheet-Rl-Platform-TwistStudent-v0")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point", help="Any agent entry the task registers; unused.")
parser.add_argument("--checkpoint", required=True, help="An rsl_rl checkpoint with a FastSAC actor (actor_state_dict).")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--episodes", type=int, default=0, help="Stop after this many finished episodes (0: until Ctrl-C).")
parser.add_argument("--real-time", dest="real_time", action="store_true", help="Run at the control rate.")
parser.add_argument("--camera", default=None, help="A camera sensor in the scene (e.g. d435i) to show in its own window.")
parser.add_argument("--depth", action="store_true", help="Also show --camera's frames as Depth-Anything-3 depth (needs the server).")
parser.add_argument("--da3_address", default="127.0.0.1:5555", help="Where run_da3_sim_server.sh listens.")
parser.add_argument("--depth_scale", choices=("affine", "scale", "inverse", "tag", "none"), default="affine",
                    help="With --depth and --marker: correct each DA3 map from the tags' PnP depths "
                         "(scripts/depth_scale.py): 'affine' scale and offset through both tags (default), 'scale' "
                         "one factor from all tags, 'inverse' the affine fit in inverse depth, 'tag' one factor from "
                         "the lowest-id tag only; the depth window shows the corrected map ('none': raw).")
parser.add_argument("--depth_scale_smoothing", type=float, default=1.0,
                    help="'tag' only: how far its factor moves towards each new frame's measurement (1, the default: "
                         "each frame its own; DA3's scale jumps frame to frame, so smoothing lags and measured worse).")
parser.add_argument("--no_depth_scale_hold", dest="depth_scale_hold", action="store_false",
                    help="Use only the tags visible in a frame (one gives a scale, none leaves the map raw); by default "
                         "the last fit is kept while the tags are hidden.")
parser.add_argument("--point_cloud", action="store_true",
                    help="With --depth: a live Open3D point cloud of the (tag-corrected) DA3 depth, coloured from the "
                         "frame, in tag 0's frame (table coordinates, z up) while the tag is visible; its own process "
                         "(scripts/point_cloud.py); s in it saves a PLY.")
parser.add_argument("--point_cloud_range", type=float, default=3.0,
                    help="Only points within this distance of the camera [m] go into the cloud.")
parser.add_argument("--point_cloud_edge", type=float, default=0.03,
                    help="Drop pixels whose depth jumps more than this [m] to a neighbour (DA3 smears object edges "
                         "into streaks); 0 keeps them.")
parser.add_argument("--point_cloud_dir", default="logs/point_clouds",
                    help="Where s in the cloud window, and the end of the run, save PLY files.")
parser.add_argument("--depth_eval", action="store_true",
                    help="With --depth: also render the camera's true depth and print how far DA3 is from it, raw "
                         "and (with --marker) corrected by every --depth_scale variant, per frame and summarised.")
parser.add_argument("--marker", action="store_true",
                    help="Measure the distance to the table's tag in --camera's frames; its own window and printout.")
parser.add_argument("--tag_family", default="apriltag_36h11", help="The tag's family (fiducial.TAG_FAMILIES).")
parser.add_argument("--tag_size", type=float, default=0.10, help="Edge of the tag's black square [m].")
parser.add_argument("--marker_every", type=int, default=2,
                    help="Measure the tag every N control steps (the detector costs tens of ms of CPU per frame).")
parser.add_argument("--camera_port", type=int, default=5556,
                    help="Serve --camera's frames to other programs (scripts/marker_distance.py) on 127.0.0.1:PORT; 0: off.")
add_launcher_args(parser)
add_frontend_args(parser)
args_cli, rest = setup_preset_cli(parser, agent_library="rsl_rl")
sys.argv = [sys.argv[0]] + rest


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    from importlib.metadata import version

    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
    from sheet_rl.algorithms.fast_sac.rsl_rl_actor import FastSACActorModel
    from sheet_rl.tasks.manager_based.sheet_twist.sheet_twist_ppo_cfg import UR10eFineTuneRunnerCfg

    runner_cfg = handle_deprecated_rsl_rl_cfg(UR10eFineTuneRunnerCfg(), version("rsl-rl-lib"))
    state = torch.load(args_cli.checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"]
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.depth_eval:
        if not (args_cli.depth and args_cli.camera and hasattr(env_cfg.scene, args_cli.camera)):
            raise SystemExit("--depth_eval needs --depth and --camera <a camera of the task>")
        # the renderer's planar depth is the same quantity as a depth map's values: the truth for --depth_eval
        getattr(env_cfg.scene, args_cli.camera).data_types = ["rgb", "distance_to_image_plane"]
    with launch_simulation(env_cfg, args_cli):
        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        inner = env.unwrapped
        obs, _ = env.reset()
        n = inner.num_envs
        obs = TensorDict(obs, batch_size=[n])
        kwargs = runner_cfg.actor.to_dict()
        kwargs.pop("class_name")
        if "learned_log_std" in state:
            kwargs["init_std"] = float(state["learned_log_std"].exp().mean())
        actor = FastSACActorModel(obs, runner_cfg.obs_groups, "actor", inner.action_manager.total_action_dim, **kwargs)
        actor.load_state_dict(state)
        actor = actor.to(inner.device).eval()
        terminations = inner.termination_manager
        if args_cli.camera:
            from camera_window import CameraWindow

            camera = inner.scene[args_cli.camera]
            window = CameraWindow(f"{args_cli.camera}: {camera.cfg.width}x{camera.cfg.height} rgb",
                                  camera.cfg.width, camera.cfg.height)
            frame_server = None
            if args_cli.camera_port:
                from camera_stream import FrameServer

                frame_server = FrameServer(args_cli.camera_port)
                print(f"[play_actor] camera frames served on {frame_server.address}", flush=True)
            marker, marker_window, marker_printed, marker_last = None, None, 0.0, (None, None)
            if args_cli.marker:
                from marker_distance import MarkerDistance

                from sheet_rl.tasks.manager_based.sheet_twist.fiducial import TagCfg

                fx = camera.cfg.width * camera.cfg.spawn.focal_length / camera.cfg.spawn.horizontal_aperture
                scene_tags = [entity.spawn for entity in vars(env_cfg.scene).values()
                              if isinstance(getattr(entity, "spawn", None), TagCfg)]
                tags_in_scene = len(scene_tags)
                marker = MarkerDistance(fx, fx, size=args_cli.tag_size, family=args_cli.tag_family,
                                        expected=tags_in_scene or None,
                                        sizes={tag.tag_id: tag.size for tag in scene_tags})
                marker_window = CameraWindow("Tag distance", camera.cfg.width, camera.cfg.height)
                print(f"[play_actor] tag distance: {args_cli.tag_family}, fx = fy = {fx:.1f} px, {tags_in_scene} tag(s) "
                      f"in the scene: " + ", ".join(f"id {t.tag_id} {t.size * 100:.0f} cm" for t in scene_tags), flush=True)
            depth_window, depth_fits, depth_errors, depth_printed = None, {}, None, 0.0
            if args_cli.depth:
                from da3_depth_window import DepthAnythingWindow, colorize
                from depth_scale import MODES, DepthErrors, TagFit, TagScale

                # the pinhole's focal length in pixels, for DA3's metric scale
                fx = camera.cfg.width * camera.cfg.spawn.focal_length / camera.cfg.spawn.horizontal_aperture
                shown_fit = args_cli.depth_scale if marker is not None and args_cli.depth_scale != "none" else None
                depth_window = DepthAnythingWindow(
                    f"{args_cli.camera}: Depth-Anything-3 metric depth" + (f", {shown_fit} from the tags" if shown_fit else ""),
                    camera.cfg.width, camera.cfg.height, fx, fx, args_cli.da3_address)
                # the correction shown, and in --depth_eval every variant, for the comparison
                depth_errors = DepthErrors() if args_cli.depth_eval else None
                if marker is not None:
                    wanted = ["tag", *MODES] if depth_errors is not None else [shown_fit] if shown_fit else []
                    depth_fits = {name: TagScale(alpha=args_cli.depth_scale_smoothing, hold=args_cli.depth_scale_hold)
                                  if name == "tag" else TagFit(name, hold=args_cli.depth_scale_hold) for name in wanted}
                print(f"[play_actor] DA3 depth from {args_cli.da3_address}, fx = fy = {fx:.1f} px"
                      + (f", shown corrected per frame from the tags ({shown_fit})" if shown_fit else "")
                      + (", compared with the renderer's depth" if depth_errors is not None else ""), flush=True)
                cloud_window = None
                if args_cli.point_cloud:
                    from point_cloud import CloudProcess, cloud_from_depth, to_tag_frame

                    cloud_window = CloudProcess(
                        f"{args_cli.camera}: DA3 point cloud" + (f" ({shown_fit} from the tags)" if shown_fit else ""),
                        args_cli.point_cloud_dir)
                    cloud_cx, cloud_cy = (camera.cfg.width - 1) / 2., (camera.cfg.height - 1) / 2.
                    print(f"[play_actor] point cloud within {args_cli.point_cloud_range:.1f} m of the camera, in tag 0's "
                          f"frame when it is visible; s in its window saves a PLY to {args_cli.point_cloud_dir}", flush=True)
        length = torch.zeros(n, dtype=torch.long)
        began = [time.time()] * n
        finished, outcomes = 0, {}
        print(f"[play_actor] {args_cli.checkpoint} on {args_cli.task}, {n} env(s), deterministic", flush=True)
        try:
            while True:
                start = time.time()
                with torch.inference_mode():
                    obs, _, terminated, truncated, _ = env.step(actor(obs))
                obs = TensorDict(obs, batch_size=[n])
                length += 1
                if args_cli.camera:
                    # env 0's frame; reading .data renders it (sensors update lazily)
                    frame = camera.data.output["rgb"][0].cpu().numpy()
                    if not window.show(frame):
                        break
                    if frame_server is not None:
                        frame_server.publish(frame)
                    if marker is not None:
                        if marker.frames == 0 or (int(length[0]) % max(1, args_cli.marker_every)) == 0:
                            result, shown = marker.measure(frame)
                            marker_last = (result, shown)
                        result, shown = marker_last
                        if not marker_window.show(shown):
                            break
                        if time.time() - marker_printed > 1.0:
                            print("[play_actor] " + MarkerDistance.describe(f"tag frame {marker.frames}", result), flush=True)
                            marker_printed = time.time()
                    if depth_window is not None:
                        truth = None
                        if depth_errors is not None:
                            truth = camera.data.output["distance_to_image_plane"][0].cpu().numpy()
                            truth = truth[..., 0] if truth.ndim == 3 else truth
                        depth_window.submit(frame, {"tag": marker_last[0] if marker is not None else None,
                                                    "truth": truth, "step": int(length[0]),
                                                    "frame": frame[..., :3].copy() if args_cli.point_cloud else None})
                        image = None
                        try:
                            result = depth_window.take()
                        except (OSError, ValueError) as exc:
                            print(f"[da3] depth server lost: {exc}", flush=True)
                            break
                        if result is not None:
                            depth, meta = result
                            variants, fit_text = {}, []
                            for name, fit in depth_fits.items():
                                if name == "tag":
                                    factor = fit.update(depth, meta["tag"])
                                    variants[name] = depth * factor if factor is not None else None
                                    fit_text.append("tag: no tag yet" if factor is None else f"tag: x{factor:.3f}")
                                else:
                                    variants[name] = fit.update(depth, meta["tag"])
                                    fit_text.append(fit.describe())
                            shown_depth = variants.get(shown_fit)
                            image = colorize(depth if shown_depth is None else shown_depth, depth_window.smoothed)
                            if cloud_window is not None:
                                points, colors = cloud_from_depth(depth if shown_depth is None else shown_depth,
                                                                  meta["frame"], fx, fx, cloud_cx, cloud_cy,
                                                                  max_range=args_cli.point_cloud_range,
                                                                  edge_m=args_cli.point_cloud_edge)
                                tag0 = next((t for t in ([] if meta["tag"] is None else meta["tag"]["all"]) if t["id"] == 0), None)
                                cloud_frame = "camera"
                                if tag0 is not None:
                                    points, cloud_frame = to_tag_frame(points, tag0["rvec"], tag0["centre"]), "tag"
                                if not cloud_window.show(points, colors, cloud_frame):
                                    break
                            if depth_errors is not None:
                                stats = depth_errors.add(depth, variants, meta["truth"], meta["tag"])
                                if time.time() - depth_printed > 1.0:
                                    lines = [f"[da3] step {meta['step']}: " + ("; ".join(fit_text) or "no correction")]
                                    lines += [f"      {DepthErrors.line(f'{name:8s}', stats[name])}" for name in stats]
                                    print("\n".join(lines), flush=True)
                                    depth_printed = time.time()
                        if not depth_window.show(image):
                            break
                for e in (terminated | truncated).nonzero().flatten().tolist():
                    ended = [name for name in terminations.active_terms if bool(terminations.get_term(name)[e])]
                    finished += 1
                    for name in ended:
                        outcomes[name] = outcomes.get(name, 0) + 1
                    steps, wall = int(length[e]), time.time() - began[e]
                    print(f"[play_actor] episode {finished} (env {e}): ended by {ended} at step {steps} "
                          f"({steps * inner.step_dt:.2f} s simulated, {wall:.1f} s wall-clock)", flush=True)
                    length[e], began[e] = 0, time.time()
                if args_cli.episodes and finished >= args_cli.episodes:
                    break
                if args_cli.real_time:
                    time.sleep(max(0.0, inner.step_dt - (time.time() - start)))
        except KeyboardInterrupt:
            pass
        print(f"[play_actor] {finished} episodes: {outcomes}", flush=True)
        if args_cli.camera:
            if marker is not None:
                mean = sum(marker.distances) / len(marker.distances) if marker.distances else float("nan")
                print(f"[play_actor] tag: found in {marker.found} of {marker.frames} frames, mean distance {mean:.3f} m; "
                      f"frames per tag: {dict(sorted(marker.found_by_id.items()))}", flush=True)
                marker_window.close()
            if frame_server is not None:
                frame_server.close()
            if depth_window is not None:
                print(f"[play_actor] DA3: {depth_window.received} depth frames for {depth_window.sent} colour frames, "
                      f"last inference {depth_window.client.last_ms:.0f} ms", flush=True)
                for name, fit in depth_fits.items():
                    print(f"[play_actor] DA3 correction '{name}': {fit.measured} frames with a tag"
                          + ("" if name == "tag" else ", frames by tags visible " + str(dict(sorted(fit.frames_with.items())))),
                          flush=True)
                if depth_errors is not None:
                    print("[play_actor] DA3 vs the renderer's depth:\n  " + "\n  ".join(depth_errors.summary()), flush=True)
                if cloud_window is not None:
                    print(f"[play_actor] point cloud: {cloud_window.sent} clouds sent to the window process", flush=True)
                    cloud_window.close()  # the window process saves the last cloud and exits
                depth_window.close()
            window.close()
        env.close()


if __name__ == "__main__":
    main()
