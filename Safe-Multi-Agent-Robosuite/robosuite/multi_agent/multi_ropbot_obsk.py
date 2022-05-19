import itertools
import numpy as np
from copy import deepcopy

class Node():
    def __init__(self, label, qpos_ids, qvel_ids, act_ids, body_fn=None, bodies=None, extra_obs=None, tendons=None):
        self.label = label
        self.qpos_ids = qpos_ids
        self.qvel_ids = qvel_ids
        self.act_ids = act_ids
        self.bodies = bodies
        self.extra_obs = {} if extra_obs is None else extra_obs
        self.body_fn = body_fn
        self.tendons = tendons
        pass

    def __str__(self):
        return self.label

    def __repr__(self):
        return self.label


class HyperEdge():
    def __init__(self, *edges):
        self.edges = set(edges)

    def __contains__(self, item):
        return item in self.edges

    def __str__(self):
        return "HyperEdge({})".format(self.edges)

    def __repr__(self):
        return "HyperEdge({})".format(self.edges)


def get_joints_at_kdist(agent_id, agent_partitions, hyperedges, k=0, kagents=False,):
    """ Identify all joints at distance <= k from agent agent_id

    :param agent_id: id of agent to be considered
    :param agent_partitions: list of joint tuples in order of agentids
    :param edges: list of tuples (joint1, joint2)
    :param k: kth degree
    :param kagents: True (observe all joints of an agent if a single one is) or False (individual joint granularity)
    :return:
        dict with k as key, and list of joints at that distance
    """
    assert not kagents, "kagents not implemented!"

    agent_joints = agent_partitions[agent_id]

    def _adjacent(lst, kagents=False):
        # return all sets adjacent to any element in lst
        ret = set([])
        for l in lst:
            ret = ret.union(set(itertools.chain(*[e.edges.difference({l}) for e in hyperedges if l in e])))
        return ret

    seen = set([])
    new = set([])
    k_dict = {}
    for _k in range(k+1):
        if not _k:
            new = set(agent_joints)
        else:
            print(hyperedges)
            new = _adjacent(new) - seen
        seen = seen.union(new)
        k_dict[_k] = sorted(list(new), key=lambda x:x.label)
    return k_dict


def build_obs(env, k_dict, k_categories, global_dict, global_categories, vec_len=None):
    """Given a k_dict from get_joints_at_kdist, extract observation vector.

    :param k_dict: k_dict
    :param qpos: qpos numpy array
    :param qvel: qvel numpy array
    :param vec_len: if None no padding, else zero-pad to vec_len
    :return:
    observation vector
    """

    # TODO: This needs to be fixed, it was designed for half-cheetah only!
    #if add_global_pos:
    #    obs_qpos_lst.append(global_qpos)
    #    obs_qvel_lst.append(global_qvel)


    body_set_dict = {}
    obs_lst = []
    # Add parts attributes
    for k in sorted(list(k_dict.keys())):
        cats = k_categories[k]
        for _t in k_dict[k]:
            for c in cats:
                if c in _t.extra_obs:
                    items = _t.extra_obs[c](env).tolist()
                    obs_lst.extend(items if isinstance(items, list) else [items])
                else:
                    if c in ["qvel","qpos"]: # this is a "joint position/velocity" item
                        items = getattr(env.sim.data, c)[getattr(_t, "{}_ids".format(c))]
                        obs_lst.extend(items if isinstance(items, list) else [items])
                    elif c in ["qfrc_actuator"]: # this is a "vel position" item
                        items = getattr(env.sim.data, c)[getattr(_t, "{}_ids".format("qvel"))]
                        obs_lst.extend(items if isinstance(items, list) else [items])
                    elif c in ["cvel", "cinert", "cfrc_ext"]:  # this is a "body position" item
                        if _t.bodies is not None:
                            for b in _t.bodies:
                                if c not in body_set_dict:
                                    body_set_dict[c] = set()
                                if b not in body_set_dict[c]:
                                    items = getattr(env.sim.data, c)[b].tolist()
                                    items = getattr(_t, "body_fn", lambda _id,x:x)(b, items)
                                    obs_lst.extend(items if isinstance(items, list) else [items])
                                    body_set_dict[c].add(b)

    # Add global attributes
    body_set_dict = {}
    for c in global_categories:
        if c in ["qvel", "qpos"]:  # this is a "joint position" item
            for j in global_dict.get("joints", []):
                items = getattr(env.sim.data, c)[getattr(j, "{}_ids".format(c))]
                obs_lst.extend(items if isinstance(items, list) else [items])
        else:
            for b in global_dict.get("bodies", []):
                if c not in body_set_dict:
                    body_set_dict[c] = set()
                if b not in body_set_dict[c]:
                    obs_lst.extend(getattr(env.sim.data, c)[b].tolist())
                    body_set_dict[c].add(b)

    if vec_len is not None:
        pad = np.array((vec_len - len(obs_lst))*[0])
        if len(pad):
            return np.concatenate([np.array(obs_lst), pad])
    return np.array(obs_lst)


def build_actions(agent_partitions, k_dict):
    # Composes agent actions output from networks
    # into coherent joint action vector to be sent to the env.
    pass

def get_parts_and_edges(label, partitioning):
    if label in ["half_cheetah", "HalfCheetah-v2"]:

        # define Mujoco graph
        bthigh = Node("bthigh", -6, -6, 0)
        bshin = Node("bshin", -5, -5, 1)
        bfoot = Node("bfoot", -4, -4, 2)
        fthigh = Node("fthigh", -3, -3, 3)
        fshin = Node("fshin", -2, -2, 4)
        ffoot = Node("ffoot", -1, -1, 5)

        edges = [HyperEdge(bfoot, bshin),
                 HyperEdge(bshin, bthigh),
                 HyperEdge(bthigh, fthigh),
                 HyperEdge(fthigh, fshin),
                 HyperEdge(fshin, ffoot)]

        root_x = Node("root_x", 0, 0, -1,
                      extra_obs={"qpos": lambda env: np.array([])})
        root_z = Node("root_z", 1, 1, -1)
        root_y = Node("root_y", 2, 2, -1)
        globals = {"joints":[root_x, root_y, root_z]}

        if partitioning == "2x3":
            parts = [(bfoot, bshin, bthigh),
                     (ffoot, fshin, fthigh)]
        elif partitioning == "6x1":
            parts = [(bfoot,), (bshin,), (bthigh,), (ffoot,), (fshin,), (fthigh,)]
        elif partitioning == "3x2":
            parts = [(bfoot, bshin,), (bthigh, ffoot,), (fshin, fthigh,)]
        else:
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        return parts, edges, globals

    elif label in ["Door", "Lift", "NutAssembly", "NutAssemblyRound", "NutAssemblySingle",
                   "NutAssemblySquare", "PickPlace", "PickPlaceBread", "PickPlaceCan", "PickPlaceCereal",
                   "PickPlaceMilk", "PickPlaceSingle", "Stack"]:
        # define Mujoco graph
        torq_j1 = Node("torq_j1", -6, -6, 0)
        torq_j2 = Node("torq_j2", -5, -5, 1)
        torq_j3 = Node("torq_j3", -4, -4, 2)
        torq_j4 = Node("torq_j4", -3, -3, 3)
        torq_j5 = Node("torq_j5", -2, -2, 4)
        torq_j6 = Node("torq_j6", -1, -1, 5)
        torq_j7 = Node("torq_j7", 0, 0, 6)
        torq_j8 = Node("torq_j8", 1, 1, 7)

        edges = [HyperEdge(torq_j1, torq_j2),
                 HyperEdge(torq_j2, torq_j3),
                 HyperEdge(torq_j3, torq_j4),
                 HyperEdge(torq_j4, torq_j5),
                 HyperEdge(torq_j5, torq_j6),
                 HyperEdge(torq_j6, torq_j7),
                 HyperEdge(torq_j7, torq_j8)]

        root_x = Node("root_x", 3, 3, -1,
                      extra_obs={"qpos": lambda env: np.array([])})
        root_z = Node("root_z", 2, 2, -1)
        root_y = Node("root_y", 1, 1, -1)
        globals = {"joints": [root_x, root_y, root_z]}

        if partitioning == "2x4":
            parts = [(torq_j1, torq_j2, torq_j3, torq_j4),
                     (torq_j5, torq_j6, torq_j7, torq_j8)]
        elif partitioning == "4x2":
            parts = [(torq_j1, torq_j2,), (torq_j3, torq_j4,),
                     (torq_j5, torq_j6,), (torq_j7, torq_j8,)]
        elif partitioning == "8x1":
            parts = [(torq_j1,), (torq_j2,), (torq_j3,), (torq_j4,), (torq_j5,), (torq_j6,), (torq_j7,), (torq_j8,)]
        else:
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        return parts, edges, globals
    elif label in ["Wipe", "Lift_Osc_Pose"]:
        # define Mujoco graph
        torq_j1 = Node("torq_j1", -6, -6, 0)
        torq_j2 = Node("torq_j2", -5, -5, 1)
        torq_j3 = Node("torq_j3", -4, -4, 2)
        torq_j4 = Node("torq_j4", -3, -3, 3)
        torq_j5 = Node("torq_j5", -2, -2, 4)
        torq_j6 = Node("torq_j6", -1, -1, 5)
        torq_j7 = Node("torq_j7", 0, 0, 6)

        edges = [HyperEdge(torq_j1, torq_j2),
                 HyperEdge(torq_j2, torq_j3),
                 HyperEdge(torq_j3, torq_j4),
                 HyperEdge(torq_j4, torq_j5),
                 HyperEdge(torq_j5, torq_j6),
                 HyperEdge(torq_j6, torq_j7),
                 ]

        root_x = Node("root_x", 3, 3, -1,
                      extra_obs={"qpos": lambda env: np.array([])})
        root_z = Node("root_z", 2, 2, -1)
        root_y = Node("root_y", 1, 1, -1)
        globals = {"joints": [root_x, root_y, root_z]}

        if partitioning == "7x1":
            parts = [(torq_j1, torq_j2, torq_j3, torq_j4,torq_j5, torq_j6, torq_j7,)]

        else:
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        return parts, edges, globals

    elif label in ["TwoArmPegInHole","TwoArmHandover_Osc_Pose", "TwoArmLift_Osc_Pose", "TwoArmPegInHole_Osc_Pose"]:
        # define Mujoco graph
        torq_j1 = Node("torq_j1", -6, -6, 0)
        torq_j2 = Node("torq_j2", -5, -5, 1)
        torq_j3 = Node("torq_j3", -4, -4, 2)
        torq_j4 = Node("torq_j4", -3, -3, 3)
        torq_j5 = Node("torq_j5", -2, -2, 4)
        torq_j6 = Node("torq_j6", -1, -1, 5)
        torq_j7 = Node("torq_j7", 0, 0, 6)
        torq_j8 = Node("torq_j8", 1, 1, 7)
        torq_j9 = Node("torq_j9", 2, 2, 8)
        torq_j10 = Node("torq_j10", 3, 3, 9)
        torq_j11 = Node("torq_j11", 4, 4, 10)
        torq_j12 = Node("torq_j12", 5, 5, 11)
        torq_j13 = Node("torq_j13", 6, 6, 12)
        torq_j14 = Node("torq_j14", 7, 7, 13)


        edges = [HyperEdge(torq_j1, torq_j2),
                 HyperEdge(torq_j2, torq_j3),
                 HyperEdge(torq_j3, torq_j4),
                 HyperEdge(torq_j4, torq_j5),
                 HyperEdge(torq_j5, torq_j6),
                 HyperEdge(torq_j6, torq_j7),
                 HyperEdge(torq_j7, torq_j8),
                 HyperEdge(torq_j8, torq_j9),
                 HyperEdge(torq_j9, torq_j10),
                 HyperEdge(torq_j10, torq_j11),
                 HyperEdge(torq_j11, torq_j12),
                 HyperEdge(torq_j12, torq_j13),
                 HyperEdge(torq_j13, torq_j14)]

        root_x = Node("root_x", 3, 3, -1,
                      extra_obs={"qpos": lambda env: np.array([])})
        root_z = Node("root_z", 2, 2, -1)
        root_y = Node("root_y", 1, 1, -1)
        globals = {"joints": [root_x, root_y, root_z]}

        if partitioning == "2x7":
            parts = [(torq_j1, torq_j2, torq_j3, torq_j4, torq_j5, torq_j6, torq_j7),
                     (torq_j8, torq_j9, torq_j10, torq_j11, torq_j12, torq_j13, torq_j14)]

        elif partitioning == "7x2":
            parts = [(torq_j1, torq_j2),
                     (torq_j3, torq_j4),
                     (torq_j5, torq_j6),
                     (torq_j7, torq_j8),
                     (torq_j9, torq_j10),
                     (torq_j11, torq_j12),
                     (torq_j13, torq_j14)]
        elif partitioning == "14x1":
            parts = [(torq_j1),
                     (torq_j2),
                     (torq_j3),
                     (torq_j4),
                     (torq_j5),
                     (torq_j6),
                     (torq_j7),
                     (torq_j8),
                     (torq_j9),
                     (torq_j10),
                     (torq_j11),
                     (torq_j12),
                     (torq_j13),
                     (torq_j14)]
        else:
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        return parts, edges, globals

    elif label in ["TwoArmHandover", "TwoArmLift"]:
        # define Mujoco graph
        torq_j1 = Node("torq_j1", -6, -6, 0)
        torq_j2 = Node("torq_j2", -5, -5, 1)
        torq_j3 = Node("torq_j3", -4, -4, 2)
        torq_j4 = Node("torq_j4", -3, -3, 3)
        torq_j5 = Node("torq_j5", -2, -2, 4)
        torq_j6 = Node("torq_j6", -1, -1, 5)
        torq_j7 = Node("torq_j7", 0, 0, 6)
        torq_j8 = Node("torq_j8", 1, 1, 7)
        torq_j9 = Node("torq_j9", 2, 2, 8)
        torq_j10 = Node("torq_j10", 3, 3, 9)
        torq_j11 = Node("torq_j11", 4, 4, 10)
        torq_j12 = Node("torq_j12", 5, 5, 11)
        torq_j13 = Node("torq_j13", 6, 6, 12)
        torq_j14 = Node("torq_j14", 7, 7, 13)
        torq_j15 = Node("torq_j15", 8, 8, 14)
        torq_j16 = Node("torq_j16", 9, 9, 15)

        edges = [HyperEdge(torq_j1, torq_j2),
                 HyperEdge(torq_j2, torq_j3),
                 HyperEdge(torq_j3, torq_j4),
                 HyperEdge(torq_j4, torq_j5),
                 HyperEdge(torq_j5, torq_j6),
                 HyperEdge(torq_j6, torq_j7),
                 HyperEdge(torq_j7, torq_j8),
                 HyperEdge(torq_j8, torq_j9),
                 HyperEdge(torq_j9, torq_j10),
                 HyperEdge(torq_j10, torq_j11),
                 HyperEdge(torq_j11, torq_j12),
                 HyperEdge(torq_j12, torq_j13),
                 HyperEdge(torq_j13, torq_j14),
                 HyperEdge(torq_j14, torq_j15),
                 HyperEdge(torq_j15, torq_j16)]

        root_x = Node("root_x", 3, 3, -1,
                      extra_obs={"qpos": lambda env: np.array([])})
        root_z = Node("root_z", 2, 2, -1)
        root_y = Node("root_y", 1, 1, -1)
        globals = {"joints": [root_x, root_y, root_z]}

        if partitioning == "2x8":
            parts = [(torq_j1, torq_j2, torq_j3, torq_j4, torq_j5, torq_j6, torq_j7, torq_j8),
                     (torq_j9, torq_j10, torq_j11, torq_j12, torq_j13, torq_j14, torq_j15, torq_j16)]
        elif partitioning == "4x4":
            parts = [(torq_j1, torq_j2, torq_j3, torq_j4),
                     (torq_j5, torq_j6, torq_j7, torq_j8),
                     (torq_j9, torq_j10, torq_j11, torq_j12),
                     (torq_j13, torq_j14, torq_j15, torq_j16)]
        elif partitioning == "8x2":
            parts = [(torq_j1, torq_j2),
                     (torq_j3, torq_j4),
                     (torq_j5, torq_j6),
                     (torq_j7, torq_j8),
                     (torq_j9, torq_j10),
                     (torq_j11, torq_j12),
                     (torq_j13, torq_j14),
                     (torq_j15, torq_j16)]
        elif partitioning == "16x1":
            parts = [(torq_j1),
                     (torq_j2),
                     (torq_j3),
                     (torq_j4),
                     (torq_j5),
                     (torq_j6),
                     (torq_j7),
                     (torq_j8),
                     (torq_j9),
                     (torq_j10),
                     (torq_j11),
                     (torq_j12),
                     (torq_j13),
                     (torq_j14),
                     (torq_j15),
                     (torq_j16)]
        else:
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        return parts, edges, globals


    elif label in ["Ant-v2"]:

        # define Mujoco graph
        torso = 1
        front_left_leg = 2
        aux_1 = 3
        ankle_1 = 4
        front_right_leg = 5
        aux_2 = 6
        ankle_2 = 7
        back_leg = 8
        aux_3 = 9
        ankle_3 = 10
        right_back_leg = 11
        aux_4 = 12
        ankle_4 = 13

        hip1 = Node("hip1", -8, -8, 2, bodies=[torso, front_left_leg], body_fn=lambda _id, x:np.clip(x, -1, 1).tolist()) #
        ankle1 = Node("ankle1", -7, -7, 3, bodies=[front_left_leg, aux_1, ankle_1], body_fn=lambda _id, x:np.clip(x, -1, 1).tolist())#,
        hip2 = Node("hip2", -6, -6, 4, bodies=[torso, front_right_leg], body_fn=lambda _id, x:np.clip(x, -1, 1).tolist())#,
        ankle2 = Node("ankle2", -5, -5, 5, bodies=[front_right_leg, aux_2, ankle_2], body_fn=lambda _id, x:np.clip(x, -1, 1).tolist())#,
        hip3 = Node("hip3", -4, -4, 6, bodies=[torso, back_leg], body_fn=lambda _id, x:np.clip(x, -1, 1).tolist())#,
        ankle3 = Node("ankle3", -3, -3, 7, bodies=[back_leg, aux_3, ankle_3], body_fn=lambda _id, x:np.clip(x, -1, 1).tolist())#,
        hip4 = Node("hip4", -2, -2, 0, bodies=[torso, right_back_leg], body_fn=lambda _id, x:np.clip(x, -1, 1).tolist())#,
        ankle4 = Node("ankle4", -1, -1, 1, bodies=[right_back_leg, aux_4, ankle_4], body_fn=lambda _id, x:np.clip(x, -1, 1).tolist())#,

        edges = [HyperEdge(ankle4, hip4),
                 HyperEdge(ankle1, hip1),
                 HyperEdge(ankle2, hip2),
                 HyperEdge(ankle3, hip3),
                 HyperEdge(hip4, hip1, hip2, hip3),
                 ]

        free_joint = Node("free", 0, 0, -1, extra_obs={"qpos": lambda env: env.sim.data.qpos[:7],
                                                       "qvel": lambda env: env.sim.data.qvel[:6],
                                                       "cfrc_ext": lambda env: np.clip(env.sim.data.cfrc_ext[0:1], -1, 1)})
        globals = {"joints": [free_joint]}

        if partitioning == "2x4": # neighbouring legs together
            parts = [(hip1, ankle1, hip2, ankle2),
                     (hip3, ankle3, hip4, ankle4)]
        elif partitioning == "2x4d": # diagonal legs together
            parts = [(hip1, ankle1, hip3, ankle3),
                     (hip2, ankle2, hip4, ankle4)]
        elif partitioning == "4x2":
            parts = [(hip1, ankle1),
                     (hip2, ankle2),
                     (hip3, ankle3),
                     (hip4, ankle4)]
        elif partitioning == "8x1":
            parts = [(hip1,), (ankle1,),
                     (hip2,), (ankle2,),
                     (hip3,), (ankle3,),
                     (hip4,), (ankle4,)]
        else:
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        return parts, edges, globals

