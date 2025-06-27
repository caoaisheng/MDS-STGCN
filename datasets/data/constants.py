import numpy as np
# 20个人，16个作为训练，4个作为测试
cross_subject_training = tuple(range(14339))
cross_subject_test = tuple(range(14339, 14839))
# 四个视角
cross_view_training = (0, 1, 2)
cross_view_test = (3,)

skeleton_rgb_max_sequence_length = 128
inertial_max_sequence_length = 128


# (num bodies, sequence length, num nodes, num channels)
skeleton_shape = (skeleton_rgb_max_sequence_length, 13, 2, 1)
rgb_shape = (skeleton_rgb_max_sequence_length, 1080, 1920, 3)


skeleton_center_joint = 0

actions = [
    "0",  # 0
    "1"
]

skeleton_joints = [
       "MidHip",  # 0
       "RHip",
       "RKnee",
       "RAnkle",
       "LHip",
       "LKnee" ,  # 5
       "LAnkle",
       "LBigToe",
       "LSmallToe",
       "LHeel",
       "RBigToe",  # 10
       "RSmallToe",
       "RHeel",   # 13
]

# FOR OPENPOSE COCO BODY
skeleton_edges = np.array([
    (0, 1),
    (2, 1),
    (3, 2),
    (10, 3),
    (12, 3),
    (0, 4),
    (5, 4),
    (6, 5),
    (7, 6),
    (9, 6),
    (8, 7),
    (11, 10)
])
center_joint = 1

action_to_index_map = {
    k: i for i, k in enumerate(actions)
}

two_people_actions = ["talking", "transferring_object"]

num_joints = len(skeleton_joints)
num_classes = len(actions)
num_subjects = 14840
num_views = 0
