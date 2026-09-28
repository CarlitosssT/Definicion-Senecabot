# Parameters for senecabot.xml
R_HIP_LEN = 0.166
R_KNEE_LEN = 0.231
R_ANKLE_LEN = 0.148

F_HIP_LEN =  0.121
F_KNEE_LEN = 0.111
F_ANKLE_LEN = (0.0802+0.0288+0.0188+0.0256)
h_difference = 0.08                 # Distance between legs from each other (Z axis)

parameters={
    "torso_w":0.12,                 # Torso Width
    "torso_h":0.06,                 # Torso Height
    "torso_l":0.21,                  # Torso Lenght
    "f_distance":0.2,               # Distance of front legs from center of torso (Y Axis)
    "b_distance":0.2,               # Distance of back legs from center of torso (Y Axis)
    "w_distance":0.1,               # Distance of all legs from center of torso (X Axis)
    "r_hip_len":R_HIP_LEN,
    "r_knee_len":R_KNEE_LEN,
    "r_ankle_len":R_ANKLE_LEN,
    "f_hip_len":R_HIP_LEN,
    "f_knee_len":F_KNEE_LEN,
    "f_ankle_len":F_ANKLE_LEN,
    "h_difference_f":h_difference/2,             
    "h_difference_b":h_difference/2,
    "z_min":-0.1,
    "z_max":0.6
}