import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

CURRENT_DISTANCE_FROM_OBTACLE = 8
CURRENT_ANGLE_WITH_OBTACLE = 20

x_distance_from_obtacle = np.arange(0, 10.5, 0.5)
print (x_distance_from_obtacle)
x_angle_with_obtacle = np.arange(0, 91, 1)
print (x_angle_with_obtacle)
x_speed  = np.arange(0, 5.2, 0.2)
print (x_speed)
x_steering_turn = np.arange(0, 91, 1)
print (x_steering_turn)

# generate membership function
# i will use triangle membership function

distance_from_obtacle_near = fuzz.trimf(x_distance_from_obtacle, [0, 0, 5])
distance_from_obtacle_far = fuzz.trimf(x_distance_from_obtacle, [0, 5, 10])
distance_from_obtacle_very_far = fuzz.trimf(x_distance_from_obtacle, [5, 10, 10])

angle_with_obtacle_small = fuzz.trimf(x_angle_with_obtacle, [0, 0, 45])
angle_with_obtacle_medium = fuzz.trimf(x_angle_with_obtacle, [0, 45, 90])
angle_with_obtacle_large = fuzz.trimf(x_angle_with_obtacle, [45, 90, 90])

speed_slow = fuzz.trimf(x_speed, [0, 0, 2.5])
speed_medium = fuzz.trimf(x_speed, [0, 2.4, 5])
speed_fast = fuzz.trimf(x_speed, [2.4, 5, 5])
speed_maximum = fuzz.trimf(x_speed, [5, 5, 5])

steering_turn_mild = fuzz.trimf(x_steering_turn, [0, 0, 45])
steering_turn_sharp = fuzz.trimf(x_steering_turn, [00, 45, 90])
steering_turn_very_sharp = fuzz.trimf(x_steering_turn, [45, 90, 90])

#visualize membership function
# Visualize these universes and membership functions
fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(8, 9))

ax0.plot(x_distance_from_obtacle, distance_from_obtacle_near, 'b', linewidth=1.5, label='Near')
ax0.plot(x_distance_from_obtacle, distance_from_obtacle_far, 'g', linewidth=1.5, label='Far')
ax0.plot(x_distance_from_obtacle, distance_from_obtacle_very_far, 'r', linewidth=1.5, label='Very far')
ax0.set_title('Distance from obtacle')
ax0.legend()

ax1.plot(x_angle_with_obtacle, angle_with_obtacle_small, 'b', linewidth=1.5, label='Small')
ax1.plot(x_angle_with_obtacle, angle_with_obtacle_medium, 'g', linewidth=1.5, label='Medium')
ax1.plot(x_angle_with_obtacle, angle_with_obtacle_large, 'r', linewidth=1.5, label='Large')
ax1.set_title('Angle with obtacle')
ax1.legend()

ax2.plot(x_speed, speed_slow, 'b', linewidth=1.5, label='Slow')
ax2.plot(x_speed, speed_medium, 'g', linewidth=1.5, label='Medium')
ax2.plot(x_speed, speed_fast, 'r', linewidth=1.5, label='Fast')
ax2.plot(x_speed, speed_maximum, 'black', linewidth=1.5, label='Maximum')
ax2.set_title('Speed')
ax2.legend()

ax3.plot(x_steering_turn, steering_turn_mild, 'b', linewidth=1.5, label='Mild')
ax3.plot(x_steering_turn, steering_turn_sharp, 'g', linewidth=1.5, label='Sharp')
ax3.plot(x_steering_turn, steering_turn_very_sharp, 'r', linewidth=1.5, label='Very sharp')
ax3.set_title('Steering turn')
ax3.legend()

for ax in (ax0, ax1, ax2,ax3):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()



#implication construction

distance_from_obtacle_near_level = fuzz.interp_membership(x_distance_from_obtacle, distance_from_obtacle_near, CURRENT_DISTANCE_FROM_OBTACLE)
print("distance_from_obtacle_near_level: " + str(distance_from_obtacle_near_level))
distance_from_obtacle_far_level  = fuzz.interp_membership(x_distance_from_obtacle, distance_from_obtacle_far, CURRENT_DISTANCE_FROM_OBTACLE)
print("distance_from_obtacle_far_level: " + str(distance_from_obtacle_far_level))
distance_from_obtacle_very_far_level = fuzz.interp_membership(x_distance_from_obtacle, distance_from_obtacle_very_far, CURRENT_DISTANCE_FROM_OBTACLE)
print("distance_from_obtacle_very_far_level: " + str(distance_from_obtacle_very_far_level))

angle_with_obtacle_small_level = fuzz.interp_membership(x_angle_with_obtacle, angle_with_obtacle_small, CURRENT_ANGLE_WITH_OBTACLE)
print("angle_with_obtacle_small_level: " + str(angle_with_obtacle_small_level))
angle_with_obtacle_medium_level  = fuzz.interp_membership(x_angle_with_obtacle, angle_with_obtacle_medium, CURRENT_ANGLE_WITH_OBTACLE)
print("angle_with_obtacle_medium_level: " + str(angle_with_obtacle_medium_level))
angle_with_obtacle_large_level = fuzz.interp_membership(x_angle_with_obtacle, angle_with_obtacle_large, CURRENT_ANGLE_WITH_OBTACLE)
print("angle_with_obtacle_large_level: " + str(angle_with_obtacle_large_level))

# if D is N, and A is S, then ST is VST
active_rule1 = np.fmin(distance_from_obtacle_near_level , angle_with_obtacle_small_level)
steering_turn_very_sharp_activation = np.fmin(active_rule1, steering_turn_very_sharp)

# else if D is N, and A is M then ST is SST
active_rule2 = np.fmin(distance_from_obtacle_near_level, angle_with_obtacle_medium_level)
steering_turn_sharp_activation_1 = np.fmin(active_rule2, steering_turn_sharp)

# else if D is N, and A is L then ST is MT
active_rule3 = np.fmin(distance_from_obtacle_near_level, angle_with_obtacle_large_level)
steering_turn_mild_activation_1 = np.fmin(active_rule3, steering_turn_mild)

# else if D is F, and A is S, then ST is SST
active_rule4 = np.fmin(distance_from_obtacle_far_level, angle_with_obtacle_small_level)
steering_turn_sharp_activation_2 = np.fmin(active_rule4, steering_turn_sharp)

# else if D is F, and A is M then ST is SST
active_rule5 = np.fmin(distance_from_obtacle_far_level, angle_with_obtacle_medium_level)
steering_turn_sharp_activation_3 = np.fmin(active_rule5, steering_turn_sharp)

# else if D is F, and A is L then ST is MT
active_rule6 = np.fmin(distance_from_obtacle_far_level, angle_with_obtacle_large_level)
steering_turn_mild_activation_2 = np.fmin(active_rule6, steering_turn_mild)

# else if D is VF, and A is S, then ST is MT
active_rule7 = np.fmin(distance_from_obtacle_very_far_level, angle_with_obtacle_small_level)
steering_turn_mild_activation_3 = np.fmin(active_rule7, steering_turn_mild)

# else if D is VF, and A is M then ST is MT
active_rule8 = np.fmin(distance_from_obtacle_very_far_level, angle_with_obtacle_medium_level)
steering_turn_mild_activation_4 = np.fmin(active_rule8, steering_turn_mild)

# else if D is VF, and A is L then ST is MT
active_rule9 = np.fmin(distance_from_obtacle_very_far_level, angle_with_obtacle_large_level)
steering_turn_mild_activation_5 = np.fmin(active_rule9, steering_turn_mild)

# aggregate for steering mild
steering_mild_activation = np.fmax(np.fmax(np.fmax(np.fmax(steering_turn_mild_activation_1, steering_turn_mild_activation_2), steering_turn_mild_activation_3),steering_turn_mild_activation_4),steering_turn_mild_activation_5)
# aggregate for steering sharp
steering_turn_sharp_activation = np.fmax(np.fmax(steering_turn_sharp_activation_1, steering_turn_sharp_activation_2), steering_turn_sharp_activation_3)
steering0 = np.zeros_like(x_steering_turn)
# Visualize steering turn
fig1, ax4 = plt.subplots(figsize=(8, 3))

ax4.fill_between(x_steering_turn, steering0, steering_mild_activation, facecolor='b', alpha=0.7)
ax4.plot(x_steering_turn, steering_turn_mild, 'b', linewidth=0.5, linestyle='--', )
ax4.fill_between(x_steering_turn, steering0, steering_turn_sharp_activation, facecolor='g', alpha=0.7)
ax4.plot(x_steering_turn, steering_turn_sharp, 'g', linewidth=0.5, linestyle='--')
ax4.fill_between(x_steering_turn, steering0, steering_turn_very_sharp_activation, facecolor='r', alpha=0.7)
ax4.plot(x_steering_turn, steering_turn_very_sharp, 'r', linewidth=0.5, linestyle='--')
ax4.set_title('Output membership activity for steering turn')
# Turn off top/right axes
for ax in (ax4,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()

# aggregate these three implication
#steering_activation = np.fmax(np.fmax(np.fmax(np.fmax(np.fmax(np.fmax(np.fmax(np.fmax(steering_turn_very_sharp_activation, steering_turn_sharp_activation_1), steering_turn_mild_activation_1), steering_turn_sharp_activation_2),
                                #steering_turn_sharp_activation_3), steering_turn_mild_activation_2), steering_turn_mild_activation_3), steering_turn_mild_activation_4), steering_turn_mild_activation_5)
steering_activation = np.fmax(np.fmax(steering_mild_activation, steering_turn_sharp_activation), steering_turn_very_sharp_activation)
steering_trun = fuzz.defuzz(x_steering_turn, steering_activation, 'mom')
#steering_trun = fuzz.defuzz(x_steering_turn, steering_activation, 'centroid')

steering_turn_mild_level = fuzz.interp_membership(x_steering_turn, steering_turn_mild, steering_trun)
print("steering_turn_mild_level: " + str(steering_turn_mild_level))
steering_turn_sharp_level  = fuzz.interp_membership(x_steering_turn, steering_turn_sharp, steering_trun)
print("steering_turn_sharp_level: " + str(steering_turn_sharp_level))
steering_turn_very_sharp_level = fuzz.interp_membership(x_steering_turn, steering_turn_very_sharp, steering_trun)
print("steering_turn_very_sharp_level: " + str(steering_turn_very_sharp_level))

# if ST is MT, then S is FS
speed_fast_activation = np.fmin(steering_turn_mild_level, speed_fast)

# else if ST is SST, then S is MS
speed_medium_activation = np.fmin(steering_turn_sharp_level, speed_medium)

# else if ST is VSP, then S is SS
speed_slow_activation = np.fmin(steering_turn_very_sharp_level, speed_slow)

speed0 = np.zeros_like(x_speed)
# Visualize speed
fig2, ax5 = plt.subplots(figsize=(8, 3))

ax5.fill_between(x_speed, speed0, speed_fast_activation, facecolor='b', alpha=0.7)
ax5.plot(x_speed, speed_maximum, 'b', linewidth=0.5, linestyle='--', )
ax5.fill_between(x_speed, speed0, speed_medium_activation, facecolor='g', alpha=0.7)
ax5.plot(x_speed, speed_fast, 'g', linewidth=0.5, linestyle='--')
ax5.fill_between(x_speed, speed0, speed_slow_activation, facecolor='r', alpha=0.7)
ax5.plot(x_speed, speed_medium, 'r', linewidth=0.5, linestyle='--')
ax5.set_title('Output membership activity for speed')
# Turn off top/right axes
for ax in (ax5,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()



speed_activation = np.fmax(np.fmax(speed_fast_activation, speed_medium_activation), speed_slow_activation)
#speed = fuzz.defuzz(x_speed, speed_activation, 'mom')
speed = fuzz.defuzz(x_speed, speed_activation, 'mom')

steer_activation_plot = fuzz.interp_membership(x_steering_turn, steering_activation, steering_trun)
speed_activation_plot = fuzz.interp_membership(x_speed, speed_activation, speed)


fig3, ax6 = plt.subplots(figsize=(8, 3))
ax6.plot(x_steering_turn, steering_turn_mild, 'b', linewidth=0.5, linestyle='--', )
ax6.plot(x_steering_turn, steering_turn_sharp, 'g', linewidth=0.5, linestyle='--')
ax6.plot(x_steering_turn, steering_turn_very_sharp, 'r', linewidth=0.5, linestyle='--')
ax6.fill_between(x_steering_turn, steering0, steering_activation, facecolor='Orange', alpha=0.7)
ax6.plot([steering_trun, steering_trun], [0, steer_activation_plot], 'k', linewidth=1.5, alpha=0.9)
ax6.set_title('Output membership activity for steering turn and result (line)')
# Turn off top/right axes
for ax in (ax6,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()

fig4, ax7 = plt.subplots(figsize=(8, 3))
ax7.plot(x_speed, speed_slow, 'b', linewidth=0.5, linestyle='--', )
ax7.plot(x_speed, speed_medium, 'g', linewidth=0.5, linestyle='--')
ax7.plot(x_speed, speed_fast, 'r', linewidth=0.5, linestyle='--')
ax7.plot(x_speed, speed_maximum, 'r', linewidth=0.5, linestyle='--')
ax7.fill_between(x_speed, speed0, speed_activation, facecolor='Orange', alpha=0.7)
ax7.plot([speed, speed], [0, speed_activation_plot], 'k', linewidth=1.5, alpha=0.9)
ax7.set_title('Output membership activity for speed and result (line)')
for ax in (ax7,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
print("steering_turn output: ",steering_trun)
print("speed output: ",speed)
plt.show()






