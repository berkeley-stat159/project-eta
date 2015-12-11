from utils.ols import *



########before preproccessing 

betas_before_process = apply_ols_to_subject(16, 3, r_outliers = False, smooth = False)

##average across all runs 

avg_before = average_betas(betas_before_process)

## shape to 4d 
avg_before = beta_2d_to_4d(betas_2d)



plt.imshow(avg_before[:, :, 16, 0], interpolation='nearest', cmap='gray')
plt.title('Middle Slice Beta(Gain) Before smoothing')

plt.savefig("Before_preproccessing_gain")

plt.imshow(avg_before[:, :, 16, 1], interpolation='nearest', cmap='gray')
plt.title('Middle Slice Beta(Loss) Before smoothing')

plt.savefig("Before_preproccessing_loss")



###### After Preproccessing 

betas_after_process = apply_ols_to_subject(16, 3, r_outliers = True, smooth = True)

##average across all runs 

avg_after = average_betas(betas_after_process)

## shape to 4d 
avg_after = beta_2d_to_4d(betas_2d)



plt.imshow(avg_after[:, :, 16, 0], interpolation='nearest', cmap='gray')
plt.title('Middle Slice Beta(Gain) After Smoothing')

plt.savefig("After_preproccessing_gain")


plt.imshow(avg_after[:, :, 16, 1], interpolation='nearest', cmap='gray')
plt.title('Middle Slice Beta(Loss) After Smoothing')

plt.savefig("Before_preproccessing_loss")

