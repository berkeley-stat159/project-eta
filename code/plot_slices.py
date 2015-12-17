from utils.ols import *
import matplotlib.pyplot as plt 

all_betas = apply_ols_to_subject(16, 3, r_outliers = True, smooth = True)
betas_2d = average_betas(all_betas)
betas_4d = beta_2d_to_4d(betas_2d)


def grid_plot(betas_4d, gain_or_loss):
	if gain_or_loss == 'gain':
		coeff = 0
	elif gain_or_loss == 'loss':
		coeff = 1
	fig, axes = plt.subplots(4,4, sharey = True, sharex = True)
	#plot in grid 
	for i,j in zip(axes.reshape(-1),xrange(2,34,2)):
		x = i.imshow(betas_4d[:,:,j,coeff])
		i.set_title(str(j))
	fig.subplots_adjust(right=0.8)
	cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
	fig.colorbar(x, cax=cbar_ax)


grid_plot(betas_4d, 'loss')
plt.savefig('loss_fit.png')

grid_plot(betas_4d, 'gain')
plt.savefig('gain_fit.png')