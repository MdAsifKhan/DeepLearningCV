from torchvision import utils
import matplotlib.pyplot as plt
import numpy as np

def vistensor(tensor, ch=0, nrow=20, padding=1): 
    '''
    https://github.com/pedrodiamel/nettutorial/blob/master/pytorch/pytorch_visualization.ipynb

    ''' 
    
    n,c,w,h = tensor.shape
    if c != 3:
      tensor = tensor.view(n*c,-1,w,h )
      tensor = tensor[:,ch,:,:].unsqueeze(dim=1)
        
    rows = np.min((tensor.shape[0]//nrow + 1, 64 ) )    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure(figsize=(nrow,rows) )
    plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)))



def plot_img(tensor, num_cols=10):
	num_kernels = tensor.shape[0]
	num_rows = 1+ num_kernels // num_cols
	fig = plt.figure(figsize=(num_cols,num_rows))
	for i in range(num_kernels):
		ax1 = fig.add_subplot(num_rows,num_cols,i+1)
		ax1.imshow(tensor[i], cmap='gray')
		ax1.axis('off')
		ax1.set_xticklabels([])
		ax1.set_yticklabels([])

	plt.subplots_adjust(wspace=0.1, hspace=0.1)
	return plt
