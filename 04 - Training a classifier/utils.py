import numpy as np
import matplotlib.pyplot as plt


# Common dataset class labels.
CLASS_LABELS = {
    'mnist': range(0, 11),
    'cifar': ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
    'cifar10': ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
}


# Helper function to plot images and labels.
def imshow(images, labels, pred=None, **kwargs):
    
    # Keyword arguments.
    normalize = kwargs.get('normalize', 0.5)
    smooth = kwargs.get('smooth', True)
    classes = kwargs.get('classes', 'cifar')
    
    # Image dimensions.
    img_shape = images.size()
    img_size = img_shape[-1]
    img_channel = img_shape[-3]
    img_batch = img_shape[0] if len(img_shape) > 3 else 1

    print(f'img_size = {img_size}\timg_batch={img_batch}'
          f'\timg_channel={img_channel}')
    
    # Get the class labels.
    if type(classes) == str:
        classes = CLASS_LABELS[classes.lower()]
    elif type(classes) == list:
        classes = classes
    else:
        raise TypeError('classes has to be a `str` or `list`')
    
    # Normalize image.
    images = images / 2 + normalize if normalize else images
    
    # Create figure with sub-plots.
    fig, axes = plt.subplots(4, 4)

    # Adjust vertical spacing if we need to print ensemble and best-net.
    wspace, hspace = 0.2, 0.8 if pred is not None else 0.4
    fig.subplots_adjust(hspace=hspace, wspace=wspace)

    for i, ax in enumerate(axes.flat):
        # cmap type.
        cmap = 'Greys' if img_channel == 1 else None
        # Interpolation type.
        smooth = 'spline16' if smooth else 'nearest'

        # Reshape image based on channel.
        if img_channel == 1:
            img = images[i].view(img_size, img_size).numpy()
        else:
            img = np.transpose(images[i].numpy(), (1, 2, 0))

        # Plot image.
        ax.imshow(img, interpolation=smooth, cmap=cmap)
            
        # Name of the true class.
        labels_name = classes[labels[i]]

        # Show true and predicted classes.
        if pred is None:
            xlabel = f'True: {labels_name}'
        else:
            # Name of the predicted class.
            pred_name = classes[pred[i]]
            
            xlabel = f'True: {labels_name}\nPred: {pred_name}'

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# Visualization function to visualize dataset.
def visualize(data, **kwargs):
    
    # Iterate over the data.
    data_iter = iter(data)
    
    # Unpack images and labels.
    images, labels = data_iter.next()
    
    # Free up memory
    del data_iter
    
    # Call to helper function for plotting images.
    imshow(images, labels=labels, **kwargs)


