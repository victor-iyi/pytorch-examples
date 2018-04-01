# NumPy and MatPlotLib.
import numpy as np
import matplotlib.pyplot as plt

# PyTorch.
import torch
from torch.autograd import Variable


# Common dataset class labels.
CLASS_LABELS = {
    # MNIST
    'mnist': ('Zero 0', 'One 1', 'Two 2', 'Three 3 ', 'Four 4',
              'Five 5', 'Six 6', 'Seven 7 ', 'Eight 8', 'Nine 9'),

    # CIFAR-10
    'cifar': ('Plane', 'Car', 'Bird', 'Cat', 'Deer',
              'Dog', 'Frog', 'Horse', 'Ship', 'Truck'),

    # Hymenoptera
    'hymenoptera': ('ants', 'bees'),
}


# Helper function to plot images and labels.
def imshow(images, labels, pred=None, **kwargs):
    """
    Plot images and associated labels. You can also show the labels and predictions
    side-by-side to visualize how your network is doing in terms of accurate
    predictions.

    === Parameters ===
        images: torch.Tensor
            Inputs (usually an image[s]) to be visualized. It must be a PyTorch Tensor
            or sub-classes like: FloatTensor, IntTensor, etc.

        labels: torch.
            Associated labels to the images. This is the (int) index of the label
            not one hot encoded labels. Must be a PyTorch tensor (just like the input).

        pred: (torch.Tensor, None), default None
            Network output or predicted labels for the input data.


    === Keyword Arguments ===
        normalize: 2D (list or array) or float, default 0.5
            Batch normalization value. To un-normalize the image, if the image
            has already been normalized. If not just set normalize to a falsy value.
            e.g normalize=False or normalize=None. Otherwise normalize takes in 2-D
            array or a single float value.

        smooth: bool, default True
            Get rid of pixelation and smooth images to make visualization pretty.

        classes: str, list, tuple, set or range. default str
            If provided a string, the string must exist as a key in the `utils.CLASS_LABELS`
            dictionary otherwise you can update the dictionary to suit your purpose. 
            Otherwise if classes is set to one of list, tuple, set or range, it uses it's
            value as the class label.

    === Raise ===
        TypeError:
            classes has to be one of `str`, `list`, `tuple`, `set` or `range`.
    """

    # Keyword arguments.
    normalize = kwargs.get('normalize', 0.5)
    smooth = kwargs.get('smooth', True)
    classes = kwargs.get('classes', 'hymenoptera')

    # Image dimensions.
    img_shape = images.size()
    img_size = img_shape[-1]
    img_channel = img_shape[-3] if len(img_shape) > 3 else 1
    img_batch = img_shape[0]

    # Get the class labels.
    if type(classes) == str:
        classes = CLASS_LABELS[classes.lower()]
    elif type(classes) in [list, range, tuple, set]:
        pass
    else:
        raise TypeError('`classes` has to be one of `str`, `list`, `tuple`, `set` or `range`.')

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

        # Normalize image.
        if normalize:
            try:
                if len(normalize) == 2:
                    # Retrieve mean and std_dev.
                    mean = np.array(normalize[0])
                    std_dev = np.array(normalize[1])
                    img = std_dev * img + mean
                else:
                    img = np.array(normalize) * img + np.array(normalize)
            except TypeError:
                img = np.array(normalize) * img + np.array(normalize)
        else:
            pass

        # Plot image.
        ax.imshow(img, interpolation=smooth, cmap=cmap)

        # Name of the true class.
        labels_name = classes[labels[i]]

        # Show true and predicted classes.
        if pred is None:
            xlabel = 'True: {}'.format(labels_name)
        else:
            # Name of the predicted class.
            pred_name = classes[pred[i]]

            xlabel = 'True: {}\nPred: {}'.format(labels_name, pred_name)

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Add center title to the figure.
    plt.suptitle('Showing ({}) images'.format(img_batch))

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# Visualization function to visualize dataset.
def visualize(data, **kwargs):
    """
    Visualize the first batch of a dataset.

    === Parameters ===
    data: torch.utils.data.DataLoader
        Dataset to be visualized.

    === Keyword Arguements ===
    See `utils.imshow`

    """

    # Iterate over the data.
    data_iter = iter(data)

    # Unpack images and labels.
    images, labels = data_iter.next()

    # Free up memory
    del data_iter

    # Call to helper function for plotting images.
    imshow(images, labels=labels, **kwargs)


# Accuracy function to compute accuracy and display predictions
def accuracy(net, data, **kwargs):
    """
    Computes each class accuracy on a dataset.

    Parameters:
    net: nn.Module
        The instance of a nn.Module subclass. Or a PyTorch
        neural network class.
    data: torch.utils.data.dataloader.DataLoader
        Data where accuracy is computed. This must
        be a DataLoader object.

    Keyword arguments:
    visualize: bool, default True
        Visualize the first `batch_size` of the data.

    classes: enum('cifar', 'mnist', 'hymenoptera') default 'hymenoptera'
        Popular dataset classnames. To add your own
        custom class, update the utils.CLASS_LABELS
        dictionary to reflect your own data classes.

    logging: bool, default False (`or` not ret).
        Pretty-print the computed accuracy, accuracy.

    ret: bool, default False
        Return the value after computing accuracy.
        If set to True, the output will not be printed,
        but you can set logging=True for pretty-print.

    === Example ===

        >>> accuracy(net=myNeuralNet, data=testset, logging=True)
        Accuracy of plane 	 = 74.60%
        Accuracy of car 	 = 62.70%
        Accuracy of bird 	 = 61.50%
        Accuracy of cat 	 = 61.40%
        Accuracy of deer 	 = 52.00%
        Accuracy of dog 	 = 51.20%
        Accuracy of frog 	 = 36.80%
        Accuracy of horse 	 = 33.40%
        Accuracy of ship 	 = 11.30%
        Accuracy of truck 	 = 9.30%
        –––––––––––––––––––––––––––––––––––
        Overall         	 = 45.42%
        –––––––––––––––––––––––––––––––––––

        >>> acc = accuracy(net=myNeuralNet, ret=True)
        >>> print(acc)

         0.7460
         0.6270
         0.6150
         0.6140
         0.5200
         0.5120
         0.3680
         0.3340
         0.1130
         0.0930
        [torch.FloatTensor of size 10]

    === Raise ===
        TypeError:
            classes has to be one of `str`, `list`, `tuple`, `set` or `range`.

    === Return ===
        accuracy: torch.Tensor
            Computed accuracy for each classes is returned in descending order,
            only if `ret=True`. Otherwise nothing is being returned.
    """
    # Keyword argument.
    visualize = kwargs.get('visualize', True)
    classes = kwargs.get('classes', 'hymenoptera')
    logging = kwargs.get('logging', False)
    ret = kwargs.get('ret', False)

    # Get the class labels.
    if type(classes) == str:
        classes = CLASS_LABELS[classes.lower()]
    elif type(classes) in [list, tuple, set, range]:
        pass
    else:
        raise TypeError('`classes` has to be one of `str`, `list`, `tuple`, `set` or `range`.')

    # Visualize the data.
    if visualize:
        # Get first batch.
        data_iter = iter(data)
        inputs, labels = data_iter.next()

        # Run the input through the network.
        outputs = net(Variable(inputs))
        _, pred = torch.max(outputs.data, dim=1)

        # Visualize images and their predictions.
        imshow(images=inputs, labels=labels, pred=pred, classes=classes)

        # Free memory.
        del data_iter, inputs, labels, pred

    # Each index in the `correct_class` stores the correct
    # classification for that class; while `total_class`
    # stores the total number of times we go through the class.
    correct_class = torch.zeros(10)
    total_class = torch.zeros(10)

    # Loop through all dataset one batch at a time.
    for (images, labels) in data:

        # Pass the images through the network.
        outputs = net(Variable(images))

        # Take the index of the maximum scores
        # returned by the network.
        _, pred = torch.max(outputs.data, dim=1)

        # Where the pred equals the labels will
        # return 1; and 0 otherwise.
        correct = (pred == labels).squeeze()

        # Loop through the batch labels.
        for i, label in enumerate(labels):
            # Add on the correct predictions
            # and total for the current label.
            correct_class[label] += correct[i]
            total_class[label] += 1

    # Calculate accuracy and sort in descending order.
    accuracy = correct_class / total_class
    accuracy, _ = torch.sort(accuracy, descending=True)

    # Print output if ret=False or logging=True.
    if not ret or logging:
        # Accuracies for each classes
        for i, acc in enumerate(accuracy):
            print('Accuracy of {} \t = {:.2%}'.format(classes[i], acc))

        # Overall accuracy
        print(35 * '–')
        print('Overall accuracy \t = {:.2%}'.format(accuracy.mean()))
        print(35 * '–')

    if ret:
        return accuracy
