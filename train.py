import math

import torch


from model import *
from torch_helper import *

import matplotlib.pyplot as plt
import numpy as np


def train(args, x_train, y_train, x_test, y_test, colours, model_mode=None, model=None):
    # Set the maximum number of threads to prevent crash in Teaching Labs
    #####################################################################################
    # TODO: Implement this function to train model and consider the below items         #
    # 0. read the utils file and use 'process' and 'get_rgb_cat' to get x and y for     #
    #    test and train dataset                                                         #
    # 1. Create train and test data loaders with respect to some hyper-parameters       #
    # 2. Get an instance of your 'model_mode' based on 'model_mode==base' or            #
    #    'model_mode==U-Net'.                                                           #
    # 3. Define an appropriate loss function (cross entropy loss)                       #
    # 4. Define an optimizers with proper hyper-parameters such as (learning_rate, ...).#
    # 5. Implement the main loop function with n_epochs iterations which the learning   #
    #    and evaluation process occurred there.                                         #
    # 6. Save the model weights                                                         #
    # Hint: Modify the predicted output form the model, to use loss function in step 3  #
    #####################################################################################
    """
    Train the model
    
    Args:
     model_mode: String
    Returns:
      model: trained model
    """
    torch.set_num_threads(5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available())
    # Numpy random seed
    np.random.seed(args.seed)

    # Save directory
    save_dir = "outputs/" + args.experiment_name

    print("Transforming data...")
    # Get X(grayscale images) and Y(the nearest Color to each pixel based on given color dictionary)
    train_rgb, train_grey = process(x_train, y_train, downsize_input=args.downsize_input, category_id=args.category_id)
    train_rgb_cat = rgb2label(train_rgb, colours, args.batch_size)
    test_rgb, test_grey = process(x_test, y_test, downsize_input=args.downsize_input, category_id=args.category_id)
    test_rgb_cat = rgb2label(test_rgb, colours, args.batch_size)

    # LOAD THE MODEL
    ##############################################################################################
    #                                            YOUR CODE                                       #
    ##############################################################################################

    num_in_channels = 1 if not args.downsize_input else 3


    if args.model == "Base":
        model = BaseModel(args.kernel, args.num_filters, 24, num_in_channels).to(device)
    elif args.model == "UNet":
        model = CustomUNET(args.kernel, args.num_filters, 24, num_in_channels).to(device)
    elif args.model == "Residual":
        model = CustomUNETResidual(args.kernel, args.num_filters, 24, num_in_channels).to(device)
    elif args.model == "UNET":
        model = UNET(args.kernel, args.num_filters, 24, ).to(device)

    # LOSS FUNCTION and Optimizer
    ##############################################################################################
    #                                            YOUR CODE                                       #
    ##############################################################################################
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)

    print("Beginning training ...")
    if args.gpu: model.cuda()



    # Create the outputs' folder if not created already
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_losses = []
    test_losses  = []
    valid_losses = []

    train_accs   = []
    test_accs   = []
    valid_accs   = []

    early_stopping_threshhold = 5
    previous_loss = math.inf
    trigger = 0
    # Training loop
    for epoch in range(args.epochs):
        correct_train = 0
        total_train = 0
        # Train the Model
        model.train()  # Change model to 'train' mode
        losses = []
        for i, (xs, ys) in enumerate(get_batch(train_grey,
                                               train_rgb_cat,
                                               args.batch_size)):
            # Convert numpy array to pytorch tensors
            images, labels = get_torch_vars(xs, ys, args.gpu)
            images, labels = images.to(device), labels.to(device)

            # Forward + Backward + Optimize
            ##############################################################################################
            #                                            YOUR CODE                                       #
            ##############################################################################################
            outputs = model(images)

            # Calculate and Print training loss for each epoch
            ##############################################################################################
            #                                            YOUR CODE                                       #
            ##############################################################################################
            train_loss = compute_loss(criterion,
                                      outputs,
                                      labels,
                                      batch_size=args.batch_size,
                                      num_colours=np.shape(colours)[0])
            optimizer.zero_grad()
            train_loss.backward()
            # Evaluate the model
            ##############################################################################################
            #                                            YOUR CODE                                       #
            ##############################################################################################
            optimizer.step()

            losses.append(train_loss.data.item())

            # Calculate and Print (validation loss, validation accuracy) for each epoch
            ##############################################################################################
            #                                            YOUR CODE                                       #
            ##############################################################################################
            _, predicted_train = torch.max(outputs.data, 1, keepdim=True)

            total_train += labels.size(0) * 32 * 32
            correct_train += (predicted_train == labels.data).sum()


        train_loss = np.mean(losses)
        train_acc = 100 * correct_train / total_train

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        losses = []
        with torch.no_grad():
            total_test = correct_test = 0
            for images, labels in get_batch(test_grey,
                                            test_rgb_cat,
                                            args.batch_size):
                images, labels = get_torch_vars(xs, ys, args.gpu)
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                test_loss = compute_loss(criterion,outputs,labels,batch_size=args.batch_size,num_colours=np.shape(colours)[0])
                losses.append(test_loss.data.item())
                _, predicted_test = torch.max(outputs.data, 1, keepdim=True)
                total_test += labels.size(0) * 32 * 32
                correct_test += (predicted_test == labels.data).sum()

        test_loss = np.mean(losses)
        test_acc = 100 * correct_test / total_test
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
        val_loss, val_acc = run_validation_step(model,
                                                criterion,
                                                test_grey,
                                                test_rgb_cat,
                                                args.batch_size,
                                                colours,
                                                save_dir + '/test_%d.png' % epoch,
                                                args.visualize,
                                                args.downsize_input)

        valid_losses.append(val_loss)
        valid_accs.append(val_acc)

        if val_loss > previous_loss:
            trigger += 1
        else:
            trigger=0
        if trigger > early_stopping_threshhold:
            break

        previous_loss = val_loss

        # Calculate and Print training loss for each epoch
        ##############################################################################################
        #                                            YOUR CODE                                       #
        ##############################################################################################

        print('Epoch [%d/%d], Train Loss: %.2f, Train Acc: %.1f%% ' % (
            epoch + 1, args.epochs, train_loss, train_acc))

        print('Epoch [%d/%d], Test Loss: %.2f, Test Acc: %.1f%% ' % (
            epoch + 1, args.epochs, test_loss, test_acc))

        # Calculate and Print (validation loss, validation accuracy) for each epoch
        ##############################################################################################
        #                                            YOUR CODE                                       #
        ##############################################################################################

        print('Epoch [%d/%d], Val Loss: %.2f, Val Acc: %.1f%% \n\n' % (
            epoch + 1, args.epochs, val_loss, val_acc))

        if args.plot:
            plt.figure(0)
            plot(xs, ys, predicted_train.cpu().numpy(), colours,
                 save_dir + '/train_%d.png' % epoch,
                 args.visualize,
                 args.downsize_input)
            plt.title("train")

        if args.plot:
            plt.figure(2)
            plot(xs, ys, predicted_test.cpu().numpy(), colours,
                 save_dir + '/test_%d.png' % epoch,
                 args.visualize,
                 args.downsize_input)
            plt.title("test")




    print(train_losses)
    # Plot training-validation curve
    plt.figure()
    plt.plot(train_losses, "ro-", label="Train")
    plt.plot(test_losses, "bo-", label="Test")
    plt.plot(valid_losses, "go-", label="Validation")
    plt.legend()
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.savefig(save_dir + "/training_curve.png")





    if args.checkpoint:
        print('Saving model...')
        torch.save(model.state_dict(), args.checkpoint)

    return model
