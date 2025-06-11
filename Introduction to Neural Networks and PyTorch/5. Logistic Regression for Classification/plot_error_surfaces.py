import numpy as np
import matplotlib.pyplot as plt


# Class for plotting error surfaces and function for plotting the results of training a model
class plot_error_surfaces(object):
    # Constructor
    def __init__(self, w_range, b_range, X, Y, n_samples=30, go=True):
        W = np.linspace(-w_range, w_range, n_samples)  # Create range for weights
        B = np.linspace(-b_range, b_range, n_samples)  # Create range for biases
        w, b = np.meshgrid(W, B)  # Create meshgrid for parameter space
        Z = np.zeros((30, 30))  # Initialize loss surface
        count1 = 0
        self.y = Y.numpy()  # Convert targets to numpy
        self.x = X.numpy()  # Convert inputs to numpy
        for w1, b1 in zip(w, b):  # Loop over meshgrid rows
            count2 = 0
            for w2, b2 in zip(w1, b1):  # Loop over meshgrid columns
                # Compute mean squared error for logistic regression output
                Z[count1, count2] = np.mean((self.y - (1 / (1 + np.exp(-1 * w2 * self.x - b2)))) ** 2)
                count2 += 1
            count1 += 1
        self.Z = Z  # Store loss surface
        self.w = w  # Store weight meshgrid
        self.b = b  # Store bias meshgrid
        self.W = []  # List to store weights during training
        self.B = []  # List to store biases during training
        self.LOSS = []  # List to store loss values during training
        self.n = 0  # Counter for iterations
        if go == True:
            plt.figure()
            plt.figure(figsize=(7.5, 5))
            # 3D surface plot of the loss surface
            plt.axes(projection='3d').plot_surface(self.w, self.b, self.Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
            plt.title('Loss Surface')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.show()
            plt.figure()
            # Contour plot of the loss surface
            plt.title('Loss Surface Contour')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.contour(self.w, self.b, self.Z)
            plt.show()

    # Setter to record parameters and loss at each step
    def set_para_loss(self, model, loss):
        self.n = self.n + 1  # Increment iteration counter
        self.W.append(list(model.parameters())[0].item())  # Store current weight
        self.B.append(list(model.parameters())[1].item())  # Store current bias
        self.LOSS.append(loss)  # Store current loss

    # Plot final parameter trajectory on loss surface
    def final_plot(self):
        ax = plt.axes(projection='3d')
        ax.plot_wireframe(self.w, self.b, self.Z)  # Wireframe of loss surface
        ax.scatter(self.W, self.B, self.LOSS, c='r', marker='x', s=200, alpha=1)  # Parameter trajectory
        plt.figure()
        plt.contour(self.w, self.b, self.Z)  # Contour plot
        plt.scatter(self.W, self.B, c='r', marker='x')  # Parameter trajectory
        plt.xlabel('w')
        plt.ylabel('b')
        plt.show()

    # Plot predictions and parameter trajectory at current step
    def plot_ps(self):
        plt.subplot(121)
        plt.ylim
        plt.plot(self.x, self.y, 'ro', label="training points")  # Plot training data
        plt.plot(self.x, self.W[-1] * self.x + self.B[-1], label="estimated line")  # Plot estimated line
        plt.plot(self.x, 1 / (1 + np.exp(-1 * (self.W[-1] * self.x + self.B[-1]))), label='sigmoid')  # Plot sigmoid
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim((-0.1, 2))
        plt.title('Data Space Iteration: ' + str(self.n))
        plt.show()
        plt.subplot(122)
        plt.contour(self.w, self.b, self.Z)  # Contour plot
        plt.scatter(self.W, self.B, c='r', marker='x')  # Parameter trajectory
        plt.title('Loss Surface Contour Iteration' + str(self.n))
        plt.xlabel('w')
        plt.ylabel('b')

# Plot the diagram

def PlotStuff(X, Y, model, epoch, leg=True):
    plt.plot(X.numpy(), model(X).detach().numpy(), label=('epoch ' + str(epoch)))  # Plot model predictions
    plt.plot(X.numpy(), Y.numpy(), 'r')  # Plot true values
    if leg == True:
        plt.legend()  # Show legend if requested
    else:
        pass