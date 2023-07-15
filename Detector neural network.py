import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

def training_set(L=1, Lw=1, n=16, m=16, Sx=10, Sy=10, Amp=1, sigma=1): 
    # L/Lw = pixel length/width, n/m = amount of pixels lengthwise/widthwise, Sx/Sy = amount of pixels in one pixel
    #lengthwise/widthwise

    #reference grid
    gridx, gridstepx = np.linspace(-(L*(n-1)), L*n, 2*n, retstep = True)
    gridy, gridstepy = np.linspace(-(Lw*(m-1)), Lw*m, 2*m, retstep = True)
    print(gridx)

    strip = np.linspace(0, -(Lw*(m-1)), m) #0 to L lengthwise

    #centres
    centrex, xstep = np.linspace(0,L,Sx+1, retstep = True)
    centrey, ystep =np.linspace(Lw,0, Sy+1, retstep = True)
    centrex = centrex + xstep/2
    centrex = centrex[:-1]
    centrey = centrey + ystep/2
    centrey = centrey[:-1]

    GRIDX=2*n-1
    GRIDY=2*m-1
    GRID = np.zeros((Sy,Sx, GRIDY, GRIDX))
    STRIP = np.zeros((GRIDY))

    #creating centre coordinate data for each reading grid (functions as labels). coord in (x,y) format
    XY = np.meshgrid(centrey, centrex) 
    coord_grid = np.array(XY).transpose()
    labels = np.zeros((m*n,Sx,Sy, 2))
    number = 0
    for i in range(m-1, -1, -1):
        for j in range(0, n):
            XY2 = np.meshgrid(centrey + i, centrex + j)
            #XY2 = np.meshgrid(centrex + j, centrey + i)
            labels[number] = np.array(XY2).transpose()
            #print(labels[number])
            number += 1
    

    array_x = -1
    for P in centrex[: int(-np.floor(Sx/2))]:
        array_y = -1
        array_x += 1
        for Q in centrey[: int(-np.floor(Sy/2))]:
            #print(P,Q)
            array_y += 1
            x1 = -1
            gaussian = lambda x, y : Amp * np.exp(-0.5 * ((np.square(x-P)+np.square(y-Q))/np.square(sigma)))
            for i in gridx[:-1]:
                y1 = -1
                x1 += 1
                for j in gridy[:-1]:
                    pixval = integrate.dblquad(gaussian, j , j+gridstepy, lambda x: i, lambda x: i+gridstepx)[0] #Bottom left quadrant
                    GRID[array_y][array_x][y1][x1] = pixval
                    GRID[(-array_x-1)][array_y][(-x1-1)][y1] = pixval #Bottom right quadrant
                    GRID[(-array_y-1)][(-array_x-1)][(-y1-1)][(-x1-1)] = pixval #Top right quadrant
                    GRID[array_x][(-array_y-1)][x1][(-y1-1)] = pixval #Top left quadrant
                    y1 -= 1

    
    plt.imshow(GRID[9][9])
    plt.imsave("referencegrid.png", GRID[9][9])
    counter = 0 
    reading = np.zeros((n*m, Sy, Sx, m, n))
    for b in range(0, m):
        for a in range(0, n):
            reading[counter] = GRID[:, :,m-1-b:2*m-1-b,n-1-a: 2*n-1-a] #storing horizontally so first row filled before filling next row
            counter += 1
    return reading, labels

reading, labels = training_set(Sx=20, Sy=20)
#plt.imshow(reading[0][0]) #first index is for the square which will contain centres (reading[0] will be the top right)
                          #square. The second index is for the centre for which the gaussian is distributed. Specifying
                          #the first and second index will give the 2D detector array for a certain square and a certain
                          #centre in that square. The centres are given rowise in terms of index so the 15th centre
                          #for a pixel is the 2nd column and 5th row of the grid of centres in a pixel.



plt.imshow(np.flipud(reading[34][0][0]),origin = "lower")
print(labels[34][5][5])
######################################


#Gaussian calibration gain filter
import numpy as np
Sx = 20
Sy = 20
m = 16
n = 16

#Below code looks sus
gain = np.random.normal(1, 0.05, (10,16,16)) #why 10 and not 100? Have changed 10 to 100
training = np.zeros((np.shape(gain)[0]*np.shape(np.reshape(reading, (Sx*Sy*m*n, m, n)))[0], 16, 16))
counter = 0
for i in range(np.shape(gain)[0]):    
    for j in range(np.shape(np.reshape(reading, (Sx*Sy*m*n, m, n)))[0]):
        #training[counter] = np.multiply(np.reshape(reading, (Sx*Sy*m*n, m, n))[j], gain[i])
        training[counter] = np.multiply(np.reshape(reading, (Sx*Sy*m*n, m, n))[j], np.random.normal(16,16))
        counter += 1
        
print(np.shape(training))
training_2 = np.concatenate((np.reshape(reading, (Sx*Sy*m*n, m, n)), training), axis=0)
print(np.shape(training_2))
labels_2 = np.reshape(labels, (Sx*Sy*m*n, 2))
labels_2 = np.tile(labels_2, (11, 1))
print(np.shape(labels_2))



#ML model
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from sklearn.model_selection import train_test_split
from contextlib import redirect_stdout

data = np.reshape(training_2, (np.shape(training_2)[0], m, n, 1))
training_final, x_valid, labels_final, y_valid = train_test_split(data, labels_2, test_size=0.2, shuffle= True)
print(np.shape(training_final))
print(np.shape(x_valid))

print(len(tf.config.list_physical_devices("GPU")))

def neural(training, label, epochs, val_x, val_y, batch, save_as = "new", n=16, m=16, Sx=10, Sy=10, l1=10, l2=0, l3=0, l4=0):
    
        model = models.Sequential()
        model.add(layers.Conv2D(32, (2,2), activation = 'relu', input_shape = (m, n, 1)))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(64, (2,2), activation = 'relu'))
        model.add(layers.MaxPooling2D((2,2)))
        #model.add(layers.Conv2D(64, (2,2), activation = 'relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(l1, activation = 'relu'))
        model.add(layers.Dense(l2, activation = 'relu'))
        #model.add(layers.Dense(l3, activation = 'relu'))
        #model.add(layers.Dense(l4, activation = 'relu'))
        #model.add(layers.Dense(128, activation = 'relu'))
        #model.add(layers.Dense(0, activation = 'relu'))
        #model.add(layers.Dense(64, activation = 'relu'))
        #model.add(layers.Dense(16, activation = 'relu'))
        #model.add(layers.Dense(64, activation = 'relu'))
        #model.add(layers.Dropout(0.2, (64,)))
        model.add(layers.Dense(2))


        model.compile(optimizer = 'adam',
                      loss = 'mean_squared_error',
                      metrics = ['accuracy'])
        
        earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                            mode ="min", patience = 5, 
                                            restore_best_weights = True)

        #val = np.reshape(data_array, (reading, coord_grid, reading))

        history = model.fit(np.reshape(training_final, (np.shape(training_final)[0], m, n, 1)), labels_final, epochs = epochs, validation_data = (x_valid, y_valid), batch_size = batch)

        #Save model
        model.save(save_as + ".model")
        numpy_loss_history = np.array(history.history["val_loss"])
        np.savetxt(save_as + " loss.txt", numpy_loss_history, delimiter=",")
        
        with open(save_as +'.txt', 'w') as f:
            with redirect_stdout(f):
                model.summary()

#neural(training = reading, label = labels, epochs = 1, val_x = x_valid, val_y = y_valid, batch = 100, Sx=20, Sy=20)
i=20
j=20
neural(Sx=20, Sy=20, training = reading, label = labels, epochs = 30, val_x = x_valid, val_y = y_valid, batch = 50, save_as = str(i) +"_"+ str(j), l1 = i, l2=j)


#Testing model with individual data
def predictor(centrex = 8, centrey = 8, A=1, s=1):
#A is amplitude and s is standard deviation (sigma)    

    AllData = []
    gaussian = lambda x, y : A * np.exp(-0.5 * (np.square(x-centrex)+np.square(y-centrey))/np.square(s))
    x1 = np.linspace(0, 16, 16 + 1)
    y = np.linspace(0, 16, 16 + 1)
    pixarray = []
    for i in range(0, 16):
        pixrow = []
        for j in range(0, 16):
            pixval = integrate.dblquad(gaussian, y[i] , y[i+1], lambda x: x1[j], lambda x: x1[j+1])
            pixrow.append(pixval[0])
        pixarray.insert(0, pixrow)
    AllData.append(pixarray)


    AllData = np.reshape(np.asarray(AllData), (1,16,16,1))

    model = tf.keras.models.load_model("new.model")
    prediction = model.predict(AllData)
    return prediction

print(predictor(centrex = 7.15, centrey = 7.15))


import multiprocessing as mp
print(mp.cpu_count())
########################################

#Testing model with individual data
def predictor(centrey = 8, centrex = 8, A=1, s=1):
#A is amplitude and s is standard deviation (sigma)    

    AllData = []
    gaussian = lambda x, y : A * np.exp(-0.5 * (np.square(x-centrex)+np.square(y-centrey))/np.square(s))
    x1 = np.linspace(0, 16, 16 + 1)
    y = np.linspace(0, 16, 16 + 1)
    pixarray = []
    for i in range(0, 16):
        pixrow = []
        for j in range(0, 16):
            pixval = integrate.dblquad(gaussian, y[i] , y[i+1], lambda x: x1[j], lambda x: x1[j+1])
            pixrow.append(pixval[0])
        pixarray.insert(0, pixrow)
    AllData.append(pixarray)


    AllData = np.reshape(np.asarray(AllData), (1,16,16,1))

    model = tf.keras.models.load_model("20_20.model")
    prediction = model.predict(AllData)
    return prediction

print(predictor(centrex = 8.15, centrey = 8.15))
######################################

for i in range(0,17):
    for j in range(0,17):
        result = predictor(centrex=j, centrey=i)
        print([(result[0][0]-i, result[0][1]-j)])





