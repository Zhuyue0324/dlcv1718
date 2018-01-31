print("Start training")

needNewEpoch= True
lastTrainLoss=10
lastValLoss=10
epoch=0
width=2048
slice_width=32
num_slices = width//slice_width    #added   "Integer Division"
printfrequence=1024  
while needNewEpoch:       
    #--------------------------------------training period---------------------------------------
    running_loss = 0.0
    epochloss = 0.0
    numsample=0
     
    net.train()
    
    print ("Slice width:  %.5d Number of Slices:  %.5d" %(width, num_slices))   #added  
    
    for inputs, labels in trainloader:  
    
        for i in range (0, num_slices):   ### added         
        
            inputs_temp = inputs[:, :, :, i*slice_width : (i+1) * slice_width]   ### added
            labels_temp = labels[:, :, i*slice_width : (i+1) * slice_width]   ### added      
            
            if useGPU:   
                inputs_temp = inputs_temp.cuda()    ### changed
                labels_temp = labels_temp.cuda()    ### changed
            
            inputs_temp, labels_temp = Variable(inputs_temp), Variable(labels_temp)     ### changed       
            # zero the parameter gradients
            optimizer.zero_grad()      
            # forward + backward + optimize
            outputs = net(inputs_temp)                  ### changed
            loss = criterion(outputs, labels_temp)      ### changed
            loss.backward()
            optimizer.step()            
            # print statistics
            running_loss += loss.data[0]
            epochloss+=loss.data[0]
            numsample += batchsize
            if numsample % printfrequence == 0: #printfrequence-1: 
                print('[%d, %5d] loss: %.5f' % (epoch+1, numsample, running_loss / printfrequence))
                running_loss = 0.0

    newTrainLoss = epochloss/(nbtrain*num_slices)   #changed
    print('The average loss of epoch ', epoch+1, ' is ', newTrainLoss)
    torch.save(net.state_dict(),weightpath)
    #--------------------------------------validation period---------------------------------------
    meanCorrectProba = 0.0
    epochloss = 0.0
    numsample=0
    printfrequence=256   
    net.eval()
    print ("Slice width:  %.5d Number of Slices:  %.5d" %(width, num_slices))   #added 
    
    for inputs, labels in valloader:

        for i in range (0, num_slices): #added

            inputs_temp = inputs[:, :, :, i*slice_width : (i+1) * slice_width]   ### added
            labels_temp = labels[:, :, i*slice_width : (i+1) * slice_width]   ### added    
            
            if useGPU:
                inputs_temp = inputs_temp.cuda()    #changed
                labels_temp = labels_temp.cuda()    #changed    
                
            inputs_temp, labels_temp = Variable(inputs_temp), Variable(labels_temp) #changed
            outputs=net(inputs_temp)    #changed
            loss = criterion(outputs, labels_temp) #changed
            meanProbability=np.exp(-loss.data[0])
            epochloss += loss.data[0]
            meanCorrectProba += meanProbability
            numsample += batchsize
       
    newValLoss = epochloss / (nbval*num_slices) 
    print('The average validation loss is ', newValLoss)
    print('The average correctness of the validation data is ', meanCorrectProba/(nbval*num_slices)*100, '%')   #changed
    #--------------------------------------evaluate the necessity of a new epoch---------------------------------------
    if (lastValLoss-newValLoss<0.01) and (lastTrainLoss-newTrainLoss<0.01):
        needNewEpoch=False
    else:
        lastLoss=newValLoss 
        epoch=epoch+1

print("End training")
