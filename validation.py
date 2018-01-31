meanCorrectProba = 0.0
numsample=0
net.eval()
for inputs, labels in valloader:
    for i in range (0, num_slices): #added

        inputs_temp = inputs[:, :, :, i*slice_width : (i+1) * slice_width]   ### added
        labels_temp = labels[:, :, i*slice_width : (i+1) * slice_width]   ### added
        if useGPU:
            inputs_temp = inputs_temp.cuda()
            labels_temp = labels_temp.cuda()
        inputs_temp, labels_temp = Variable(inputs_temp), Variable(labels_temp)
        outputs=net(inputs_temp)
        loss = criterion(outputs, labels_temp)
        meanProbability=np.exp(-loss.data[0])
        
    # print statistics
        meanCorrectProba += meanProbability
        numsample += batchsize
        
#    if numsample % printfrequence == 0: #printfrequence-1: 
#        print('[%d, %5d] loss: %.5f' % (epoch+1, numsample, running_loss / printfrequence))
#        running_loss = 0.0
print('The average correctness of the validation data is ', meanCorrectProba/(nbval*num_slices)*100, '%')
